"""Barcode pipelines: learned detector + zxing-cpp decode on crops, or zxing-cpp on the whole frame.

  detect: JPEG decode -> letterbox -> TensorRT oriented detector -> rotated NMS
          -> deskewed crop from the full-res frame -> zxing-cpp per crop
  zxing:  JPEG decode -> grayscale -> zxing-cpp over the whole frame (the classic CPU scanner)

zxing-cpp's 1D and PDF417 readers only scan rows and columns, so on the whole frame they miss
codes tilted away from 0/90/180/270 degrees. The oriented detector supplies each code's angle
and the crop is rotated upright first, which is what makes 1D reading omnidirectional here.
"""
import html
import json
import re
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from . import yolo
from .jpeg import CpuJpeg, make_decoder
from .pipeline import Clock

F = zxingcpp.BarcodeFormat
GS1_HRI = re.compile(r"(\(\d{2,4}\)[^()]+)+")
# Detector classes, matching host/make_barcodes.py: 0 = 1D (linear), 1 = 2D (matrix + stacked).
CLASS_FORMATS = [F.AllLinear, F.AllMatrix | F.PDF417 | F.MicroPDF417 | F.CompactPDF417]


def load_frames(root):
    """Test set written by host/make_barcodes.py: [{jpeg, barcodes: [{format, text, polygon, cls}]}]."""
    root = Path(root)
    frames = []
    for line in (root / "gt.jsonl").read_text().splitlines():
        g = json.loads(line)
        g["jpeg"] = (root / g["image"]).read_bytes()
        frames.append(g)
    return frames


def _rect(rbox):
    cx, cy, w, h, a = (float(v) for v in rbox)
    return (cx, cy), (w, h), float(np.degrees(a))


def _riou(a, b):
    """IoU of two cv2 RotatedRects."""
    kind, pts = cv2.rotatedRectangleIntersection(a, b)
    if kind == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    inter = cv2.contourArea(pts)
    union = a[1][0] * a[1][1] + b[1][0] * b[1][1] - inter
    return inter / union if union > 0 else 0.0


def canon(text):
    """Comparable form of a barcode string. Annotated real-photo sets (BarBeR) write Code 39 with its
    '*' start/stop characters and UPC-A with or without the leading 0 that zxing-cpp adds when it
    reports the code as EAN-13; GS1 strings may carry FNC1 separators or (AI) brackets, and some
    annotations are HTML-escaped. Applied to both ground truth and
    reads, so synthetic sets (whose ground truth is zxing's own read) score exactly as before."""
    t = html.unescape(text.strip()).replace("\x1d", "")
    if len(t) > 2 and t[0] == t[-1] == "*":
        t = t[1:-1]
    if GS1_HRI.fullmatch(t):  # "(90)17699": zxing's human-readable GS1 form of "9017699"
        t = t.replace("(", "").replace(")", "")
    return (t.lstrip("0") or "0") if t.isdigit() else t


def score(frames, outputs):
    """Decode rate against the ground-truth strings, plus detection recall when boxes exist."""
    n_gt = decoded = detected = misreads = n_boxes = 0
    by_format = {}
    has_boxes = any(o.get("boxes") is not None for o in outputs)
    for frame, out in zip(frames, outputs):
        texts = [canon(d["text"]) for d in out["decodes"]]
        truth = {canon(b["text"]) for b in frame["barcodes"]}
        misreads += sum(t not in truth for t in set(texts))
        for b in frame["barcodes"]:
            n_gt += 1
            ok = canon(b["text"]) in texts
            decoded += ok
            f = by_format.setdefault(b["format"], [0, 0])
            f[0] += ok
            f[1] += 1
            if has_boxes:
                gt = cv2.minAreaRect(np.float32(b["polygon"]))
                detected += any(_riou(gt, _rect(bx)) >= 0.5 for bx in out["boxes"])
        if has_boxes:
            n_boxes += len(out["boxes"])
    acc = {"barcodes": n_gt, "decode_rate": decoded / n_gt, "misreads": misreads,
           "decode_rate_by_format": {k: v[0] / v[1] for k, v in sorted(by_format.items())}}
    if has_boxes:
        acc["detect_recall"] = detected / n_gt
        acc["boxes_per_frame"] = n_boxes / len(frames)
    return acc


class ZxingPipeline:
    stages = ["jpeg_decode", "decode"]

    def __init__(self, try_downscale=True, jpeg=None, jpeg_name="cpu"):
        self.name = "barcode"
        self.try_downscale = try_downscale
        self.jpeg = jpeg or CpuJpeg()
        self.config = {"mode": "zxing", "jpeg": jpeg_name, "detector": None, "precision": None, "input_size": None}

    def process(self, jpeg, times):
        clock = Clock(times)
        with clock("jpeg_decode"):
            gray = self.jpeg.decode(jpeg, gray=True)
        with clock("decode"):
            res = zxingcpp.read_barcodes(gray, try_downscale=self.try_downscale)
        return {"boxes": None, "decodes": [{"text": r.text, "format": r.format.name} for r in res]}

    def clone(self):
        return ZxingPipeline(self.try_downscale, make_decoder(self.config["jpeg"]), self.config["jpeg"])

    def score(self, frames, outputs):
        return score(frames, outputs)

    def close(self):
        self.jpeg.close()


class DetectPipeline:
    stages = ["jpeg_decode", "preprocess", "infer", "postprocess", "decode"]

    def __init__(self, runner, onnx_name, precision, input_size, conf=0.25, margin=0.15, int8_calibrated=None,
                 jpeg=None, jpeg_name="cpu", prep="cpu"):
        self.name = "barcode"
        self.runner = runner
        self.input = runner.inputs[0].host
        self.output = runner.outputs[0].host
        self.conf, self.margin = conf, margin
        self.jpeg = jpeg or CpuJpeg()
        # prep="gpu": the letterbox runs as one CUDA kernel on the GPU (gpuprep), bit-exact
        # with yolo.letterbox_into at these sizes.
        self.gprep = None
        if prep == "gpu":
            from .gpuprep import GpuLetterbox
            self.gprep = GpuLetterbox(runner.inputs[0], runner.stream, yolo.lut(), rgb=True, center=True, pad=114)
        self._kw = dict(onnx_name=onnx_name, precision=precision, input_size=input_size, conf=conf, margin=margin,
                        int8_calibrated=int8_calibrated, jpeg_name=jpeg_name, prep=prep)
        self.config = {"mode": "detect", "jpeg": jpeg_name, "prep": prep, "detector": onnx_name, "precision": precision,
                       "input_size": input_size, "int8_calibrated": int8_calibrated}
        self.gpu_ms = []

    def clone(self):
        """An independent copy (own engine context, stream, buffers, decoder) for another worker."""
        from .runner import TrtRunner

        c = DetectPipeline(TrtRunner(self.runner.engine_file), jpeg=make_decoder(self.config["jpeg"]), **self._kw)
        c.config = dict(self.config)
        return c

    def process(self, jpeg, times):
        gp = self.gprep
        with Clock(times)("jpeg_decode"):
            img = self.jpeg.decode(jpeg, out=gp.frame if gp else None)
        return self.process_image(img, times)

    def process_image(self, img, times):
        """Everything after JPEG decode, on a BGR frame (in gprep.frame when prep is gpu)."""
        clock = Clock(times)
        gp = self.gprep
        with clock("preprocess"):
            if gp:
                *lb, times["preprocess_gpu"] = gp(img)
            else:
                lb = yolo.letterbox_into(img, self.input)
        with clock("infer"):
            gpu_ms, _ = self.runner.infer(upload=gp is None)
        times["infer_gpu"] = gpu_ms
        with clock("postprocess"):
            boxes, scores, cls = yolo.postprocess_obb(self.output, *lb, conf=self.conf)
        with clock("decode"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            decodes = []
            for (cx, cy, w, h, a), c in zip(boxes.tolist(), cls):
                # Rotate the box upright and cut it out with a quiet-zone margin, in one warp.
                cw, ch = int(w * (1 + 2 * self.margin) + 16), int(h * (1 + 2 * self.margin) + 16)
                M = cv2.getRotationMatrix2D((cx, cy), np.degrees(a), 1.0)
                M[:, 2] += (cw / 2 - cx, ch / 2 - cy)
                crop = cv2.warpAffine(gray, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=255)
                res = zxingcpp.read_barcodes(crop, formats=CLASS_FORMATS[int(c)])
                if not res:  # wrong 1D/2D class guess: retry with every format
                    res = zxingcpp.read_barcodes(crop)
                decodes += [{"text": r.text, "format": r.format.name} for r in res]
        return {"boxes": boxes, "scores": scores, "cls": cls, "decodes": decodes}

    def score(self, frames, outputs):
        return score(frames, outputs)

    def close(self):
        if self.gprep:
            self.gprep.close()
        self.runner.close()
        self.jpeg.close()
