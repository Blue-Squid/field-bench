"""Several models per frame: barcode and OCR on every frame, the "handheld assistant" case.

A handheld that doesn't know what the user is pointing at runs every reader on each frame:

  jpeg_decode (once) -> barcode branch: letterbox -> YOLO11n-OBB -> NMS -> deskewed crops -> zxing-cpp
                     -> ocr branch:     det_pre -> PP-OCRv5 det -> DB post -> line crops -> rec -> CTC

order="sequential" runs the barcode branch, then the OCR branch. order="concurrent" runs them
in two threads, each with its own CUDA stream and TensorRT context, so one branch's CPU work
(zxing, DB postprocess, crops) overlaps the other's GPU work. With GPU preprocessing both
letterbox kernels read the one pinned frame the decoder wrote (GpuLetterbox.share_frame).

The test set mixes barcode frames and OCR label frames, and every frame goes through both
branches. Barcodes are scored on the barcode frames and text lines on the OCR frames; the
work done on the other frames (text under a barcode, nothing found on a label) is real load.
Stage keys: "barcode" and "ocr" are each branch's wall time (they overlap when concurrent);
each branch's own stages are kept as "bc.<stage>" and "ocr.<stage>".
"""
import threading

from . import barcode, ocr
from .jpeg import make_decoder
from .pipeline import Clock


class AssistantPipeline:
    stages = ["jpeg_decode", "barcode", "ocr"]

    def __init__(self, bc, oc, order="sequential", jpeg_name="cpu"):
        """bc: barcode.DetectPipeline, oc: ocr.OcrPipeline (their own decoders go unused)."""
        assert order in ("sequential", "concurrent")
        self.name = "assistant"
        self.bc, self.oc, self.order = bc, oc, order
        self.jpeg = make_decoder(jpeg_name)
        self.gp = bc.gprep
        if bc.gprep and oc.gprep:
            oc.gprep.share_frame(bc.gprep)
        self.config = {"mode": order, "jpeg": jpeg_name, "prep": bc.config["prep"],
                       "detector": bc.config["detector"], "input_size": bc.config["input_size"],
                       "precision": bc.config["precision"], "calibrator": bc.config.get("calibrator"),
                       "ocr_detector": oc.config["detector"], "ocr_input_size": oc.config["input_size"],
                       "ocr_precision": oc.config["precision"]}

    def _branch(self, pipe, key, img, times, results):
        sub = {}
        with Clock(sub)(key):
            results[key] = pipe.process_image(img, sub)
        times[key] = sub.pop(key)
        prefix = "bc." if key == "barcode" else "ocr."
        times.update({prefix + k: v for k, v in sub.items()})

    def process(self, jpeg, times):
        with Clock(times)("jpeg_decode"):
            img = self.jpeg.decode(jpeg, out=self.gp.frame if self.gp else None)
        results, bt = {}, {}
        if self.order == "sequential":
            self._branch(self.bc, "barcode", img, bt, results)
            self._branch(self.oc, "ocr", img, bt, results)
        else:
            other = threading.Thread(target=self._branch, args=(self.bc, "barcode", img, bt, results))
            other.start()
            self._branch(self.oc, "ocr", img, bt, results)
            other.join()
            if "barcode" not in results:
                raise RuntimeError("barcode branch failed")
        times.update(bt)
        return results

    def score(self, frames, outputs):
        bc_idx = [i for i, f in enumerate(frames) if "barcodes" in f]
        oc_idx = [i for i, f in enumerate(frames) if "lines" in f]
        b = barcode.score([frames[i] for i in bc_idx], [outputs[i]["barcode"] for i in bc_idx])
        o = ocr.score([frames[i] for i in oc_idx], [outputs[i]["ocr"] for i in oc_idx])
        return {"decode_rate": b["decode_rate"], "misreads": b["misreads"], "line_exact": o["line_exact"],
                "cer": o["cer"], "barcode_frames": len(bc_idx), "ocr_frames": len(oc_idx),
                "barcode": b, "ocr": o}

    def clone(self):
        c = AssistantPipeline(self.bc.clone(), self.oc.clone(), self.order, self.config["jpeg"])
        c.config = dict(self.config)
        return c

    def close(self):
        self.oc.close()  # oc's kernel reads bc's frame: release it first
        self.bc.close()
        self.jpeg.close()


def load_frames(bc_root, oc_root, every=1):
    """Barcode and OCR test frames interleaved (every n-th of each, to bound the accuracy pass)."""
    b = barcode.load_frames(bc_root)[::every]
    o = ocr.load_frames(oc_root)[::every]
    out = []
    for i in range(max(len(b), len(o))):
        out += ([b[i]] if i < len(b) else []) + ([o[i]] if i < len(o) else [])
    return out
