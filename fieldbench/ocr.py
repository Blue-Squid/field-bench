"""OCR pipeline: PP-OCRv5 mobile text detector (DB) -> line crops -> batched recognizer -> CTC decode.

  jpeg_decode -> det_pre (resize + normalize) -> det_infer -> det_post (threshold, contours, boxes)
  -> rec_pre (deskewed crops, 48 px high) -> rec_infer (batches of 8) -> rec_post (CTC greedy decode)

Recognizer width buckets (optional): with engines at several widths (e.g. 320, 480, 640), a
frame's lines are sorted widest first and cut into batches of 8; each batch runs on the
narrowest engine that holds its widest line instead of padding every line to 640 px.

Postprocessing follows PaddleOCR's DBPostProcess and CTCLabelDecode with their default
thresholds; "unclip" is done analytically on the box rectangle instead of with pyclipper.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .jpeg import CpuJpeg, make_decoder
from .pipeline import Clock

DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # applied to BGR as PaddleOCR does
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_frames(root):
    """Test set written by host/make_text.py: [{jpeg, lines: [{text, polygon, cap_px}]}]."""
    root = Path(root)
    frames = []
    for line in (root / "gt.jsonl").read_text().splitlines():
        g = json.loads(line)
        g["jpeg"] = (root / g["image"]).read_bytes()
        frames.append(g)
    return frames


def det_normalize(u8):
    """uint8 BGR (..., 3) -> float32, PaddleOCR's (x / 255 - mean) / std."""
    return (u8.astype(np.float32) * (1 / 255) - DET_MEAN) / DET_STD


def det_lut():
    """det_normalize of every uint8 level, per channel: the GPU kernel's table (gpuprep)."""
    return det_normalize(np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1))


def det_preprocess_into(img, out):
    """Fit a BGR image into the (1, 3, S, S) buffer, top-left aligned, zero padding. Returns scale."""
    S = out.shape[-1]
    h, w = img.shape[:2]
    scale = min(S / w, S / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out[0, :, :nh, :nw] = det_normalize(resized).transpose(2, 0, 1)
    out[0, :, nh:, :] = 0
    out[0, :, :nh, nw:] = 0
    return scale


def det_postprocess(prob, scale, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5, min_size=3, max_candidates=1000):
    """DB probability map (S, S) -> list of rotated rects ((cx, cy), (w, h), angle) in image coordinates."""
    bitmap = (prob > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours[:max_candidates]:
        (cx, cy), (w, h), a = cv2.minAreaRect(c)
        if min(w, h) < min_size:
            continue
        # Box score: mean probability inside the rectangle (PaddleOCR's box_score_fast).
        pts = cv2.boxPoints(((cx, cy), (w, h), a))
        x0, y0 = np.floor(pts.min(0)).astype(int).clip(0)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x1, y1 = min(x1, prob.shape[1] - 1), min(y1, prob.shape[0] - 1)
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(mask, [np.int32(pts - (x0, y0))], 1)
        if cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0] < box_thresh:
            continue
        # Unclip: grow the shrunk DB kernel back to the text extent. For a rectangle, offsetting
        # by d = area * ratio / perimeter adds 2d to each side.
        d = w * h * unclip_ratio / (2 * (w + h))
        w, h = w + 2 * d, h + 2 * d
        if min(w, h) < min_size + 2:
            continue
        rects.append(((cx / scale, cy / scale), (w / scale, h / scale), a))
    return rects


def crop_line(img, rect):
    """Perspective-crop a text rect upright, as PaddleOCR's get_rotate_crop_image does."""
    pts = cv2.boxPoints(rect)
    xs = pts[np.argsort(pts[:, 0])]
    left, right = xs[:2][np.argsort(xs[:2, 1])], xs[2:][np.argsort(xs[2:, 1])]
    tl, bl, tr, br = left[0], left[1], right[0], right[1]
    w = int(max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br)))
    h = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
    if w < 2 or h < 2:
        return None
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h / w >= 1.5:
        crop = np.rot90(crop)
    return crop


def rec_width(crop, H=48, W=640):
    """Width a crop resizes to at height H (capped at W), as rec_preprocess_into computes it."""
    h, w = crop.shape[:2]
    return min(W, max(1, int(np.ceil(H * w / h))))


def rec_batches(widths, B, buckets):
    """Plan recognizer batches: [(bucket width, [line indices])], widest lines first, B per batch,
    each batch on the narrowest bucket that holds its widest line."""
    order = sorted(range(len(widths)), key=lambda k: -widths[k])
    plan = []
    for start in range(0, len(order), B):
        chunk = order[start:start + B]
        plan.append((next(b for b in buckets if b >= widths[chunk[0]]), chunk))
    return plan


def rec_preprocess_into(crop, out_slot):
    """One crop -> (3, 48, W) slot of the recognizer batch: height 48, aspect kept, zero padded."""
    _, H, W = out_slot.shape
    h, w = crop.shape[:2]
    nw = min(W, max(1, int(np.ceil(H * w / h))))
    resized = cv2.resize(crop, (nw, H), interpolation=cv2.INTER_LINEAR)
    out_slot[:, :, :nw] = (resized.astype(np.float32) * (2 / 255) - 1).transpose(2, 0, 1)
    out_slot[:, :, nw:] = 0


def ctc_decode(probs, charset, drop_score=0.5):
    """(T, C) softmax output -> (text, confidence) with PaddleOCR's greedy CTC rules; None if low."""
    idx = probs.argmax(1)
    conf = probs[np.arange(len(idx)), idx]
    keep = idx != 0
    keep[1:] &= idx[1:] != idx[:-1]
    if not keep.any():
        return None
    text = "".join(charset[i] for i in idx[keep])
    score = float(conf[keep].mean())
    return (text, score) if score >= drop_score else None


def _norm(s):
    return " ".join(s.split())


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _inter(a, b):
    kind, pts = cv2.rotatedRectangleIntersection(a, b)
    return cv2.contourArea(pts) if kind != cv2.INTERSECT_NONE and pts is not None else 0.0


def _assign(lines, rects):
    """Map predicted boxes to GT lines. Ground-truth polygons are tight ink boxes while DB boxes carry
    a margin (~1.5x as tall), so IoU undercounts good boxes; instead a box belongs to the line its
    centre falls in, unless it covers half of two or more lines (a merge), which then gets no credit."""
    polys = [np.float32(g["polygon"]) for g in lines]
    gts = [cv2.minAreaRect(p) for p in polys]
    areas = [r[1][0] * r[1][1] for r in gts]
    owner, merged = [None] * len(rects), set()
    for k, rect in enumerate(rects):
        inter = [_inter(gr, rect) for gr in gts]
        covered = [i for i, a in enumerate(inter) if a >= 0.5 * areas[i]]
        if len(covered) >= 2:
            merged.update(covered)
            continue
        inside = [i for i, p in enumerate(polys) if cv2.pointPolygonTest(p, tuple(map(float, rect[0])), False) >= 0]
        if inside:
            owner[k] = max(inside, key=lambda i: inter[i])
    cover = [0.0] * len(lines)
    for k, i in enumerate(owner):
        if i is not None:
            cover[i] += _inter(gts[i], rects[k]) / areas[i]
    return owner, cover, merged


def score(frames, outputs):
    """Line detection recall, exact line reads, character error rate and word recall vs ground truth.

    A line is detected when the boxes assigned to it (see _assign) cover >= 70% of it; when DB split
    it into several boxes, their texts are joined left to right."""
    n = detected = exact = split = merged = 0
    chars = errs = 0
    words_gt = words_hit = 0
    by_size = {}
    for frame, out in zip(frames, outputs):
        pred_words = [w for t in out["texts"] if t for w in t[0].split()]
        pool = list(pred_words)
        owner, cover, merged_lines = _assign(frame["lines"], out["rects"])
        merged += len(merged_lines)
        for i, g in enumerate(frame["lines"]):
            n += 1
            p = np.float32(g["polygon"])
            along = p[1] - p[0]
            mine = sorted((k for k, o in enumerate(owner) if o == i), key=lambda k: np.dot(out["rects"][k][0], along))
            truth = _norm(g["text"])
            hit_det = cover[i] >= 0.7
            detected += hit_det
            split += hit_det and len(mine) > 1
            read = _norm(" ".join(out["texts"][k][0] for k in mine if out["texts"][k])) if hit_det else ""
            ok = read == truth
            exact += ok
            chars += len(truth)
            errs += min(_edit(read, truth), len(truth))
            bucket = "cap<20" if g["cap_px"] < 20 else ("cap20-32" if g["cap_px"] < 32 else "cap32+")
            s = by_size.setdefault(bucket, [0, 0])
            s[0] += ok
            s[1] += 1
            for w in truth.split():
                words_gt += 1
                if w in pool:
                    pool.remove(w)
                    words_hit += 1
    return {"lines": n, "line_exact": exact / n, "cer": errs / chars, "detect_recall": detected / n,
            "word_recall": words_hit / words_gt, "lines_per_frame_pred": sum(len(o["rects"]) for o in outputs) / len(frames),
            "lines_split": split / n, "lines_merged": merged / n,
            "line_exact_by_size": {k: v[0] / v[1] for k, v in sorted(by_size.items())}}


class OcrPipeline:
    stages = ["jpeg_decode", "det_pre", "det_infer", "det_post", "rec_pre", "rec_infer", "rec_post"]

    def __init__(self, det_runner, rec_runner, charset, det_name, det_size, precision, rec_precision,
                 jpeg=None, jpeg_name="cpu", prep="cpu", rec_narrow=()):
        """rec_runner: the full-width (640) recognizer; rec_narrow: optional narrower ones (width buckets)."""
        self.name = "ocr"
        self.det, self.rec = det_runner, rec_runner
        self.det_in, self.det_out = det_runner.inputs[0].host, det_runner.outputs[0].host
        self.rec_in, self.rec_out = rec_runner.inputs[0].host, rec_runner.outputs[0].host
        self.recs = {r.inputs[0].host.shape[-1]: r for r in (*rec_narrow, rec_runner)}  # width -> runner
        self.buckets = sorted(self.recs)
        self.charset = charset
        self.jpeg = jpeg or CpuJpeg()
        # prep="gpu": det_pre runs as one CUDA kernel writing the detector's input on the GPU
        # (gpuprep), bit-exact with det_preprocess_into below 2560 px.
        self.gprep = None
        if prep == "gpu":
            from .gpuprep import GpuLetterbox
            self.gprep = GpuLetterbox(det_runner.inputs[0], det_runner.stream, det_lut())
        self._kw = dict(charset=charset, det_name=det_name, det_size=det_size, precision=precision,
                        rec_precision=rec_precision, jpeg_name=jpeg_name, prep=prep)
        self.config = {"mode": "ppocr5", "jpeg": jpeg_name, "prep": prep, "detector": det_name, "input_size": det_size, "precision": precision,
                       "rec_precision": rec_precision, "rec_batch": self.rec_in.shape[0],
                       "rec_width": self.rec_in.shape[-1], "rec_buckets": self.buckets}

    def clone(self):
        """An independent copy (own engine contexts, streams, buffers, decoder) for another worker."""
        from .runner import TrtRunner

        c = OcrPipeline(TrtRunner(self.det.engine_file), TrtRunner(self.rec.engine_file),
                        jpeg=make_decoder(self.config["jpeg"]),
                        rec_narrow=[TrtRunner(r.engine_file) for w, r in self.recs.items() if r is not self.rec],
                        **self._kw)
        c.config = dict(self.config)
        return c

    def process(self, jpeg, times):
        gp = self.gprep
        with Clock(times)("jpeg_decode"):
            # NVJPG decodes straight into the kernel's pinned frame; the CPU decoder ignores out=.
            img = self.jpeg.decode(jpeg, out=gp.frame if gp else None)
        return self.process_image(img, times)

    def process_image(self, img, times):
        """Everything after JPEG decode, on a BGR frame (in gprep.frame when prep is gpu)."""
        clock = Clock(times)
        gp = self.gprep
        with clock("det_pre"):
            if gp:
                scale, _, _, times["det_pre_gpu"] = gp(img)
            else:
                scale = det_preprocess_into(img, self.det_in)
        with clock("det_infer"):
            g, _ = self.det.infer(upload=gp is None)
        times["det_infer_gpu"] = g
        with clock("det_post"):
            rects = det_postprocess(self.det_out[0, 0], scale)
        texts = [None] * len(rects)  # stays None for crops too thin to cut out
        B = self.rec_in.shape[0]
        times["rec_batches"] = 0
        with clock("rec_pre"):
            crops = [crop_line(img, rect) for rect in rects]
            idx = [k for k, c in enumerate(crops) if c is not None]
            plan = rec_batches([rec_width(crops[k]) for k in idx], B, self.buckets)
        for width, chunk in plan:
            rec = self.recs[width]
            rec_in, rec_out = rec.inputs[0].host, rec.outputs[0].host
            with clock("rec_pre"):
                for slot, j in enumerate(chunk):
                    rec_preprocess_into(crops[idx[j]], rec_in[slot])
                rec_in[len(chunk):] = 0
            with clock("rec_infer"):
                g, _ = rec.infer()
            times["rec_infer_gpu"] = times.get("rec_infer_gpu", 0.0) + g
            times["rec_batches"] += 1
            times[f"rec_w{width}"] = times.get(f"rec_w{width}", 0) + 1
            with clock("rec_post"):
                for slot, j in enumerate(chunk):
                    texts[idx[j]] = ctc_decode(rec_out[slot], self.charset)
        return {"rects": rects, "texts": texts}

    def score(self, frames, outputs):
        return score(frames, outputs)

    def close(self):
        if self.gprep:
            self.gprep.close()
        self.det.close()
        for r in self.recs.values():
            r.close()
        self.jpeg.close()


def load_charset(path):
    """CTC classes: blank, the model's dictionary, then space (PaddleOCR's use_space_char)."""
    chars = Path(path).read_text().split("\n")
    if chars and chars[-1] == "":
        chars = chars[:-1]
    return ["<blank>", *chars, " "]
