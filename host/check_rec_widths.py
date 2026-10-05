"""Does padding text lines to a narrower recognizer width change what PP-OCRv5 reads?

The pipeline pads every line crop to 48x640 (static engine). Width buckets would pad each line
only to the smallest bucket that holds it. This runs the OCR test set through the detector
(ONNX Runtime, CPU, the pipeline's own pre/postprocessing) once, then recognizes every crop
with the original dynamic-width recognizer at: 640 (today), the bucket that fits, and the
crop's own width rounded up to 8 px. Batches don't interact, so lines are run one at a time.

  .venv/bin/python host/check_rec_widths.py [--size 1280] [--buckets 160 320 480 640]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fieldbench import ocr  # noqa: E402


def needed_width(crop, H=48, W=640):
    """Width the crop resizes to at height H, as rec_preprocess_into computes it."""
    h, w = crop.shape[:2]
    return min(W, max(1, int(np.ceil(H * w / h))))


def recognize(sess, crop, width, charset):
    slot = np.zeros((1, 3, 48, width), np.float32)
    ocr.rec_preprocess_into(crop, slot[0])
    probs = sess.run(None, {"x": slot})[0][0]
    return ocr.ctc_decode(probs, charset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=1280, help="detector input size")
    ap.add_argument("--buckets", type=int, nargs="+", default=[160, 320, 480, 640])
    ap.add_argument("--frames", type=int, default=0, help="first n frames only (0 = all)")
    args = ap.parse_args()

    import cv2

    frames = ocr.load_frames(ROOT / "data" / "ocr" / "test")
    if args.frames:
        frames = frames[:args.frames]
    det = ort.InferenceSession(str(ROOT / "models" / f"ppocr5_det_{args.size}.onnx"), providers=["CPUExecutionProvider"])
    rec = ort.InferenceSession(str(ROOT / "models" / "_ppocr" / "en_PP-OCRv5_rec_mobile.onnx"),
                               providers=["CPUExecutionProvider"])
    charset = ocr.load_charset(ROOT / "models" / "ppocr5_rec_en.chars.txt")
    buf = np.zeros((1, 3, args.size, args.size), np.float32)

    strategies = {"pad640": lambda nw: 640,
                  "bucket": lambda nw: next(b for b in sorted(args.buckets) if b >= nw),
                  "exact8": lambda nw: -(-nw // 8) * 8}
    outputs = {k: [] for k in strategies}
    widths, changed = [], {k: 0 for k in strategies}
    for i, f in enumerate(frames):
        img = cv2.imdecode(np.frombuffer(f["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        scale = ocr.det_preprocess_into(img, buf)
        rects = ocr.det_postprocess(det.run(None, {"x": buf})[0][0, 0], scale)
        crops = [ocr.crop_line(img, r) for r in rects]
        texts = {k: [] for k in strategies}
        for c in crops:
            if c is None:
                for k in strategies:
                    texts[k].append(None)
                continue
            nw = needed_width(c)
            widths.append(nw)
            for k, fn in strategies.items():
                texts[k].append(recognize(rec, c, fn(nw), charset))
            for k in strategies:
                a, b = texts[k][-1], texts["pad640"][-1]
                changed[k] += (a and a[0]) != (b and b[0])
        for k in strategies:
            outputs[k].append({"rects": rects, "texts": texts[k]})
        if i % 20 == 0:
            print(f"  {i}/{len(frames)} frames", flush=True)

    w = np.array(widths)
    print(f"\n{len(w)} detected lines; needed width p50 {np.median(w):.0f}, p90 {np.percentile(w, 90):.0f} px")
    print("share of lines per bucket:", {b: f"{np.mean((w <= b) & (w > p)):.2f}"
                                         for p, b in zip([0, *sorted(args.buckets)], sorted(args.buckets))})
    print(f"\n{'strategy':<10}{'line exact':>11}{'CER':>8}{'words':>8}{'reads != pad640':>17}")
    for k in strategies:
        s = ocr.score(frames, outputs[k])
        print(f"{k:<10}{s['line_exact']:>11.4f}{s['cer']:>8.4f}{s['word_recall']:>8.4f}{changed[k]:>17}")


if __name__ == "__main__":
    main()
