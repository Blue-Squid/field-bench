"""Barcode detector accuracy on the BarBeR test sample, on the host (ONNX Runtime, CPU), by photo size.

Runs fieldbench's own DetectPipeline (letterbox, OBB postprocess, deskewed crops, zxing-cpp)
with ONNX Runtime in place of TensorRT, plus whole-photo zxing, over data/barber/fieldbench,
and breaks the decode rate down by photo size. That breakdown is what an input-size rule
("run the detector at the size nearest the photo") would be chosen from.

  .venv/bin/python host/eval_barber.py --models barcode_yolo11n barcode_real_yolo11n --sizes 640 1280
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "host")]

from fieldbench import barcode  # noqa: E402
from ort_runner import OrtRunner  # noqa: E402

BINS = [(0, 0.5), (0.5, 1.5), (1.5, 4), (4, 1e9)]  # megapixels


class _Runner(OrtRunner):
    def infer(self, upload=True):  # DetectPipeline passes upload=; ONNX Runtime has no device copy
        return super().infer()


def _mp(frame):
    import cv2

    h, w = cv2.imdecode(np.frombuffer(frame["jpeg"], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape
    return h * w * 64 / 1e6


def by_bin(frames, outputs, mps):
    out = {}
    for lo, hi in BINS:
        idx = [i for i, m in enumerate(mps) if lo <= m < hi]
        if idx:
            s = barcode.score([frames[i] for i in idx], [outputs[i] for i in idx])
            out[f"{lo}-{hi if hi < 1e9 else ''}MP"] = (s["decode_rate"], s["barcodes"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["barcode_yolo11n", "barcode_real_yolo11n"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 1280])
    ap.add_argument("--data", default=str(ROOT / "data" / "barber" / "fieldbench"))
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models", help="where <model>_<size>.onnx live")
    args = ap.parse_args()

    frames = barcode.load_frames(args.data)
    mps = [_mp(f) for f in frames]
    print(f"{len(frames)} photos; by size: " + ", ".join(
        f"{lo}-{hi if hi < 1e9 else ''} MP: {sum(lo <= m < hi for m in mps)}" for lo, hi in BINS))
    results = {}
    z = barcode.ZxingPipeline()
    results["zxing whole photo"] = [z.process(f["jpeg"], {}) for f in frames]
    for model in args.models:
        for size in args.sizes:
            pipe = barcode.DetectPipeline(_Runner(args.models_dir / f"{model}_{size}.onnx"), model, "fp32", size)
            results[f"{model} {size}"] = [pipe.process(f["jpeg"], {}) for f in frames]
            print(f"  done {model} {size}", flush=True)

    print(f"\n{'configuration':<30}{'read':>7}{'recall':>8}{'boxes':>7}{'miss':>6}  read by photo size (n barcodes)")
    for name, outs in results.items():
        s = barcode.score(frames, outs)
        bins = "  ".join(f"{k} {v:.1%} ({n})" for k, (v, n) in by_bin(frames, outs, mps).items())
        print(f"{name:<30}{s['decode_rate']:>7.1%}{s.get('detect_recall', float('nan')):>8.1%}"
              f"{s.get('boxes_per_frame', float('nan')):>7.2f}{s['misreads']:>6}  {bins}")
    # Best detector size per photo-size bin: the input-size rule this data supports.
    print("\nbest detector configuration per photo-size bin:")
    det = {k: by_bin(frames, v, mps) for k, v in results.items() if k != "zxing whole photo"}
    for b in next(iter(det.values())):
        best = max(det, key=lambda k: det[k][b][0])
        print(f"  {b:<10} {best} ({det[best][b][0]:.1%})")


if __name__ == "__main__":
    main()
