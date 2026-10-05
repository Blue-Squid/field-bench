"""Export benchmark models to static-shape ONNX on the host (run with .venv/bin/python).

TensorRT engines are device-specific, so only ONNX is produced here; the Jetson
builds engines from these files.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OPSET = 17


def export_yolo11n():
    from ultralytics import YOLO

    work = MODELS / "_ultralytics"
    work.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(work / "yolo11n.pt"))
    # Raw head output (1x84x8400), no NMS: keeps the engine data-independent for timing.
    out = model.export(format="onnx", imgsz=640, opset=OPSET, simplify=True, dynamic=False, batch=1)
    shutil.move(out, MODELS / "yolo11n.onnx")


def export_mobilenetv3l():
    import torch
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, MODELS / "mobilenetv3l.onnx",
        input_names=["images"], output_names=["logits"], opset_version=OPSET, dynamo=False,
    )


def export_mobilenetv3l_embed():
    """MobileNetV3-L without its classifier head: features -> avgpool -> flatten, a 960-d embedding
    for product recognition by kNN lookup (fieldbench/product.py). Checked against PyTorch."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).eval()
    model = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten(1)).eval()
    x = torch.from_numpy(np.random.default_rng(0).uniform(-2, 2, (1, 3, 224, 224)).astype(np.float32))
    dst = MODELS / "mobilenetv3l_embed.onnx"
    tmp = dst.with_suffix(".tmp.onnx")
    torch.onnx.export(model, x, tmp, input_names=["images"], output_names=["embedding"],
                      opset_version=OPSET, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
    err = float(np.abs(ref - got).max()) if got.shape == ref.shape == (1, 960) else float("inf")
    if err > 1e-3:
        tmp.unlink()
        raise SystemExit(f"{dst.name}: ONNX output differs from PyTorch (shape {got.shape}, max abs diff {err:.3g})")
    tmp.replace(dst)
    print(f"  ONNX vs PyTorch: max abs diff {err:.3g} (|embedding| max {np.abs(ref).max():.3g})")


def export_barcode(size, family="barcode_yolo11n"):
    """YOLO11n fine-tuned on barcodes (host/train_barcode.py), at one of Zebra's input sizes."""
    from ultralytics import YOLO

    weights = MODELS / "_ultralytics" / f"{family}.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} missing: run host/make_barcodes.py and host/train_barcode.py first")
    out = YOLO(str(weights)).export(format="onnx", imgsz=size, opset=OPSET, simplify=True, dynamic=False, batch=1)
    shutil.move(out, MODELS / f"{family}_{size}.onnx")


PPOCR = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5"
PPOCR_FILES = {"det": "det/ch_PP-OCRv5_det_mobile.onnx", "rec": "rec/en_PP-OCRv5_rec_mobile.onnx"}
OCR_REC_SHAPE = (8, 3, 48, 640)  # 8 lines per batch, 48 px high, up to 640 px wide (80 CTC steps)


def _ppocr_source(kind):
    """Download a PP-OCRv5 mobile ONNX (RapidOCR's conversion of PaddleOCR, Apache-2.0) once."""
    import urllib.request

    src = MODELS / "_ppocr" / Path(PPOCR_FILES[kind]).name
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{PPOCR}/{PPOCR_FILES[kind]}", src)
    return src


def _static(src, dst, shape):
    """Pin a dynamic-shape ONNX to one input shape and fold the shape arithmetic away."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxslim

    # onnxslim 0.1.97's EliminationReshape + FusionGemm rewrite of the SVTR neck in the PP-OCRv5
    # recognizer silently corrupts its output (dropped spaces, misread characters).
    model = onnxslim.slim(str(src), input_shapes=[f"x:{','.join(map(str, shape))}"],
                          skip_fusion_patterns=["FusionGemm"])
    onnx.save(model, dst)
    x = np.random.default_rng(0).uniform(-1, 1, shape).astype(np.float32)
    ref, got = (ort.InferenceSession(str(p)).run(None, {"x": x})[0] for p in (src, dst))
    err = float(np.abs(ref - got).max())
    if err > 1e-3:
        raise SystemExit(f"{dst.name}: static model differs from {src.name} (max abs diff {err:.3g})")


def export_ocr_det(size):
    _static(_ppocr_source("det"), MODELS / f"ppocr5_det_{size}.onnx", (1, 3, size, size))


REC_BUCKETS = (320, 480)  # extra recognizer widths; 640 is ppocr5_rec_en itself (fieldbench/ocr.py)


def export_ocr_rec(width=640):
    """PP-OCRv5 recognizer at batch 8 x 48 px x width. Narrower widths are buckets for short lines."""
    import onnx

    src = _ppocr_source("rec")
    dst = MODELS / ("ppocr5_rec_en.onnx" if width == 640 else f"ppocr5_rec_en_w{width}.onnx")
    _static(src, dst, (*OCR_REC_SHAPE[:3], width))
    if width != 640:
        return
    # Keep the character list next to the model: TensorRT engines don't carry ONNX metadata.
    chars = {p.key: p.value for p in onnx.load(src).metadata_props}["character"]
    (MODELS / "ppocr5_rec_en.chars.txt").write_text(chars)


EXPORTERS = {"yolo11n": export_yolo11n, "mobilenetv3l": export_mobilenetv3l,
             "mobilenetv3l_embed": export_mobilenetv3l_embed,
             **{f"ppocr5_det_{s}": (lambda s=s: export_ocr_det(s)) for s in (640, 1280, 1600, 2560)},
             "ppocr5_rec_en": export_ocr_rec,
             **{f"ppocr5_rec_en_w{w}": (lambda w=w: export_ocr_rec(w)) for w in REC_BUCKETS},
             **{f"{fam}_{s}": (lambda s=s, fam=fam: export_barcode(s, fam))
                for fam in ("barcode_yolo11n", "barcode_real_yolo11n") for s in (640, 1280, 1600)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", default=list(EXPORTERS), choices=list(EXPORTERS))
    ap.add_argument("--force", action="store_true", help="re-export even if the ONNX exists")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    for name in args.models:
        target = MODELS / f"{name}.onnx"
        if target.exists() and not args.force:
            print(f"skip {name}: {target.name} exists")
            continue
        print(f"export {name} ...")
        EXPORTERS[name]()
        print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
