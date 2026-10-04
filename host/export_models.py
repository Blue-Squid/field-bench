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


EXPORTERS = {"yolo11n": export_yolo11n, "mobilenetv3l": export_mobilenetv3l}


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
