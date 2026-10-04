"""Fine-tune YOLO11n-OBB on the synthetic barcode set (run with .venv/bin/python, CUDA PyTorch).

Oriented boxes, not axis-aligned ones: the pipeline rotates each crop upright before decoding,
because zxing-cpp's 1D/PDF417 readers only scan rows and columns.

  .venv/bin/python host/make_barcodes.py      # dataset first
  .venv/bin/python host/train_barcode.py      # -> models/_ultralytics/barcode_yolo11n.pt
  make export                                 # -> models/barcode_yolo11n_{640,1280,1600}.onnx
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "barcodes" / "barcodes.yaml"
WEIGHTS = ROOT / "models" / "_ultralytics" / "barcode_yolo11n.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from runs/barcode/weights/last.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(str(ROOT / "runs" / "barcode" / "weights" / "last.pt")).train(resume=True)
    else:
        model = YOLO(str(ROOT / "models" / "_ultralytics" / "yolo11n-obb.pt"))  # DOTA-pretrained start
        model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0, workers=12,
                    project=str(ROOT / "runs"), name="barcode", exist_ok=True, degrees=10, plots=True)
    best = ROOT / "runs" / "barcode" / "weights" / "best.pt"
    shutil.copy(best, WEIGHTS)
    print(f"-> {WEIGHTS}")


if __name__ == "__main__":
    main()
