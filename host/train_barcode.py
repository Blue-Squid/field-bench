"""Fine-tune YOLO11n-OBB on the synthetic barcode set (run with .venv/bin/python, CUDA PyTorch).

Oriented boxes, not axis-aligned ones: the pipeline rotates each crop upright before decoding,
because zxing-cpp's 1D/PDF417 readers only scan rows and columns.

  .venv/bin/python host/make_barcodes.py      # dataset first
  .venv/bin/python host/train_barcode.py      # -> models/_ultralytics/barcode_yolo11n.pt
  make export                                 # -> models/barcode_yolo11n_{640,1280,1600}.onnx

Real-photo fine-tune (BarBeR train split + the synthetic set, from the synthetic weights):
  .venv/bin/python host/make_barber.py --yolo data/barber/yolo
  .venv/bin/python host/train_barcode.py --data data/barber/yolo/mix.yaml --init models/_ultralytics/barcode_yolo11n.pt \
      --name barcode_real --epochs 40          # -> models/_ultralytics/barcode_real_yolo11n.pt
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "barcodes" / "barcodes.yaml"
DOTA = ROOT / "models" / "_ultralytics" / "yolo11n-obb.pt"  # DOTA-pretrained start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--init", type=Path, default=DOTA, help="starting weights")
    ap.add_argument("--name", default="barcode", help="run name: runs/<name>, models/_ultralytics/<name>_yolo11n.pt")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from runs/<name>/weights/last.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    run = ROOT / "runs" / args.name
    if args.resume:
        YOLO(str(run / "weights" / "last.pt")).train(resume=True)
    else:
        model = YOLO(str(args.init))
        model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0, workers=12,
                    project=str(ROOT / "runs"), name=args.name, exist_ok=True, degrees=10, plots=True)
    weights = ROOT / "models" / "_ultralytics" / f"{args.name}_yolo11n.pt"
    shutil.copy(run / "weights" / "best.pt", weights)
    print(f"-> {weights}")


if __name__ == "__main__":
    main()
