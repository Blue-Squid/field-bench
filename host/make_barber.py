"""BarBeR real photos -> a fieldbench barcode test set (gt.jsonl + images) for the real-photo check.

BarBeR (Vezzali, Bolelli, Santi, Grana, "BarBeR: A Barcode Benchmarking Repository", ICPR 2024)
pools 12 public barcode datasets: 8,748 real photos with a polygon, symbology, pixels per
element and encoded string for each barcode. Download it (free account) from
https://ditto.ing.unimore.it/barber/ and unzip it so that data/barber/BarBeR - Dataset/
holds Annotations/VIA/*.json and dataset/images/.

Kept as ground truth: barcodes whose symbology zxing-cpp can read and whose string is known.
Postal codes (RoyalMail, KIX, Japan Post, PostNet, Intelligent Mail), IATA 2 of 5, EAN-2
add-ons and codes marked undecodable ("-1", "^", type "1D") stay in the photo but are not
scored. A frame needs at least one scored barcode. A stratified sample (same share of each
source dataset, seeded) keeps the accuracy pass and the copy to the board small.
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "barber" / "BarBeR - Dataset"
OUT = ROOT / "data" / "barber" / "fieldbench"

# BarBeR type -> (zxing-cpp format name, detector class: 0 = 1D, 1 = 2D)
SUPPORTED = {
    "EAN13": ("EAN13", 0), "EAN8": ("EAN8", 0), "UPCA": ("UPCA", 0), "UPCS": ("UPCA", 0),
    "C128": ("Code128", 0), "UCC128": ("Code128", 0), "C39": ("Code39", 0), "I2O5": ("ITF", 0),
    "QR": ("QRCode", 1), "DATAMATRIX": ("DataMatrix", 1), "PDF417": ("PDF417", 1), "AZTEC": ("Aztec", 1),
}
UNKNOWN = {"-1", "^", ""}


def regions(src):
    """(source dataset, filename, [region dicts]) for every annotated image."""
    for f in sorted((src / "Annotations" / "VIA").glob("*.json")):
        meta = json.loads(f.read_text())
        meta = meta.get("_via_img_metadata", meta)
        for v in meta.values():
            yield f.stem, v["filename"], v["regions"]


def convert(src, limit, seed):
    images = {p.name: p for p in (src / "dataset" / "images").iterdir()}
    frames, skipped = defaultdict(list), Counter()
    for source, name, regs in regions(src):
        if name not in images:
            skipped["image missing"] += 1
            continue
        codes, ignored = [], 0
        for r in regs:
            a, s = r["region_attributes"], r["shape_attributes"]
            text = str(a.get("String", "")).strip()
            if a.get("Type") not in SUPPORTED or text in UNKNOWN or s.get("name") != "polygon":
                ignored += 1
                continue
            fmt, cls = SUPPORTED[a["Type"]]
            codes.append({"format": fmt, "text": text, "cls": cls, "module_px": float(a.get("PPE", -1)),
                          "polygon": [[float(x), float(y)] for x, y in zip(s["all_points_x"], s["all_points_y"])]})
        if not codes:
            skipped["no scorable barcode"] += 1
            continue
        frames[source].append({"src": images[name], "source": source, "barcodes": codes, "ignored_barcodes": ignored})

    total = sum(len(v) for v in frames.values())
    rng = random.Random(seed)
    picked = []
    for source, items in sorted(frames.items()):
        n = max(1, round(limit * len(items) / total)) if limit else len(items)
        picked += rng.sample(items, min(n, len(items)))
    return picked, frames, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=600, help="frames to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    picked, frames, skipped = convert(args.src, args.limit, args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    with (args.out / "gt.jsonl").open("w") as f:
        for i, fr in enumerate(picked):
            dst = f"images/{i:04d}_{fr['src'].stem}.jpg"
            shutil.copyfile(fr["src"], args.out / dst)
            f.write(json.dumps({"image": dst, "source": fr["source"], "original": fr["src"].name,
                                "ignored_barcodes": fr["ignored_barcodes"], "barcodes": fr["barcodes"]}) + "\n")
    fmts = Counter(b["format"] for fr in picked for b in fr["barcodes"])
    print(f"{sum(len(v) for v in frames.values())} scorable frames; skipped {dict(skipped)}")
    print(f"wrote {len(picked)} frames, {sum(fmts.values())} barcodes -> {args.out}")
    print("by source:", dict(Counter(fr["source"] for fr in picked)))
    print("by format:", dict(fmts.most_common()))


if __name__ == "__main__":
    main()
