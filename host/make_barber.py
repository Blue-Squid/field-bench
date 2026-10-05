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

--yolo DIR writes a detector training set from every other photo (YOLO-OBB labels, classes
1D/2D, train/val split per source). Every annotated barcode is labelled, postal and
undecodable ones included: the detector should still find them. The test sample is held out,
together with any photo whose perceptual hash is within a few bits of a test photo (several
sources are video-like sequences of the same product).
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
MATRIX = {"QR", "DATAMATRIX", "PDF417", "AZTEC"}  # detector class 1 (2D: matrix and stacked)


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


def dhash(path, size=8):
    """64-bit difference hash of a photo, for near-duplicate detection."""
    import cv2
    import numpy as np

    g = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def yolo_set(src, test_gt, out, val_frac, seed, max_dist=6):
    """Every annotated photo outside the test sample (and its near-duplicates) -> YOLO-OBB train/val."""
    import cv2
    import numpy as np

    images = {p.name: p for p in (src / "dataset" / "images").iterdir()}
    regs_by_file, source_of = defaultdict(list), {}
    for source, name, regs in regions(src):
        regs_by_file[name] += regs  # a few photos are annotated in two source files
        source_of.setdefault(name, source)
    test = {json.loads(line)["original"] for line in Path(test_gt).read_text().splitlines()}
    test_hashes = [dhash(images[n]) for n in sorted(test)]
    skipped, kept = Counter(), defaultdict(list)
    for name, regs in sorted(regs_by_file.items()):
        if name in test:
            skipped["test sample"] += 1
            continue
        if name not in images:
            skipped["image missing"] += 1
            continue
        if any(r["shape_attributes"].get("name") != "polygon" for r in regs):
            skipped["non-polygon region"] += 1  # dropping the region would teach "background" there
            continue
        h = dhash(images[name])
        if min(bin(h ^ t).count("1") for t in test_hashes) <= max_dist:
            skipped["near-duplicate of a test photo"] += 1
            continue
        kept[source_of[name]].append(name)

    rng = random.Random(seed)
    split = {}
    for source, names in sorted(kept.items()):
        rng.shuffle(names)
        n_val = max(1, round(val_frac * len(names)))
        split.update({n: "val" for n in names[:n_val]}, **{n: "train" for n in names[n_val:]})
    if out.exists():
        shutil.rmtree(out)
    counts = Counter()
    for name, part in sorted(split.items()):
        img = images[name]
        w, h = _size(img)
        lines, seen = [], set()
        for r in regs_by_file[name]:
            s, t = r["shape_attributes"], r["region_attributes"].get("Type", "")
            pts = np.float32(list(zip(s["all_points_x"], s["all_points_y"])))
            box = cv2.boxPoints(cv2.minAreaRect(pts))  # 5-7 point polygons -> their oriented box
            key = tuple(np.round(box).astype(int).flatten())
            if key in seen:  # same barcode annotated by both source files
                continue
            seen.add(key)
            box = (box / (w, h)).clip(0, 1)
            lines.append(f"{1 if t in MATRIX else 0} " + " ".join(f"{v:.6f}" for v in box.flatten()))
            counts[f"{part} barcodes"] += 1
        for sub in ("images", "labels"):
            (out / part / sub).mkdir(parents=True, exist_ok=True)
        (out / part / "images" / name).symlink_to(img.resolve())
        (out / part / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
        counts[f"{part} photos"] += 1
    print(f"skipped {dict(skipped)}")
    print(f"wrote {dict(counts)} -> {out}")
    print("train photos by source:", dict(Counter(source_of[n] for n, p in split.items() if p == "train")))
    data = ROOT / "data"
    mix = out / "mix.yaml"
    mix.write_text("# BarBeR real photos (host/make_barber.py --yolo) + the synthetic set, validated on BarBeR only.\n"
                   f"path: {data}\n"
                   f"train: [{out.resolve().relative_to(data)}/train/images, barcodes/train/images]\n"
                   f"val: {out.resolve().relative_to(data)}/val/images\n"
                   "names:\n  0: 1d\n  1: 2d\n")
    print(f"-> {mix} (training config: real + synthetic)")


def _size(path):
    """Displayed (width, height) from the JPEG header. BarBeR's polygons are in the EXIF-rotated
    frame, which is also what cv2 and Ultralytics load, so orientations 5-8 swap the sides."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        return (h, w) if im.getexif().get(0x0112, 1) in (5, 6, 7, 8) else (w, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=600, help="frames to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yolo", type=Path, metavar="DIR",
                    help="instead: write a YOLO-OBB training set (train/val) from every photo outside --out's test sample")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    if args.yolo:
        yolo_set(args.src, args.out / "gt.jsonl", args.yolo, args.val_frac, args.seed)
        return

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
