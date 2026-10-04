"""Product-recognition gallery + test set from the Grocery Store Dataset (run with .venv/bin/python).

Source: Klasson, Zhang, Kjellstrom, "A Hierarchical Grocery Store Image Dataset with Visual and
Semantic Labels", WACV 2019. https://github.com/marcusklasson/GroceryStoreDataset, MIT License.
5,125 smartphone photos of fruit, vegetables and carton products taken in grocery stores,
81 fine-grained classes grouped into 43 coarse classes (classes.csv ids 0-42; the README says 42).

The photos are already centred on the product (point-and-shoot), so the pipeline needs no
localizer: the whole frame is embedded and looked up against the gallery.

  data/products/gallery/  train + val splits: the "known products" the device embeds once
  data/products/test/     test split: the query images
Each has images/*.jpg (original JPEG bytes, not re-encoded) and gt.jsonl lines
{"image": "images/<split>_<file>.jpg", "label": <fine class>, "coarse": <coarse class>}.
"""
import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "_src" / "GroceryStoreDataset"
OUT = ROOT / "data" / "products"
REPO = "https://github.com/marcusklasson/GroceryStoreDataset"
CITATION = ('Klasson, M., Zhang, C., Kjellstrom, H. "A Hierarchical Grocery Store Image Dataset with '
            'Visual and Semantic Labels." IEEE Winter Conference on Applications of Computer Vision (WACV), 2019. '
            'arXiv:1901.00711')
SPLITS = {"gallery": ["train", "val"], "test": ["test"]}


def fetch():
    if not (SRC / "dataset" / "classes.csv").exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
    lic = (SRC / "LICENSE").read_text()
    if not lic.startswith("MIT License"):
        raise SystemExit(f"{SRC / 'LICENSE'}: expected the MIT License, found:\n{lic[:300]}")
    return lic


def classes():
    """class id -> (fine name, coarse name, coarse id), from dataset/classes.csv."""
    with open(SRC / "dataset" / "classes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["Class ID (int)"]): (r["Class Name (str)"], r["Coarse Class Name (str)"], int(r["Coarse Class ID (int)"]))
            for r in rows}


def jpeg_size(data):
    """(width, height) from the JPEG's SOF marker, without decoding."""
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError("bad JPEG marker")
        marker, seglen = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + seglen
    raise ValueError("no SOF marker")


def write(name, splits, names):
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    rows, sizes = [], Counter()
    for split in splits:
        for line in (SRC / "dataset" / f"{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            path, fine, coarse = (s.strip() for s in line.split(","))
            fine_name, coarse_name, coarse_id = names[int(fine)]
            if int(coarse) != coarse_id:
                raise SystemExit(f"{split}.txt: {path} has coarse id {coarse}, classes.csv says {coarse_id}")
            data = (SRC / "dataset" / path).read_bytes()
            sizes[jpeg_size(data)] += 1
            dst = f"images/{split}_{Path(path).name}"
            (out / dst).write_bytes(data)  # original bytes: no re-encode
            rows.append({"image": dst, "label": fine_name, "coarse": coarse_name})
    if len({r["image"] for r in rows}) != len(rows):
        raise SystemExit(f"{name}: duplicate image names")
    (out / "gt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{name}: {len(rows)} images ({'+'.join(splits)}), {len({r['label'] for r in rows})} fine / "
          f"{len({r['coarse'] for r in rows})} coarse classes")
    print("  sizes (w x h: count): " + ", ".join(f"{w}x{h}: {n}" for (w, h), n in sizes.most_common(6))
          + (f", ... ({len(sizes)} distinct)" if len(sizes) > 6 else ""))
    return rows


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    lic = fetch()
    names = classes()
    print(f"{len(names)} fine classes, {len({c for _, c, _ in names.values()})} coarse classes")
    gallery = write("gallery", SPLITS["gallery"], names)
    test = write("test", SPLITS["test"], names)
    missing = {r["label"] for r in test} - {r["label"] for r in gallery}
    if missing:
        raise SystemExit(f"test classes with no gallery images: {sorted(missing)}")
    (OUT / "SOURCE.txt").write_text(f"Grocery Store Dataset\n{REPO}\n\nCite: {CITATION}\n\n{lic}")
    print(f"license: MIT (copied to {OUT / 'SOURCE.txt'})")


if __name__ == "__main__":
    main()
