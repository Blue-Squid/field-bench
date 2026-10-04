"""Generate a synthetic OCR test set: printed labels with known text on real photos.

Each 4 MP frame (2304x1728) gets 1-4 labels (product, shipping, lot/expiry, price tags)
pasted under mild rotation and perspective, then the same lighting/blur/noise as the
barcode set. Ground truth is every text line's string and polygon.

  data/ocr/test/images + gt.jsonl     scored test frames
  data/ocr/calib/*.jpg                separate frames for INT8 calibration

  .venv/bin/python host/make_text.py --test 200 --calib 100
"""
import argparse
import json
import random
import string
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_barcodes as mb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ocr"
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/freefont"]
FONT_SKIP = ("Oblique", "Italic", "Narrow", "Serif")  # keep to upright sans/mono label fonts

WORDS = """organic whole milk fresh orange juice natural spring water greek yogurt chicken breast
ground coffee whole grain bread cheddar cheese olive oil brown rice pasta sauce peanut butter
apple cider vinegar sea salt black pepper green tea dark chocolate almond milk tomato soup
frozen peas baby spinach sweet potato red onion garlic lemon banana avocado blueberries
paper towels dish soap laundry detergent hand sanitizer batteries light bulbs printer paper
fragile handle with care this side up keep dry store below refrigerate after opening
best before use by product of packed on net weight serving size ingredients warning""".split()
CITIES = ["SPRINGFIELD IL", "RIVERSIDE CA", "FRANKLIN TN", "GREENVILLE SC", "MADISON WI", "SALEM OR"]


def _d(n):
    return "".join(random.choices(string.digits, k=n))


def _A(n):
    return "".join(random.choices(string.ascii_uppercase, k=n))


def _words(a, b):
    return " ".join(random.choices(WORDS, k=random.randint(a, b)))


def label_lines():
    """Text lines for one label, in the styles a handheld OCR job reads."""
    kind = random.choice(["product", "shipping", "lot", "price", "asset"])
    if kind == "product":
        lines = [_words(1, 3).title(), _words(2, 4), f"Net Wt {random.randint(50, 2000)}g",
                 f"BEST BEFORE {random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2026, 2029)}"]
    elif kind == "shipping":
        lines = ["SHIP TO:", f"{_A(1)}. {_A(1)}{''.join(random.choices(string.ascii_lowercase, k=5))}",
                 f"{random.randint(10, 9999)} {random.choice(['MAIN', 'OAK', 'PINE', 'ELM', 'LAKE'])} ST",
                 f"{random.choice(CITIES)} {_d(5)}", f"TRACKING 1Z {_d(3)} {_A(2)}{_d(1)} {_d(2)} {_d(4)} {_d(3)}"]
    elif kind == "lot":
        lines = [f"LOT {_A(1)}{_d(6)}", f"EXP {random.randint(2026, 2030)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                 f"REF {_d(4)}-{_d(3)}", f"SN {_A(2)}{_d(8)}"]
    elif kind == "price":
        lines = [_words(1, 2).title(), f"${random.randint(0, 99)}.{_d(2)}", f"SKU {_A(2)}-{_d(4)}-{_A(2)}",
                 f"UNIT PRICE ${random.randint(0, 9)}.{_d(2)} / {random.choice(['lb', 'kg', 'oz', 'ea'])}"]
    else:
        lines = ["PROPERTY OF", f"{_A(random.randint(3, 6))} {random.choice(['LOGISTICS', 'MEDICAL', 'FACILITIES'])}",
                 f"ASSET {_d(6)}", f"CAL DUE {random.randint(1, 12):02d}/{random.randint(2026, 2029)}"]
    return random.sample(lines, k=random.randint(min(2, len(lines)), len(lines))) if kind != "shipping" else lines


def make_label(fonts):
    """Render a label with PIL. Returns (BGR label, line corner array (n*4, 2), line texts, cap height px)."""
    lines = label_lines()
    font_path = random.choice(fonts)
    cap = random.uniform(14, 56)  # cap height in the full-res frame, ~ 6-24 pt from 20-60 cm
    font = ImageFont.truetype(font_path, int(cap / 0.72))
    pad = int(cap * random.uniform(0.6, 1.2))
    spacing = int(cap * random.uniform(0.5, 0.9))
    boxes, y = [], pad
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    for text in lines:
        x0, y0, x1, y1 = draw.textbbox((pad, y), text, font=font)
        boxes.append((x0, y0, x1, y1))
        y = y1 + spacing
    W = max(b[2] for b in boxes) + pad
    H = y - spacing + pad
    paper = random.randint(205, 255)
    img = Image.new("RGB", (W, H), (paper,) * 3)
    d = ImageDraw.Draw(img)
    ink = (random.randint(0, 60),) * 3
    y = pad
    for text, (x0, y0, x1, y1) in zip(lines, boxes):
        d.text((pad, y), text, font=font, fill=ink)
        y = y1 + spacing
    corners = np.float32([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in boxes]).reshape(-1, 2)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), corners, lines, cap


def scene(seed, fonts):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    frame = mb.background(mb.FULL)
    taken, lines_gt = [], []
    for _ in range(random.choices([1, 2, 3, 4], [35, 35, 20, 10])[0]):
        label, corners, texts, cap = make_label(fonts)
        pts = mb.place(frame, label, corners, False, taken, max_angle=random.choice([5, 15, 30]))
        if pts is None:
            continue
        for text, poly in zip(texts, pts.reshape(-1, 4, 2)):
            lines_gt.append({"text": text, "cap_px": round(cap, 1), "polygon": poly.astype(float).round(1).tolist()})
    img, info = mb.degrade(frame)
    return img, lines_gt, info


def _write(job):
    split, i, seed, out = job
    img, lines, info = scene(seed, _write.fonts)
    name = f"{split}_{i:05d}.jpg"
    if split == "calib":
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return None
    cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(80, 95)])
    return {"image": f"images/{name}", "width": mb.FULL[0], "height": mb.FULL[1], **info, "lines": lines}


def _init():
    mb._init()
    _write.fonts = sorted(str(p) for d in FONT_DIRS for p in Path(d).glob("*.ttf")
                          if not any(k in p.name for k in FONT_SKIP))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for split, n, base in [("test", args.test, 3 * 10**6), ("calib", args.calib, 4 * 10**6)]:
            out = args.out / split
            (out / "images" if split == "test" else out).mkdir(parents=True, exist_ok=True)
            gt = [r for r in pool.map(_write, [(split, i, base + i, out) for i in range(n)], chunksize=4) if r]
            if split == "test":
                (out / "gt.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gt))
                print(f"test: {n} frames, {sum(len(g['lines']) for g in gt)} text lines -> {out}")
            else:
                print(f"calib: {n} frames -> {out}")


if __name__ == "__main__":
    main()
