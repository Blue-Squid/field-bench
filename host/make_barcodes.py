"""Generate a synthetic barcode-scene dataset with known contents (run with .venv/bin/python).

Each scene is a COCO photo with 1-6 printed labels pasted in under perspective, each label
carrying one barcode (1D or 2D) plus distractor text, then degraded with lighting, blur,
noise and JPEG compression. Every barcode is rendered by zxing-cpp and read back from the
clean render, so its ground-truth string is exactly what a decoder should return.

  data/barcodes/{train,val}/images|labels   YOLO OBB format, classes 0=1d 1=2d, 1600x1200
  data/barcodes/test/images + gt.jsonl      4 MP frames (2304x1728) like Zebra's test setup

  .venv/bin/python host/make_barcodes.py --train 4000 --val 400 --test 300
"""
import argparse
import json
import random
import string
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

ROOT = Path(__file__).resolve().parent.parent
BACKGROUNDS = ROOT / "data" / "coco" / "val2017"
OUT = ROOT / "data" / "barcodes"
F = zxingcpp.BarcodeFormat
CLASSES = ["1d", "2d"]
FULL = (2304, 1728)   # render size; 4 MP, as in Zebra's published test conditions
TRAIN = (1600, 1200)  # train/val are downscaled to keep the dataset light

# (format, weight, content generator, is_2d)
_digits = lambda n: "".join(random.choices(string.digits, k=n))  # noqa: E731
_alnum = lambda a, b: "".join(random.choices(string.ascii_uppercase + string.digits + "-", k=random.randint(a, b)))  # noqa: E731
SYMBOLOGIES = [
    (F.EAN13, 22, lambda: _digits(12), False),
    (F.UPCA, 8, lambda: _digits(11), False),
    (F.EAN8, 4, lambda: _digits(7), False),
    (F.Code128, 20, lambda: _alnum(6, 18), False),
    (F.Code39, 5, lambda: _alnum(5, 12).replace("-", ""), False),
    (F.ITF, 4, lambda: _digits(random.choice([10, 14])), False),
    (F.QRCode, 17, lambda: random.choice([
        lambda: f"https://ex.co/p/{_alnum(6, 20)}",
        lambda: f"SN:{_alnum(8, 16)};LOT:{_digits(6)}",
        lambda: _alnum(10, 60)])(), True),
    (F.DataMatrix, 12, lambda: f"(01){_digits(14)}(21){_alnum(6, 12)}", True),
    (F.PDF417, 8, lambda: f"{_alnum(10, 40)}", True),
]


def render_symbol():
    """Pick a symbology and content; return (uint8 image at 1 px/module, truth dict, is_2d)."""
    fmt, _, gen, is_2d = random.choices(SYMBOLOGIES, weights=[s[1] for s in SYMBOLOGIES])[0]
    for _ in range(10):
        try:
            bc = zxingcpp.create_barcode(gen(), fmt)
            img = np.array(zxingcpp.write_barcode_to_image(bc, scale=4, add_quiet_zones=False))
            res = zxingcpp.read_barcodes(cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255))
            if res:
                return img, {"format": res[0].format.name, "text": res[0].text}, is_2d
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError(f"could not render {fmt}")


def make_label(sym, is_2d, frame_w):
    """Paper label with the symbol, quiet zone and some text. Returns (BGR label, symbol corners)."""
    # Module size in the full-res frame: what a handheld sees from ~10-60 cm.
    module_px = random.uniform(1.4, 4.5) if not is_2d else random.uniform(2.5, 10.0)
    scale = module_px / 4.0 * frame_w / FULL[0]
    # Area interpolation when shrinking, like a sensor integrating light over each pixel;
    # nearest-neighbour would quantize bar widths and break 1D width ratios.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    if not is_2d:
        # 1D bars only need to be tall enough; vary height independently of width.
        h = max(12, int(sym.shape[1] * scale * random.uniform(0.25, 0.55)))
        sym = cv2.resize(sym, (max(8, round(sym.shape[1] * scale)), h), interpolation=interp)
    else:
        sym = cv2.resize(sym, None, fx=scale, fy=scale, interpolation=interp)
    sh, sw = sym.shape
    q = int(max(sh, sw) * 0.12) + 6
    pad_txt = int(sh * random.uniform(0.3, 0.9)) + 10
    lh, lw = sh + 2 * q + pad_txt, sw + 2 * q
    paper = np.array([random.randint(200, 255)] * 3, np.uint8)
    if random.random() < 0.2:  # tinted labels
        paper = np.clip(paper.astype(int) - np.random.randint(0, 50, 3), 0, 255).astype(np.uint8)
    label = np.full((lh, lw, 3), paper, np.uint8)
    ink = random.randint(0, 60)
    dark = 1 - sym[..., None].astype(np.float32) / 255  # keep the anti-aliased edges
    label[q:q + sh, q:q + sw] = (paper + (ink - paper.astype(np.float32)) * dark).astype(np.uint8)
    font_scale = max(0.3, pad_txt / 60)
    for i in range(random.randint(1, 2)):
        y = q + sh + int(pad_txt * (0.45 + 0.4 * i))
        if y < lh - 2:
            cv2.putText(label, _alnum(4, 14), (q, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (ink,) * 3, max(1, int(font_scale * 1.5)), cv2.LINE_AA)
    corners = np.float32([[q, q], [q + sw, q], [q + sw, q + sh], [q, q + sh]])
    return label, corners, module_px


def place(frame, label, corners, is_2d, taken, max_angle=None):
    """Warp a label into the frame under random rotation/perspective. Returns `corners` mapped
    into the frame (any number of points), or None if it didn't fit."""
    H, W = frame.shape[:2]
    lh, lw = label.shape[:2]
    src = np.float32([[0, 0], [lw, 0], [lw, lh], [0, lh]])
    if max_angle is not None:
        angle = random.uniform(-max_angle, max_angle)
    else:
        angle = random.uniform(-180, 180) if (is_2d or random.random() < 0.5) else random.uniform(-25, 25)
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = src.mean(0)
    dst = (src - c) @ rot.T
    jitter = random.uniform(0, 0.12) * max(lw, lh)
    dst = dst + np.random.uniform(-jitter, jitter, dst.shape)
    span = dst.max(0) - dst.min(0)
    if span[0] >= W * 0.95 or span[1] >= H * 0.95:
        return None
    for _ in range(30):
        off = np.array([random.uniform(-dst[:, 0].min(), W - dst[:, 0].max()),
                        random.uniform(-dst[:, 1].min(), H - dst[:, 1].max())])
        d = (dst + off).astype(np.float32)
        box = (*d.min(0), *d.max(0))
        if not any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            break
    else:
        return None
    M = cv2.getPerspectiveTransform(src, d)
    warped = cv2.warpPerspective(label, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=frame.copy())
    mask = cv2.warpPerspective(np.full((lh, lw), 255, np.uint8), M, (W, H))
    # Soft label shadow, then paste.
    shadow = cv2.GaussianBlur(mask, (0, 0), 6).astype(np.float32)[..., None] / 255 * random.uniform(0, 0.35)
    np.copyto(frame, (frame * (1 - shadow)).astype(np.uint8))
    np.copyto(frame, warped, where=mask[..., None] > 127)
    taken.append(box)
    return cv2.perspectiveTransform(corners[None], M)[0]


def degrade(img):
    """Lighting, blur and sensor noise. Returns (image, {blur, blur_px, noise})."""
    h, w = img.shape[:2]
    info = {"blur": "none", "blur_px": 0.0}
    out = img.astype(np.float32)
    # Uneven lighting: a linear gradient plus global gain/offset.
    gx, gy = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
    v = np.random.uniform(-0.35, 0.35, 2)
    out *= (1 + gx * v[0] + gy * v[1])[..., None] * random.uniform(0.55, 1.2)
    out += random.uniform(-25, 25)
    r = random.random()
    if r < 0.2:  # defocus
        sigma = random.uniform(0.5, 1.6)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
        info.update(blur="defocus", blur_px=round(sigma, 2))
    elif r < 0.3:  # motion blur from hand shake
        k = random.randint(3, 9)
        info.update(blur="motion", blur_px=float(k))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2] = 1 / k
        kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1), (k, k))
        out = cv2.filter2D(out, -1, kern / max(kern.sum(), 1e-6))
    info["noise"] = round(random.uniform(1, 7), 2)
    out += np.random.normal(0, info["noise"], out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), info


def background(size):
    bgs = background.files
    img = cv2.imread(str(random.choice(bgs)))
    W, H = size
    s = max(W / img.shape[1], H / img.shape[0]) * random.uniform(1.0, 1.4)
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    y, x = random.randint(0, img.shape[0] - H), random.randint(0, img.shape[1] - W)
    return img[y:y + H, x:x + W].copy()


def scene(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    frame = background(FULL)
    taken, objects = [], []
    for _ in range(random.choices([0, 1, 2, 3, 4, 5, 6], [5, 30, 25, 18, 12, 8, 7])[0]):
        sym, truth, is_2d = render_symbol()
        label, corners, module_px = make_label(sym, is_2d, FULL[0])
        poly = place(frame, label, corners, is_2d, taken)
        if poly is not None:
            objects.append({**truth, "cls": int(is_2d), "module_px": round(module_px, 2),
                            "polygon": poly.astype(float).round(1).tolist()})
    # Decoys: text-only labels and stripe patterns, so the detector doesn't fire on any sticker.
    for _ in range(random.choices([0, 1, 2], [50, 35, 15])[0]):
        place(frame, decoy(), np.zeros((4, 2), np.float32), False, taken)
    img, info = degrade(frame)
    return img, objects, info


def decoy():
    h, w = random.randint(60, 260), random.randint(120, 520)
    paper = random.randint(190, 255)
    label = np.full((h, w, 3), paper, np.uint8)
    ink = (random.randint(0, 70),) * 3
    if random.random() < 0.3:  # regular stripes: barcode-like texture with no symbol
        step = random.randint(4, 16)
        for x in range(10, w - 10, step):
            cv2.rectangle(label, (x, 10), (x + step // 2, h - 10), ink, -1)
        return label
    lines = random.randint(1, 5)
    fs = max(0.4, (h - 10) / lines / 35)
    for i in range(lines):
        cv2.putText(label, _alnum(3, 16), (8, int(10 + (i + 0.8) * (h - 10) / lines)), random.choice(
            [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]), fs, ink,
            max(1, int(fs * 1.5)), cv2.LINE_AA)
    return label


def _write(job):
    split, i, seed, out = job
    img, objects, info = scene(seed)
    name = f"{split}_{i:05d}"
    if split == "test":
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(80, 95)])
        return {"image": f"images/{name}.jpg", "width": FULL[0], "height": FULL[1], **info, "barcodes": objects}
    img = cv2.resize(img, TRAIN, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 95)])
    # Oriented boxes (Ultralytics OBB format: class + 4 normalized corners): the decoder needs
    # each barcode's angle, because zxing-cpp's 1D readers only scan rows and columns.
    lines = []
    for o in objects:
        p = (np.array(o["polygon"]) / FULL).clip(0, 1)
        lines.append(f"{o['cls']} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))
    (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n" if lines else "")
    return None


def _init():
    background.files = sorted(BACKGROUNDS.glob("*.jpg"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not BACKGROUNDS.exists():
        raise SystemExit(f"{BACKGROUNDS} missing: download COCO val2017 there first")

    # Seeds are disjoint per split so test scenes never appear in training.
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for split, n, base in [("train", args.train, 0), ("val", args.val, 10**6), ("test", args.test, 2 * 10**6)]:
            if n == 0:
                continue
            out = args.out / split
            (out / "images").mkdir(parents=True, exist_ok=True)
            if split != "test":
                (out / "labels").mkdir(parents=True, exist_ok=True)
            jobs = [(split, i, base + i, out) for i in range(n)]
            gt = [r for r in pool.map(_write, jobs, chunksize=8) if r]
            if split == "test":
                (out / "gt.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gt))
            print(f"{split}: {n} images -> {out}")
    (args.out / "barcodes.yaml").write_text(
        f"path: {args.out}\ntrain: train/images\nval: val/images\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))


if __name__ == "__main__":
    main()
