"""Product recognition: embed the image, then look it up in a gallery of known products (kNN).

  jpeg_decode -> preprocess (BGR->RGB, shorter side to 256, centre crop 224, ImageNet mean/std)
  -> infer (MobileNetV3-L backbone, 960-d embedding) -> match (cosine similarity vs the gallery)

This is the retrieval approach to SKU recognition: adding or changing a product means embedding
a few photos of it into the gallery, with no retraining. There is no localizer: the test photos
(Grocery Store Dataset, host/make_products.py) are already centred on the product, which matches
a point-and-shoot handheld trigger.

The gallery is built in __init__ by running every gallery frame through the same decode,
preprocess and infer path (untimed). Matching is either the nearest class centroid ("centroid",
the default: mean of each class's normalized embeddings) or the nearest gallery image ("knn",
with an optional k-neighbour vote by summed similarity). On the Grocery Store test set (host
check, ONNX Runtime FP32) centroid gets 71.1% top-1 / 96.5% top-5 / 82.6% coarse top-1 against
the nearest neighbour's 67.8% / 95.9% / 81.4%, and it compares against 81 vectors instead of 2,936.

Resize is 256 -> crop 224 (the classic ImageNet recipe). torchvision's own transform for these
IMAGENET1K_V2 weights resizes to 232 with antialiasing; on this set that scored lower (69.0% centroid).
"""
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .jpeg import CpuJpeg, make_decoder
from .pipeline import Clock

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255  # ImageNet, RGB order, on 0..255 pixels
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
RESIZE, CROP = 256, 224
TOPN = 5


def load_frames(root):
    """Set written by host/make_products.py: [{jpeg, image, label, coarse}]."""
    root = Path(root)
    frames = []
    for line in (root / "gt.jsonl").read_text().splitlines():
        g = json.loads(line)
        g["jpeg"] = (root / g["image"]).read_bytes()
        frames.append(g)
    return frames


def preprocess_into(img, out):
    """BGR uint8 -> (1, 3, 224, 224) float32 in `out`: torchvision's Resize(256) + CenterCrop(224)
    + Normalize, with cv2 INTER_LINEAR (no antialias) for the resize."""
    h, w = img.shape[:2]
    if h <= w:  # torchvision truncates the long side
        nh, nw = RESIZE, int(RESIZE * w / h)
    else:
        nh, nw = int(RESIZE * h / w), RESIZE
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = int(round((nh - CROP) / 2.0)), int(round((nw - CROP) / 2.0))
    crop = cv2.cvtColor(small[top:top + CROP, left:left + CROP], cv2.COLOR_BGR2RGB)
    out[0] = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)


def _l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def score(frames, outputs):
    n = len(frames)
    top1 = sum(o["labels"][0] == f["label"] for f, o in zip(frames, outputs))
    top5 = sum(f["label"] in o["labels"][:TOPN] for f, o in zip(frames, outputs))
    coarse = sum(o["coarse"] == f["coarse"] for f, o in zip(frames, outputs))
    return {"images": n, "top1": top1 / n, "top5": top5 / n, "top1_coarse": coarse / n}


class ProductPipeline:
    stages = ["jpeg_decode", "preprocess", "infer", "match"]

    def __init__(self, runner, gallery_frames, onnx_name, precision, jpeg=None, jpeg_name="cpu", k=1,
                 method="centroid", engine_path=None):
        if method not in ("knn", "centroid"):
            raise ValueError(f"method must be knn or centroid, not {method!r}")
        self.name = "product"
        self.runner = runner
        self.engine_path = engine_path
        self.input = runner.inputs[0].host
        self.output = runner.outputs[0].host
        self.jpeg = jpeg or CpuJpeg()
        self.k, self.method = int(k), method

        # Gallery: every known-product image through the same decode/preprocess/infer path.
        t0 = time.monotonic()
        emb = np.stack([self._embed(f["jpeg"], {}) for f in gallery_frames])
        self.gallery_build_s = time.monotonic() - t0
        self.gallery = np.ascontiguousarray(_l2n(emb), dtype=np.float32)  # N x D, unit rows
        self.classes = sorted({f["label"] for f in gallery_frames})
        cid = {c: i for i, c in enumerate(self.classes)}
        self.gallery_ids = np.array([cid[f["label"]] for f in gallery_frames], np.int64)
        self.coarse_of = {f["label"]: f.get("coarse") for f in gallery_frames}
        self.centroids = np.ascontiguousarray(_l2n(np.stack(
            [self.gallery[self.gallery_ids == i].mean(0) for i in range(len(self.classes))])), dtype=np.float32)

        self.config = {"mode": f"embed-{method}", "jpeg": jpeg_name, "prep": "cpu", "detector": onnx_name,
                       "precision": precision, "input_size": CROP, "k": self.k if method == "knn" else None,
                       "gallery_size": len(gallery_frames), "gallery_classes": len(self.classes),
                       "embed_dim": int(self.gallery.shape[1])}

    def _embed(self, jpeg, times):
        clock = Clock(times)
        with clock("jpeg_decode"):
            img = self.jpeg.decode(jpeg)
        with clock("preprocess"):
            preprocess_into(img, self.input)
        with clock("infer"):
            gpu_ms, _ = self.runner.infer()
        times["infer_gpu"] = gpu_ms
        return np.array(self.output, dtype=np.float32).reshape(-1)

    def _rank(self, q):
        """Unit query embedding -> (top labels, their similarities), best first."""
        if self.method == "centroid":
            sims = self.centroids @ q
            order = np.argsort(-sims)[:TOPN]
            return [self.classes[i] for i in order], sims[order].tolist()
        sims = self.gallery @ q
        m = min(len(sims), max(64, self.k))
        order = np.argpartition(-sims, m - 1)[:m]
        order = order[np.argsort(-sims[order])]
        if len(set(self.gallery_ids[order].tolist())) < TOPN:  # rare: fewer than 5 classes in the top 64
            order = np.argsort(-sims)
        # Classes in neighbour order with their best similarity (enough for the top 5 and the k-vote).
        best, votes = {}, {}
        for rank, i in enumerate(order):
            c = int(self.gallery_ids[i])
            best.setdefault(c, float(sims[i]))
            if rank < self.k:  # vote over the k nearest neighbours by summed similarity
                votes[c] = votes.get(c, 0.0) + float(sims[i])
            elif len(best) >= TOPN:
                break
        ranked = sorted(votes, key=votes.get, reverse=True)  # k=1: just the nearest neighbour's class
        ranked = (ranked + [c for c in best if c not in votes])[:TOPN]
        return [self.classes[c] for c in ranked], [best[c] for c in ranked]

    def process(self, jpeg, times):
        e = self._embed(jpeg, times)
        with Clock(times)("match"):
            labels, sims = self._rank(_l2n(e))
        return {"labels": labels, "sims": sims, "coarse": self.coarse_of[labels[0]]}

    def score(self, frames, outputs):
        return score(frames, outputs)

    def clone(self):
        """Independent copy for another worker: its own runner (same engine) and decoder, same gallery."""
        if self.engine_path is None:
            raise RuntimeError("clone() needs the engine path: pass engine_path= to ProductPipeline")
        from .runner import TrtRunner

        new = copy.copy(self)  # shares the read-only gallery arrays and class tables
        new.runner = TrtRunner(self.engine_path)
        new.input = new.runner.inputs[0].host
        new.output = new.runner.outputs[0].host
        new.jpeg = make_decoder(self.config["jpeg"])
        new.config = dict(self.config)
        return new

    def close(self):
        self.runner.close()
        self.jpeg.close()
