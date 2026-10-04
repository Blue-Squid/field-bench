"""YOLO (Ultralytics head layout) pre- and postprocessing in numpy/OpenCV."""
import cv2
import numpy as np


def letterbox_into(img, out, pad_value=114):
    """Resize a BGR image into the (1, 3, S, S) float buffer `out`, keeping aspect ratio.

    Writes RGB, scaled to [0, 1], centred with gray padding. Returns (scale, pad_x, pad_y),
    which map network coordinates back to the image: x_img = (x_net - pad_x) / scale.
    """
    S = out.shape[-1]
    h, w = img.shape[:2]
    scale = min(S / w, S / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (S - nw) // 2, (S - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((S, S, 3), pad_value, np.uint8)
    canvas[py:py + nh, px:px + nw] = resized
    # HWC BGR uint8 -> CHW RGB float, straight into the (pinned) input buffer.
    chw = canvas[..., ::-1].transpose(2, 0, 1)
    _normalize_into(chw, out[0])
    return scale, px, py


def _normalize_into(u8, out):
    np.multiply(u8, 1 / 255, out=out, casting="unsafe")


def lut():
    """The normalization of every uint8 level, per channel: the GPU kernel's table (gpuprep)."""
    out = np.empty((256, 3), np.float32)
    _normalize_into(np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1), out)
    return out


def postprocess(pred, scale, px, py, conf=0.25, iou=0.5, max_det=100):
    """Raw head output (1, 4 + nc, N) -> (boxes xyxy in image coords, scores, class ids)."""
    p = pred[0]
    scores_all = p[4:]
    cls = scores_all.argmax(0)
    scores = scores_all[cls, np.arange(p.shape[1])]
    keep = scores > conf
    if not keep.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    cx, cy, w, h = p[:4, keep]
    scores, cls = scores[keep], cls[keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    # Class-aware NMS: offset boxes per class so different classes never suppress each other.
    shifted = boxes.copy()
    shifted[:, :2] += cls[:, None].astype(np.float32) * 4096
    idx = cv2.dnn.NMSBoxes(shifted.tolist(), scores.tolist(), conf, iou)
    idx = np.array(idx, dtype=np.int64).reshape(-1)[:max_det]
    b = boxes[idx]
    xyxy = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / scale
    return xyxy.astype(np.float32), scores[idx].astype(np.float32), cls[idx]


def postprocess_obb(pred, scale, px, py, conf=0.25, iou=0.5, max_det=100):
    """Oriented head output (1, 4 + nc + 1, N) -> (rboxes [cx, cy, w, h, angle_rad] in image coords,
    scores, class ids). Angle is the last channel, in radians, rotating clockwise on screen."""
    p = pred[0]
    scores_all = p[4:-1]
    cls = scores_all.argmax(0)
    scores = scores_all[cls, np.arange(p.shape[1])]
    keep = scores > conf
    if not keep.any():
        return np.zeros((0, 5), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    r = np.concatenate([p[:4, keep], p[-1:, keep]]).T  # (n, 5)
    scores, cls = scores[keep], cls[keep]
    off = cls.astype(np.float32) * 4096  # class-aware NMS
    rects = [((x + o, y + o), (w, h), np.degrees(a)) for (x, y, w, h, a), o in zip(r.tolist(), off.tolist())]
    idx = np.array(cv2.dnn.NMSBoxesRotated(rects, scores.tolist(), conf, iou), dtype=np.int64).reshape(-1)[:max_det]
    r = r[idx]
    r[:, 0] = (r[:, 0] - px) / scale
    r[:, 1] = (r[:, 1] - py) / scale
    r[:, 2:4] /= scale
    return r.astype(np.float32), scores[idx].astype(np.float32), cls[idx]
