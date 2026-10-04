"""Build and cache TensorRT engines from ONNX using trtexec."""
import hashlib
import os
import subprocess
import time
from pathlib import Path

import tensorrt as trt

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# int8 keeps fp16 enabled so layers without int8 kernels fall back to fp16, not fp32.
PRECISION_FLAGS = {"fp32": [], "fp16": ["--fp16"], "int8": ["--int8", "--fp16"]}


def _digest(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def engine_path(onnx, precision, engine_dir):
    onnx = Path(onnx)
    return Path(engine_dir) / f"{onnx.stem}.{precision}.trt{trt.__version__}.{_digest(onnx)}.engine"


def build(onnx, precision, engine_dir="engines", force=False):
    """Return (engine path, build seconds or None if served from cache)."""
    target = engine_path(onnx, precision, engine_dir)
    if target.exists() and not force:
        return target, None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial")
    cmd = [TRTEXEC, f"--onnx={onnx}", f"--saveEngine={tmp}", "--skipInference", *PRECISION_FLAGS[precision]]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.monotonic() - t0
    target.with_suffix(".log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise RuntimeError(f"trtexec failed for {onnx} [{precision}]:\n{tail}")
    os.replace(tmp, target)
    return target, seconds
