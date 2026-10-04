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


class _CalibratorMixin:
    """Feeds preprocessed (1, C, H, W) float32 batches to TensorRT's INT8 calibration.

    Mixed into one of TensorRT's calibrator classes (CALIBRATORS); the class decides how
    the recorded activation ranges become scales.
    """

    def __init__(self, batches, cache_file):
        super().__init__()
        from cuda.bindings import runtime as cudart

        self.cudart = cudart
        self.batches = iter(batches)
        self.cache_file = Path(cache_file)
        self.device = None
        self.nbytes = None

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        try:
            batch = next(self.batches)
        except StopIteration:
            return None
        cudart = self.cudart
        if self.device is None:
            self.nbytes = batch.nbytes
            _, self.device = cudart.cudaMalloc(self.nbytes)
        cudart.cudaMemcpy(self.device, batch.ctypes.data, self.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        return [int(self.device)]

    def read_calibration_cache(self):
        return self.cache_file.read_bytes() if self.cache_file.exists() else None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(bytes(cache))

    def free(self):
        if self.device is not None:
            self.cudart.cudaFree(self.device)


# How activation ranges become INT8 scales. No single choice fits every model:
#   minmax   never clips. Right for YOLO, whose head mixes box coordinates (0..S px) with
#            scores (0..1) in one tensor, where clipping the large values breaks boxes.
#   entropy  clips rare outliers (KL-divergence threshold) for finer steps in the bulk. Right
#            for PP-OCR's DB detector: its neck has activations up to ~1,500, and MinMax scales
#            built on those crushed the probability map (lines found 90% -> 47% at 1280 px).
CALIBRATORS = {"minmax": trt.IInt8MinMaxCalibrator, "entropy": trt.IInt8EntropyCalibrator2}


def build_calibrated(onnx, batches, calib_id, calibrator="minmax", workspace_mb=1024, engine_dir="engines",
                     force=False, log=print):
    """INT8 engine calibrated on real inputs (FP16 fallback). Returns (engine path, build seconds or None).

    `batches` yields preprocessed float32 arrays shaped like the network input; `calib_id`
    names the calibration set so engines calibrated on different data don't collide.
    `workspace_mb` caps tactic scratch memory: on the 8 GB Orin Nano, bigger requests fail
    in NvMap (ENOMEM) and TensorRT skips those tactics anyway, just noisily.
    """
    onnx = Path(onnx)
    target = (Path(engine_dir) /
              f"{onnx.stem}.int8cal-{calibrator}-{calib_id}.trt{trt.__version__}.{_digest(onnx)}.engine")
    if target.exists() and not force:
        return target, None
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError(f"ONNX parse failed: {[str(parser.get_error(i)) for i in range(parser.num_errors)]}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    cls = type("Calibrator", (_CalibratorMixin, CALIBRATORS[calibrator]), {})
    calib = cls(batches, target.with_suffix(".calib"))
    config.int8_calibrator = calib
    log(f"  calibrating + building {target.name} (several minutes) ...")
    t0 = time.monotonic()
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        calib.free()
    if plan is None:
        raise RuntimeError(f"INT8 calibrated build failed for {onnx}")
    tmp = target.with_suffix(".partial")
    tmp.write_bytes(bytes(plan))
    os.replace(tmp, target)
    return target, time.monotonic() - t0


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
