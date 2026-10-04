"""Detector preprocessing on the GPU: one fused CUDA kernel, compiled at runtime with NVRTC.

    uint8 HWC BGR frame (pinned host memory, read in place by the GPU)
      -> bilinear resize -> place on an S x S canvas (pad) -> per-channel lookup table
      -> float32 CHW, written straight into the TensorRT input buffer (no host->device copy)

The CPU path is cv2.resize + numpy (see ocr.det_preprocess_into, yolo.letterbox_into), and
most of its cost is numpy's float passes over the canvas, not the resize. Orin's GPU shares
DRAM with the CPU and pinned memory is CPU-cached there (equal read/write speed to numpy
memory), so the decoder can write the frame where the kernel reads it.

Output is bit-exact with the CPU path, except at 2560 px (below). The resize reproduces
cv2.resize INTER_LINEAR for uint8 as OpenCV runs it on NEON: the same source taps and 11-bit
weights (tables built on the host the way OpenCV builds them) and the SIMD vertical pass's
rounding, which differs from OpenCV's scalar formula in ~9% of pixels. Normalization is a
256-entry table per channel, filled by the CPU path's own numpy expression, so the floats
match too. At 2560 px the frame is upscaled and OpenCV rounds its two clamped edge rows
some other way: 0.01% of values, one level each. `python -m fieldbench.gpuprep` checks all this.
"""
import ctypes
import time
from pathlib import Path

import numpy as np
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart

COEF_BITS = 11  # OpenCV's INTER_RESIZE_COEF_BITS
ONE = 1 << COEF_BITS

KERNEL = r"""
extern "C" __global__ void letterbox(
    const unsigned char* __restrict__ src, int pitch, int sw, int sh,
    const int2* __restrict__ xtab, const int2* __restrict__ ytab,   // (source index, weight of it)
    const float* __restrict__ lut,                                 // [level * 3 + out channel]
    float* __restrict__ dst, int S, int ox, int oy, int nw, int nh, int swap_rb,
    float p0, float p1, float p2)                                  // padding, already normalized
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= S || y >= S) return;
    size_t plane = (size_t)S * S, o = (size_t)y * S + x;
    int rx = x - ox, ry = y - oy;
    if (rx < 0 || ry < 0 || rx >= nw || ry >= nh) {
        dst[o] = p0; dst[o + plane] = p1; dst[o + 2 * plane] = p2;
        return;
    }
    int2 tx = xtab[rx], ty = ytab[ry];
    int x0 = tx.x * 3, x1 = min(tx.x + 1, sw - 1) * 3, a0 = tx.y, a1 = 2048 - a0;
    const unsigned char* r0 = src + (size_t)ty.x * pitch;
    const unsigned char* r1 = src + (size_t)min(ty.x + 1, sh - 1) * pitch;
    int b0 = ty.y, b1 = 2048 - b0;
#pragma unroll
    for (int ch = 0; ch < 3; ch++) {
        int sc = swap_rb ? 2 - ch : ch;
        int h0 = r0[x0 + sc] * a0 + r0[x1 + sc] * a1;   // horizontal pass, Q11
        int h1 = r1[x0 + sc] * a0 + r1[x1 + sc] * a1;
        // Vertical pass -> uint8, rounded as OpenCV's VResizeLinearVec_32s8u does.
        int v = (((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2;
        dst[o + ch * plane] = lut[v * 3 + ch];
    }
}
"""


def _ck(ret):
    """Unwrap cuda-python's (status, *values) convention for runtime, driver and NVRTC calls."""
    err, *vals = ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    return vals[0] if len(vals) == 1 else (vals or None)


def resize_taps(src_len, dst_len):
    """cv2.resize INTER_LINEAR's per-output (source index, Q11 weight of that index) table.

    Mirrors OpenCV's resize setup: fx = float((d + 0.5) * scale - 0.5) with a double scale,
    clamped at the edges, weight saturate_cast<short>((1 - fx) * 2048) (round half to even)."""
    d = np.arange(dst_len, dtype=np.float64)
    f = ((d + 0.5) * (src_len / dst_len) - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int32)
    f = f - s.astype(np.float32)
    lo, hi = s < 0, s >= src_len - 1
    f[lo], s[lo] = 0, 0
    f[hi], s[hi] = 0, src_len - 1
    w = np.rint((np.float32(1) - f) * np.float32(ONE)).astype(np.int32)
    return np.stack([s, w], 1)


_FUNC = None


def _kernel():
    """Compile once per process for this GPU's architecture (~0.3 s) and load the cubin."""
    global _FUNC
    if _FUNC is None:
        _ck(cudart.cudaFree(0))  # make the runtime's primary context current for the driver API
        dev = _ck(cudart.cudaGetDevice())
        major = _ck(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev))
        minor = _ck(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev))
        prog = _ck(nvrtc.nvrtcCreateProgram(KERNEL.encode(), b"letterbox.cu", 0, [], []))
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(err) != 0:
            size = _ck(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC compile failed:\n{log.decode(errors='replace')}")
        cubin = b" " * _ck(nvrtc.nvrtcGetCUBINSize(prog))
        _ck(nvrtc.nvrtcGetCUBIN(prog, cubin))
        nvrtc.nvrtcDestroyProgram(prog)
        module = _ck(cu.cuModuleLoadData(cubin))
        _FUNC = _ck(cu.cuModuleGetFunction(module, b"letterbox"))
    return _FUNC


class GpuLetterbox:
    """Fill a TrtRunner input tensor (1, 3, S, S) float32 from a uint8 BGR frame on the GPU.

    lut: (256, 3) float32, the normalized value of each uint8 level per *output* channel;
        build it with the CPU path's own expression (ocr.det_lut, yolo.lut) to match it exactly.
    rgb: output channels in RGB order (YOLO) instead of the frame's BGR (PaddleOCR).
    center: centre the resized image (YOLO letterbox) instead of top-left (PaddleOCR).
    pad: padding as a uint8 pixel level (YOLO 114, normalized through lut), or None for 0.0.

    .frame is a pinned uint8 buffer the GPU reads in place; decode into it (decoder `out=`)
    to skip a 12 MB copy per 4 MP frame. Any other array is copied into it first.
    """

    def __init__(self, tensor, stream, lut, rgb=False, center=False, pad=None):
        assert tensor.shape[:2] == (1, 3) and tensor.shape[2] == tensor.shape[3] and tensor.dtype == np.float32
        self.dst, self.S = tensor.device, tensor.shape[-1]
        self.stream = stream
        self.cu_stream = cu.CUstream(int(stream))
        self.rgb, self.center = rgb, center
        self.lut = np.ascontiguousarray(lut, np.float32)
        assert self.lut.shape == (256, 3)
        self.p = np.zeros(3, np.float32) if pad is None else self.lut[pad]
        self._lut_dev = _ck(cudart.cudaMalloc(self.lut.nbytes))
        _ck(cudart.cudaMemcpy(self._lut_dev, self.lut.ctypes.data, self.lut.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        self.func = _kernel()
        self.frame = None
        self._frame_ptr = None
        self.source = None  # another GpuLetterbox whose pinned frame this one reads (share_frame)
        self._tabs = {}  # (h, w) -> (device ptr, nw, nh, scale)
        self.ev0, self.ev1 = _ck(cudart.cudaEventCreate()), _ck(cudart.cudaEventCreate())

    def share_frame(self, other):
        """Read `other`'s pinned frame instead of owning one: several models on one decoded frame."""
        self.source = other

    def _pinned(self, shape):
        if self.source is not None:
            self.frame = self.source._pinned(shape)
            self._frame_dev = self.source._frame_dev
            return self.frame
        if self.frame is None or self.frame.shape != shape:
            if self._frame_ptr is not None:
                cudart.cudaFreeHost(self._frame_ptr)
            n = int(np.prod(shape))
            self._frame_ptr = _ck(cudart.cudaMallocHost(n))
            self.frame = np.frombuffer((ctypes.c_uint8 * n).from_address(self._frame_ptr), np.uint8).reshape(shape)
            self._frame_dev = _ck(cudart.cudaHostGetDevicePointer(self._frame_ptr, 0))
        return self.frame

    def _tables(self, h, w):
        key = (h, w)
        if key not in self._tabs:
            S = self.S
            scale = min(S / w, S / h)
            nw, nh = round(w * scale), round(h * scale)
            tab = np.ascontiguousarray(np.concatenate([resize_taps(w, nw), resize_taps(h, nh)]), np.int32)
            ptr = _ck(cudart.cudaMalloc(tab.nbytes))
            _ck(cudart.cudaMemcpy(ptr, tab.ctypes.data, tab.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            self._tabs[key] = (ptr, nw, nh, scale)
        return self._tabs[key]

    def __call__(self, img):
        """Preprocess `img` into the tensor; synchronous. Returns (scale, pad_x, pad_y, gpu_ms)."""
        h, w = img.shape[:2]
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        frame = self._pinned(img.shape)
        if img.ctypes.data != frame.ctypes.data:
            np.copyto(frame, img)
        tab, nw, nh, scale = self._tables(h, w)
        ox, oy = ((self.S - nw) // 2, (self.S - nh) // 2) if self.center else (0, 0)
        args = ((self._frame_dev, w * 3, w, h, tab, tab + 8 * nw, self._lut_dev,
                 self.dst, self.S, ox, oy, nw, nh, int(self.rgb), *self.p.tolist()),
                (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int, *[ctypes.c_float] * 3))
        bx, by = 32, 8
        _ck(cudart.cudaEventRecord(self.ev0, self.stream))
        _ck(cu.cuLaunchKernel(self.func, (self.S + bx - 1) // bx, (self.S + by - 1) // by, 1, bx, by, 1,
                              0, self.cu_stream, args, 0))
        _ck(cudart.cudaEventRecord(self.ev1, self.stream))
        _ck(cudart.cudaStreamSynchronize(self.stream))
        return scale, ox, oy, _ck(cudart.cudaEventElapsedTime(self.ev0, self.ev1))

    def close(self):
        for ptr, *_ in self._tabs.values():
            cudart.cudaFree(ptr)
        self._tabs.clear()
        cudart.cudaFree(self._lut_dev)
        if self._frame_ptr is not None:
            cudart.cudaFreeHost(self._frame_ptr)
            self._frame_ptr = None
        self.frame = None
        cudart.cudaEventDestroy(self.ev0)
        cudart.cudaEventDestroy(self.ev1)


def _check(n_frames=20, sizes=(640, 1280, 1600, 2560)):
    """CPU path vs GPU kernel on real test frames: numeric agreement and per-frame cost."""
    import cv2

    from . import ocr, yolo
    from .runner import Tensor

    stream = _ck(cudart.cudaStreamCreate())
    sets = {"ocr": sorted(Path("data/ocr/test/images").glob("*.jpg")),
            "barcode": sorted(Path("data/barcodes/test/images").glob("*.jpg"))}
    variants = {"ocr": (ocr.det_preprocess_into, dict(lut=ocr.det_lut())),
                "barcode": (yolo.letterbox_into, dict(lut=yolo.lut(), rgb=True, center=True, pad=114))}
    print(f"{'workload':<9}{'S':>6}{'max lvl':>9}{'differ %':>10}{'max abs':>10}"
          f"{'cpu ms':>9}{'gpu wall':>10}{'kernel':>8}")
    for wl, files in sets.items():
        imgs = [cv2.imread(str(f)) for f in files[:n_frames]]
        cpu_prep, kw = variants[wl]
        for S in sizes:
            t = Tensor("x", (1, 3, S, S), np.float32, True)
            gpu = GpuLetterbox(t, stream, **kw)
            ref = np.empty(t.shape, np.float32)
            worst, differ, max_abs, t_cpu, t_gpu, k_gpu = 0.0, [], 0.0, [], [], []
            for img in imgs:
                gpu(img)  # warm (tables, pinned frame)
                h0 = time.perf_counter()
                cpu_prep(img, ref)
                h1 = time.perf_counter()
                *_, g = gpu(img)
                h2 = time.perf_counter()
                _ck(cudart.cudaMemcpy(t.host_ptr, t.device, t.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
                d = np.abs(t.host - ref)[0]
                lv = d / (gpu.lut[1] - gpu.lut[0])[:, None, None]  # in uint8 levels
                worst, max_abs = max(worst, float(lv.max())), max(max_abs, float(d.max()))
                differ.append(float((lv > 0.5).mean()))
                t_cpu.append((h1 - h0) * 1e3)
                t_gpu.append((h2 - h1) * 1e3)
                k_gpu.append(g)
            print(f"{wl:<9}{S:>6}{worst:>9.2f}{100 * np.mean(differ):>10.4f}{max_abs:>10.2e}"
                  f"{np.median(t_cpu):>9.2f}{np.median(t_gpu):>10.2f}{np.median(k_gpu):>8.2f}")
            gpu.close()
            t.free()
    print("max lvl: largest CPU-vs-GPU difference in uint8 levels; differ %: values off by >0.5 level;"
          "\ncpu ms: cv2+numpy path; gpu wall: copy into pinned frame + kernel + sync; kernel: CUDA events (p50)")


if __name__ == "__main__":
    _check()
