"""JPEG decoders with one interface, so pipelines can compare CPU and hardware decode per frame.

    dec = make_decoder("cpu" | "nvjpg" | "nvjpg-vic" | "nvjpg-gst")
    img = dec.decode(jpeg_bytes)             # BGR uint8 HxWx3, C-contiguous
    y   = dec.decode(jpeg_bytes, gray=True)  # luma uint8 HxW
    dec.close()

CpuJpeg   cv2.imdecode (libjpeg-turbo on the CPU), exactly what the pipelines did before.
NvJpeg    the Jetson NVJPG engine through Tegra libnvjpeg.so (libjpeg-8b API with NVIDIA's
          hardware extensions, the path MMAPI's NvJPEGDecoder::decodeToFd uses), driven by a
          small C++ shim compiled on first use with g++ against /usr/src/jetson_multimedia_api
          and cached in ~/.cache/fieldbench/ (needs no sudo). Output is bit-exact with
          cv2.imdecode for both gray and BGR (see the NvJpeg docstring for the variants).
          JPEGs the hardware path does not take (grayscale, progressive, CMYK; 4:2:2/4:4:4
          for BGR) fall back to cv2 and are counted in .fallbacks.
GstNvJpeg the same engine through GStreamer (appsrc ! jpegparse ! nvjpegdec ! I420 ! appsink),
          kept for comparison: no compiler needed, but it copies out of NVMM on the CPU and
          converts BGR with OpenCV, so it is slower than CpuJpeg for BGR.

Measured on Orin Nano Super, MAXN_SUPER, 2304x1728 4:2:0, p50 per frame (wall / CPU time):
    cpu    gray 21 / 21.6 ms   bgr 36 / 36 ms
    nvjpg  gray 17 / 2.9 ms    bgr 28 / 22 ms     (NVJPG decode itself ~7 ms of that)
Not thread-safe: use one decoder per thread. ctypes releases the GIL, so two decoders in two
threads overlap (gray 56 -> 156 fps). The first NvJpeg decode is slow (~100 ms setup), so warm
up before timing. Unlike cv2.IMREAD_COLOR, the hardware paths ignore EXIF orientation.
"""
import ctypes as C
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

MMAPI = Path("/usr/src/jetson_multimedia_api")
TEGRA_LIB = Path("/usr/lib/aarch64-linux-gnu/tegra")
CACHE = Path(os.environ.get("FIELDBENCH_CACHE", Path.home() / ".cache" / "fieldbench"))


class CpuJpeg:
    name = "cpu"

    def decode(self, jpeg, gray=False, out=None):  # out: accepted for API parity, unused
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), flag)
        if img is None:
            raise ValueError("cv2.imdecode failed")
        return img

    def close(self):
        pass


_SHIM = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csetjmp>
#include <cstdint>
#include <ctime>
#include <arm_neon.h>
#include "jpeglib.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

extern "C" {

struct ErrMgr { jpeg_error_mgr pub; jmp_buf jb; char msg[JMSG_LENGTH_MAX]; };
static void err_exit(j_common_ptr c) {
    ErrMgr *e = (ErrMgr *)c->err;
    (*c->err->format_message)(c, e->msg);
    longjmp(e->jb, 1);
}
static void quiet(j_common_ptr, int) {}

struct Dec {
    jpeg_decompress_struct cinfo;
    ErrMgr jerr;
    NvBufSurface *src = nullptr;   // decoder-owned output of the last decode
    NvBufSurface *dst = nullptr;   // our packed BGRx surface for the VIC conversion
    int dst_w = 0, dst_h = 0;
    int w = 0, h = 0;
    NvBufSurface *mid = nullptr;   // our cached staging copy of the decode (same format)
    int mid_w = 0, mid_h = 0, mid_fmt = -1;
    NvBufSurface *rd = nullptr;    // surface being read: mid (staged) or src (direct)
    int stage = 1;
    unsigned char *scratch = nullptr;  // rows / planes for the CPU color path
    size_t scratch_n = 0;
    double t[4] = {0, 0, 0, 0};
    char err[JMSG_LENGTH_MAX + 64];
};

void *fbj_create(void) {
    Dec *d = new Dec();
    memset(&d->cinfo, 0, sizeof(d->cinfo));
    d->cinfo.err = jpeg_std_error(&d->jerr.pub);
    d->jerr.pub.error_exit = err_exit;   // libjpeg's default calls exit()
    d->jerr.pub.emit_message = quiet;
    d->err[0] = 0;
    jpeg_create_decompress(&d->cinfo);
    d->cinfo.mjpeg_decode = TRUE;        // as MMAPI's NvJPEGDecoder does
    return d;
}

const char *fbj_error(void *h) { return ((Dec *)h)->err; }

// Decode one JPEG on NVJPG (mirrors NvJPEGDecoder::decodeToFd). 0 on success, -1 on error,
// -2 if the JPEG is a kind the hardware path does not take (caller falls back to CPU).
int fbj_decode(void *h, const unsigned char *buf, unsigned long len, int *w, int *hh, int *colorfmt) {
    Dec *d = (Dec *)h;
    jpeg_decompress_struct &c = d->cinfo;
    NvBufSurface vendor;
    d->src = nullptr;
    if (setjmp(d->jerr.jb)) {
        snprintf(d->err, sizeof d->err, "libnvjpeg: %s", d->jerr.msg);
        jpeg_abort_decompress(&c);
        return -1;
    }
    jpeg_mem_src(&c, (unsigned char *)buf, len);
    c.out_color_space = JCS_YCbCr;
    jpeg_read_header(&c, TRUE);
    if (c.num_components != 3 || c.jpeg_color_space != JCS_YCbCr || c.progressive_mode) {
        snprintf(d->err, sizeof d->err, "unsupported JPEG (components=%d colorspace=%d progressive=%d)",
                 c.num_components, (int)c.jpeg_color_space, (int)c.progressive_mode);
        jpeg_abort_decompress(&c);
        return -2;
    }
    c.out_color_space = JCS_YCbCr;
    c.IsVendorbuf = TRUE;
    c.pVendor_buf = (unsigned char *)&vendor;
    jpeg_start_decompress(&c);
    if (c.global_state != 202 /* DSTATE_READY */) {
        snprintf(d->err, sizeof d->err, "format not supported by the libnvjpeg hardware path");
        jpeg_abort_decompress(&c);
        return -2;
    }
    jpeg_read_raw_data(&c, NULL, c.comp_info[0].v_samp_factor * DCTSIZE);
    jpeg_finish_decompress(&c);
    NvBufSurface *s = nullptr;
    if (NvBufSurfaceFromFd(c.fd, (void **)&s) != 0 || !s) {
        snprintf(d->err, sizeof d->err, "NvBufSurfaceFromFd(%d) failed", c.fd);
        return -1;
    }
    d->src = s;
    d->w = c.image_width;
    d->h = c.image_height;
    *w = d->w; *hh = d->h;
    *colorfmt = s->surfaceList[0].colorFormat;
    return 0;
}

static inline double now_ms() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

// Timing of the last fbj_get_* call: [map+sync, copy/convert, VIC, total] in ms.
void fbj_times(void *h, double *t) { memcpy(t, ((Dec *)h)->t, sizeof ((Dec *)h)->t); }

// Make the last decode readable by the CPU; sets d->rd to the surface to read.
// The decoder's own buffer maps uncached (CPU reads run at ~0.6 GB/s). With d->stage = 1, VIC
// first copies it, same format, into our cached surface (costs VIC time, saves CPU; VIC also
// perturbs chroma slightly). With stage = 0 the decoder buffer is mapped and read directly.
// Either way, call release_src() afterwards.
static int map_src(Dec *d) {
    if (!d->src) { snprintf(d->err, sizeof d->err, "no decoded frame"); return -1; }
    NvBufSurfaceParams &sp = d->src->surfaceList[0];
    if (d->stage) {
        double t0 = now_ms();
        if (!d->mid || d->mid_w != d->w || d->mid_h != d->h || d->mid_fmt != (int)sp.colorFormat) {
            if (d->mid) { NvBufSurfaceUnMap(d->mid, 0, -1); NvBufSurfaceDestroy(d->mid); d->mid = nullptr; }
            NvBufSurfaceCreateParams cp;
            memset(&cp, 0, sizeof cp);
            cp.width = d->w;
            cp.height = d->h;
            cp.colorFormat = sp.colorFormat;
            cp.layout = NVBUF_LAYOUT_PITCH;
            cp.memType = NVBUF_MEM_SURFACE_ARRAY;
            if (NvBufSurfaceCreate(&d->mid, 1, &cp) != 0) {
                snprintf(d->err, sizeof d->err, "NvBufSurfaceCreate(stage, fmt %d) failed", (int)sp.colorFormat);
                d->mid = nullptr;
                return -1;
            }
            d->mid->numFilled = 1;
            if (NvBufSurfaceMap(d->mid, 0, -1, NVBUF_MAP_READ) != 0) {
                snprintf(d->err, sizeof d->err, "NvBufSurfaceMap(stage) failed");
                NvBufSurfaceDestroy(d->mid); d->mid = nullptr;
                return -1;
            }
            d->mid_w = d->w; d->mid_h = d->h; d->mid_fmt = sp.colorFormat;
        }
        NvBufSurfTransformParams tp;
        memset(&tp, 0, sizeof tp);
        NvBufSurfTransform_Error e = NvBufSurfTransform(d->src, d->mid, &tp);
        if (e != NvBufSurfTransformError_Success) {
            snprintf(d->err, sizeof d->err, "NvBufSurfTransform(stage) failed (%d)", (int)e);
            return -1;
        }
        double t1 = now_ms();
        NvBufSurfaceSyncForCpu(d->mid, 0, -1);
        d->rd = d->mid;
        d->t[2] = t1 - t0;
        d->t[0] = now_ms() - t1;
        return 0;
    }
    double t0 = now_ms();
    if (NvBufSurfaceMap(d->src, 0, -1, NVBUF_MAP_READ) != 0) {
        snprintf(d->err, sizeof d->err, "NvBufSurfaceMap(src) failed");
        return -1;
    }
    // The decoder recycles one buffer under the same fd, so map per call: a mapping
    // kept across decodes returned stale pixels.
    NvBufSurfaceSyncForCpu(d->src, 0, -1);
    d->rd = d->src;
    d->t[2] = 0;
    d->t[0] = now_ms() - t0;
    return 0;
}

static void release_src(Dec *d) {
    if (d->rd == d->src) NvBufSurfaceUnMap(d->src, 0, -1);
    d->rd = nullptr;
}

void fbj_set_stage(void *h, int stage) { ((Dec *)h)->stage = stage; }

// Copy the luma plane of the last decode into out (rows of out_stride bytes).
int fbj_get_gray(void *h, unsigned char *out, int out_stride) {
    Dec *d = (Dec *)h;
    double t0 = now_ms();
    if (map_src(d) != 0) return -1;
    double t1 = now_ms();
    NvBufSurfaceParams &p = d->rd->surfaceList[0];
    int pitch = p.planeParams.pitch[0];
    const unsigned char *s = (const unsigned char *)p.mappedAddr.addr[0];
    if (pitch == out_stride && pitch == d->w)
        memcpy(out, s, (size_t)d->w * d->h);
    else
        for (int y = 0; y < d->h; y++) memcpy(out + (size_t)y * out_stride, s + (size_t)y * pitch, d->w);
    double t2 = now_ms();
    release_src(d);
    d->t[1] = t2 - t1; d->t[3] = now_ms() - t0;
    return 0;
}

// libjpeg's h2v2 "fancy" (triangle) chroma upsampling of one output row (jdsample.c):
// vertical 3:1 blend of the nearer and farther chroma rows, then horizontal 3:1 blends.
static void upsample_row(const unsigned char *nr, const unsigned char *fr, int cw, int W,
                         uint16_t *cs, unsigned char *o) {
    int i = 0;
    const uint8x8_t three8 = vdup_n_u8(3);
    for (; i + 8 <= cw; i += 8) vst1q_u16(cs + i, vmlal_u8(vmovl_u8(vld1_u8(fr + i)), vld1_u8(nr + i), three8));
    for (; i < cw; i++) cs[i] = 3 * nr[i] + fr[i];
    if (cw == 1) { o[0] = (cs[0] * 4 + 8) >> 4; if (W > 1) o[1] = (cs[0] * 4 + 7) >> 4; return; }
    o[0] = (cs[0] * 4 + 8) >> 4;
    o[1] = (3 * cs[0] + cs[1] + 7) >> 4;
    i = 1;
    const uint16x8_t three = vdupq_n_u16(3), k8 = vdupq_n_u16(8), k7 = vdupq_n_u16(7);
    for (; i + 8 <= cw - 1; i += 8) {
        uint16x8_t c3 = vmulq_u16(vld1q_u16(cs + i), three);
        uint16x8_t ev = vaddq_u16(vaddq_u16(c3, vld1q_u16(cs + i - 1)), k8);
        uint16x8_t od = vaddq_u16(vaddq_u16(c3, vld1q_u16(cs + i + 1)), k7);
        uint8x8x2_t r;
        r.val[0] = vshrn_n_u16(ev, 4);
        r.val[1] = vshrn_n_u16(od, 4);
        vst2_u8(o + 2 * i, r);
    }
    for (; i < cw - 1; i++) {
        o[2 * i] = (3 * cs[i] + cs[i - 1] + 8) >> 4;
        o[2 * i + 1] = (3 * cs[i] + cs[i + 1] + 7) >> 4;
    }
    i = cw - 1;
    o[2 * i] = (3 * cs[i] + cs[i - 1] + 8) >> 4;
    if (2 * i + 1 < W) o[2 * i + 1] = (cs[i] * 4 + 7) >> 4;
}

static inline unsigned char clamp8(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

// JFIF full-range YCbCr -> BGR, bit-exact with libjpeg's fixed-point math (jdcolor.c):
//   R = y + ((91881*cr + 32768) >> 16), B = y + ((116130*cb + 32768) >> 16),
//   G = y + ((-22554*cb - 46802*cr + 32768) >> 16). Constants above 32767 are split
// into a multiple of 65536 (an exact integer term) plus an int16 remainder.
static inline int16x8_t mulhi_round(int16x8_t a, int16_t ka, int16x8_t b, int16_t kb) {
    int32x4_t lo = vmull_n_s16(vget_low_s16(a), ka), hi = vmull_n_s16(vget_high_s16(a), ka);
    if (kb) { lo = vmlal_n_s16(lo, vget_low_s16(b), kb); hi = vmlal_n_s16(hi, vget_high_s16(b), kb); }
    const int32x4_t half = vdupq_n_s32(32768);
    return vcombine_s16(vshrn_n_s32(vaddq_s32(lo, half), 16), vshrn_n_s32(vaddq_s32(hi, half), 16));
}

static void ycc_to_bgr_row(const unsigned char *Y, const unsigned char *Cb, const unsigned char *Cr,
                           unsigned char *o, int W) {
    int x = 0;
    const int16x8_t k128 = vdupq_n_s16(128);
    for (; x + 8 <= W; x += 8) {
        int16x8_t y = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(Y + x)));
        int16x8_t cb = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(Cb + x))), k128);
        int16x8_t cr = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(Cr + x))), k128);
        int16x8_t r = vaddq_s16(vaddq_s16(y, cr), mulhi_round(cr, 26345, cr, 0));                // 91881 = 65536 + 26345
        int16x8_t b = vaddq_s16(vaddq_s16(y, vshlq_n_s16(cb, 1)), mulhi_round(cb, -14942, cb, 0)); // 116130 = 131072 - 14942
        int16x8_t g = vsubq_s16(y, cr);
        g = vaddq_s16(g, mulhi_round(cb, -22554, cr, 18734));                                  // -46802 = -65536 + 18734
        uint8x8x3_t px;
        px.val[0] = vqmovun_s16(b);
        px.val[1] = vqmovun_s16(g);
        px.val[2] = vqmovun_s16(r);
        vst3_u8(o + 3 * x, px);
    }
    for (; x < W; x++) {
        int y = Y[x], cb = Cb[x] - 128, cr = Cr[x] - 128;
        o[3 * x + 2] = clamp8(y + ((91881 * cr + 32768) >> 16));
        o[3 * x + 1] = clamp8(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
        o[3 * x + 0] = clamp8(y + ((116130 * cb + 32768) >> 16));
    }
}

// Full-range color conversion of the last decode on the CPU (4:2:0 only): libjpeg's fancy
// chroma upsampling + YCbCr->BGR, so the result matches cv2.imdecode (libjpeg-turbo).
// Reading straight from the uncached decoder buffer is slow for this access pattern, so in
// direct mode the chroma planes and each luma row are first bulk-copied into cached scratch.
int fbj_get_bgr(void *h, unsigned char *out, int out_stride) {
    Dec *d = (Dec *)h;
    if (!d->src) { snprintf(d->err, sizeof d->err, "no decoded frame"); return -1; }
    NvBufSurfaceParams &p = d->src->surfaceList[0];
    if (p.colorFormat != NVBUF_COLOR_FORMAT_YUV420 || p.planeParams.num_planes != 3) {
        snprintf(d->err, sizeof d->err, "CPU color path handles 4:2:0 only (format %d)", (int)p.colorFormat);
        return -2;
    }
    double t0 = now_ms();
    if (map_src(d) != 0) return -1;
    double t1 = now_ms();
    NvBufSurfaceParams &q = d->rd->surfaceList[0];
    int W = d->w, H = d->h, cw = (W + 1) / 2, ch = (H + 1) / 2;
    const unsigned char *Y = (const unsigned char *)q.mappedAddr.addr[0];
    const unsigned char *U = (const unsigned char *)q.mappedAddr.addr[1];
    const unsigned char *V = (const unsigned char *)q.mappedAddr.addr[2];
    int py = q.planeParams.pitch[0], pu = q.planeParams.pitch[1], pv = q.planeParams.pitch[2];
    bool direct = d->rd == d->src;
    size_t need = (size_t)(W + 32) * 3 + (size_t)(cw + 16) * 2 + (direct ? (size_t)cw * ch * 2 : 0) + 64;
    if (d->scratch_n < need) {
        free(d->scratch);
        d->scratch = (unsigned char *)malloc(need);
        d->scratch_n = need;
    }
    unsigned char *cb = d->scratch, *cr = cb + W + 32, *yrow = cr + W + 32;
    uint16_t *cs = (uint16_t *)(((uintptr_t)(yrow + W + 32) + 15) & ~(uintptr_t)15);
    if (direct) {
        unsigned char *Uc = (unsigned char *)(cs + cw + 16), *Vc = Uc + (size_t)cw * ch;
        for (int r = 0; r < ch; r++) {
            memcpy(Uc + (size_t)r * cw, U + (size_t)r * pu, cw);
            memcpy(Vc + (size_t)r * cw, V + (size_t)r * pv, cw);
        }
        U = Uc; V = Vc; pu = pv = cw;
    }
    for (int y = 0; y < H; y++) {
        int n = y >> 1, f = (y & 1) ? n + 1 : n - 1;
        if (f < 0) f = 0;
        if (f > ch - 1) f = ch - 1;
        upsample_row(U + (size_t)n * pu, U + (size_t)f * pu, cw, W, cs, cb);
        upsample_row(V + (size_t)n * pv, V + (size_t)f * pv, cw, W, cs, cr);
        const unsigned char *yr = Y + (size_t)y * py;
        if (direct) { memcpy(yrow, yr, W); yr = yrow; }
        ycc_to_bgr_row(yr, cb, cr, out + (size_t)y * out_stride, W);
    }
    double t2 = now_ms();
    release_src(d);
    d->t[1] = t2 - t1; d->t[3] = now_ms() - t0;
    return 0;
}

static int ensure_dst(Dec *d) {
    if (d->dst && d->dst_w == d->w && d->dst_h == d->h) return 0;
    if (d->dst) { NvBufSurfaceUnMap(d->dst, 0, 0); NvBufSurfaceDestroy(d->dst); d->dst = nullptr; }
    NvBufSurfaceCreateParams cp;
    memset(&cp, 0, sizeof cp);
    cp.width = d->w;
    cp.height = d->h;
    cp.colorFormat = NVBUF_COLOR_FORMAT_BGRx;
    cp.layout = NVBUF_LAYOUT_PITCH;
    cp.memType = NVBUF_MEM_SURFACE_ARRAY;
    if (NvBufSurfaceCreate(&d->dst, 1, &cp) != 0) {
        snprintf(d->err, sizeof d->err, "NvBufSurfaceCreate(BGRx) failed");
        d->dst = nullptr;
        return -1;
    }
    d->dst->numFilled = 1;
    if (NvBufSurfaceMap(d->dst, 0, 0, NVBUF_MAP_READ) != 0) {
        snprintf(d->err, sizeof d->err, "NvBufSurfaceMap(dst) failed");
        NvBufSurfaceDestroy(d->dst); d->dst = nullptr;
        return -1;
    }
    d->dst_w = d->w; d->dst_h = d->h;
    return 0;
}

static void bgrx_to_bgr_row(const unsigned char *s, unsigned char *o, int w) {
    int x = 0;
    for (; x + 16 <= w; x += 16) {
        uint8x16x4_t v = vld4q_u8(s + 4 * x);
        uint8x16x3_t t; t.val[0] = v.val[0]; t.val[1] = v.val[1]; t.val[2] = v.val[2];
        vst3q_u8(o + 3 * x, t);
    }
    for (; x < w; x++) { o[3*x] = s[4*x]; o[3*x+1] = s[4*x+1]; o[3*x+2] = s[4*x+2]; }
}

// VIC color conversion (BGRx), then copy out as BGR (channels 3) or BGRx (4). Fast, but
// VIC takes the decoder's surface as limited-range BT.601 (relabeling it does not help), so
// colors are contrast-stretched and clipped vs JFIF full range. Kept for comparison.
int fbj_get_bgr_vic(void *h, unsigned char *out, int out_stride, int channels) {
    Dec *d = (Dec *)h;
    if (!d->src) { snprintf(d->err, sizeof d->err, "no decoded frame"); return -1; }
    double t0 = now_ms();
    if (ensure_dst(d) != 0) return -1;
    NvBufSurfTransformParams tp;
    memset(&tp, 0, sizeof tp);
    tp.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    tp.transform_filter = NvBufSurfTransformInter_Nearest;
    NvBufSurfTransform_Error e = NvBufSurfTransform(d->src, d->dst, &tp);
    if (e != NvBufSurfTransformError_Success) {
        snprintf(d->err, sizeof d->err, "NvBufSurfTransform failed (%d)", (int)e);
        return -1;
    }
    double t1 = now_ms();
    NvBufSurfaceSyncForCpu(d->dst, 0, 0);
    double t2 = now_ms();
    NvBufSurfaceParams &dp = d->dst->surfaceList[0];
    int pitch = dp.planeParams.pitch[0];
    const unsigned char *s = (const unsigned char *)dp.mappedAddr.addr[0];
    for (int y = 0; y < d->h; y++) {
        const unsigned char *sr = s + (size_t)y * pitch;
        unsigned char *dr = out + (size_t)y * out_stride;
        if (channels == 3) bgrx_to_bgr_row(sr, dr, d->w);
        else memcpy(dr, sr, (size_t)d->w * 4);
    }
    double t3 = now_ms();
    d->t[0] = t2 - t1; d->t[1] = t3 - t2; d->t[2] = t1 - t0; d->t[3] = t3 - t0;
    return 0;
}

void fbj_destroy(void *h) {
    Dec *d = (Dec *)h;
    if (!d) return;
    if (d->dst) { NvBufSurfaceUnMap(d->dst, 0, 0); NvBufSurfaceDestroy(d->dst); }
    if (d->mid) { NvBufSurfaceUnMap(d->mid, 0, -1); NvBufSurfaceDestroy(d->mid); }
    free(d->scratch);
    jpeg_destroy_decompress(&d->cinfo);
    delete d;
}

}  // extern "C"
"""


def _build_shim():
    """Compile the shim once per source version; returns the .so path."""
    tag = hashlib.sha1(_SHIM.encode()).hexdigest()[:12]
    so = CACHE / f"libfbnvjpg-{tag}.so"
    if so.exists():
        return so
    if not (MMAPI / "include" / "libjpeg-8b" / "jpeglib.h").exists():
        raise RuntimeError(f"Jetson Multimedia API headers not found under {MMAPI}")
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        src = Path(tmp) / "fbnvjpg.cpp"
        src.write_text(_SHIM)
        out = Path(tmp) / so.name
        cmd = ["g++", "-O3", "-shared", "-fPIC", "-o", str(out), str(src),
               f"-I{MMAPI / 'include'}", f"-I{MMAPI / 'include' / 'libjpeg-8b'}",
               f"-L{TEGRA_LIB}", f"-Wl,-rpath,{TEGRA_LIB}",
               "-lnvjpeg", "-lnvbufsurface", "-lnvbufsurftransform"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"building the NVJPG shim failed:\n{r.stderr}")
        os.replace(out, so)  # atomic, so concurrent first uses don't see a partial file
    for stale in CACHE.glob("libfbnvjpg-*.so"):  # builds of older shim versions
        if stale != so:
            stale.unlink(missing_ok=True)
    return so


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        L = C.CDLL(str(_build_shim()))
        vp, ip = C.c_void_p, C.POINTER(C.c_int)
        L.fbj_create.restype = vp
        L.fbj_create.argtypes = []
        L.fbj_error.restype = C.c_char_p
        L.fbj_error.argtypes = [vp]
        L.fbj_decode.argtypes = [vp, C.c_char_p, C.c_ulong, ip, ip, ip]
        L.fbj_get_gray.argtypes = [vp, vp, C.c_int]
        L.fbj_get_bgr.argtypes = [vp, vp, C.c_int]
        L.fbj_get_bgr_vic.argtypes = [vp, vp, C.c_int, C.c_int]
        L.fbj_times.argtypes = [vp, C.POINTER(C.c_double)]
        L.fbj_times.restype = None
        L.fbj_set_stage.argtypes = [vp, C.c_int]
        L.fbj_set_stage.restype = None
        L.fbj_destroy.argtypes = [vp]
        L.fbj_destroy.restype = None
        _LIB = L
    return _LIB


class NvJpeg:
    """NVJPG hardware decode via libnvjpeg.

    The decoded YUV lands in a decoder-owned buffer that the CPU can only map uncached.
    gray: the Y plane is copied out (bit-exact with cv2.imdecode IMREAD_GRAYSCALE).
    color="cpu" (default): NEON full-range YCbCr->BGR with libjpeg's fancy chroma
        upsampling (bit-exact with cv2.imdecode IMREAD_COLOR when read directly).
    color="vic": VIC does YCbCr->BGRx (little CPU), but it treats the JPEG as limited-range
        BT.601, so colors are stretched/clipped (~8 levels mean error vs cv2). bgrx=True with
        color="vic" returns HxWx4 BGRx and skips the 4->3 repack.
    stage: whether VIC first copies the decode into a cached surface before the CPU reads it.
        "auto" (default) stages gray (CPU ~3 ms instead of ~9 ms per 4 MP frame, ~2 ms more
        wall) and reads BGR directly (staging would cost wall time and break bit-exactness).
        True / False force it either way.
    After each decode, .times holds the breakdown in ms: {"decode": NVJPG decode call,
        "map": map + cache sync, "copy": copy or color conversion, "vic": VIC, "total": readback}.
    """
    name = "nvjpg"

    def __init__(self, color="cpu", bgrx=False, stage="auto"):
        if color not in ("cpu", "vic"):
            raise ValueError(f"color must be 'cpu' or 'vic', not {color!r}")
        if bgrx and color != "vic":
            raise ValueError("bgrx=True needs color='vic'")
        self._L = _lib()
        self._h = self._L.fbj_create()
        self._stage, self._staged = stage, None
        self._color = color
        self._bgrx = bgrx
        self._cpu = CpuJpeg()
        self._t = (C.c_double * 4)()
        self.fallbacks = 0
        self.times = {}

    def _err(self):
        return self._L.fbj_error(self._h).decode(errors="replace")

    def _fallback(self, jpeg, gray):
        self.fallbacks += 1
        self.times = {}
        img = self._cpu.decode(jpeg, gray)
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) if self._bgrx and not gray else img

    def _buf(self, out, shape):
        if out is not None and out.shape == shape and out.dtype == np.uint8 and out.flags.c_contiguous:
            return out
        return np.empty(shape, np.uint8)

    def decode(self, jpeg, gray=False, out=None):
        """out: optional preallocated uint8 array to fill (used if its shape matches);
        reusing one avoids page-faulting a fresh 4-12 MB array on every frame."""
        if self._h is None:
            raise RuntimeError("decoder is closed")
        jpeg = bytes(jpeg)
        w, h, fmt = C.c_int(), C.c_int(), C.c_int()
        t0 = time.perf_counter()
        r = self._L.fbj_decode(self._h, jpeg, len(jpeg), C.byref(w), C.byref(h), C.byref(fmt))
        t_dec = (time.perf_counter() - t0) * 1e3
        if r == -2:
            return self._fallback(jpeg, gray)
        if r != 0:
            raise ValueError(f"NVJPG decode failed: {self._err()}")
        H, W = h.value, w.value
        stage = bool(gray) if self._stage == "auto" else bool(self._stage)
        if stage != self._staged:
            self._L.fbj_set_stage(self._h, int(stage))
            self._staged = stage
        if gray:
            out = self._buf(out, (H, W))
            r = self._L.fbj_get_gray(self._h, out.ctypes.data, out.strides[0])
        elif self._color == "cpu":
            out = self._buf(out, (H, W, 3))
            r = self._L.fbj_get_bgr(self._h, out.ctypes.data, out.strides[0])
        else:
            out = self._buf(out, (H, W, 4 if self._bgrx else 3))
            r = self._L.fbj_get_bgr_vic(self._h, out.ctypes.data, out.strides[0], out.shape[2])
        if r == -2:
            return self._fallback(jpeg, gray)
        if r != 0:
            raise RuntimeError(f"NVJPG readback failed: {self._err()}")
        self._L.fbj_times(self._h, self._t)
        self.times = {"decode": t_dec, **dict(zip(("map", "copy", "vic", "total"), self._t))}
        return out

    def close(self):
        if self._h is not None:
            self._L.fbj_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _i420_to_bgr(y, u, v):
    """JFIF full-range I420 -> BGR with OpenCV (bilinear chroma upsampling)."""
    h, w = y.shape
    cr = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    cb = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)


class GstNvJpeg:
    """NVJPG through GStreamer: a persistent `appsrc ! jpegparse ! nvjpegdec !
    video/x-raw,format=I420 ! appsink` pipeline, one buffer pushed and one sample pulled per
    call. nvjpegdec copies the decode from NVMM into a system-memory I420 buffer; gray takes
    the Y plane, BGR is converted on the CPU with OpenCV (full range). The NVMM -> nvvidconv
    -> BGRx route was not used: VIC treats the decode as limited range (wrong colors)."""
    name = "nvjpg-gst"

    def __init__(self):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        self._Gst = Gst
        self._p = Gst.parse_launch(
            "appsrc name=src format=time caps=image/jpeg ! jpegparse ! nvjpegdec ! "
            "video/x-raw,format=I420 ! appsink name=sink sync=false max-buffers=1")
        self._src, self._sink = self._p.get_by_name("src"), self._p.get_by_name("sink")
        self._p.set_state(Gst.State.PLAYING)

    def decode(self, jpeg, gray=False, out=None):  # out: accepted for API parity, unused
        Gst = self._Gst
        self._src.emit("push-buffer", Gst.Buffer.new_wrapped(bytes(jpeg)))
        sample = self._sink.emit("pull-sample")
        if sample is None:
            raise ValueError("nvjpegdec produced no frame")
        s = sample.get_caps().get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        buf = sample.get_buffer()
        ok, m = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("could not map GStreamer buffer")
        try:
            a = np.frombuffer(m.data, np.uint8)
            # GStreamer's default I420 layout: strides rounded up to 4, planes back to back.
            ys, cs = (w + 3) & ~3, (((w + 1) // 2) + 3) & ~3
            cw, ch = (w + 1) // 2, (h + 1) // 2
            y = a[: ys * h].reshape(h, ys)[:, :w]
            if gray:
                return y.copy()
            u = a[ys * h: ys * h + cs * ch].reshape(ch, cs)[:, :cw]
            v = a[ys * h + cs * ch: ys * h + 2 * cs * ch].reshape(ch, cs)[:, :cw]
            return _i420_to_bgr(np.ascontiguousarray(y), np.ascontiguousarray(u),
                                np.ascontiguousarray(v))
        finally:
            buf.unmap(m)

    def close(self):
        if self._p is not None:
            self._src.emit("end-of-stream")
            self._p.set_state(self._Gst.State.NULL)
            self._p = None


DECODERS = {"cpu": CpuJpeg, "nvjpg": NvJpeg, "nvjpg-gst": GstNvJpeg}


def make_decoder(name="cpu", **kw):
    """name: cpu | nvjpg | nvjpg-vic | nvjpg-gst"""
    if name == "nvjpg-vic":
        return NvJpeg(color="vic", **kw)
    return DECODERS[name](**kw)
