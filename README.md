# fieldbench

**Handheld-class computer-vision benchmarks for NVIDIA Jetson.**

fieldbench measures how an embedded GPU board handles the vision workloads of frontline handheld devices: the rugged mobile computers that scan barcodes and read labels all day in warehouses, stores and hospitals. It runs real end-to-end pipelines on 4 MP camera frames with exact ground truth, times every stage, records board power, energy per frame, clocks and thermals, and puts each result next to the published figures of a production handheld at the same job.

The reference platform is the **NVIDIA Jetson Orin Nano Super (8 GB)**. The reference handheld is the **Zebra TC53/TC58 class** on the **Qualcomm QCS6490** SoC. Both are swappable: [Adapting fieldbench](docs/adapting.md) explains how to benchmark another Jetson, or to compare against your own handheld.

Author: **Siddhartha Boppana**

---

## Contents

1. [What it measures](#1-what-it-measures)
2. [Headline results](#2-headline-results)
3. [Target devices](#3-target-devices)
4. [Requirements](#4-requirements)
5. [Installation](#5-installation)
6. [Workflow](#6-workflow)
7. [Workloads and pipelines](#7-workloads-and-pipelines)
8. [Measurement protocol](#8-measurement-protocol)
9. [Output: result files and report](#9-output-result-files-and-report)
10. [Repository layout](#10-repository-layout)
11. [Documentation](#11-documentation)
12. [Licenses and attribution](#12-licenses-and-attribution)

---

## 1. What it measures

| Question | How fieldbench answers it |
|---|---|
| **Latency parity:** can the board match handheld per-frame latency, and at which precision? | Bare-network benchmarks at FP32, FP16 and INT8 against Qualcomm AI Hub's QCS6490 figures (Phase 1); whole pipelines against Zebra's published TC53 times (Phase 2). |
| **Energy:** what does each frame cost? | Whole-board input power sampled every 50 ms during the timed loop, with an idle baseline, gives millijoules per frame, both total and above idle. |
| **Where does the time go?** | Every pipeline stage is timed separately (JPEG decode, preprocessing, network, postprocessing, decoding), with CUDA events for GPU work. |
| **What helps?** | Each optimization is an A/B switch on the same frames: hardware JPEG decode (`--jpeg`), GPU preprocessing (`--prep`), calibrated INT8 (`--precisions`), locked clocks, and overlapping frames across workers (`--workers`). |
| **Is the answer still right?** | Every configuration first makes an untimed accuracy pass over the full test set, so a faster configuration that reads fewer barcodes or text lines is caught. |

Work is organized in phases:

- **Phase 1: model benchmarks.** YOLO11n and MobileNetV3-Large at FP32/FP16/INT8, in every power mode.
- **Phase 2: handheld pipelines.** Barcode reading (oriented detector + deskewed crops + zxing-cpp) and OCR (PP-OCRv5 mobile) on synthetic 4 MP frames with exact ground truth. This phase also covers calibrated INT8, NVJPG hardware decode, a bit-exact CUDA preprocessing kernel, DVFS effects, and stage overlap.

**Status and known gaps.** Phase 1 and Phase 2 are complete. These are deliberately left open and are stated wherever they affect a number:

- **Recognizer width buckets (OCR).** Every text line is padded to 640 px for the recognizer, a flat ~33 ms per frame. Bucketing lines by width (for example 160/320/640 px engines) would cut that.
- **Real-photo barcode validation.** Barcode accuracy is measured on synthetic frames only. [BarBeR](https://ditto.ing.unimore.it/barber/) (8,748 annotated real images, free account) is the intended check.
- **7W pipelines.** The 7W mode needs a reboot to enter, so pipeline power-mode results cover 15W, 25W and MAXN_SUPER. Model benchmarks include 7W.

## 2. Headline results

Jetson Orin Nano Super, MAXN_SUPER power mode, TensorRT 10.3, batch 1. Full tables and analysis: [Phase 1 results](docs/results-phase1.md), [Phase 2 results](docs/results-phase2.md).

**Bare networks against the QCS6490 NPU** (best published QCS6490 latency ÷ Jetson GPU p50; above 1× means the Jetson is faster):

| Model | FP16 | INT8 | Energy per inference, FP32 → INT8 |
|---|---|---|---|
| YOLO11n (640×640) | 4.04 ms, 1.01× | 3.30 ms, **1.24×** | 145 → 56 mJ |
| MobileNetV3-L (224×224) | 1.41 ms, 0.83× | 1.16 ms, **1.00×** | 28 → 13 mJ |

**End-to-end pipelines on 4 MP JPEGs** against the Zebra TC53's published times for the same job at the same detector input size:

| Pipeline | Best configuration | Total p50 | Zebra TC53 | Accuracy |
|---|---|---|---|---|
| Barcode, 640 px detector | INT8, NVJPG decode | 56.2 ms | 57 ms | 92.3% of barcodes read |
| Barcode, 1600 px detector | INT8, NVJPG, GPU preprocessing | 82.6 ms | 124 ms (**1.50×**) | 92.5% |
| OCR, 1280 px detector | FP16, NVJPG, GPU preprocessing | 135.9 ms | 180 ms (**1.32×**) | 83.8% of lines exact |
| OCR, 2560 px detector | FP16, NVJPG, GPU preprocessing | 203.0 ms | 480 ms (**2.36×**) | 89.3% |
| OCR, 1280 px, locked clocks | as above + `jetson_clocks` | 76.8 ms | 180 ms (**2.34×**) | 83.8% |
| Barcode 640 px, 3 workers | FP16, NVJPG, GPU preprocessing | 53.3 ms, **55 frames/s** | 57 ms, one frame at a time | 93.1% |

What the numbers say:

1. **With INT8, the Jetson matches or beats the handheld NPU on latency.** It needs 25 W or more of board power to do so, though. At 15 W it runs at 0.7–0.8× the QCS6490.
2. **In a real pipeline, CPU stages dominate, not the network.** JPEG decode is about half of a 640 px barcode frame. The OCR detector's resize and normalize took 25–41% of the frame until it moved into a fused CUDA kernel, which matches OpenCV bit for bit.
3. **Precision is the biggest energy lever (2–3× per frame).** Between 15 W and MAXN_SUPER, the power mode changes energy per frame by less than ±7%.
4. **Calibration has to fit the model.** MinMax INT8 keeps the barcode detector within a point of FP16, but it wrecks the OCR text detector (lines read exactly: 84% → 31%). The OCR detector stays FP16.
5. **The default DVFS governors cost these pipelines 18–45% of their latency.** The units take turns, so each one's governor keeps clocks low. Locking clocks uses the same or *less* energy per frame.
6. **Overlapping frames fixes that without root.** Three barcode-pipeline workers run 3.3× the throughput of one (55 frames/s) at *lower* per-frame latency and 57% less energy per frame. OCR saturates the GPU at two workers, with 1.8× throughput ([section 11](docs/results-phase2.md#11-stage-overlap)).

## 3. Target devices

| Role | Device | Notes |
|---|---|---|
| Device under test | **NVIDIA Jetson Orin Nano Super Developer Kit, 8 GB** | 6× Cortex-A78AE, 1024-core Ampere GPU (sm_87), 8 GB LPDDR5 shared by CPU and GPU, NVJPG and VIC engines, no DLA. JetPack 6.2.1 (L4T R36.4.7), TensorRT 10.3. Power modes 7W / 15W / 25W / MAXN_SUPER. Full snapshot: [docs/device-orin-nano-super.md](docs/device-orin-nano-super.md). |
| Reference handheld | **Zebra TC53 / TC58** (Qualcomm **QCS6490**) | Kryo 670 CPU, Adreno 643 GPU, Hexagon NPU. Model latencies come from [Qualcomm AI Hub](https://aihub.qualcomm.com/) model cards (QCS6490, w8a8/w8a16). Pipeline latencies come from Zebra's [AI Data Capture SDK](https://techdocs.zebra.com/ai-datacapture/latest/) documentation (barcode localizer, TextOCR). |
| Build host | Any Linux x86-64 machine | Generates the datasets, exports ONNX, renders the report. A CUDA GPU is needed only to train the barcode detector. |

**Other Jetsons** (Orin NX, AGX Orin, other Orin Nano variants) work with the same code. Power-mode names, sysfs paths and clock caps differ, and [docs/adapting.md](docs/adapting.md#2-another-jetson) lists exactly what to check. **Other handhelds** plug in as reference figures in [`fieldbench/catalog.py`](fieldbench/catalog.py). To measure them yourself, run the same test sets and metric definitions on the handheld ([docs/adapting.md](docs/adapting.md#3-your-own-handheld-as-the-reference)).

## 4. Requirements

**Jetson**
- JetPack 6.x (tested on 6.2.1 / L4T R36.4.7) with its system TensorRT, CUDA and OpenCV.
- Python 3.10 (JetPack's system Python).
- [`jetson-stats`](https://github.com/rbonghi/jetson_stats) (jtop) running as a service, with your user in the `jtop` group. fieldbench uses it to switch power modes and read the memory clock without root.
- About 2 GB free disk for engines, test data and results.
- For hardware JPEG decode: `g++` and the Jetson Multimedia API headers (`/usr/src/jetson_multimedia_api`, part of JetPack). The decoder shim compiles itself on first use; no root needed.

**Host**
- Linux, Python 3.10+ (tested on 3.14), `rsync` and `ssh` access to the Jetson under a host alias (default `jetson`; override with `make JETSON=myboard`).
- For training the barcode detector: an NVIDIA GPU with a CUDA build of PyTorch (tested on an RTX 5080 Laptop GPU, about 30 minutes).
- [COCO val2017](https://cocodataset.org/#download) images (1 GB). They are only used as photo backgrounds for the synthetic scenes.

**Root on the Jetson** is never needed for benchmarking. You only need it for optional locked-clock runs (`sudo jetson_clocks`) and for switching the desktop off to free memory.

## 5. Installation

### 5.1 Host

```bash
git clone https://github.com/Blue-Squid/field-bench.git && cd field-bench
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
.venv/bin/pip install ultralytics onnx onnxslim onnxscript onnxruntime zxing-cpp opencv-python
```

Add the board to `~/.ssh/config` so that `ssh jetson` works without a password prompt:

```
Host jetson
    HostName <board IP or hostname>
    User <your user on the board>
```

### 5.2 Jetson

```bash
ssh jetson
sudo apt install python3-venv g++                   # if missing
sudo pip3 install -U jetson-stats && sudo usermod -aG jtop $USER   # then log out and back in
mkdir -p ~/fieldbench
python3 -m venv --system-site-packages ~/fieldbench/.venv           # reuses JetPack's TensorRT and OpenCV
~/fieldbench/.venv/bin/python -m pip install "cuda-python>=12.6,<12.7" zxing-cpp
```

Match `cuda-python` to the CUDA version of your JetPack (12.6 for JetPack 6.2). The venv must use `--system-site-packages`, because TensorRT's Python bindings come from JetPack and are not on PyPI for Jetson.

### 5.3 Check the setup

```bash
make info        # syncs the code, prints board facts, power mode, clocks and one telemetry sample
make modes       # lists the nvpmodel power modes and which ones switch without a reboot
```

`make info` should show the board model, `power_mode`, `gpu_max_mhz` and non-null `p_VDD_IN`, `temp_*` and `emc_mhz` readings. If `emc_mhz` is null, jtop is not reachable (see [Troubleshooting](docs/troubleshooting.md)).

## 6. Workflow

Everything runs from the host through `make`. Code and data are pushed to `~/fieldbench` on the Jetson, long runs execute there in a tmux session you can watch, and results come back as JSONL. The full reference for every target and flag is in [docs/cli.md](docs/cli.md).

### 6.1 Phase 1: model benchmarks

```bash
make export                                   # YOLO11n + MobileNetV3-L → models/*.onnx (static shapes)
make live ARGS="--label baseline"             # all models × FP32/FP16/INT8 in tmux: benchmark | jtop
make attach                                   # watch; Ctrl-b d detaches and leaves it running
make report                                   # pull results/*.jsonl, render report/index.html
```

Power-mode sweep (live-switchable modes, lowest budget first; the original mode is restored at the end):

```bash
make live ARGS="--power-modes 15W 25W MAXN_SUPER --label power-sweep"
make set-mode MODE=7W REBOOT=1                # 7W power-gates half the GPU: reboot required
make live ARGS="--label power-7w"             # after the board comes back
make set-mode MODE=MAXN_SUPER REBOOT=1
```

The first build of each TensorRT engine takes 2–10 minutes for these models. Engines are cached in `~/fieldbench/engines`, keyed by ONNX hash, precision and TensorRT version.

### 6.2 Phase 2: pipelines

Prepare data and models on the host (once):

```bash
# COCO val2017 images must be in data/coco/val2017 first
make barcodes          # 4,000 train / 400 val / 300 test barcode scenes with oriented-box labels and decoded strings
make train-barcode     # fine-tune YOLO11n-OBB on the host GPU, export ONNX at 640/1280/1600 (checked against PyTorch)
make ocr-data          # 200 OCR test frames + 100 calibration frames with ground-truth lines
.venv/bin/python host/export_models.py ppocr5_det_640 ppocr5_det_1280 ppocr5_det_1600 ppocr5_det_2560 ppocr5_rec_en
```

Run on the Jetson (each `make live` syncs code, models and test data first):

```bash
# Barcode: whole-frame zxing vs detector, FP16 vs calibrated INT8, CPU vs NVJPG decode
make live CMD=pipeline ARGS="barcode --modes zxing detect --sizes 640 1280 1600 --precisions fp16 int8 --jpeg cpu nvjpg"

# OCR at Zebra's four input sizes, CPU vs GPU preprocessing
make live CMD=pipeline ARGS="ocr --sizes 640 1280 1600 2560 --jpeg nvjpg --prep cpu gpu"

# Stage overlap: 1, 2 and 3 pipeline workers on consecutive frames
make live CMD=pipeline ARGS="ocr --sizes 1280 --jpeg nvjpg --prep gpu --workers 1 2 3"

make report
```

The first INT8 run builds a calibrated engine from held-out images (15–45 min per engine). The calibration cache is stored next to the engine, so later builds reuse it. See [INT8 calibration in fieldbench](docs/int8-calibration-in-fieldbench.md).

Check that the GPU preprocessing kernel still matches the CPU path, for example after changing OpenCV or JetPack:

```bash
ssh jetson 'cd fieldbench && .venv/bin/python -m fieldbench.gpuprep'
```

### 6.3 Rules for clean measurements

- **One benchmark at a time** on the board, with nothing heavy alongside it. `make live` refuses to start while a session is already running.
- **Never lock or unlock clocks during a sweep.** `jetson_clocks` mid-run makes the rows on either side incomparable. Lock or restore only between runs:
  `sudo jetson_clocks --store /tmp/jc.conf && sudo jetson_clocks` … `sudo jetson_clocks --restore /tmp/jc.conf`.
  Rows record `device.clocks_locked`, and the report keeps locked and unlocked rows apart.
- **Compare default-governor numbers only within one session.** With DVFS active, the same configuration can move 10–25% between sessions (see [Phase 2, finding 10](docs/results-phase2.md#10-dvfs)).
- **Re-export means rebuild.** Changing an ONNX file changes its hash and therefore its engine name, so expect a 2–45 min build.

## 7. Workloads and pipelines

### 7.1 Models

| Name | Task | Input | Source / license | Reference |
|---|---|---|---|---|
| `yolo11n` | Object detection (COCO) | 1×3×640×640 | Ultralytics, AGPL-3.0 | QCS6490: 4.10 ms (TFLite w8a8) |
| `mobilenetv3l` | Classification | 1×3×224×224 | torchvision, BSD-3-Clause | QCS6490: 1.17 ms (TFLite w8a8) |
| `barcode_yolo11n_{640,1280,1600}` | Oriented barcode detection, 1D + 2D | 1×3×S×S | YOLO11n-OBB fine-tuned here; AGPL-3.0 | Zebra TC53: 57 / 94 / 124 ms detect + decode |
| `ppocr5_det_{640,1280,1600,2560}` | Text detection (DB) | 1×3×S×S | PP-OCRv5 mobile, Apache-2.0 | Zebra TC53 TextOCR: 110 / 180 / 270 / 480 ms |
| `ppocr5_rec_en` | Text-line recognition (CTC) | 8×3×48×640 | PP-OCRv5 mobile English, Apache-2.0 | – |

All engines are built from **static-shape ONNX**. Every export is checked against the original framework (maximum absolute difference ≤ 1e-3) and refuses to save on a mismatch.

### 7.2 Pipelines

```
barcode, detect   JPEG decode ─▶ letterbox ─▶ YOLO11n-OBB (TensorRT) ─▶ rotated NMS ─▶ per box: rotate upright + crop ─▶ zxing-cpp
barcode, zxing    JPEG decode (gray) ─▶ zxing-cpp over the whole frame                     (classic CPU scanner, no network)
ocr               JPEG decode ─▶ resize + normalize ─▶ DB text detector ─▶ boxes ─▶ line crops ─▶ recognizer ×8 ─▶ CTC decode
```

| Switch | Values | Effect |
|---|---|---|
| `--jpeg` | `cpu`, `nvjpg` | `cv2.imdecode` (libjpeg-turbo), or the NVJPG hardware engine through a ctypes shim over Tegra `libnvjpeg` (bit-exact with OpenCV). |
| `--prep` | `cpu`, `gpu` | Detector preprocessing in OpenCV + NumPy, or one fused NVRTC kernel that reads the decoded frame in place from pinned memory and writes the TensorRT input directly (bit-exact up to 1600 px). |
| `--precisions` | `fp16`, `int8`, `fp32` | INT8 detectors are **calibrated** on held-out images; `--calibrator minmax|entropy` (per-workload default). |
| `--workers` | `1 2 3 …` | N independent pipeline copies as threads on consecutive frames (stage overlap). |

### 7.3 Test data

Both test sets are **synthetic and generated on the host**, so every barcode string and text line is known exactly and nothing has redistribution restrictions.

| Set | Frames | Content | Degradations |
|---|---|---|---|
| Barcode test | 300 × 4 MP | 677 barcodes: EAN-13, UPC-A, EAN-8, Code 128, Code 39, ITF, QR, DataMatrix, PDF417, on printed labels over COCO photos; text and stripe decoys | Rotation, perspective, uneven light, defocus/motion blur, noise, JPEG; 1D modules 1.4–4.5 px, 2D 2.5–10 px |
| OCR test | 200 × 4 MP | 1,394 lines on product, shipping, lot/expiry, price and asset labels; sans and mono fonts, cap height 14–56 px | Tilt up to 30°, same degradations |
| Calibration | 300 + 100 | The first 300 barcode validation frames and 100 separate OCR frames; never from a test set | – |

Barcode ground truth is the string zxing-cpp reads from each code's clean render, so the scoring is consistent with the decoder.

## 8. Measurement protocol

Each configuration runs the same sequence:

1. **Accuracy pass** (untimed) over every test frame, scored against ground truth.
2. **Idle baseline**, 2 s: board power with engines loaded and nothing running.
3. **Warmup**, 3 s.
4. **Timed loop**: at least `--duration` seconds (default 20 for pipelines, 10 for models) **and** at least `--min-frames` frames (100), cycling through the test frames. JPEG bytes are held in RAM, so storage speed is excluded and JPEG decode is included.
5. **Cooldown**, 5 s.

| Metric | Definition |
|---|---|
| `total_ms`, `stage_ms` | Wall-clock time per frame and per stage, with p50/p90/p95/p99/mean. GPU stages also record CUDA-event time (`infer_gpu`, `det_infer_gpu`, `preprocess_gpu`). |
| `gpu_ms` (models) | Network execution only, between CUDA events around `execute_async_v3`. Use this against vendor NPU numbers. |
| `fps` | Frames completed ÷ wall time of the timed loop (aggregate across workers). |
| `p_VDD_IN` | Whole-board input power from the on-board INA3221 monitor, sampled every 50 ms. |
| `mj_per_frame_total` / `_dynamic` | Mean VDD_IN × wall time ÷ frames; `_dynamic` subtracts the idle baseline first. |
| `gpu_mhz`, `emc_mhz`, `cpu_mhz`, `temp_*` | Clocks and thermal zones over the same window. |
| Barcode accuracy | Share of ground-truth barcodes decoded to the exact string; detection recall (rotated IoU ≥ 0.5); misreads (decoded strings not in the frame). |
| OCR accuracy | Lines read exactly (whitespace-normalized); character error rate; word recall; line detection recall (assigned boxes cover ≥ 70% of the line). |

The caveats that matter when you compare with a handheld are covered in detail in [docs/methodology.md](docs/methodology.md): NPU vs GPU, w8a8 vs TensorRT INT8, board power vs SoC power, dynamic clocks, and one engine across power modes.

## 9. Output: result files and report

- `results/<timestamp>-<label>.jsonl` has one JSON object per configuration. Each row contains the configuration, latency statistics, per-stage statistics, accuracy, energy, idle and load telemetry, device facts (power mode, clock caps, `clocks_locked`), and the label. Files are append-only and are never rewritten.
- `make report` renders `report/index.html`, a self-contained page with no external requests. It has latency against the handheld references, energy per inference, power-mode charts, per-stage pipeline breakdowns, the stage-overlap tables and a table of every row. When a configuration appears in several files, the most recent row wins.
- `python3 host/report.py results/a.jsonl results/b.jsonl -o report/subset.html` renders a subset.

The `results/` and `report/` folders in this repository hold the Orin Nano Super measurements behind every number in this README.

## 10. Repository layout

```
Makefile                 host-side workflow: export, data, train, sync, live, attach, pull, report
host/                    runs on the build host
  export_models.py       PyTorch / PaddleOCR → static ONNX, verified against the source model
  make_barcodes.py       synthetic barcode scenes, oriented-box labels, ground-truth strings
  make_text.py           synthetic OCR label scenes and ground-truth lines
  train_barcode.py       fine-tune YOLO11n-OBB on the barcode set
  ort_runner.py          ONNX Runtime stand-in for the TensorRT runner (host-side pipeline checks)
  report.py              results JSONL → report/index.html (template: report_template.html)
fieldbench/              runs on the Jetson
  __main__.py            CLI: info, bench, power, pipeline
  catalog.py             models and published handheld reference latencies
  engine.py              trtexec engine build/cache, calibrated INT8 builds
  runner.py              TensorRT executor with CUDA-event timing
  bench.py               model benchmark: baseline → warmup → timed loop → stats
  pipeline.py            pipeline runner: accuracy pass, per-stage timing, worker pool, telemetry
  barcode.py             barcode pipelines and scoring
  ocr.py                 PP-OCRv5 pipeline (DB postprocess, line crops, CTC) and scoring
  yolo.py                letterbox, axis-aligned and oriented-box postprocessing
  jpeg.py                JPEG decoders: OpenCV, and NVJPG via a ctypes shim compiled on first use
  gpuprep.py             fused letterbox/normalize CUDA kernel (NVRTC); `python -m fieldbench.gpuprep` verifies it
  telemetry.py           sysfs power/thermal/clock sampler, EMC clock via jtop
  power.py               nvpmodel power modes via the jtop service
  stats.py               latency percentiles
docs/                    manuals and results (see below)
results/  report/        measurements from the reference board
models/ engines/ data/ runs/   generated locally, not in git
```

## 11. Documentation

| Document | Contents |
|---|---|
| [docs/cli.md](docs/cli.md) | Every `make` target and CLI flag, with examples. |
| [docs/methodology.md](docs/methodology.md) | Timing, telemetry and energy definitions; comparison caveats; how DVFS affects results. |
| [docs/adapting.md](docs/adapting.md) | Running on another Jetson, adding your own handheld's reference numbers, porting the pipelines to a handheld, using your own models and data. |
| [docs/results-phase1.md](docs/results-phase1.md) | Model benchmarks: precision and power-mode sweeps. |
| [docs/results-phase2.md](docs/results-phase2.md) | Pipeline findings: accuracy, stage breakdowns, INT8, NVJPG, GPU preprocessing, DVFS, stage overlap. |
| [docs/int8-quantization-theory.md](docs/int8-quantization-theory.md) | INT8 inference, scale selection, TensorRT calibrators, implicit vs explicit quantization. |
| [docs/int8-calibration-in-fieldbench.md](docs/int8-calibration-in-fieldbench.md) | How the calibrated engines are built here, and what worked for which model. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Known failure modes and their fixes. |
| [docs/device-orin-nano-super.md](docs/device-orin-nano-super.md) | Hardware and software snapshot of the reference board. |

## 12. Licenses and attribution

- **YOLO11 / Ultralytics: AGPL-3.0.** Fine for benchmarking. A commercial product needs an Ultralytics enterprise license or an Apache-licensed detector (RT-DETR, YOLOX, D-FINE).
- **PP-OCRv5 (PaddleOCR): Apache-2.0.** ONNX conversions from [RapidOCR](https://github.com/RapidAI/RapidOCR).
- **MobileNetV3 weights (torchvision): BSD-3-Clause.**
- **COCO val2017** images are used only as backgrounds in generated scenes (CC BY 4.0 annotations; images under their Flickr licenses).
- **zxing-cpp: Apache-2.0.**
- Reference latencies are © their publishers: Qualcomm AI Hub model cards and Zebra Technologies' AI Data Capture SDK documentation. They are quoted for comparison, and every row in `fieldbench/catalog.py` links its source.

Zebra, TC53, TC58, Qualcomm, Snapdragon, NVIDIA and Jetson are trademarks of their respective owners. This project is independent and not affiliated with any of them.
