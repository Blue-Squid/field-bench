# Command reference

fieldbench has two entry points: `make` targets on the host, which sync and drive the Jetson over SSH, and the `python -m fieldbench` CLI, which runs on the Jetson. Every `make` target that touches the board runs `make sync` first.

## 1. Make targets (host)

| Target | What it does |
|---|---|
| `make export` | Export the Phase 1 models (`yolo11n`, `mobilenetv3l`) to static ONNX in `models/`. |
| `make barcodes` | Generate the synthetic barcode dataset in `data/barcodes/{train,val,test}` (needs `data/coco/val2017`). |
| `make train-barcode` | Fine-tune YOLO11n-OBB on it (CUDA), then export `barcode_yolo11n_{640,1280,1600}.onnx`, each verified against PyTorch. |
| `make ocr-data` | Generate the OCR test set (200 frames) and calibration set (100 frames) in `data/ocr/`. |
| `make sync` | rsync `fieldbench/`, `models/*.onnx` and the pipeline test and calibration images to `~/fieldbench` on the board. **`fieldbench/` is synced with `--delete`**, so edit code on the host, never on the board. |
| `make info` | Print device facts and one telemetry sample. |
| `make modes` | List nvpmodel power modes and whether each switches live or needs a reboot. |
| `make set-mode MODE=<name> [REBOOT=1]` | Switch the power mode through jtop. Modes that need a reboot (7W on the Orin Nano) refuse without `REBOOT=1`. |
| `make bench ARGS="…"` | Run `fieldbench bench` in the foreground over SSH, then `make pull`. Good for short checks. |
| `make live [CMD=bench\|pipeline] ARGS="…"` | Start `fieldbench $CMD $ARGS` in a tmux session on the board, with jtop in a side pane. Refuses to start if the session already exists. |
| `make attach` | Attach to that session (detach with `Ctrl-b d`; the run continues). On the board's own desktop: `tmux attach -t fieldbench`. |
| `make pull` | rsync `results/` back from the board. |
| `make report` | `make pull`, then render `report/index.html`. |

Variables: `JETSON` (SSH host alias, default `jetson`), `REMOTE_DIR` (default `fieldbench`, relative to the remote home), `SESSION` (tmux session name, default `fieldbench`).

## 2. `python -m fieldbench info`

Prints `device_info()` (board model, L4T, power mode, GPU min/max clock, CPU max clock and online cores, EMC cap, TPC power-gating mask, `clocks_locked`), one telemetry sample, and the discovered power rails and thermal zones. Use it to check that telemetry works before a long run.

## 3. `python -m fieldbench bench` (Phase 1)

Benchmarks bare networks on random input.

| Flag | Default | Meaning |
|---|---|---|
| `--models M …` | `yolo11n mobilenetv3l` | Any model in `fieldbench/catalog.py`. |
| `--precisions P …` | `fp32 fp16 int8` | INT8 here is **uncalibrated** (timing only). |
| `--duration S` | 10 | Timed seconds per configuration. |
| `--warmup S` | 3 | Warmup seconds. |
| `--min-iters N` | 200 | Minimum timed inferences, even if `--duration` has passed. |
| `--cooldown S` | 5 | Rest between configurations. |
| `--power-modes M …` | current only | Sweep these nvpmodel modes, ordered by budget (lowest first), restoring the original mode at the end. Reboot-only modes are rejected; measure them separately after `make set-mode … REBOOT=1`. |
| `--settle S` | 10 | Seconds to wait after a mode switch. |
| `--label L` | – | Tag stored in every row; also used in the output file name. |
| `--out PATH` | `results/<timestamp>[-label].jsonl` | Append to this file instead. |

Example: `make live ARGS="--models yolo11n --precisions fp16 int8 --power-modes 15W 25W MAXN_SUPER --label sweep"`.

## 4. `python -m fieldbench power`

Without arguments, lists the modes. `--set MODE` switches to a mode. `--reboot` permits the immediate reboot that some modes need. Switching goes through the jtop service, so it needs no sudo.

## 5. `python -m fieldbench pipeline {barcode,ocr}` (Phase 2)

Runs end-to-end pipelines on the test set. Each combination of the list-valued flags is one configuration, and therefore one result row.

| Flag | Default | Meaning |
|---|---|---|
| `--modes {zxing,detect} …` | both | Barcode only. `zxing`: zxing-cpp over the whole frame. `detect`: oriented detector, deskewed crops, zxing-cpp per crop. |
| `--sizes S …` | barcode `640 1280`, OCR `1280 1600` | Detector input size; one ONNX file and one engine per size. Barcode: 640/1280/1600; OCR: 640/1280/1600/2560. |
| `--precisions P …` | barcode `fp16 int8`, OCR `fp16` | Detector precision. INT8 is **calibrated** on `--calib`. The OCR recognizer is always FP16. |
| `--calibrator {minmax,entropy}` | barcode `minmax`, OCR `entropy` | INT8 calibration algorithm. |
| `--calib DIR` | barcode `data/barcodes/val/images`, OCR `data/ocr/calib` | Calibration JPEGs, preprocessed exactly like the pipeline. Never the test set. |
| `--calib-images N` | 300 | Calibration images used, at most. The engine name records the count actually used. |
| `--jpeg {cpu,nvjpg} …` | `cpu` | JPEG decoder: `cv2.imdecode`, or the NVJPG engine. |
| `--prep {cpu,gpu} …` | `cpu` | Detector preprocessing: OpenCV + NumPy, or the fused CUDA kernel. |
| `--power-modes M …` | current only | Sweep these nvpmodel modes (lowest budget first), rebuilding the pipelines in each and restoring the original mode at the end. Reboot-only modes (7W) are rejected. |
| `--settle S` | 10 | Seconds to wait after a mode switch. |
| `--workers N …` | `1` | Pipeline copies running concurrently on consecutive frames (stage overlap). A list sweeps worker counts, e.g. `1 2 3`. |
| `--series S` | 0 (off) | Also store per-window throughput, latency, power, clocks and temperatures every S seconds (`row["series"]`), for soak runs. |
| `--data DIR` | per workload | Test set folder containing `gt.jsonl` and `images/`. |
| `--duration S` / `--warmup S` / `--min-frames N` / `--cooldown S` | 20 / 3 / 100 / 5 | As for `bench`. |
| `--label L`, `--out PATH` | – | As for `bench`. |

At the end, a summary table prints p50 and p90 totals, per-stage p50s, fps, board power, energy per frame, accuracy, and the ratio against Zebra's published TC53 time.

Examples:

```bash
# Detector vs whole-frame zxing, both decoders, FP16 and calibrated INT8
make live CMD=pipeline ARGS="barcode --modes zxing detect --sizes 640 --precisions fp16 int8 --jpeg cpu nvjpg"

# OCR with GPU preprocessing at every size
make live CMD=pipeline ARGS="ocr --sizes 640 1280 1600 2560 --jpeg nvjpg --prep gpu"

# Throughput with overlapping frames
make live CMD=pipeline ARGS="barcode --modes detect --sizes 640 --precisions fp16 --jpeg nvjpg --prep gpu --workers 1 2 3"

# 20-minute soak with 30-second telemetry windows
make live CMD=pipeline ARGS="ocr --sizes 1280 --jpeg nvjpg --prep gpu --workers 2 --duration 1200 --series 30 --label soak"
```

## 6. Other entry points

| Command | Where | What |
|---|---|---|
| `python -m fieldbench.gpuprep` | Jetson | Runs the CPU preprocessing and the CUDA kernel on 20 real frames per workload at every size. Prints the largest difference in uint8 levels, the share of values that differ, and the cost of each path. |
| `python host/export_models.py [names…] [--force]` | Host | Export the named models (default: all). The `ppocr5_*` models download the RapidOCR ONNX once into `models/_ppocr/`. |
| `python host/report.py [files…] [-o out.html]` | Host | Render a report from any subset of result files. |
| `python host/make_barcodes.py`, `python host/make_text.py` | Host | Dataset generators. `--train/--val/--test` (barcodes) or `--test/--calib` (OCR) set the frame counts, `--workers` the parallel processes, `--out` the folder. |

## 7. Engine files

`engines/<model>.<precision>.trt<version>.<sha1[:10]>.engine`, and for calibrated INT8 `engines/<model>.int8cal-<calibrator>-<set><count>.trt<version>.<sha1[:10]>.engine` with a matching `.calib`. Builds take 2–10 min for the Phase 1 models, 9–33 min for FP16 pipeline detectors, and 15–45 min for calibrated INT8. Calibrated builds cap the TensorRT workspace at 1 GB, which avoids `NvMap` allocation failures on the 8 GB board. An engine is only valid for the TensorRT version and GPU it was built on.
