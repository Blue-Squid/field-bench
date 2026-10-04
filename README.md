# fieldbench

**Handheld-class computer-vision benchmarks for NVIDIA Jetson.**

fieldbench measures how an embedded GPU board handles the vision workloads of frontline handheld devices: the rugged mobile computers that scan barcodes and read labels all day. It records latency, board power, energy per inference, clocks and thermals, and puts every result next to the published figures of a production handheld's NPU running the same model.

The reference platform is the **NVIDIA Jetson Orin Nano Super (8 GB)**. The reference handheld silicon is the **Qualcomm QCS6490**, the SoC in Zebra TC53/TC58-class devices.

Author: **Siddhartha Boppana**

> This revision covers **Phase 1: model benchmarks**. End-to-end handheld pipelines (barcode reading and OCR on camera frames) follow in Phase 2.

## 1. What it measures

| Question | How |
|---|---|
| **Latency parity:** can the board match a handheld NPU's per-frame latency, and at which precision? | YOLO11n and MobileNetV3-Large at FP32, FP16 and INT8 (TensorRT), compared with Qualcomm AI Hub's QCS6490 figures. |
| **Energy:** what does each inference cost? | Whole-board input power, sampled every 50 ms during the timed loop, with an idle baseline. The result is millijoules per inference, both total and above idle. |
| **Power modes:** what changes between 7 W and MAXN_SUPER? | The same engines in every nvpmodel mode, switched through the jtop service (no root needed). |

## 2. Results

Jetson Orin Nano Super, MAXN_SUPER, TensorRT 10.3, batch 1. The ratio is the best published QCS6490 latency ÷ the Jetson GPU p50; above 1× means the Jetson is faster.

| Model | FP32 | FP16 | INT8 | Energy per inference, FP32 → INT8 |
|---|---|---|---|---|
| YOLO11n (640×640) | 7.29 ms, 0.56× | 4.04 ms, 1.01× | 3.30 ms, **1.24×** | 145 → 56 mJ |
| MobileNetV3-L (224×224) | 1.77 ms, 0.66× | 1.41 ms, 0.83× | 1.16 ms, **1.00×** | 28 → 13 mJ |

- **With INT8, the Jetson matches or beats the handheld NPU,** but only at 25 W or more of board power. At 15 W it runs at 0.7–0.8× the QCS6490.
- **Precision is the energy lever (2–3× per inference).** Between 15 W and MAXN_SUPER, the power mode changes energy per inference by less than ±7%. 7 W doubles it.
- INT8 engines in this benchmark are uncalibrated: the timing is representative, but the outputs aren't.

Full tables and analysis: [docs/results-phase1.md](docs/results-phase1.md). Raw rows: `results/*.jsonl`. Charts: `report/index.html`.

## 3. Requirements

**Jetson:** JetPack 6.x (tested on 6.2.1 / L4T R36.4.7) with its system TensorRT and CUDA, and Python 3.10. Also [`jetson-stats`](https://github.com/rbonghi/jetson_stats) (jtop) running as a service, with your user in the `jtop` group.

**Host:** Linux, Python 3.10+, and `rsync` and `ssh` access to the board under the host alias `jetson` (override with `make JETSON=myboard`).

## 4. Installation

```bash
# Host
git clone https://github.com/Blue-Squid/field-bench.git && cd field-bench
python3 -m venv .venv
.venv/bin/pip install torch torchvision ultralytics onnx onnxslim onnxscript

# Jetson
ssh jetson 'sudo pip3 install -U jetson-stats && sudo usermod -aG jtop $USER'   # then log in again
ssh jetson 'mkdir -p ~/fieldbench && python3 -m venv --system-site-packages ~/fieldbench/.venv &&
            ~/fieldbench/.venv/bin/python -m pip install "cuda-python>=12.6,<12.7"'
make info      # board facts, power mode, clocks and one telemetry sample
```

The venv must use `--system-site-packages`, because TensorRT's Python bindings come from JetPack. Match `cuda-python` to the board's CUDA version.

## 5. Workflow

```bash
make export                                   # YOLO11n + MobileNetV3-L → models/*.onnx (static shapes)
make live ARGS="--label baseline"             # all models × FP32/FP16/INT8, in tmux on the board (benchmark | jtop)
make attach                                   # watch; Ctrl-b d detaches and leaves it running
make report                                   # pull results/*.jsonl, render report/index.html

make live ARGS="--power-modes 15W 25W MAXN_SUPER --label power-sweep"   # live-switchable modes
make set-mode MODE=7W REBOOT=1                # 7W power-gates half the GPU: reboot required
make live ARGS="--label power-7w"
make set-mode MODE=MAXN_SUPER REBOOT=1
```

`python -m fieldbench bench` flags: `--models`, `--precisions fp32 fp16 int8`, `--duration` (10 s), `--warmup` (3 s), `--min-iters` (200), `--cooldown` (5 s), `--power-modes`, `--settle` (10 s), `--label`, `--out`. Engines are built once with `trtexec` and cached in `engines/`, named by model, precision, TensorRT version and ONNX hash.

## 6. Measurement protocol

1. **Engine:** built on first use, cached afterwards.
2. **Idle baseline, 2 s:** board power with the engine loaded and nothing running.
3. **Warmup, 3 s.**
4. **Timed loop:** at least `--duration` seconds and `--min-iters` inferences, synchronous at batch 1.
5. **Cooldown, 5 s.**

| Metric | Meaning |
|---|---|
| `gpu_ms` | Network execution only, measured with CUDA events around `execute_async_v3`. Compare this against vendor NPU numbers. |
| `e2e_ms` | Adds host↔device copies (pinned memory) and launch overhead. |
| `p_VDD_IN` | Whole-board input power from the INA3221 monitor. |
| `mj_per_inf_total` / `_dynamic` | Mean VDD_IN × time ÷ inferences; `_dynamic` subtracts the idle baseline. |
| `gpu_mhz`, `temp_*`, `throttle_events` | GPU clock, thermal zones and SoC over-current alarms during the run. |

Caveats: QCS6490 figures come from its Hexagon NPU (w8a8/w8a16), Jetson figures from the Ampere GPU (the Orin Nano has no DLA). `VDD_IN` covers the whole developer kit (about 5.5 W idle), not just the SoC. Engines are built at MAXN_SUPER and reused in every mode. Clocks are dynamic unless a row records `clocks_locked`.

## 7. Repository layout

```
Makefile                 host workflow: export, sync, live, attach, bench, modes, set-mode, pull, report
host/export_models.py    PyTorch → static ONNX
host/report.py           results JSONL → report/index.html (template: report_template.html)
fieldbench/__main__.py   CLI: info, bench, power
fieldbench/catalog.py    models and published QCS6490 reference latencies
fieldbench/engine.py     trtexec engine build and cache
fieldbench/runner.py     TensorRT executor with CUDA-event timing
fieldbench/bench.py      baseline → warmup → timed loop → statistics
fieldbench/telemetry.py  sysfs power, thermal and clock sampler
fieldbench/power.py      nvpmodel power modes via jtop
fieldbench/stats.py      latency percentiles
docs/                    results and the reference board's snapshot
results/ report/         measurements from the reference board
```

## 8. Licenses

- YOLO11 / Ultralytics: **AGPL-3.0**. Fine for benchmarking; a commercial product needs an Ultralytics enterprise license or an Apache-licensed detector.
- MobileNetV3 weights (torchvision): BSD-3-Clause.
- Reference latencies are © Qualcomm (AI Hub model cards) and are quoted for comparison with links in `fieldbench/catalog.py`.

Qualcomm, Zebra, NVIDIA and Jetson are trademarks of their respective owners. This project is independent and not affiliated with any of them.
