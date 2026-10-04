# Methodology

How fieldbench produces each number, what each number covers, and what to watch out for when you compare it with a handheld's published figures.

## 1. Two kinds of measurement

| | Model benchmark (`bench`) | Pipeline benchmark (`pipeline`) |
|---|---|---|
| Input | Random tensors, refilled once | Real 4 MP JPEG frames with ground truth, cycled in order |
| Work timed | One network, batch 1 | JPEG decode → preprocessing → network(s) → postprocessing → barcode/text decoding |
| Main latency metric | `gpu_ms` (CUDA events around the network) | `total_ms` (wall clock per frame) plus `stage_ms` for each stage |
| Accuracy | None (INT8 engines here are uncalibrated) | Full untimed pass over the test set before timing |
| Compared with | Qualcomm AI Hub QCS6490 model latencies (NPU) | Zebra TC53 published pipeline times |

## 2. Protocol

Each configuration (model × precision × power mode, or pipeline × size × precision × decoder × preprocessing × workers) runs this sequence:

1. **Engine.** Built with `trtexec` from static ONNX on first use, then loaded from the cache in `engines/`. The file name carries the model, precision, TensorRT version and the first 10 hex digits of the ONNX file's SHA-1, so a changed model can never silently reuse a stale engine. Calibrated INT8 engines also carry the calibrator and calibration set, plus a `.calib` cache next to them.
2. **Accuracy pass** (pipelines only). Every test frame goes through the pipeline once and is scored. With `--workers N`, this pass also runs through the N workers, so a race between workers would show up as an accuracy change.
3. **Idle baseline**, 2 s. Telemetry with engines loaded and nothing running.
4. **Warmup**, 3 s. Clocks ramp, caches and allocators settle, and lazy initialization (NVRTC compile, NVJPG shim, pinned buffers) is excluded.
5. **Timed loop.** At least `--duration` s **and** at least `--min-iters` / `--min-frames`. Pipelines cycle through the test frames, with JPEG bytes held in RAM.
6. **Cooldown**, `--cooldown` s (default 5) before the next configuration.

Telemetry is sampled every 50 ms by a background thread for the whole run. The idle and load windows are cut from the same sample stream by timestamp.

## 3. Timing

- **Wall clock:** `time.perf_counter()` around each stage and around the whole frame. Stages are recorded with a small stopwatch (`pipeline.Clock`) so that every frame reports every stage, even when a stage does no work (zero is recorded).
- **GPU time:** CUDA events recorded on the runner's stream immediately before and after `execute_async_v3` (`infer_gpu`, `det_infer_gpu`, `rec_infer_gpu`) and around the preprocessing kernel (`preprocess_gpu`, `det_pre_gpu`). The difference between the wall time and the GPU time of a stage is launch overhead, copies and synchronization.
- **Statistics:** p50, p90, p95, p99, mean, min and max per metric. Stage p50s don't add up to the total p50, because a percentile of a sum isn't the sum of percentiles. Stage *means* do add up, and the report's stacked bars use them.
- **Synchronous by default.** One frame at a time, the way a handheld responds to a trigger pull. `--workers N` runs N independent pipeline copies (each with its own TensorRT execution context, CUDA stream, pinned buffers and JPEG decoder) as threads that each take the next frame as soon as they finish one. `fps` is then the aggregate throughput, and `total_ms` is per-frame latency under that load. OpenCV, zxing-cpp, the NVJPG shim, TensorRT execution and CUDA synchronization all release Python's GIL, so the threads overlap for real.

## 4. Power and energy

| Field | Source |
|---|---|
| `p_VDD_IN` | INA3221 channel for the whole board's input: the `/sys/class/hwmon/hwmon*` device named `ina3221`, `in<i>_input` (mV) × `curr<i>_input` (mA), rails found by `in<i>_label`. |
| `p_VDD_CPU_GPU_CV`, `p_VDD_SOC` | INA3221 channels for the CPU+GPU+CV rail and the SoC rail. |
| `gpu_mhz`, `gpu_load` | `/sys/class/devfreq/17000000.gpu/cur_freq` and `/sys/devices/platform/bus@0/17000000.gpu/load` (permille). |
| `cpu_mhz` | `cpufreq` of CPU 0. |
| `emc_mhz` | Memory controller clock from the jtop service (the debugfs node needs root). |
| `temp_*` | Every readable `/sys/class/thermal/thermal_zone*`, named by its `type` (`cpu`, `gpu`, `soc0`–`soc2`, `tj`, …). |
| `throttle_events` | Change in the SoC over-current alarm counters (`soctherm_oc` hwmon, `oc*_event_cnt`) during the run. |

Energy per frame: `mj_per_frame_total = mean(p_VDD_IN) × wall_s ÷ frames`. `mj_per_frame_dynamic` subtracts the idle baseline's mean power first, which leaves the energy attributable to the work itself. Model benchmarks report the same per inference (`mj_per_inf_*`).

## 5. Clocks and DVFS

By default the Jetson runs dynamic governors: `schedutil` for the CPU, `nvhost_podgov` for the GPU, and an actmon-driven EMC governor. A pipeline that runs one stage at a time leaves each unit partly idle, so its governor keeps clocks low, and the pipeline runs 18–45% slower than with locked clocks ([Phase 2, section 10](results-phase2.md#10-dvfs)). Consequences:

- Every row records `device.clocks_locked`, which is true when the GPU's min and max frequencies are equal (what `jetson_clocks` sets). Locked and default-governor rows are never mixed in a comparison.
- Default-governor results vary between sessions by up to about 25% for the same configuration. Comparisons in this repository are made within one session. When you reproduce a result, compare it against a baseline you measure yourself in the same session.
- A handheld OS does something similar to locking: Android raises clocks on touch and camera events. Locked-clock numbers approximate a device that boosts on a trigger pull. Default-governor numbers approximate one that doesn't.

## 6. Accuracy scoring

**Barcodes** ([`fieldbench/barcode.py`](../fieldbench/barcode.py), `score`)
- *Read rate:* the share of ground-truth barcodes whose exact string appears among the frame's decodes.
- *Detection recall:* the share of ground-truth barcodes matched by a predicted oriented box with rotated IoU ≥ 0.5.
- *Misreads:* distinct decoded strings that aren't on the frame.
- Ground truth is the string zxing-cpp reads from each code's clean render. zxing reports UPC-A as EAN-13 with a leading zero, and the scoring is consistent with that.

**OCR** ([`fieldbench/ocr.py`](../fieldbench/ocr.py), `score`)
- Each predicted box is assigned to the ground-truth line that contains its centre. A box that covers half of two or more lines counts as a merge and earns no credit.
- A line counts as *detected* when its assigned boxes cover ≥ 70% of it. DB boxes are drawn about 1.5× as tall as the ink, so IoU against tight ground-truth polygons would undercount them.
- *Lines exact:* the texts of the line's boxes are joined in reading order, whitespace is normalized, and the result must equal the truth.
- *CER:* edit distance ÷ truth length, capped at 1 per line. *Word recall:* multiset match of words per frame.

## 7. Comparing with handheld figures

- **Different accelerators.** QCS6490 model figures come from the Hexagon NPU with w8a8 or w8a16 quantization. Jetson figures come from the Ampere GPU (the Orin Nano has no DLA). Compare `gpu_ms` with them, not `e2e_ms`.
- **Precision.** Qualcomm's w8a8 is closest to TensorRT INT8. w8a16 (8-bit weights, 16-bit activations) has no exact TensorRT equivalent. FP16 is the "no quantization effort" baseline.
- **Uncalibrated INT8 in `bench`.** Phase 1 INT8 engines use placeholder scales, so their latency is realistic and their outputs aren't. Pipeline INT8 detectors are calibrated on held-out images, so their accuracy is real.
- **What Zebra's numbers cover.** The TC53 barcode figures are "detection" and "detection + decode" with Zebra's own model at the stated input size. OCR figures are whole-pipeline TextOCR times. fieldbench compares its `total_ms` (including JPEG decode) against "detection + decode" and the OCR total, which is conservative toward the Jetson if Zebra's figure starts from an already-decoded frame.
- **Board vs SoC power.** `VDD_IN` covers the whole developer kit: carrier board, NVMe, Wi-Fi and a desktop session if one is running. About 5.2–5.6 W of it is idle draw. A handheld SoC idles far lower, and Qualcomm and Zebra don't publish power for these figures, so energy comparisons are Jetson-only.
- **One engine across power modes.** Engines are built once (at MAXN_SUPER) and reused in every mode, the way a deployed app ships one engine. TensorRT warns "Using an engine plan file across different models of devices is not recommended" when a process starts in a mode with a different GPU clock cap. The CUDA driver reports that cap as `memoryClockRate` and TensorRT compares it on load. The engine binary is the same and runs correctly. Only kernel tuning may be suboptimal in the lower modes.
- **Model variants.** Qualcomm's exported graphs can differ slightly from the ones exported here (head layout, postprocessing). Treat sub-millisecond differences as noise.

## 8. Reproducibility checklist

1. `make info`: record the power mode, clock caps and `clocks_locked`.
2. Close other workloads on the board. With a desktop session running, expect about 0.5–1 GB less memory and some background CPU.
3. Run every configuration you want to compare in **one** `make live` invocation, so they share the session.
4. Keep the result files. They're append-only, and each row carries the full device state.
5. When a result changes, check `telemetry_load.gpu_mhz_mean` and `emc_mhz_mean` before blaming the code.
