# Adapting fieldbench

fieldbench ships configured for one board (Jetson Orin Nano Super 8 GB) and one reference handheld (Zebra TC53/TC58 on the Qualcomm QCS6490). This guide covers the four ways people usually need to change that:

1. [Target devices and what they stand for](#1-target-devices-and-what-they-stand-for)
2. [Another Jetson](#2-another-jetson)
3. [Your own handheld as the reference](#3-your-own-handheld-as-the-reference)
4. [Measuring on the handheld itself](#4-measuring-on-the-handheld-itself)
5. [Your own models](#5-your-own-models)
6. [Your own test data](#6-your-own-test-data)

## 1. Target devices and what they stand for

| | Jetson Orin Nano Super 8 GB | Zebra TC53 / TC58 (QCS6490) |
|---|---|---|
| Role | Device under test | Published reference |
| CPU | 6× Cortex-A78AE @ 1.73 GHz | 8× Kryo 670 (1× A78 @ 2.7 GHz, 3× A78 @ 2.4 GHz, 4× A55) |
| ML accelerator | Ampere GPU, 1024 CUDA cores, 32 tensor cores, sm_87 (no DLA) | Hexagon NPU (12 TOPS class), Adreno 643 GPU |
| Memory | 8 GB LPDDR5, 102 GB/s, shared CPU/GPU | 4 or 8 GB RAM depending on SKU, shared |
| JPEG / imaging | NVJPG, VIC | Spectra ISP, hardware JPEG |
| Power envelope | 7–25 W board modes | A few watts, battery |
| Software stack | JetPack 6.2.1, TensorRT 10.3, CUDA 12.6 | Android, Zebra AI Data Capture SDK, QNN / TFLite |

The comparison asks whether a GPU module in the size class of a handheld's compute board can run the same jobs at handheld latency, and at what energy cost. It doesn't claim the Jetson is a drop-in handheld SoC: its idle power alone (≈5.5 W for the developer kit) exceeds a handheld's active budget.

## 2. Another Jetson

The code runs unchanged on any JetPack 6 Orin (Orin Nano 4/8 GB, Orin NX 8/16 GB, AGX Orin). Check these points, in order:

| What | Where | What to check |
|---|---|---|
| CUDA bindings | Jetson venv | Install the `cuda-python` release that matches the board's CUDA (12.6 on JetPack 6.2). The NVRTC kernel picks the GPU's compute capability at runtime, so no code change is needed. |
| Engines | `engines/` | **Rebuild them on the new board.** TensorRT engines are tied to the GPU and the TensorRT version. Copying `.engine` files between boards is not supported; `.calib` caches can be copied. |
| Power modes | `fieldbench/power.py` `BUDGET_ORDER` | Mode names differ by module and JetPack release (for example 10W/15W/25W/MAXN on Orin NX, 15W/30W/50W/MAXN on AGX Orin). Add your board's names in ascending budget order so that sweeps run lowest first. Run `make modes` to see which switch live. |
| Power rails | `fieldbench/telemetry.py` `Sensors` | Rails are discovered from the `ina3221` hwmon labels. Orin Nano/NX label the board input `VDD_IN`. AGX Orin uses other labels and a second monitor. If `p_VDD_IN` is null in `make info`, map your board's total-input rail to the `VDD_IN` key, because energy per frame is computed from it. |
| GPU nodes | `telemetry.py` constants | `GPU_DEVFREQ = /sys/class/devfreq/17000000.gpu` and `GPU_LOAD = /sys/devices/platform/bus@0/17000000.gpu/load` are correct for every Orin. Xavier-generation boards (JetPack 5) use a different node name (`17000000.gv11b`) and are untested. |
| Memory clock | jtop service | `emc_mhz` comes from jtop. Without jtop it is recorded as null and everything else still works. |
| NVJPG | `fieldbench/jpeg.py` | The shim builds against `/usr/src/jetson_multimedia_api` and Tegra `libnvjpeg.so`. AGX Orin has two NVJPG engines, so `--workers` can decode two frames truly in parallel there. The Orin Nano has one. |
| Memory headroom | – | Only the 8 GB Orin Nano has been tested. On smaller boards, stop the desktop session (`sudo systemctl isolate multi-user.target`), keep the 1 GB workspace cap for calibrated builds, and add workers one at a time while you watch memory in jtop. |

Then repeat the reference runs (README section 6) and compare your `results/*.jsonl` with the ones in this repository using `host/report.py`.

## 3. Your own handheld as the reference

Reference figures live in [`fieldbench/catalog.py`](../fieldbench/catalog.py), as plain Python data that the CLI tables and the report both read.

**Model-level reference** (compared against `gpu_ms` in `bench`). Append to a model's `references` list:

```python
MODELS["yolo11n"]["references"].append({
    "device": "Honeywell CT47 (QCS6490)",    # shown in tables and the report
    "runtime": "TFLite",                     # QNN, SNPE, NNAPI, TFLite, ONNX, ...
    "precision": "w8a8",                     # as published
    "unit": "NPU",                           # NPU, GPU, DSP or CPU
    "latency_ms": 4.1,
    "source": "https://...",                 # always link the source
})
```

The table's "vs" column uses the **lowest** reference latency for each model, so a stronger reference automatically becomes the bar to beat.

**Pipeline-level reference** (compared against the pipeline's `total_ms`). Set `pipeline_references` on the detector model the pipeline uses, one per input size:

```python
MODELS["barcode_yolo11n_640"]["pipeline_references"] = [{
    "device": "Datalogic Memor 17", "runtime": "vendor SDK", "precision": "?", "unit": "NPU",
    "latency_ms": 48, "covers": "detection + decode", "source": "https://...",
}]
```

Match what the vendor's figure covers: whether it starts from a decoded frame, whether it includes decoding the barcode or text, and at which input size. Record that in `covers`. Where to find numbers:

- **Qualcomm AI Hub** model cards list per-device latencies for many SoCs (QCS6490, QCS8550, Snapdragon 8 Gen 2/3, …), with runtime and precision. Pick the device your handheld uses.
- **Handheld vendor SDK documentation** (Zebra AI Data Capture, Honeywell, Datalogic, Scandit) sometimes publishes per-model or per-pipeline times by device.
- **Your own measurement** on the handheld (next section). This is the only way to get power figures for the handheld side.

## 4. Measuring on the handheld itself

fieldbench doesn't ship an Android harness. Its datasets, outputs and scoring are portable, though, so you can produce numbers on a handheld that compare directly with the Jetson's:

1. **Copy the test sets.** `data/barcodes/test` and `data/ocr/test` are plain JPEG files plus `gt.jsonl` (schema in section 6). Copy them to the device unchanged, because recompressing changes the task.
2. **Run the same stages.** For barcodes: decode the JPEG, run the detector at the same input size, and decode each detected region. For OCR: run the detector at the same input size, then the recognizer. The ONNX files in `models/` convert to TFLite or QNN with the vendor's tools (Qualcomm AI Hub compiles ONNX for QCS6490 directly). Calibrate INT8 on the same calibration sets (`data/barcodes/val/images`, `data/ocr/calib`) so the quantization matches.
3. **Time it the same way.** Hold the JPEG bytes in memory, make one untimed pass, warm up for 3 s, then time at least 20 s and 100 frames, cycling through the test set. Record each stage's wall time and report p50/p90.
4. **Write one JSON line per frame** in fieldbench's output format, in test-set order:
   - Barcode: `{"boxes": [[cx, cy, w, h, angle_rad], ...] or null, "decodes": [{"text": "...", "format": "EAN13"}, ...]}`
   - OCR: `{"rects": [[[cx, cy], [w, h], angle_deg], ...], "texts": [["text", confidence] or null, ...]}`, where `texts[k]` belongs to `rects[k]`.
5. **Score it on the host** with the same functions the Jetson uses:
   ```python
   import json
   from fieldbench import barcode
   frames = barcode.load_frames("data/barcodes/test")
   outputs = [json.loads(l) for l in open("handheld_outputs.jsonl")]
   print(barcode.score(frames, outputs))      # ocr.score for OCR; same frame order
   ```
6. **Energy (optional).** On Android, sample battery current and voltage every 50–100 ms: `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW` from an app, or `/sys/class/power_supply/battery/current_now` and `voltage_now` where the build exposes them (often root only). Take a 2 s idle baseline with the screen in the same state, then compute `mean power × wall time ÷ frames`, exactly as in [methodology](methodology.md#4-power-and-energy). Battery-side power includes the display and radios, so hold those constant across runs.

To try a different accelerator or runtime **on the Jetson side**, implement the runner contract and pass it to a pipeline. `host/ort_runner.py` is a complete 30-line example over ONNX Runtime:

```python
class MyRunner:
    inputs:  list   # objects with .host (numpy array, model input, filled by the pipeline)
    outputs: list   # objects with .host (numpy array, filled by infer())
    def infer(self, upload=True) -> tuple[float, float]: ...   # (accelerator_ms, end_to_end_ms)
    def close(self): ...
```

GPU preprocessing (`--prep gpu`) and `--workers` additionally need `.stream` (a CUDA stream), `.inputs[0].device` and `.engine_file`, so use `--prep cpu` and one worker with other runtimes.

## 5. Your own models

1. **Export static ONNX.** Add an exporter to `host/export_models.py` (register it in `EXPORTERS`). Export with fixed batch and resolution, and verify the ONNX against the source framework before saving. The existing exporters show the pattern, including the 1e-3 tolerance check. Avoid graph simplifiers you haven't verified: one of them silently corrupted a recognizer in this project ([troubleshooting](troubleshooting.md)).
2. **Register it** in `fieldbench/catalog.py` with `onnx`, `task`, `workload`, `input`, `params_m`, `license` and `references`. `bench` can then time it at every precision immediately.
3. **Use it in a pipeline.** A detector with a different output layout needs its own postprocessing (see `yolo.postprocess_obb` for the oriented-box YOLO layout `[cx, cy, w, h, class scores…, angle]`). Preprocessing must match training exactly. For GPU preprocessing, build the kernel's lookup table with your CPU normalization expression (`yolo.lut`, `ocr.det_lut`), and the output will match the CPU path bit for bit.
4. **Calibrate INT8 per model.** MinMax suits detectors that regress pixel coordinates. Entropy or percentile calibration suits feature maps with rare large activations. Always compare accuracy against FP16 on the test set before trusting an INT8 engine ([INT8 calibration](int8-calibration-in-fieldbench.md)).

## 6. Your own test data

Any folder with `images/*.jpg` and a `gt.jsonl` works with `--data`. One line per frame:

```json
{"image": "images/0001.jpg", "barcodes": [{"format": "EAN13", "text": "4006381333931", "cls": 0,
  "polygon": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]}]}
{"image": "images/0002.jpg", "lines": [{"text": "LOT 24A-117", "cap_px": 22.9,
  "polygon": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]}]}
```

- `polygon` is the four corners in pixel coordinates, clockwise from top-left of the content. For OCR, the first edge runs along the text direction, which the scorer uses for reading order.
- Barcode `cls` is 0 for linear (1D) codes and 1 for 2D/stacked codes, matching the detector's classes. `format` is informational.
- OCR `cap_px` (capital-letter height in pixels) is used only for the accuracy-by-size breakdown.
- Extra keys (`width`, `blur`, `noise`, `module_px`, …) are ignored by the scorers and kept for analysis.

For real photos, the strings must be verified by hand or by a decoder at high resolution. For public real-image sets, see BarBeR (Vezzali et al., ICPR 2024; 8,748 annotated barcode images, free account required) for barcodes. For text, use a scene-text set with line-level transcriptions, converted to the schema above.

The generators (`host/make_barcodes.py`, `host/make_text.py`) are also a starting point: change the symbologies, fonts, label templates, module sizes and degradations to match your site's labels and camera, and regenerate.
