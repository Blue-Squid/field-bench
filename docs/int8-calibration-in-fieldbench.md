# INT8 calibration in fieldbench

How fieldbench builds INT8 engines, what they are calibrated on, and what we know about the result. For the theory behind the terms used here (amax, MinMax vs entropy, implicit quantization), see [int8-quantization-theory.md](int8-quantization-theory.md).

## Two kinds of INT8 in this repo

| Where | How it's built | Calibrated? | What its numbers are good for |
|---|---|---|---|
| `bench` (Phase 1 model benchmark) | `trtexec --int8 --fp16`, no calibration data ([engine.py](../fieldbench/engine.py) `build`) | **No**: placeholder ranges | Timing and energy only. Outputs are garbage, so no accuracy can be read from them. Result rows carry `int8_calibrated: false`. |
| `pipeline` (Phase 2 handheld pipelines) | TensorRT Python builder with a calibrator ([engine.py](../fieldbench/engine.py) `build_calibrated`) | **Yes**, on held-out images of the same kind as the test set | Timing *and* accuracy: each pipeline's accuracy pass scores the INT8 engine against ground truth. Rows carry `int8_calibrated: true`. |

The Phase 1 shortcut was deliberate: it gives realistic INT8 latency without having calibration data for COCO or ImageNet on the device. Pipelines can't take that shortcut. An uncalibrated detector finds nothing, so every downstream decode or read would fail.

## The implementation

### Build path

`build_calibrated(onnx, batches, calib_id)` in [fieldbench/engine.py](../fieldbench/engine.py):

1. Parses the ONNX into a TensorRT network (static input shape, batch 1).
2. Enables **both INT8 and FP16** in the builder config. Layers without a fast INT8 kernel then fall back to FP16 rather than FP32. TensorRT picks each layer's precision by speed (implicit quantization).
3. Attaches a calibrator: `_CalibratorMixin` combined with one of TensorRT's calibrator classes, chosen by `--calibrator` (`CALIBRATORS` in engine.py: `minmax` → `IInt8MinMaxCalibrator`, `entropy` → `IInt8EntropyCalibrator2`). It also caps the builder's tactic scratch memory at 1 GB (`workspace_mb`). On the 8 GB Orin Nano, larger requests fail in NvMap with ENOMEM (`NvMapMemAllocInternalTagged ... error 12`), and TensorRT skips those tactics anyway, just noisily. The mixin:
   - `get_batch_size()` returns 1.
   - `get_batch()` takes the next preprocessed image (a float32 array shaped like the network input), copies it into one device buffer with `cudaMemcpy`, and returns that pointer. It returns `None` when the images run out, which ends calibration.
   - `read_calibration_cache()` / `write_calibration_cache()` read and write a cache file next to the engine.
4. Calls `build_serialized_network`. TensorRT first runs the calibration pass, executing the network once per image and recording each activation tensor's range, then builds and tunes the INT8/FP16 engine.
5. Writes the engine atomically (to `.partial`, then renamed).

The calibration pass and the build both happen on the Jetson, inside the `pipeline` command, the first time a calibrated configuration is requested. It's slow: **15–24 minutes per engine** for the barcode detector (640 → 1280 px), against 9–16 minutes for an FP16 build. After that, the engine is cached.

### Choosing the calibrator: MinMax for barcodes, entropy for OCR

The calibrator is chosen per workload (`WORKLOADS` in [fieldbench/__main__.py](../fieldbench/__main__.py), overridable with `--calibrator`). The first sweep used MinMax for everything, and that showed why one choice doesn't fit all.

**Barcode detector (YOLO11n-OBB): MinMax works.** The detector's single output tensor concatenates box geometry in **pixels** (0 to about the input size) with class scores (0–1). Entropy calibration clips rare large values to gain resolution in the bulk, which would clip box coordinates. MinMax never clips. The calibration cache shows the output range growing with input size:

| Detector input | Output amax | Step size if quantized |
|---|---|---|
| 640 px | 730 | 5.7 |
| 1280 px | 1,328 | 10.5 |
| 1600 px | 1,656 | 13.0 |

If the layers producing that tensor ran in INT8, scores between 0 and 1 would collapse to zero. The detector still finds every test barcode in INT8 (recall 1.000 at 640 and 1280, 0.999 at 1600), so TensorRT evidently keeps the head's tail in FP16. That inference is consistent with the accuracy but hasn't been confirmed by inspecting per-layer precisions.

**OCR text detector (PP-OCRv5 DB): MinMax fails.** The detector's neck concatenates upsampled feature maps, and those tensors reach amax of about 1,150–1,550 on the calibration set. MinMax sets each tensor's scale from that extreme, so one INT8 step is about 12 units wide and ordinary activations are crushed into a handful of levels. The probability map degrades, and the box extraction fragments text into twice as many boxes:

| PP-OCRv5 detector | 1280 FP16 | 1280 INT8 MinMax | 1600 FP16 | 1600 INT8 MinMax |
|---|---|---|---|---|
| Detector GPU time | 31.9 ms | 23.2 ms | 55.2 ms | 39.8 ms |
| Boxes per frame | 8.8 | 16.8 | 9.4 | 17.2 |
| Lines found | 90.5% | 47.4% | 92.5% | 62.8% |
| Lines read exactly | 83.8% | 30.8% | 86.4% | 47.1% |
| Recognizer time | 35.0 ms | 68.8 ms | 33.0 ms | 66.0 ms |
| **Pipeline total p50** | **174.8 ms** | **220.5 ms** | **209.5 ms** | **262.0 ms** |

The detector got 27% faster, yet the whole pipeline got **slower**, because every extra box has to be cropped and recognized. This is the clearest argument in this repo for judging a precision change on end-to-end accuracy and time, never on the network's own latency.

Entropy calibration was designed for exactly this distribution shape: it clips rare outliers to keep resolution in the bulk. It's the OCR default now, and it helped, but not enough (1280 px, 100 calibration frames):

| PP-OCRv5 detector, 1280 px | FP16 | INT8 MinMax | INT8 entropy |
|---|---|---|---|
| Detector GPU time | 31.9 ms | 23.2 ms | 31.5 ms |
| Boxes per frame | 8.8 | 16.8 | 7.7 |
| Lines found | 90.5% | 47.4% | 66.3% |
| Lines read exactly | 83.8% | 30.8% | 55.7% |
| Pipeline total p50 | 174.8 ms | 220.5 ms | 173.0 ms |

- Entropy stops the fragmentation but over-corrects: clipping weakens the probability map, so lines are missed instead.
- It also brought no detector speed-up. That result is confounded: the entropy build was the first with the 1 GB workspace cap, which may have excluded the fastest INT8 tactics (MinMax was built without the cap).
- **Conclusion for now: the OCR detector stays FP16.** Even a perfect INT8 detector would save about 9 ms of a 175 ms frame (5%). The CPU preprocessing before it costs 44 ms at 1280 px and 157 ms at 2560 px, so that's the better target.
- If INT8 is revisited, it should be done properly, in one of these ways:
  - percentile calibration
  - FP16 precision constraints on the DB head and neck (`OBEY_PRECISION_CONSTRAINTS`)
  - explicit Q/DQ quantization with per-layer sensitivity analysis (TensorRT Model Optimizer)

  Each would be built once with and once without the workspace cap, to separate the two effects.

### Calibration data: same preprocessing, never the test set

`_calib_batches()` in [fieldbench/__main__.py](../fieldbench/__main__.py) reads JPEGs from a calibration folder in sorted order. It runs each through **the pipeline's own preprocessing function** and yields the result. So calibration sees exactly the tensors inference sees:

| Workload | Model calibrated | Calibration images | Preprocessing (shared with the pipeline) | Test set (never used for calibration) |
|---|---|---|---|---|
| Barcode | YOLO11n-OBB detector, 640 / 1280 / 1600 px (MinMax) | First 300 of the 400 frames in `data/barcodes/val/images` (1600×1200 synthetic scenes from the same generator as the test set) | `yolo.letterbox_into`: aspect-preserving resize, gray padding, RGB, scaled to [0, 1] | `data/barcodes/test` (300 frames, 4 MP) |
| OCR | PP-OCRv5 DB text detector, 1280 / 1600 px (entropy; MinMax tried first, see above) | All 100 frames in `data/ocr/calib` (4 MP synthetic label scenes, separate seeds from the test set) | `ocr.det_preprocess_into`: resize, top-left zero padding, ImageNet mean/std on BGR | `data/ocr/test` (200 frames, 4 MP) |
| OCR | PP-OCRv5 recognizer | not calibrated: always FP16 | – | – |

Notes:

- The barcode validation split was also used to pick the best training epoch. It's held out from the **test** set, which is what matters for honest accuracy, but it isn't fully untouched data.
- Calibration frames are smaller (1600×1200) than test frames (2304×1728). After letterboxing to the detector's input size, objects appear at the same relative scale, so the activation ranges are comparable.
- The recognizer stays FP16. Calibrating it would need a set of cropped text lines rather than whole frames, and it isn't the stage that grows with input size. INT8 for it can come later if its time matters.

### Files and naming

Calibrated engines and their caches live in `engines/` on the Jetson:

```
barcode_yolo11n_640.int8cal-minmax-bc300.trt10.3.0.9646a86785.engine
barcode_yolo11n_640.int8cal-minmax-bc300.trt10.3.0.9646a86785.calib
└─ model ──────────┘ └ calibrator ┘└ set ┘ └ TRT ┘ └ ONNX sha1 ┘
```

- **calibrator**: `minmax` or `entropy`, so engines calibrated differently never collide.
- **calibration set**: a workload tag (`bc` for barcode, `ocr` for OCR) plus the number of images **actually used** (the first `--calib-images` files, default 300). The OCR folder has only 100 frames, so OCR engines are `ocr100`. The first sweep's engines were named before these two fields existed and were renamed in place on the Jetson (`int8cal-bc300` → `int8cal-minmax-bc300`, `int8cal-ocr300` → `int8cal-minmax-ocr100`), so nothing had to be rebuilt.
- **TRT version and ONNX hash**: a re-exported model or a JetPack upgrade can never pick up a stale engine.
- **The `.calib` cache** is TensorRT's text format: a header naming the TensorRT version and algorithm (`TRT-100300-MinMaxCalibration`), then one line per tensor with its scale as a hex-encoded float32:

  ```
  images: 3c010204        -> scale 0.007874 = 1/127, amax 1.0 (input in [0, 1])
  output0: 40b7daf4       -> scale 5.745, amax 730 (box coordinates in pixels, at 640)
  ```

  When the cache exists, TensorRT reads it and **skips the calibration pass**. It still builds and tunes the engine.

### How to (re)calibrate

```bash
# barcode: FP16 and calibrated INT8 by default (MinMax)
make live CMD=pipeline ARGS="barcode --sizes 640 1280"
# OCR: FP16 by default; INT8 only on request (entropy by default, or pick one)
make live CMD=pipeline ARGS="ocr --sizes 1280 --precisions int8 --calibrator entropy"

# other calibration data or count
make live CMD=pipeline ARGS="barcode --precisions int8 --calib data/barcodes/val/images --calib-images 400"
```

To force a fresh calibration with the same settings, delete both files on the Jetson: `engines/<name>.int8cal-*.engine` and the matching `.calib`. Deleting only the engine rebuilds it from the cached scales.

## Results so far

Jetson Orin Nano Super, MAXN_SUPER. Times are p50 per frame, with CPU JPEG decode.

**Barcode detector (MinMax), 300 test frames / 677 barcodes:**

| Input | Precision | Detector GPU time | Pipeline total | Barcodes read | Found | Misreads |
|---|---|---|---|---|---|---|
| 640 | FP16 | 12.4 ms | 64.0 ms | 93.1% | 100% | 2 |
| 640 | INT8 | 10.3 ms | 58.8 ms | 92.3% | 100% | 0 |
| 1280 | FP16 | 38.3 ms | 101.5 ms | 93.1% | 100% | 1 |
| 1280 | INT8 | 27.4 ms | 92.5 ms | 92.8% | 100% | 0 |
| 1600 | FP16 | 38.7 ms | 115.8 ms | 92.9% | 100% | 1 |
| 1600 | INT8 | 30.4 ms | 105.1 ms | 92.5% | 99.9% | 3 |

- INT8 cuts detector GPU time by 17% at 640, 28% at 1280 and 21% at 1600.
- Accuracy drops by 0.3–0.8 points. At 640 and 1280 INT8 gives fewer misreads, but at 1600 it loses one barcode and adds misreads. FP16 remains the safer default; INT8 is worth it at 1280 and above, where the detector is a larger share of the frame.
- At 640, the 2 ms saving is small next to JPEG decode (about 30 ms of the frame).

**OCR detector:** both calibrators lose too much accuracy (MinMax: lines read exactly 84% → 31%; entropy: 84% → 56%), so it stays FP16. See the tables above.

## Open items

- **OCR detector INT8, done properly**: percentile calibration, FP16 precision constraints on the DB head and neck, or explicit Q/DQ with sensitivity analysis. Build each with and without the workspace cap, to separate tactic availability from calibration.
- **Entropy on the barcode detector**, for completeness: does it really clip box coordinates in practice?
- **Confirm layer precisions**: build with detailed profiling verbosity and read the engine inspector's per-layer precision, to verify the detection head's tail runs in FP16.
- **Explicit quantization** (Q/DQ via TensorRT Model Optimizer) if a model loses too much in implicit INT8. Implicit calibration is deprecated in TensorRT 10.1+ and will eventually have to move there anyway.
- **Calibrate on real photos** (BarBeR) once that dataset is available, and check how much synthetic-only calibration costs on real images.
- **Recognizer INT8**, calibrated on cropped text lines, if its share of OCR time justifies it.
