# Phase 2 results: handheld pipelines

End-to-end pipelines on 4 MP JPEG frames with exact ground truth: what a handheld does on each trigger pull. Jetson Orin Nano Super, MAXN_SUPER, default DVFS governors unless a table says *locked clocks*. Zebra TC53 figures are Zebra's published times for the same job at the same detector input size. Test sets, pipelines and metrics: [README sections 7–8](../README.md#7-workloads-and-pipelines). Raw rows: `results/*-barcode-*.jsonl`, `results/*-ocr-*.jsonl`.

| Section | Topic |
|---|---|
| [1](#1-whole-frame-zxing-and-tilted-1d-codes) | Whole-frame zxing and tilted 1D codes |
| [2](#2-detector-accuracy) | Detector accuracy |
| [3](#3-export-integrity) | Export integrity |
| [4](#4-ocr-accuracy-vs-input-size) | OCR accuracy vs input size |
| [5](#5-hardware-jpeg-decode-nvjpg) | Hardware JPEG decode (NVJPG) |
| [6](#6-barcode-pipeline-on-the-jetson) | Barcode pipeline on the Jetson |
| [7](#7-ocr-pipeline-on-the-jetson) | OCR pipeline on the Jetson |
| [8](#8-int8-calibration-and-the-ocr-detector) | INT8 calibration and the OCR detector |
| [9](#9-gpu-preprocessing) | GPU preprocessing |
| [10](#10-dvfs) | DVFS |
| [11](#11-stage-overlap) | Stage overlap |
| [12](#12-pipelines-across-power-modes) | Pipelines across power modes |
| [13](#13-real-photos-barber) | Real photos (BarBeR) |


## 1. Whole-frame zxing and tilted 1D codes

**zxing-cpp can't read tilted 1D codes on its own.** Its 1D and PDF417 readers scan rows and columns only. Over the whole frame it reads 63% of the test barcodes. Cropping each barcode with its true polygon raises that only to 66%, while rotating each crop upright first gives 93% (98% on sharp frames, about 80% under blur, 67% for modules under 2 px). QR and DataMatrix are rotation-invariant and read at about 94% either way. So the detector predicts **oriented** boxes (YOLO11n-OBB), and the pipeline deskews each crop. On a handheld, omnidirectional 1D reading is table stakes, so this is the pipeline Zebra's barcode model competes with.

## 2. Detector accuracy

**The detector pipeline reads as well as perfect crops would.** YOLO11n-OBB (from DOTA weights, 1024 px, 50 epochs, about 30 min on an RTX 5080) reaches validation mAP50 0.995 / mAP50-95 0.987. On the test set (host, ONNX Runtime) it finds every barcode at 640, 1280 and 1600 px and decodes 92.6–92.9% of them. Decoding the ground-truth crops reaches 92.9%. Whole-frame zxing reaches 62.6%:

| | Read | Misreads | EAN-13 | Code 128 | PDF417 | QR |
|---|---|---|---|---|---|---|
| zxing-cpp, whole frame | 62.6% | 6 | 50.5% | 58.7% | 14.3% | 93.9% |
| Detector 640 + deskewed crops | 92.6% | 1 | 90.3% | 92.1% | 100% | 93.0% |
| Detector 1280 + deskewed crops | 92.9% | 0 | 91.3% | 92.1% | 100% | 93.9% |
| Ground-truth crops (ceiling) | 92.9% | 2 | 91.7% | 90.5% | 100% | 93.9% |

The codes that still fail are blurred codes with 1.5–2 px bars, and the ground-truth crops miss those too. The detector draws few extra boxes: 2.29 boxes per frame against 2.26 barcodes. Text decoys never trigger it; two striped decoys and one dot-grid background did, without producing a decode. Accuracy on real photos has not been measured; [BarBeR](https://ditto.ing.unimore.it/barber/) (Vezzali et al., ICPR 2024; 8,748 annotated real barcode images, free account required) is the natural check. The exported ONNX matches PyTorch to float noise.

## 3. Export integrity

**onnxslim corrupted the PP-OCRv5 recognizer** (0.1.97, its FusionGemm and EliminationReshape rewrites) while pinning static shapes. Outputs drifted by up to 0.99, so spaces vanished and letters changed case. The export now skips that fusion and refuses to save a static model whose output differs from the original by more than 1e-3. It was caught because the test set has ground truth.

## 4. OCR accuracy vs input size

**OCR accuracy scales with detector input size** (host, ONNX Runtime FP32, all 1,394 lines):

| Detector input | Lines found | Lines read exactly | Char. error rate | Words found |
|---|---|---|---|---|
| 640 | 74.8% | 68.3% | 25.8% | 74.5% |
| 1280 | 90.4% | 83.9% | 11.2% | 87.7% |
| 1600 | 92.5% | 86.4% | 8.8% | 90.3% |
| 2560 | 96.6% | 89.3% | 4.2% | 93.8% |

Of the remaining errors, about 4.7% of lines are recognizer limits: O/0 and I/1/l in monospace fonts, a dropped space, heavy motion blur. About 7% are the detector missing or merging lines, which falls from 15% at 640 to 2% at 2560. The scorer gives a detection credit when the boxes assigned to a line cover at least 70% of it (DB boxes are drawn about 1.5× as tall as the ink, so IoU against tight ground-truth boxes undercounts them). It joins a line split into several boxes. It doesn't penalize stray boxes on text that is part of the background photo.

## 5. Hardware JPEG decode (NVJPG)

**Hardware JPEG decode works and is bit-exact** ([fieldbench/jpeg.py](../fieldbench/jpeg.py)). Python has no direct route to NVJPG, so a small C++ shim (compiled with g++ on first use, no sudo) drives Tegra `libnvjpeg.so` the same way the Multimedia API's `NvJPEGDecoder::decodeToFd` does, and Python calls it through ctypes. The NVJPG decode itself is about 7 ms. The output lands in a buffer the CPU can only read uncached (about 0.6 GB/s), which shapes the rest. For grayscale, VIC first copies the frame into normal memory. For BGR, the CPU converts with NEON, reproducing libjpeg's arithmetic exactly. On a 4 MP frame (MAXN_SUPER, p50):

| | Gray: wall / CPU time | BGR: wall / CPU time |
|---|---|---|
| `cv2.imdecode` (CPU) | 21 / 21.6 ms | 36 / 36 ms |
| NVJPG | 17 / **2.9 ms** | 28 / 22 ms |

The win is CPU time, which the rest of a handheld pipeline needs. Gray decode, all that whole-frame zxing uses, drops from 21.6 to 2.9 ms of CPU. Converting to BGR on the CPU keeps most of the cost, and is slower in practice than in isolation because the CPU governor lowers clocks while the CPU waits on NVJPG. Letting VIC do the conversion cuts CPU time to 5 ms but gives wrong colours (it treats the JPEG as limited-range BT.601), so it's kept only for comparison. GStreamer's `nvjpegdec` is slower than plain CPU decode for BGR (49 ms). Each pipeline chooses its decoder: `--jpeg cpu nvjpg`.

## 6. Barcode pipeline on the Jetson

**On the Jetson, the barcode pipeline matches or beats the Zebra TC53 while reading 93% of barcodes** (MAXN_SUPER, 300 test frames at 4 MP; p50 per frame with CPU / NVJPG JPEG decode):

| Configuration | Total p50 | Barcodes read | vs TC53 |
|---|---|---|---|
| zxing-cpp, whole frame | 149 / 151 ms | 62.6% | – |
| 640 px, FP16 | 64.0 / 57.8 ms | 93.1% | 0.89× / 0.99× (57 ms) |
| 640 px, INT8 | 58.8 / **56.2 ms** | 92.3% | 0.97× / **1.02×** |
| 1280 px, FP16 | 101.5 / 101.9 ms | 93.1% | 0.93× / 0.92× (94 ms) |
| 1280 px, INT8 | 92.5 / 91.1 ms | 92.8% | 1.02× / 1.03× |
| 1600 px, FP16 | 115.8 / 116.2 ms | 92.9% | 1.07× / 1.07× (124 ms) |
| 1600 px, INT8 | 105.1 / 104.4 ms | 92.5% | 1.18× / 1.19× |

- The detector pipeline is 2.5× faster than whole-frame zxing and uses about 40% of the energy per frame (0.39–0.47 J vs 1.08 J at 640 px).
- JPEG decode is the largest stage at 640 px: 29–35 ms of a 56–64 ms frame, against 10–12 ms for the detector on the GPU.
- INT8 (MinMax calibration) cuts detector GPU time by 17–28% and costs 0.3–0.8 points of accuracy. At 1600 px it also loses one barcode and adds 3 misreads.
- TensorRT on the device matches the host's ONNX Runtime accuracy, and FP16 even reads slightly more (93.1% vs 92.6–92.9%).
- Board power stays at 7–8 W: the stages run one after another on the CPU, so the GPU sits idle most of each frame.

## 7. OCR pipeline on the Jetson

**OCR on the Jetson ties the TC53 at 640–1280 px and beats it by about 27% at 1600–2560 px** (FP16; p50 with CPU decode):

| Detector input | Total p50 | Detector resize (CPU) | Detector (GPU) | Recognizer | Lines read exactly | vs TC53 |
|---|---|---|---|---|---|---|
| 640 px | 110.3 ms | 10.1 ms | 8.7 ms | 33.1 ms | 68.2% | 1.00× (110 ms) |
| 1280 px | 174.8 ms | 43.9 ms | 31.9 ms | 35.0 ms | 83.8% | 1.03× (180 ms) |
| 1600 px | 209.5 ms | 62.1 ms | 55.2 ms | 33.0 ms | 86.4% | 1.29× (270 ms) |
| 2560 px | 381.9 ms | **156.7 ms** | 115.2 ms | 33.9 ms | 89.3% | 1.26× (480 ms) |

- **CPU preprocessing is the bottleneck at large sizes.** At 2560 px, resizing and normalizing the frame on the CPU takes longer than the detector on the GPU and is 41% of the frame. Moving it into a GPU kernel is the clearest optimization in this data (done in section 9).
- The recognizer costs a flat ~33 ms at every size, because every line is padded to 640 px wide. Width buckets would cut that.
- NVJPG doesn't help OCR, which needs BGR: the CPU still does the color conversion. Peak Tj was 79 °C at 2560 px, with no throttling.

## 8. INT8 calibration and the OCR detector

**MinMax INT8 calibration breaks the OCR text detector.** Its detector GPU time drops 27%, but its output degrades:

| | 1280 FP16 | 1280 INT8 MinMax | 1600 FP16 | 1600 INT8 MinMax |
|---|---|---|---|---|
| Lines read exactly | 83.8% | 30.8% | 86.4% | 47.1% |
| Boxes per frame | 8.8 | 16.8 | 9.4 | 17.2 |
| Total p50 | 174.8 ms | 220.5 ms | 209.5 ms | 262.0 ms |

- The detector fragments text into twice as many boxes, each of which must be recognized. So the INT8 pipeline ends up **slower and far less accurate** than FP16.
- Cause: the detector's neck has activations up to about 1,500. MinMax sets each tensor's scale from its extreme value, which crushes ordinary activations into a few INT8 levels.
- MinMax suits the YOLO head (it doesn't clip pixel-coordinate box values) but not this model. The calibrator is chosen per workload (`--calibrator`, OCR defaults to entropy).
- Entropy calibration avoids the fragmentation but misses lines instead: 55.7% read exactly at 1280 px, with no detector speed-up. That build was also the first with a 1 GB workspace cap, so its speed is confounded.
- **The OCR detector stays FP16.** Even a perfect INT8 detector would save about 5% of the frame, while CPU preprocessing costs 25–41%. Details in [docs/int8-calibration-in-fieldbench.md](../docs/int8-calibration-in-fieldbench.md).

## 9. GPU preprocessing

**Detector preprocessing runs as one CUDA kernel, bit-exact with the CPU path** ([fieldbench/gpuprep.py](../fieldbench/gpuprep.py), `--prep cpu gpu`). The kernel does the bilinear resize, the padding, the normalization and the HWC→CHW transpose in one pass, and writes the float32 result straight into the TensorRT input buffer, so the 4–79 MB host-to-device copy of the input tensor disappears. It is compiled at runtime with NVRTC (about 0.3 s, once per process), so the engines don't change and nothing needed rebuilding.
- **No copies.** On Orin, `cudaMallocHost` memory is CPU-cached: filling and reading a 12 MB frame costs the same as with ordinary numpy memory (1.7 ms to fill it), and the GPU reads it in place. NVJPG decodes straight into that buffer. The CPU decoder can't write into a caller's buffer, so its frame is copied in (1.7 ms).
- **Bit-exact.** OpenCV on the Jetson resizes with NEON, and its SIMD vertical pass rounds differently from its scalar code: a kernel using OpenCV's scalar formula disagreed with `cv2.resize` on 8.5–11% of pixels (always by one level). The kernel uses the SIMD formula, `((b0·(h0>>4))>>16 + (b1·(h1>>4))>>16 + 2)>>2`, with source taps and 11-bit weights built on the host the way OpenCV builds them. Normalization is a 256-entry table per channel, filled by the CPU path's own numpy expression. Over 20 test frames per workload, the output matches the CPU path exactly at 640, 1280 and 1600 px for both OCR and barcode (max abs diff 0). At 2560 px the frame is upscaled, and OpenCV rounds the two clamped edge rows some other way: 0.0089% of values, one level each. End to end, accuracy is identical in all 28 GPU-prep configurations (OCR exact lines and CER, barcode reads and misreads).
- **Cost of the step** (`python -m fieldbench.gpuprep`, p50 per 4 MP frame; GPU wall includes the copy into the pinned frame):

  | | OCR 640 | OCR 1280 | OCR 1600 | OCR 2560 | Barcode 640 | Barcode 1280 | Barcode 1600 |
  |---|---|---|---|---|---|---|---|
  | CPU (cv2 + numpy) | 9.0 ms | 33.4 ms | 50.5 ms | 149.0 ms | 4.6 ms | 13.0 ms | 21.0 ms |
  | GPU, wall | 3.2 ms | 4.1 ms | 4.8 ms | 6.8 ms | 3.1 ms | 4.1 ms | 4.6 ms |
  | GPU, kernel only | 0.49 ms | 1.41 ms | 1.94 ms | 4.09 ms | 0.49 ms | 1.41 ms | 1.94 ms |

  Most of the CPU cost was numpy's float passes over the canvas, not the resize: `cv2.resize` alone to 1280 takes about 4 ms. The kernel moves about 25 MB at 1280 px in 1.41 ms, roughly 18 GB/s against the board's 102 GB/s. It isn't tuned (one thread per output pixel, byte gathers from pinned memory), but it is 1–2% of a frame.
- **With clocks locked** (`jetson_clocks`, so the governors can't confound the comparison), every other stage stays within about 4 ms (most within 1 ms), and the frame gets faster by about the preprocessing time removed:

  | Locked clocks, p50 | CPU prep | GPU prep | Detector prep (mean) | vs TC53 |
  |---|---|---|---|---|
  | OCR 1280, CPU decode | 119.5 ms | 83.9 ms (−30%) | 36.0 → 2.3 ms | 2.15× |
  | OCR 1280, NVJPG | 112.2 ms | **76.8 ms** (−32%) | 35.6 → 1.0 ms | **2.34×** |
  | OCR 2560, CPU decode | 320.8 ms | 174.1 ms (−46%) | 142.2 → 3.4 ms | 2.76× |
  | OCR 2560, NVJPG | 310.0 ms | **166.7 ms** (−46%) | 137.7 → 2.0 ms | **2.88×** |
  | Barcode 640 FP16, NVJPG | 41.6 ms | **38.6 ms** (−7%) | 3.3 → 0.7 ms | **1.48×** |
  | Barcode 1280 FP16, NVJPG | 60.2 ms | **47.9 ms** (−20%) | 12.1 → 1.0 ms | **1.96×** |

- **With the default governors**, as in sections 6–8 (MAXN_SUPER, same session for each pair; best configuration per size, NVJPG decode):

  | Default governors, p50 | CPU prep | GPU prep | vs TC53 | mJ / frame |
  |---|---|---|---|---|
  | OCR 640 | 130.2 ms | 121.1 ms | 0.91× | 1048 → 992 |
  | OCR 1280 | 172.5 ms | 135.9 ms | 1.32× | 1723 → 1437 |
  | OCR 1600 | 207.7 ms | 152.2 ms | 1.77× | 2245 → 1810 |
  | OCR 2560 | 385.0 ms | **203.0 ms** | **2.36×** | 4185 → 3131 |
  | Barcode 640, INT8 | 55.2 ms | 52.3 ms | 1.09× | 376 → 346 |
  | Barcode 1280, INT8 | 91.1 ms | 71.8 ms | 1.31× | 724 → 573 |
  | Barcode 1600, INT8 | 104.7 ms | **82.6 ms** | **1.50×** | 904 → 733 |

  Energy per frame falls 5–25% in every pair, although board power rises at the large OCR sizes (10.7 → 15.0 W at 2560 px): the frame finishes sooner. Under the governors the gains are noisier, because the clocks move with the load (section 10). At 2560 px the GPU-prep run had the GPU at 1012 vs 833 MHz and the memory at 3188 vs 2161 MHz, so its detector also got faster (114 → 90 ms). A short smoke run at 1280 px, compared against an earlier session, went the other way (GPU 756 → 515 MHz).

## 10. DVFS

**The default DVFS governors cost these pipelines 18–45% of their latency.** The stages run one after another, so the CPU, the GPU and the memory controller each sit partly idle, and their governors keep the clocks low. At barcode 640 px, the GPU averages its 306 MHz minimum, and the detector takes 12–13 ms instead of 4.3–4.8 ms locked. Same configuration, default governors → locked clocks (GPU 1020 MHz, EMC 3199 MHz, CPU 1728 MHz):

| p50 | Default | Locked | Board W | mJ / frame |
|---|---|---|---|---|
| OCR 1280, NVJPG, GPU prep | 135.9 ms | 76.8 ms (−43%) | 10.1 → 14.6 | 1437 → **1263** |
| OCR 2560, NVJPG, GPU prep | 203.0 ms | 166.7 ms (−18%) | 15.1 → 16.5 | 3131 → **2883** |
| Barcode 640 FP16, NVJPG, GPU prep | 54.7 ms | 38.6 ms (−29%) | 6.8 → 9.6 | 376 → 373 |
| Barcode 1280 FP16, NVJPG, GPU prep | 87.0 ms | 47.9 ms (−45%) | 7.5 → 12.1 | 662 → **591** |

- Locked clocks draw more power but use the **same or less energy per frame**: finishing sooner and idling saves more than the higher clocks cost.
- **Default-governor numbers move between sessions.** OCR 640 px with CPU prep and CPU decode measured 110.3 ms in section 7 and 137.1 ms in a later session. Comparisons in these documents are only made within one session.
- A handheld does the equivalent of locking: Android raises clocks on input and camera events (boost hints). On the Jetson, `jetson_clocks` needs root; raising only the GPU and EMC minimum frequencies would be a gentler middle ground to measure. Overlapping the stages (section 11) keeps the units busy instead.
- Telemetry now records the memory (EMC) clock, read through the jtop service because debugfs needs root, and the report shows GPU and EMC clocks per pipeline row.

## 11. Stage overlap

**Running two or three copies of the pipeline on consecutive frames multiplies barcode throughput by 3.3× at lower latency, and nearly doubles OCR throughput until the GPU saturates** (`--workers 1 2 3`; MAXN_SUPER, default governors, GPU preprocessing, one session per workload). Each worker is a thread with its own TensorRT execution context, CUDA stream, pinned buffers and JPEG decoder. While one worker decodes a JPEG or runs zxing on the CPU, another's network runs on the GPU. Accuracy is measured through the same workers and is identical to one worker in every configuration (barcode 93.1% read, 2 misreads; OCR 1280 83.8% exact; OCR 2560 89.3% exact).

| Configuration | Workers | Frames/s | vs 1 worker | Latency p50 | Board W | mJ / frame | GPU MHz (mean) | GPU load | Over-current events |
|---|---|---|---|---|---|---|---|---|---|
| Barcode 640 FP16, NVJPG | 1 | 16.7 | 1.00× | 59.9 ms | 7.3 | 438 | 363 | 53% | 0 |
| | 2 | 34.8 | 2.08× | 56.6 ms | 8.7 | 251 | 493 | 56% | 0 |
| | 3 | **55.2** | **3.30×** | **53.3 ms** | 10.5 | **190** | 604 | 59% | 0 |
| Barcode 640 FP16, CPU decode | 1 | 15.0 | 1.00× | 67.1 ms | 7.6 | 507 | 344 | 46% | 0 |
| | 3 | 48.7 | 3.25× | 60.9 ms | 10.5 | 215 | 537 | 57% | 0 |
| OCR 1280 FP16, NVJPG | 1 | 7.3 | 1.00× | 124.4 ms | 10.5 | 1441 | 842 | 56% | 1 |
| | 2 | **13.0** | **1.78×** | 147.1 ms | 16.3 | 1259 | 1020 | 74% | 892 |
| | 3 | 14.2 | 1.95× | 197.9 ms | 17.5 | 1230 | 1020 | 86% | 1,476 |
| OCR 2560 FP16, NVJPG | 1 | 4.6 | 1.00× | 208.5 ms | 14.7 | 3206 | 1004 | 64% | 608 |
| | 2 | 6.3 | 1.37× | 301.5 ms | 17.8 | 2806 | 1020 | 86% | 1,907 |
| | 3 | 6.8 | 1.48× | 424.8 ms | 18.9 | 2777 | 1020 | 96% | 2,609 |

- **Barcode scales better than linearly.** With one worker the GPU is idle for most of each frame, so its governor holds it near the 306 MHz floor (section 10). Overlapping frames keeps it busy, so the clock rises (363 → 604 MHz), and the detector itself gets faster (GPU time 15.1 → 10.9 ms mean). Throughput rises 3.3×, per-frame latency *falls* 11%, and energy per frame falls 57% (438 → 190 mJ), because the board's ≈5.5 W floor is shared by three times as many frames. Overlap gets most of the locked-clock benefit without root.
- **OCR becomes GPU-bound.** At 1280 px, two workers reach 1.78× and the GPU clock pins at its 1020 MHz maximum. A third worker adds only 9% more throughput, and latency grows by 59% over one worker, because frames queue for the GPU (detector GPU time 29 → 53 ms as kernels from different workers interleave). At 2560 px the GPU is the bottleneck even with one worker, and three workers give 1.48× for twice the latency. For OCR, two workers is the sweet spot: about 1.8× throughput for +18% latency.
- **The power limit appears.** Above about 16 W board power, the SoC's over-current alarm counters (`throttle_events`: hardware over-current alarms, which can briefly throttle clocks) climb into the hundreds or thousands per run. One-worker runs stayed at zero to one, except OCR 2560 at 14.7 W. Those OCR rows are at the edge of the module's power budget, so their throughput includes throttling. Peak junction temperature reached 85 °C (OCR 2560, three workers).
- **NVJPG still matters at high throughput.** With three barcode workers, the hardware decoder gives 13% more throughput than CPU decode (55.2 vs 48.7 fps), because the six CPU cores are shared by decode, zxing and Python.

What this means for a handheld: a scanner that processes a camera stream (continuous scanning, "pick list" or multi-barcode modes) should pipeline frames rather than run one at a time. On this board, that turns a 60 ms single-frame barcode reader into a 55 frames/s stream reader, at a lower cost per frame.

## 12. Pipelines across power modes

**Under the default governors, the barcode pipeline runs at the same speed and energy in every power mode, and OCR stays ahead of the TC53 even at 15W** (`--power-modes 15W 25W MAXN_SUPER`; FP16, NVJPG decode, GPU preprocessing, one worker, one session; 7W needs a reboot and is not included).

| Pipeline | Mode | Total p50 | vs TC53 | Board W | mJ / frame | GPU MHz (mean) | Accuracy |
|---|---|---|---|---|---|---|---|
| Barcode 640 | 15W | 61.8 ms | 0.92× | 7.1 | 443 | 334 | 93.1% read |
| | 25W | 62.5 ms | 0.91× | 7.0 | 443 | 369 | 93.1% |
| | MAXN_SUPER | 58.5 ms | 0.97× | 7.3 | 439 | 400 | 93.1% |
| OCR 1280 | 15W | 151.0 ms | **1.19×** | 8.9 | 1493 | 612 | 83.8% exact |
| | 25W | 136.7 ms | 1.32× | 9.6 | 1457 | 781 | 83.8% |
| | MAXN_SUPER | 130.1 ms | 1.38× | 10.3 | 1455 | 840 | 83.8% |

- **The barcode pipeline never reaches the mode caps.** Its GPU averages 334–400 MHz in every mode, close to the 306 MHz floor and far below even the 15W cap (612 MHz), because the GPU is idle for most of each frame (section 10). The power mode only matters when something keeps the units busy. With one frame at a time, the governor decides the speed, not the mode.
- **OCR is busy enough to feel the cap.** At 15W the GPU pins at its 612 MHz cap, and the frame takes 16% longer than at MAXN_SUPER, yet it still beats the TC53 by 1.19×.
- **Energy per frame is flat across modes** (within 1% for barcode and 3% for OCR), as Phase 1 found for the bare networks. A lower mode caps peak power without saving energy on these workloads.

## 13. Real photos (BarBeR)

**On real photos, the synthetic-trained barcode detector reads fewer barcodes than whole-frame zxing, and gets worse at higher input sizes: a domain gap, not a decoder limit.**

[BarBeR](https://ditto.ing.unimore.it/barber/) (Vezzali, Bolelli, Santi and Grana, "BarBeR: A Barcode Benchmarking Repository", ICPR 2024) pools 12 public datasets of real barcode photos, with an annotated polygon, symbology and encoded string for each code. `host/make_barber.py` converts it to fieldbench's format and keeps the barcodes zxing-cpp can read whose string is known. Postal codes, IATA 2-of-5, add-ons and codes marked undecodable stay in the photo unscored. It then draws a stratified sample: **600 photos with 659 barcodes**, the same share from each source dataset (EAN-13 393, Code 128 105, QR 94, UPC-A 18, Code 39 15, DataMatrix 13, PDF417 10, others 11). The photos range up to 15 MP (median 1.9 MP, 10th percentile 0.3 MP), with a median of 4.1 px per module (10th percentile 1.5). Annotated strings and zxing reads are compared in a canonical form: Code 39 `*` delimiters, GS1 `(AI)` brackets, a UPC-A leading zero and HTML escapes are normalized on both sides. MAXN_SUPER, CPU decode and CPU preprocessing (the frame size changes with every photo).

| Configuration | Barcodes read | Detection recall | Boxes per photo | QR read | EAN-13 read |
|---|---|---|---|---|---|
| Ground-truth crops + zxing (ceiling, host) | 71.2% | – | – | 82% | 80% |
| **zxing-cpp, whole photo** | **70.4%** | – | – | 85% | 78% |
| Detector 640 px, FP16 | 60.4% | 83.3% | 1.00 | 50% | 70% |
| Detector 640 px, INT8 | 60.2% | 82.4% | 0.97 | 53% | 69% |
| Detector 1280 px, FP16 | 45.7% | 69.0% | 0.97 | 16% | 54% |
| Detector 1600 px, FP16 | 39.3% | 60.8% | 0.85 | 17% | 44% |

- **The ceiling is low because the photos are hard.** Decoding each annotated barcode from its own upright crop reads only 71.2%. Many codes have 1.1–1.5 px modules (Deal Kaist, ParcelBar) or are blurred. Whole-photo zxing is already within a point of that ceiling, because most of these photos are framed around a single code, which is the case zxing's row scanner was built for.
- **The detector misses real codes the synthetic set never showed it.** At 640 px it finds 83% of barcodes, against 100% on the synthetic test set. Larger input sizes make it worse (recall 83% → 61%), the opposite of the synthetic results. The training scenes were 4 MP frames with small labels, while many BarBeR photos are close-ups in which one code fills the frame. Upscaling such a photo to 1280–1600 px makes the code far larger than anything in training. QR codes, usually photographed close, drop from 50% to 16–17%.
- **INT8 costs nothing extra here** (60.2% vs 60.4% at 640 px), consistent with the synthetic results (section 6).
- **What fixes it:** fine-tune the detector on real images (BarBeR's own training split, plus close-up and low-resolution synthetic scenes), and pick the detector input size from the photo size instead of always upscaling. A product pipeline would also fall back to whole-frame zxing when the detector finds nothing. The timing of these rows isn't compared with the TC53, because photo sizes vary by more than 20×.

