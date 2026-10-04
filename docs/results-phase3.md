# Phase 3 results: several models per frame, product recognition, sustained load

Phase 2 timed one reader at a time. A handheld assistant that doesn't know what the user is pointing at runs every reader on each frame. A retail handheld also identifies the product itself, and a shift lasts hours, not 20 seconds. Jetson Orin Nano Super, MAXN_SUPER, default governors, NVJPG decode and GPU preprocessing throughout. Raw rows: `results/*-assistant-*.jsonl`, `results/*-product-*.jsonl`, `results/*-soak-*.jsonl`.

## 1. Barcode and OCR on every frame

**Reading barcodes and text on the same frame takes 136–151 ms, 1.57–1.74× faster than a TC53 doing the two jobs one after the other** (`pipeline assistant`). The JPEG is decoded once into pinned memory, and both letterbox kernels read it in place. Then the barcode branch (YOLO11n-OBB 640 px FP16 → deskewed crops → zxing-cpp) and the OCR branch (PP-OCRv5 detector 1280 px FP16 → recognizer) run either one after the other (`--order sequential`) or as two threads, each with its own CUDA stream and TensorRT context (`--order concurrent`). The test set interleaves 150 barcode frames and 100 OCR frames (every second frame of each set), and every frame goes through both branches. The TC53 reference is Zebra's barcode (57 ms at 640 px) plus OCR (180 ms at 1280 px) published times, which add up to 237 ms.

| Order | Workers | Total p50 | p90 | Frames/s | vs TC53 (237 ms) | Board W | mJ / frame | Barcode branch (mean) | OCR branch (mean) | Over-current events |
|---|---|---|---|---|---|---|---|---|---|---|
| Sequential | 1 | 151.3 ms | 209.9 ms | 6.2 | 1.57× | 10.0 | 1603 | 17.8 ms | 100.9 ms | 0 |
| **Concurrent** | 1 | **136.1 ms** | 196.1 ms | 6.8 | **1.74×** | 10.6 | 1555 | 47.9 ms (overlapped) | 104.1 ms | 0 |
| Sequential | 2 | 156.5 ms | 217.8 ms | 12.1 | – | 16.0 | 1324 | 24.1 ms | 103.4 ms | 848 |
| Concurrent | 2 | 152.3 ms | 213.8 ms | **12.4** | – | 16.4 | 1325 | 62.9 ms (overlapped) | 121.8 ms | 1,296 |

Accuracy is the same in all four rows: 93.8% of barcodes read with 1 misread, and 83.0% of text lines read exactly. These are the same models as Phase 2, scored on the half-size subsets.

- **The barcode reader is almost free next to OCR.** Sequentially, the barcode branch adds 17.8 ms (a 6.7 ms detector and 8.2 ms of zxing) to a 100.9 ms OCR branch. On this board, a "scan anything" mode costs about 12% more per frame than OCR alone (17.8 ms on top of 142 ms of decode and OCR).
- **Concurrent branches save 15 ms, not the full 18.** The barcode detector's GPU time rises from 6.7 ms to 37.5 ms when it runs alongside OCR. The two TensorRT contexts time-share the GPU, and the barcode kernels wait behind the OCR detector's. What the overlap really hides is the barcode branch's CPU work (crops and zxing). CUDA stream priorities (`cudaStreamCreateWithPriority`) would let the short barcode kernels jump the queue, which is the next thing to try.
- **Two workers double throughput to 12 frames/s at 3–12% more latency per frame.** That pushes board power to 16 W, where the over-current alarms start firing (as in [Phase 2, section 11](results-phase2.md#11-stage-overlap)).

## 2. Product recognition

**A MobileNetV3-Large embedding with a class-centroid lookup identifies 71% of grocery products exactly and has the right one in its top 5 for 96.5%, at 7.4 ms per image** (`pipeline product`). This is the retrieval approach used for SKU recognition: an image is embedded (here the 960-d pooled features of ImageNet-pretrained MobileNetV3-L, with no fine-tuning) and compared by cosine similarity against stored embeddings of known products. New products need gallery images, not retraining. There's no localizer: the test images are already centred on the product, which matches a point-and-shoot trigger.

- **Data:** [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset) (Klasson, Zhang and Kjellström, WACV 2019; MIT license). Phone photos taken in stores of fruit, vegetables and packaged goods, mostly 348×348 px. 81 fine classes in 43 coarse groups. Gallery: the train and val splits, 2,936 images. Test: 2,485 images. `make products` downloads and lays it out.
- **Matcher:** each class's gallery embeddings are averaged into one unit-length centroid, and the query takes the nearest centroid. On the host, centroid matching beat nearest-neighbour (71.1% vs 67.8% top-1) and is 4× cheaper (81 comparisons instead of 2,936). `ProductPipeline(method="knn")` gives the neighbour variant.
- **Preprocessing:** resize the shorter side to 256, centre crop 224, ImageNet normalization, all on the CPU.

| Workers | Images/s | Latency p50 | p90 | p99 | Board W | mJ / image (above idle) | GPU MHz (mean) |
|---|---|---|---|---|---|---|---|
| 1 | 112.9 | **7.4 ms** | 13.7 ms | 21.3 ms | 10.4 | 92 (30) | 409 |
| 2 | 191.8 | 9.4 ms | 15.7 ms | 23.1 ms | 11.2 | 58 (21) | 538 |
| 3 | **238.5** | 11.5 ms | 19.1 ms | 28.6 ms | 11.6 | **49** (18) | 614 |

| Accuracy (2,485 test images) | Jetson, TensorRT FP16 | Host, ONNX Runtime FP32 |
|---|---|---|
| Top-1 (81 classes) | 71.0% | 71.1% |
| Top-5 | 96.5% | 96.5% |
| Top-1 coarse (43 groups) | 82.6% | 82.6% |

- **FP16 is free here.** Accuracy matches the FP32 reference to within one image.
- **The network is half the frame.** Per image (one worker, means): JPEG decode 1.5 ms, preprocessing 2.5 ms, inference 4.1 ms wall (3.6 ms on the GPU), centroid match 0.7 ms. The 3.6 ms GPU time is 2.5× the Phase 1 MobileNetV3-L FP16 figure (1.41 ms), because the GPU governor sits at 409 MHz on this light, bursty load (Phase 2, section 10). The QCS6490 NPU runs the classifier version in 1.17 ms (w8a8), so on a single image the handheld NPU is faster. With three workers the Jetson recognizes 239 images/s, enough to classify every product crop of a shelf photo in one pass.
- **Accuracy is the limit, not speed.** 71% top-1 with an off-the-shelf ImageNet backbone is a baseline. Retail systems fine-tune the embedding with metric learning on their own catalogue. Coarse accuracy (82.6%) shows that about 40% of the errors (11.6 of 29 points) pick another product from the right group, such as a different Arla milk or yoghurt.

## 3. Sustained load (thermal soak)

**Two 20-minute soaks at the board's power limit both hung after about 20 minutes, with the same signature; the JPEG decoder is not the cause.** Configuration: assistant, concurrent branches, 2 workers, GPU preprocessing, about 16 W board power, where the SoC's over-current alarms fire by the thousands (section 1).

| Run | JPEG decode | Stalled after | Rows written |
|---|---|---|---|
| 1 | NVJPG | ≤ 25 min (found stalled at 25 min) | none |
| 2 | CPU (`cv2.imdecode`) | 19–20 min (stall watcher, 30 s resolution) | none |

The signature was the same both times:

- One thread was blocked in the kernel in `dma_fence_default_wait`, and the others sat in futex waits.
- The process used almost no CPU.
- GPU load read 99.9% at the 306 MHz floor clock.
- No nvgpu errors were in the kernel log.
- Killing the process released the GPU immediately, with no reboot needed.

20-second runs of the same configuration complete normally (section 1), and 20-minute runs at lower power never got the chance to fail. The common factor is sustained operation at the power limit with four TensorRT contexts and several CUDA streams in flight. Run 2 rules out the NVJPG path. A GPU work item that never completes under prolonged over-current throttling would produce exactly this. The cause is not identified. To pin it down, the runner needs a stall watchdog that writes a partial row with the time series so far (rows are currently written only at the end of a configuration), then reruns at 2 workers with locked clocks, and with one worker.

**Below the power limit, sustained load is completely stable.** A 10-minute soak with one worker (assistant, concurrent branches, CPU JPEG decode, GPU preprocessing) completed normally:

| | Whole run | Range of the 30-s windows |
|---|---|---|
| Frames | 3,292 in 600 s | – |
| Throughput | 5.48 frames/s | 5.37–5.67 |
| Latency p50 | 172.1 ms (p90 234.8, p99 328.3) | 165–176 ms |
| Board power | 8.5 W | 8.42–8.62 W |
| Junction temperature (max) | 69.6 °C | 68.8–69.6 °C |
| GPU clock (mean) | 527 MHz | 513–544 MHz |
| Over-current alarms | 0 | – |
| Accuracy | 93.8% barcodes, 83.0% lines exact | unchanged |

- **No drift.** Throughput, latency, power and temperature stay flat from the first window to the last. The temperature never moves by more than 1 °C, so the fan and heatsink hold this load indefinitely. The last window covers only the moments after the 600 s mark and is left out.
- **Latency is higher than in section 1** (172 vs 136 ms p50) because this run decodes JPEGs on the CPU (about 40 ms, against about 11 ms of CPU time for NVJPG) and the GPU runs at a lower clock under the default governors. It's a different session, so compare only within each table.
- **The safe envelope:** one assistant worker at about 8.5 W runs indefinitely. Two workers at about 16 W hit the over-current limit and hung twice after about 20 minutes. A deployment on this module should cap sustained load below the over-current threshold, for example with fewer workers or the 15W mode, until the hang is understood.
