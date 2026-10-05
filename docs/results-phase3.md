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

**Two workers with concurrent branches can hang the GPU at MAXN_SUPER. The cause is not identified; it is not NVJPG and not the over-current limit.** Configuration: assistant, concurrent branches, 2 workers, GPU preprocessing, MAXN_SUPER.

The first two soaks (NVJPG, then CPU decode) stopped producing frames after about 20 minutes and wrote no rows. `pipeline` now has a stall watchdog (`--stall-s`). When a worker finishes no frame for that long, the watchdog writes a partial row with the telemetry series so far, every thread's Python stack and kernel wait channel, and two snapshots of the host1x/GPU interrupt counters. Then it exits with code 3. With the watchdog armed, the same configuration hung twice more:

| Run | JPEG decode | Hung after (timed loop) | Frames before | Last 30-s window |
|---|---|---|---|---|
| 1 | NVJPG | ≤ 25 min | – (no watchdog yet) | – |
| 2 | CPU | 19–20 min | – (no watchdog yet) | – |
| 3 (`hang-repro`) | CPU | about 40 min | 15,346 | – |
| 4 (`hang-repro2`) | CPU | about 2.4 min | 2,296 | 11.5 W, 0 over-current alarms, Tj 80 °C, GPU 1020 MHz, 99.9% load |

The signature was the same every time:

- A worker is blocked in `cudaStreamSynchronize` after a TensorRT inference, and its kernel wait channel is `dma_fence_default_wait`. The other threads sit in futex or poll waits.
- The host1x syncpoint interrupt counters keep advancing, so the host1x interrupt freeze reported on JetPack 6.2.1 is not the cause.
- The process uses almost no CPU, and killing it releases the GPU with no reboot needed.

To narrow it down, five 10-minute runs each changed one thing (`scripts/hang_matrix.sh`, CPU decode, GPU preprocessing, 2 workers, `--stall-s 60`):

| Run | Change | Result | Throughput | p50 | Over-current alarms |
|---|---|---|---|---|---|
| A | 15W power mode | completed | 9.7 frames/s | 195 ms | 0 |
| B | OCR pipeline alone (1280) | completed | 13.1 | 146 ms | 7,846 |
| C | OCR recognizer inferences under one process-wide lock (`--serialize rec`) | completed | 13.7 | 138 ms | 40,274 |
| D | every TensorRT inference under the lock (`--serialize all`) | completed | 13.9 | 136 ms | 32,266 |
| E | branches in turn (`--order sequential`) | completed | 12.1 | 158 ms | 11,093 |

- **Over-current throttling is not required.** Run 4 hung at 11.5 W with no alarms in its last windows, while C and D threw tens of thousands of alarms and completed.
- **Serializing the recognizer costs nothing.** C's p50 (138 ms) and throughput are as good as those of the unlocked runs, so a recognizer lock is a cheap mitigation if it holds.
- **The matrix doesn't prove a fix.** The unlocked runs hung anywhere from 2.4 to 40 minutes in, so a single clean 10-minute run is weak evidence. Confirming C needs repeated soaks of an hour or more, and naming the kernel that never completes needs an Nsight Systems trace of a hang. Both are left open.

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
- **The safe envelope:** one assistant worker at about 8.5 W runs indefinitely. Two workers with concurrent branches at MAXN_SUPER hung four times, after 2 to 40 minutes. Until the hang is understood, a deployment on this module should run one worker, or serialize the TensorRT calls, and use a watchdog.
