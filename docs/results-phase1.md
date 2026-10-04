# Phase 1 results: model benchmarks

Bare networks on random input tensors, batch 1, TensorRT 10.3 on the Jetson Orin Nano Super (JetPack 6.2.1), compared with the best published Qualcomm QCS6490 NPU latency for the same model. How the numbers are produced: [methodology](methodology.md). Raw rows: `results/*-full-sweep.jsonl`, `*-power-sweep.jsonl`, `*-power-7w.jsonl`.

## 1. Precision sweep at MAXN_SUPER
| Model | Precision | GPU p50 | GPU p99 | E2E p50 | FPS | Board power | Energy / inf (work) | Tj max | vs QCS6490 best |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11n | FP32 | 7.29 ms | 10.32 ms | 7.65 ms | 129 | 18.7 W | 145.0 mJ (101.2) | 70.9 °C | 0.56× |
| YOLO11n | FP16 | 4.04 ms | 5.18 ms | 4.39 ms | 224 | 17.3 W | 77.0 mJ (52.5) | 72.0 °C | 1.01× |
| YOLO11n | INT8\* | **3.30 ms** | 4.47 ms | 3.65 ms | 269 | 15.0 W | **55.6 mJ** (34.5) | 71.4 °C | **1.24×** |
| MobileNetV3-L | FP32 | 1.77 ms | 1.93 ms | 1.87 ms | 523 | 14.8 W | 28.2 mJ (17.4) | 70.1 °C | 0.66× |
| MobileNetV3-L | FP16 | 1.41 ms | 1.48 ms | 1.50 ms | 652 | 11.3 W | 17.4 mJ (9.0) | 69.7 °C | 0.83× |
| MobileNetV3-L | INT8\* | **1.16 ms** | 1.21 ms | 1.25 ms | 781 | 10.1 W | **12.9 mJ** (5.8) | 67.0 °C | **1.00×** |

*Run `full-sweep`, MAXN_SUPER, dynamic clocks (GPU held at 1020 MHz under load), 10 s per configuration, board idle ≈ 5.6 W, zero throttle events, 2026-10-04. QCS6490 best = 4.10 ms (YOLO11n) and 1.17 ms (MobileNetV3-L), both TFLite w8a8 on the NPU; > 1× means the Jetson is faster. \*INT8 uncalibrated, so the timing is valid but the accuracy is not. Engine builds: 92–592 s, cached afterwards. Full charts: `report/index.html`.*

### Findings

1. **INT8 puts the Jetson at or ahead of the handheld NPU.** YOLO11n runs 24% faster than the QCS6490's best published result; MobileNetV3-L ties it. FP16 alone already matches on YOLO11n.
2. **Precision is the main energy lever.** Going FP32 → FP16 → INT8 cuts energy per frame by 2.6× for YOLO11n (145 → 56 mJ) and 2.2× for MobileNet (28 → 13 mJ), because each frame finishes sooner *and* the board draws less power while it runs.
3. **The idle board is a big share of every frame.** About 5.6 W goes to just keeping the dev kit on: 38% of YOLO11n INT8's energy per frame, and 55% of MobileNet INT8's. The higher the throughput per watt, the more that overhead dominates. Lower power modes (section 2) and doing more work per wake-up (batching, several models per frame) are the levers that follow from this.
4. **Small models leave the GPU under-used.** MobileNet holds GPU load at 83–87% versus about 95% for YOLO11n, consistent with per-launch overhead mattering at ~1 ms per frame. CUDA graphs or batching should help, and that's where a dedicated NPU has a structural edge.
5. **Thermals are a non-issue at this duration.** Tj peaked at 72 °C with no throttling. Sustained handheld-style use needs longer soak runs (`--duration 1200 --series 30`).
6. **Copy overhead is small.** End-to-end adds 0.1–0.4 ms over GPU time for these input sizes.

## 2. Power modes

| Model | Precision | Mode | GPU p50 | Board power | Energy / inf (work) | vs QCS6490 best |
|---|---|---|---|---|---|---|
| YOLO11n | INT8\* | 7W | 15.29 ms | 7.0 W | 117.6 mJ (28.2) | 0.27× |
| YOLO11n | INT8\* | 15W | 5.33 ms | 9.1 W | 53.9 mJ (22.0) | 0.77× |
| YOLO11n | INT8\* | 25W | 3.54 ms | 13.8 W | 54.6 mJ (33.0) | 1.16× |
| YOLO11n | INT8\* | MAXN_SUPER | 3.26 ms | 15.2 W | 55.4 mJ (35.0) | 1.26× |
| YOLO11n | FP16 | 7W | 14.27 ms | 7.5 W | 115.0 mJ (33.5) | 0.29× |
| YOLO11n | FP16 | 15W | 6.63 ms | 10.1 W | 72.9 mJ (33.6) | 0.62× |
| YOLO11n | FP16 | 25W | 4.42 ms | 15.7 W | 75.9 mJ (49.6) | 0.93× |
| YOLO11n | FP16 | MAXN_SUPER | 4.05 ms | 17.4 W | 77.3 mJ (52.9) | 1.01× |
| MobileNetV3-L | INT8\* | 7W | 3.07 ms | 6.5 W | 30.8 mJ (4.6) | 0.38× |
| MobileNetV3-L | INT8\* | 15W | 1.69 ms | 7.7 W | 14.1 mJ (4.2) | 0.69× |
| MobileNetV3-L | INT8\* | 25W | 1.25 ms | 9.5 W | 12.9 mJ (5.4) | 0.93× |
| MobileNetV3-L | INT8\* | MAXN_SUPER | 1.16 ms | 10.4 W | 13.2 mJ (6.1) | 1.00× |

*Runs `power-sweep` (15W → 25W → MAXN_SUPER in one session) and `power-7w` (after a reboot into 7W), dynamic clocks, 10 s per configuration, board idle 5.2–5.6 W in every mode, zero throttle events, 2026-10-04. FP32 rows and the full table are in `report/index.html`. The MAXN_SUPER rows reproduce the earlier `full-sweep` within about 1%.*

### Findings

1. **Energy per frame barely moves between 15W and MAXN_SUPER.** Across 15W, 25W and MAXN_SUPER, total energy per inference stays within ±7% for every model and precision. Precision changes it by 2–3×. In that range, the power mode is a latency knob, not an energy knob.
2. **That's because the work gets cheaper but the board's floor doesn't.** Energy above idle drops 31–37% from MAXN_SUPER to 15W (YOLO11n INT8: 35 → 22 mJ; MobileNet INT8: 6.1 → 4.2 mJ), as expected from lower voltage at lower clocks. But the ~5.4 W idle draw is paid for 45–65% longer per frame, which cancels the saving. A handheld SoC idles far lower than a dev kit, so on a real device the lower modes would come out ahead. These numbers suggest roughly a third less energy for the work itself.
3. **25W is the practical sweet spot.** It is within 6–9% of MAXN_SUPER latency at about 9% less board power. YOLO11n INT8 still runs 1.16× faster than the QCS6490 there.
4. **At 15W the Jetson falls behind the handheld NPU.** YOLO11n INT8 runs at 0.77× and MobileNet INT8 at 0.69× the QCS6490's best published latency. Matching a handheld NPU on latency takes 25W or more of board power. That's the clearest statement yet of the efficiency gap.
5. **Latency scales with GPU clock, not memory.** From MAXN_SUPER to 15W the GPU clock drops 40% (1020 → 612 MHz) and YOLO11n latency rises 61–64%, close to the 67% that clock alone predicts. MobileNet FP16/INT8 rise only 43–46%, because fixed launch overhead is a larger share of their ~1.5 ms frames. GPU load stays at 84–97% in every mode.
6. **Cooler at lower modes.** Peak Tj drops from 73 °C (MAXN_SUPER) to 66 °C (15W) and 64 °C (7W).
7. **7W is the worst mode for energy.** Energy per frame roughly doubles compared with every other mode (YOLO11n INT8: 118 mJ vs 54–55 mJ; MobileNet INT8: 31 mJ vs 13–14 mJ). With half the TPCs gated and the GPU at 408 MHz, frames take 2–3× longer than at 15W. The work energy also stops falling (YOLO11n INT8 28 mJ vs 22 mJ at 15W), so the board's ~5.3 W floor dominates. On a dev kit, 7W only makes sense as a power cap, not as a way to save energy.
8. **7W is also jittery, and INT8 loses its edge there.** p90 is 1.5–2.4× p50 for most configs (MobileNet INT8: 3.07 ms p50, 7.37 ms p90), versus near-flat distributions in the other modes. YOLO11n INT8 runs slower than FP16 (15.3 vs 14.3 ms). Likely causes are kernels chosen for 8 SMs running on 4, and launch latency from 4 CPU cores at ~800 MHz. Building engines inside 7W would tell these apart.
