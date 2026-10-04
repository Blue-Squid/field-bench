# fieldbench documentation

Start with the [main README](../README.md) for installation and the workflow. These documents go deeper.

| Document | Read it when you want to… |
|---|---|
| [Command reference](cli.md) | look up a `make` target or CLI flag, or the engine file naming. |
| [Methodology](methodology.md) | know exactly what a number measures: protocol, timing, power and energy, DVFS, accuracy scoring, and caveats when comparing with handhelds. |
| [Adapting fieldbench](adapting.md) | run on another Jetson, add your handheld's reference numbers, measure on a handheld, or bring your own models and data. |
| [Phase 1 results](results-phase1.md) | see the model benchmarks: precision and power-mode sweeps against the QCS6490 NPU. |
| [Phase 2 results](results-phase2.md) | see the pipeline findings: accuracy, stage breakdowns, INT8, NVJPG, GPU preprocessing, DVFS, stage overlap. |
| [Phase 3 results](results-phase3.md) | see several models per frame, product recognition, and how the board holds up under sustained load. |
| [INT8 quantization: the theory](int8-quantization-theory.md) | understand what INT8 does to a network, how scales are chosen, and TensorRT's calibrators. |
| [INT8 calibration in fieldbench](int8-calibration-in-fieldbench.md) | rebuild or change the calibrated engines, and see which calibrator worked for which model. |
| [Troubleshooting](troubleshooting.md) | fix a setup, engine, measurement or operations problem. |
| [Reference board](device-orin-nano-super.md) | check the hardware and software of the board behind the published results. |
