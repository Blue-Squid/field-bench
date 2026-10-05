# Troubleshooting

Known failure modes and their fixes. Most of them were found the hard way while building the reference results.

## Setup

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tensorrt` on the Jetson | The venv was created without system packages | Recreate it with `python3 -m venv --system-site-packages`. TensorRT's bindings come from JetPack. |
| `.venv/bin/pip: bad interpreter` | The venv folder was moved or renamed after creation (stale shebang) | Use `.venv/bin/python -m pip …`, or recreate the venv. |
| `emc_mhz` is null, power-mode switching fails | jtop service not running, or your user is not in the `jtop` group | `sudo systemctl enable --now jtop`, `sudo usermod -aG jtop $USER`, then log in again. |
| `p_VDD_IN` is null | Board's INA3221 rails have other labels | See [adapting, section 2](adapting.md#2-another-jetson) and map the total-input rail to `VDD_IN`. |
| NVJPG shim fails to compile | Missing `g++` or Multimedia API headers | `sudo apt install g++ nvidia-l4t-jetson-multimedia-api`. The shim is cached in `~/.cache/fieldbench/libfbnvjpg-<hash>.so`; delete it to force a rebuild. |

## Engines and models

| Symptom | Cause | Fix |
|---|---|---|
| A 10–45 min build appears unexpectedly | The ONNX file changed (re-export), so its hash and engine name changed | Expected. Avoid needless re-exports. Engines may be renamed if `.engine` and `.calib` move together. |
| `NvMap` / `ENOMEM` errors during INT8 builds | TensorRT workspace plus calibration buffers exceed the shared 8 GB | Calibrated builds cap the workspace at 1 GB already. Also close the desktop session and other GPU users. |
| "Using an engine plan file across different models of devices is not recommended" | The process started in a power mode with a different GPU clock cap than the build | Benign. See [methodology, section 7](methodology.md#7-comparing-with-handheld-figures). |
| OCR output is plausible but wrong (missing spaces, wrong case) | A graph simplifier corrupted the recognizer during static-shape export (onnxslim 0.1.97's `FusionGemm` rewrite did) | `host/export_models.py` skips that fusion and refuses to save when ONNX and source differ by more than 1e-3. Keep that check for any new model. |
| INT8 pipeline is *slower* and less accurate than FP16 | Calibration crushed the detector's activations: a fragmented text map produces 2× as many boxes for the recognizer | Use entropy calibration for the OCR detector, or keep it FP16 (the default). Always compare against FP16 accuracy on the test set. |

## Measurements

| Symptom | Cause | Fix |
|---|---|---|
| Same configuration is 10–25% slower or faster than yesterday | Default DVFS governors respond to load pattern and temperature | Compare only within one session, or lock clocks (`sudo jetson_clocks`) for both sides. See [methodology, section 5](methodology.md#5-clocks-and-dvfs). |
| Rows on either side of a moment are incomparable | Clocks were locked or restored during a sweep | Only run `jetson_clocks` between runs. Discard a row whose timed window spans the change. |
| Barcode detector at 640 px takes 12 ms on GPU instead of 4–5 ms | The GPU governor sits at its 306 MHz minimum, because the GPU is idle most of each frame | Expected under default governors. Use locked clocks or `--workers 2` to keep the GPU busy. |
| A multi-worker run stops making progress: no new frames, GPU load near 100%, the process barely uses CPU | The unresolved two-worker GPU hang: a worker blocks in `cudaStreamSynchronize` (kernel wait `dma_fence_default_wait`) | Keep the watchdog on (`--stall-s`, default 120 s); it writes a partial row with diagnostics and exits. Killing the process frees the GPU, with no reboot needed. Use one worker for long runs. See [Phase 3, section 3](results-phase3.md#3-sustained-load-thermal-soak). |
| Stage p50s don't add up to the total p50 | Percentiles don't add | Use stage means (the report's bars do). |
| A UPC-A barcode "misreads" as EAN-13 | zxing-cpp returns UPC-A as EAN-13 with a leading 0 | Ground truth is taken from zxing's own read of the clean render, so scoring is consistent. |
| GPU preprocessing differs from the CPU path | OpenCV on aarch64 resizes with NEON (carotene), whose rounding differs from its scalar code | The kernel reproduces the SIMD formula. Run `python -m fieldbench.gpuprep` after any OpenCV or JetPack upgrade. At 2560 px (upscaling) two edge rows differ by one level, which is expected. |

## Operations

| Symptom | Cause | Fix |
|---|---|---|
| A file you created on the board disappeared | `make sync` runs `rsync --delete` on `fieldbench/` | Write code on the host. Keep board-only files outside `~/fieldbench/fieldbench/`. |
| `pkill -f pattern` killed your own SSH command | The pattern appears in the shell's own command line | `pgrep -f pattern` first, check the PIDs, then `kill`. |
| `make live` says the session already exists | A previous run is still open (or finished and waiting for Enter) | `make attach` to check it. Close it with Enter, or `ssh jetson tmux kill-session -t fieldbench`. |
| Out of memory with a desktop session running | GDM and the compositor use 0.5–1 GB of the shared memory | `sudo systemctl isolate multi-user.target` (back with `graphical.target`). |
