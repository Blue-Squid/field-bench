# Reference board: Jetson Orin Nano Super 8 GB

Snapshot of the board behind every result in this repository, taken 2026-10-04. Values marked *(spec)* come from NVIDIA's published specs; everything else was read from the device.

## At a glance

| | |
|---|---|
| **Board** | NVIDIA Jetson **Orin Nano** Developer Kit **Super** (8 GB) |
| **Module / carrier** | P3767-0005 module on P3768-0000 carrier (`tegra234` SoC) |
| **Access** | SSH (host alias `jetson`) |
| **OS** | Ubuntu 22.04.5 LTS (Jammy), aarch64 |
| **L4T / JetPack** | L4T R36.4.7 (≈ JetPack 6.2.1) |
| **Kernel** | `5.15.148-tegra` (OOT kernel variant) |
| **Power mode** | `MAXN_SUPER` (mode 2), the highest |

> Note: this is an **Orin Nano**, not the original 2019 Jetson Nano (Maxwell, 4 GB, JetPack 4). The two have different CUDA versions, tooling and performance.

## Compute

### CPU
| | |
|---|---|
| Cores | 6× Arm Cortex-A78AE (2 clusters × 3 cores), 1 thread/core |
| Clock | 115.2 MHz – **1.728 GHz** |
| Governor | `schedutil` |
| Cache | L1d 384 KiB · L1i 384 KiB · L2 1.5 MiB · L3 4 MiB |
| ISA extras | NEON/ASIMD, dot-product (`asimddp`), fp16, AES/SHA/CRC32, atomics |

### GPU
| | |
|---|---|
| Architecture | Ampere *(spec)*, compute capability 8.7 |
| CUDA / Tensor cores | 1024 / 32 *(spec)* |
| Clock (sysfs) | 306 – **1020 MHz** (steps: 306, 408, 510, 612, 714, 816, 918, 1020) |
| Governor | `nvhost_podgov` (scales on load) |
| AI perf | ~67 TOPS sparse INT8 in Super mode *(spec)* |

### Fixed-function engines
| Engine | Max clock | Notes |
|---|---|---|
| NVDEC (video decode) | 524.8 MHz | H.264/H.265/AV1 decode |
| NVJPG ×2 (JPEG) | 499.2 MHz | hardware JPEG encode/decode |
| VIC (video image compositor) | 435.2 MHz | scale / color-convert / composite |
| OFA (optical flow) | 537.6 MHz | optical-flow accelerator |
| **NVENC** | — | **None.** Orin Nano has no hardware video encoder *(spec)*; encoding runs on the CPU |
| **DLA** | — | **None** on Orin Nano *(spec)*; the `cv0/1/2` thermal zones are present but report n/a |

## Memory and storage

| | |
|---|---|
| RAM | 8 GB LPDDR5, shared CPU+GPU (7.4 GiB visible to the OS) |
| Bandwidth | 102 GB/s in Super mode *(spec)* |
| Swap | 12 GiB total: 8 GiB file at `/mnt/nvme/swapfile` + 6 × 635 MiB zram |

| Device | Size | Mount | Notes |
|---|---|---|---|
| `mmcblk0` (microSD) | 238.8 GB | `/` (ext4), 15% used | **Root filesystem boots from SD** |
| `nvme0n1` Samsung 990 PRO | 1 TB | `/mnt/nvme` (ext4), 4% used | Holds Docker data-root and swapfile |

> The OS runs from the microSD card while a fast Gen4 NVMe sits mostly empty. Moving the rootfs to NVMe would speed up package installs, Python imports and model loading.

## Power and thermal

**Power modes** (`nvpmodel`):

| ID | Name |
|---|---|
| 0 | 15W |
| 1 | 25W |
| 2 | **MAXN_SUPER** ← current |
| 3 | 7W |

`jetson_clocks` is installed at `/usr/bin/jetson_clocks` (pins clocks to max).

**Idle readings** (from `tegrastats`, 20 min uptime, ~1% CPU load):

| Rail (INA3221) | Power |
|---|---|
| `VDD_IN` (total board input) | ~5.2 W |
| `VDD_CPU_GPU_CV` | ~0.5 W |
| `VDD_SOC` | ~1.6 W |

| Sensor | Temp |
|---|---|
| CPU | 60 °C |
| GPU | 60 °C |
| SoC0 / SoC1 / SoC2 | 61 / 60 / 60 °C |
| Tj (junction) | 61 °C |

**Fan:** PWM fan at duty 109/255, ~2580 RPM. Managed by `nvfancontrol`: closed-loop, profile `quiet` (a `cool` profile is also available).

## Software stack

| Component | Version |
|---|---|
| CUDA toolkit | 12.6 (`nvcc` V12.6.68) at `/usr/local/cuda` |
| cuDNN | 9.3.0 |
| TensorRT | 10.3.0 (Python bindings installed) |
| VPI | 3.2.4 (Python 3.10 bindings installed) |
| OpenCV | 4.8.0 (NVIDIA build), **built without CUDA** (`cv2.cuda` reports 0 devices) |
| Python | 3.10.12 |
| Notable pip packages | `jetson-stats` 4.3.2 (`jtop`), `Jetson.GPIO` 2.1.7, `numpy` 1.21.5, `tensorrt` 10.3.0 |
| Docker | 29.8.2, `nvidia` runtime configured, data-root `/mnt/nvme/docker` |
| NVIDIA Container Toolkit | 1.16.2 |

**Not installed:** PyTorch, TensorFlow, ONNX Runtime, DeepStream. The `nvidia-jetpack` meta-package isn't installed either, though its components above are present individually.

**Monitoring tools present:** `tegrastats`, `jtop` (the `jtop.service` is running).

## Connectivity and I/O

| Interface | State | Details |
|---|---|---|
| `wlP1p1s0` Wi-Fi | **UP** | Realtek RTL8822CE (802.11ac), used for all runs |
| `enP8p1s0` Ethernet | down | Realtek RTL8111 Gigabit |
| `can0` | down | CAN bus available on the carrier |
| `usb0` / `usb1` / `l4tbr0` | down | USB device-mode networking |
| Bluetooth | present | IMC Networks radio (USB) |

- **USB:** Realtek 4-port USB 3.0 + USB 2.0 hubs.
- **Cameras:** none attached (no `/dev/video*`). `nvargus-daemon` is running and ready for CSI cameras.
- **GPIO / I²C:** `/dev/gpiochip0-1`, `/dev/i2c-{0,1,2,4,5,7,9}`. A user in the `gpio`, `i2c`, `video`, `render` and `jtop` groups needs no sudo for these.
- **Serial consoles:** `ttyTCU0` (debug UART, 115200) and `ttyGS0` (USB gadget serial).

## Running services worth knowing

`docker`, `containerd`, `jtop`, `nvfancontrol`, `nvargus-daemon` (CSI camera), `nvphs` (power hinting), `nvidia-pva-allowd`, `gdm` (boots to `graphical.target`, so a desktop is running), `ssh`, `bluetooth`, `avahi-daemon`.

## Observations and opportunities

1. **Boots from microSD.** Migrating the rootfs to the 990 PRO is the biggest general speedup available.
2. **No NVENC.** Video pipelines that need to *encode* (recording, RTSP out) will load the CPU. Decoding is hardware-accelerated.
3. **OpenCV lacks CUDA.** Use VPI, TensorRT or a CUDA-enabled OpenCV build for GPU image processing.
4. **The desktop (GDM) is running** on a headless SSH-only box. Switching to `multi-user.target` frees roughly 0.5–1 GB of shared RAM/VRAM.
5. **Rich telemetry is exposed** (INA3221 power rails, 9 thermal zones, fan tach, devfreq clocks for every engine, `jtop` socket). That's a good foundation for a monitoring or control utility.

## Handy commands

```bash
ssh jetson 'sudo nvpmodel -q'                          # current power mode
ssh jetson 'sudo nvpmodel -m 0'                        # switch to 15W
ssh jetson 'tegrastats --interval 1000'                # live stats stream
ssh jetson 'cat /sys/class/hwmon/hwmon1/in1_input'     # VDD_IN voltage (mV)
ssh jetson 'cat /sys/class/hwmon/hwmon1/curr1_input'   # VDD_IN current (mA)
ssh jetson 'cat /sys/class/thermal/thermal_zone*/temp' # temps (m°C)
ssh jetson 'cat /sys/class/devfreq/17000000.gpu/cur_freq'  # GPU clock (Hz)
ssh -t jetson jtop                                     # interactive TUI
```
