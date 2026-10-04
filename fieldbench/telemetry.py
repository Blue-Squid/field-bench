"""Jetson power, thermal and clock telemetry read straight from sysfs (no sudo needed).

The memory (EMC) clock is only readable as root (debugfs), so it comes from the jtop service,
which runs as root, when jtop is installed; it updates at the service's interval (~1 s).
"""
import atexit
import glob
import os
import statistics
import subprocess
import threading
import time

GPU_DEVFREQ = "/sys/class/devfreq/17000000.gpu"
GPU_LOAD = "/sys/devices/platform/bus@0/17000000.gpu/load"  # permille
CPU_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"  # kHz
CPU_MAX_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"  # kHz
CPUS_ONLINE = "/sys/devices/system/cpu/online"  # e.g. "0-5" or "0-3"
EMC_CAP = "/sys/kernel/nvpmodel_clk_cap/emc"  # Hz, memory clock cap set by nvpmodel
TPC_PG_MASK = "/sys/devices/platform/bus@0/17000000.gpu/tpc_pg_mask"  # power-gated GPU TPCs


def _read(path, cast=int):
    try:
        with open(path) as f:
            return cast(f.read().strip().strip("\0"))
    except (OSError, ValueError, TypeError):
        return None


def _find_hwmon(name):
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _read(f"{h}/name", str) == name:
            return h
    return None


class Sensors:
    """Discovers sensor paths once; read() returns one flat sample dict."""

    def __init__(self):
        self.rails = {}  # INA3221 label -> (mV path, mA path)
        ina = _find_hwmon("ina3221")
        if ina:
            for i in range(1, 9):
                label = _read(f"{ina}/in{i}_label", str)
                if label and os.path.exists(f"{ina}/curr{i}_input"):
                    self.rails[label] = (f"{ina}/in{i}_input", f"{ina}/curr{i}_input")

        self.thermal = {}  # zone name -> temp path; zones that refuse reads (cv0..2) are skipped
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            kind = _read(f"{zone}/type", str)
            if kind and _read(f"{zone}/temp") is not None:
                self.thermal[kind.removesuffix("-thermal")] = f"{zone}/temp"

        oc = _find_hwmon("soctherm_oc")
        self.oc_counters = sorted(glob.glob(f"{oc}/oc*_event_cnt")) if oc else []

        self.jtop = None  # for the EMC clock; DVFS lowers it when the pipeline leaves memory idle
        try:
            from jtop import jtop
            self.jtop = jtop()
            self.jtop.start()
            atexit.register(self.jtop.close)
        except Exception:
            self.jtop = None

    def read(self):
        s = {"t": time.monotonic()}
        for label, (vp, cp) in self.rails.items():
            mv, ma = _read(vp), _read(cp)
            s[f"p_{label}"] = mv * ma / 1000 if mv is not None and ma is not None else None  # mW
        for name, path in self.thermal.items():
            v = _read(path)
            s[f"temp_{name}"] = v / 1000 if v is not None else None
        hz = _read(f"{GPU_DEVFREQ}/cur_freq")
        s["gpu_mhz"] = hz / 1e6 if hz else None
        load = _read(GPU_LOAD)
        s["gpu_load"] = load / 10 if load is not None else None  # percent
        khz = _read(CPU_FREQ)
        s["cpu_mhz"] = khz / 1e3 if khz else None
        try:
            s["emc_mhz"] = self.jtop.memory["EMC"]["cur"] / 1e3  # kHz
        except Exception:
            s["emc_mhz"] = None
        return s

    def throttle_events(self):
        """Cumulative SoC over-current throttle events (hardware OC alarms)."""
        return sum(_read(p) or 0 for p in self.oc_counters)


class Sampler:
    """Background thread that samples Sensors at a fixed interval while active."""

    def __init__(self, sensors, interval=0.05):
        self.sensors = sensors
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self.sensors.read())
            self._stop.wait(self.interval)

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s["t"] <= t1]


def summarize(samples):
    """Mean and max of every numeric field across a list of samples."""
    out = {"n_samples": len(samples)}
    if not samples:
        return out
    for key in samples[0]:
        if key == "t":
            continue
        vals = [s[key] for s in samples if s.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = round(statistics.fmean(vals), 3)
            out[f"{key}_max"] = round(max(vals), 3)
            out[f"{key}_min"] = round(min(vals), 3)
    return out


def device_info():
    """Static facts about the board and its current power configuration."""
    info = {
        "model": _read("/proc/device-tree/model", str),
        "l4t": (_read("/etc/nv_tegra_release", str) or "").split(",")[:2],
        "gpu_min_mhz": (_read(f"{GPU_DEVFREQ}/min_freq") or 0) / 1e6,
        "gpu_max_mhz": (_read(f"{GPU_DEVFREQ}/max_freq") or 0) / 1e6,
        "gpu_tpc_pg_mask": _read(TPC_PG_MASK),
        "cpu_max_mhz": (_read(CPU_MAX_FREQ) or 0) / 1e3,
        "cpus_online": _read(CPUS_ONLINE, str),
        "emc_cap_mhz": (_read(EMC_CAP) or 0) / 1e6,
    }
    info["l4t"] = " ".join(p.strip("# ") for p in info["l4t"])
    # Locked clocks (jetson_clocks) show up as min == max on the GPU devfreq node.
    info["clocks_locked"] = info["gpu_min_mhz"] == info["gpu_max_mhz"]
    try:
        q = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
        info["power_mode"] = q.splitlines()[0].split(":", 1)[1].strip()
    except (OSError, IndexError, subprocess.TimeoutExpired):
        info["power_mode"] = None
    return info
