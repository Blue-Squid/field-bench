"""Run an end-to-end pipeline over real image files with per-stage timing and telemetry.

A pipeline object exposes:
  name, config (dict), stages (ordered stage names)
  process(jpeg_bytes, times) -> output     fills times[stage] = ms for each stage (extra keys,
                                           e.g. infer_gpu, are summarized too)
  score(frames, outputs) -> accuracy dict  one output per frame, in order
  close()

Protocol per configuration: one untimed accuracy pass over every frame, idle baseline,
warmup, then a timed loop that cycles through the frames (JPEG bytes held in RAM, so
storage speed is not measured; JPEG decode is).

workers > 1 overlaps the stages across frames: N independent copies of the pipeline
(pipe.clone(): own TensorRT context, CUDA stream, buffers and JPEG decoder), each a thread
that takes the next frame as soon as it finishes the last one (closed loop). While one
worker decodes or runs zxing on the CPU, another's network runs on the GPU. total_ms is then
the per-frame latency under that load and fps the aggregate throughput. The accuracy pass
also runs through the workers, so a concurrency bug shows up as an accuracy change.
"""
import itertools
import threading
import time

from .stats import latency_stats
from .telemetry import Sampler, summarize


def _pool(pipes, jpegs, keep_going, on_frame):
    """Run frames through len(pipes) worker threads. Frame i goes to whichever worker is free;
    on_frame(i, times, ms, t_end, output) is called after each one. Re-raises the first worker error."""
    counter, lock, errors = itertools.count(), threading.Lock(), []

    def work(p):
        try:
            while True:
                with lock:
                    i = next(counter)
                if not keep_going(i):
                    return
                times = {}
                h0 = time.perf_counter()
                out = p.process(jpegs[i % len(jpegs)], times)
                h1 = time.perf_counter()
                on_frame(i, times, (h1 - h0) * 1e3, time.monotonic(), out)
        except BaseException as exc:  # noqa: BLE001  surfaced in the caller
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(p,), daemon=True) for p in pipes[1:]]
    for t in threads:
        t.start()
    work(pipes[0])
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def _series(ends, lat, samples, t0, t1, step):
    """Per-window throughput, latency and telemetry over the timed loop (for soak runs)."""
    out = []
    k = 0
    while t0 + k * step < t1:
        a, b = t0 + k * step, min(t0 + (k + 1) * step, t1)
        win = [ms for e, ms in zip(ends, lat) if a <= e < b]
        tel = summarize([s for s in samples if a <= s["t"] < b])
        out.append({"t_s": round(a - t0, 1), "fps": len(win) / (b - a) if b > a else 0.0,
                    "p50_ms": latency_stats(win)["p50"] if win else None,
                    **{key: tel.get(key) for key in ("p_VDD_IN_mean", "temp_tj_max", "temp_gpu_max",
                                                     "temp_cpu_max", "gpu_mhz_mean", "gpu_mhz_min",
                                                     "cpu_mhz_mean", "emc_mhz_mean", "gpu_load_mean")}})
        k += 1
    return out


def run(pipe, frames, sensors, *, warmup_s=3.0, duration_s=10.0, min_frames=100, idle_s=2.0, workers=1,
        series_s=0, log=print):
    jpegs = [f["jpeg"] for f in frames]
    pipes = [pipe] + [pipe.clone() for _ in range(workers - 1)]
    try:
        return _run(pipes, frames, jpegs, sensors, warmup_s, duration_s, min_frames, idle_s, series_s, log)
    finally:
        for p in pipes[1:]:
            p.close()


def _run(pipes, frames, jpegs, sensors, warmup_s, duration_s, min_frames, idle_s, series_s, log):
    pipe, workers = pipes[0], len(pipes)
    t_acc = time.monotonic()
    if workers == 1:
        outputs = [pipe.process(j, {}) for j in jpegs]
    else:
        outputs = [None] * len(jpegs)
        _pool(pipes, jpegs, lambda i: i < len(jpegs),
              lambda i, times, ms, t, out: outputs.__setitem__(i, out))
    accuracy = pipe.score(frames, outputs)
    accuracy_pass_s = time.monotonic() - t_acc
    log("  accuracy: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in accuracy.items() if not isinstance(v, dict)))

    throttle_before = sensors.throttle_events()
    per_stage = {}  # every key a pipeline records (stages, plus extras such as infer_gpu)
    total, ends = [], []
    lock = threading.Lock()

    def record(i, times, ms, t_end, out):
        with lock:
            total.append(ms)
            ends.append(t_end)
            for s in {*pipe.stages, *times}:
                per_stage.setdefault(s, []).append(times.get(s, 0.0))

    with Sampler(sensors) as sampler:
        t_idle = time.monotonic()
        time.sleep(idle_s)
        t_warm = time.monotonic()
        if workers == 1:
            i = 0
            while time.monotonic() - t_warm < warmup_s:
                pipe.process(jpegs[i % len(jpegs)], {})
                i += 1
            t0 = time.monotonic()
            i = 0
            while time.monotonic() - t0 < duration_s or i < min_frames:
                times = {}
                h0 = time.perf_counter()
                pipe.process(jpegs[i % len(jpegs)], times)
                record(i, times, (time.perf_counter() - h0) * 1e3, time.monotonic(), None)
                i += 1
        else:
            _pool(pipes, jpegs, lambda i: time.monotonic() - t_warm < warmup_s, lambda *a: None)
            t0 = time.monotonic()
            # Workers stop taking frames once both limits are met; frames in flight still finish.
            _pool(pipes, jpegs, lambda i: time.monotonic() - t0 < duration_s or i < min_frames, record)
        t1 = time.monotonic()

    idle = summarize(sampler.window(t_idle, t_warm))
    load = summarize(sampler.window(t0, t1))
    n, wall_s = len(total), t1 - t0
    energy = {}
    p_in, p_idle = load.get("p_VDD_IN_mean"), idle.get("p_VDD_IN_mean")
    if p_in is not None:
        energy["mj_per_frame_total"] = p_in * wall_s / n
        if p_idle is not None:
            energy["mj_per_frame_dynamic"] = (p_in - p_idle) * wall_s / n

    row = {
        "kind": "pipeline",
        "pipeline": pipe.name,
        **pipe.config,
        "workers": workers,
        "stages": pipe.stages,
        "frames_in_set": len(frames),
        "frames_timed": n,
        "wall_s": wall_s,
        "fps": n / wall_s,
        "total_ms": latency_stats(total),
        "stage_ms": {s: latency_stats(v) for s, v in per_stage.items()},
        "accuracy": accuracy,
        "accuracy_pass_s": accuracy_pass_s,
        "energy": energy,
        "telemetry_idle": idle,
        "telemetry_load": load,
        "throttle_events": sensors.throttle_events() - throttle_before,
    }
    if series_s:
        row["series_s"] = series_s
        row["series"] = _series(ends, total, sampler.window(t0, t1), t0, t1, series_s)
    return row


class Clock:
    """Stage stopwatch: `with clock("decode"): ...` adds elapsed ms to times["decode"]."""

    def __init__(self, times):
        self.times = times

    def __call__(self, stage):
        self.stage = stage
        return self

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        self.times[self.stage] = self.times.get(self.stage, 0.0) + (time.perf_counter() - self.t) * 1e3
