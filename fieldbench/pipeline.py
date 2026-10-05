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

stall_s > 0 arms a watchdog: if any worker goes that long without finishing a frame (the
others may still be running), it writes a partial row
(kind "pipeline-stalled": the series so far plus every thread's Python stack and kernel wait
channel) through on_stall and ends the process, since a thread stuck inside a CUDA call
can't be unwound.
"""
import contextlib
import itertools
import os
import subprocess
import sys
import threading
import time
import traceback

from .stats import latency_stats
from .telemetry import Sampler, summarize


def _pool(pipes, jpegs, keep_going, on_frame, watch=None):
    """Run frames through len(pipes) worker threads. Frame i goes to whichever worker is free;
    on_frame(i, times, ms, t_end, output) is called after each one. Re-raises the first worker error.
    watch (a Watchdog) hears from each worker after every frame and when it leaves."""
    counter, lock, errors = itertools.count(), threading.Lock(), []

    def work(p):
        try:
            if watch:
                watch.beat()
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
        finally:
            if watch:
                watch.retire()

    threads = [threading.Thread(target=work, args=(p,), daemon=True) for p in pipes[1:]]
    for t in threads:
        t.start()
    work(pipes[0])
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


class Watchdog:
    """Calls on_stall(stalled_s) from its own thread when any thread that beat() hasn't beaten again
    for stall_s. Per thread, so one stuck worker is caught while the others keep finishing frames."""

    def __init__(self, stall_s, on_stall, poll_s=5.0):
        self.stall_s, self.on_stall, self.poll_s = stall_s, on_stall, poll_s
        self.last = {threading.get_ident(): time.monotonic()}  # thread -> last beat
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="watchdog")

    def beat(self):
        self.last[threading.get_ident()] = time.monotonic()

    def retire(self):
        """The calling thread is done (a pool worker leaving); stop watching it."""
        self.last.pop(threading.get_ident(), None)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            idle = max((time.monotonic() - t for t in list(self.last.values())), default=0.0)
            if idle >= self.stall_s:
                self.on_stall(idle)
                return


def _proc(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as exc:
        return f"<{exc.__class__.__name__}>"


def _irq_counts():
    """host1x/GPU interrupt lines of /proc/interrupts and the softirq totals, as text lines."""
    lines = [l.split(None, 1)[1] if ":" in l.split(None, 1)[0] else l
             for l in _proc("/proc/interrupts").splitlines() if any(k in l for k in ("host1x", "gk20a", "nvgpu"))]
    return lines + [l for l in _proc("/proc/softirqs").splitlines() if l.strip().startswith(("HI:", "TASKLET:"))]


def stall_diagnostics():
    """Where every thread is: Python stack (for Python threads) and kernel state + wait channel
    for every task in the process (CUDA's own threads included), plus whether the host1x and GPU
    interrupts still advance (two snapshots 5 s apart; a reported JetPack 6.2.1 hang freezes the
    host1x interrupt counts on CPU0). No root needed."""
    irq_before = _irq_counts()
    time.sleep(5)
    irq_after = _irq_counts()
    names = {t.ident: t for t in threading.enumerate()}
    py = {}
    for ident, frame in sys._current_frames().items():
        t = names.get(ident)
        key = f"{t.name if t else ident} (tid {t.native_id if t else '?'})"
        py[key] = traceback.format_stack(frame)[-8:]
    tasks = []
    for tid in sorted(os.listdir("/proc/self/task"), key=int):
        stat = _proc(f"/proc/self/task/{tid}/stat")
        state = stat.rsplit(")", 1)[-1].split()[0] if ")" in stat else "?"
        tasks.append({"tid": int(tid), "comm": _proc(f"/proc/self/task/{tid}/comm"), "state": state,
                      "wchan": _proc(f"/proc/self/task/{tid}/wchan")})
    try:
        dmesg = subprocess.run(["dmesg", "--ctime"], capture_output=True, text=True, timeout=5)
        kernel_log = (dmesg.stdout.splitlines()[-40:] if dmesg.returncode == 0 else [dmesg.stderr.strip()])
    except (OSError, subprocess.TimeoutExpired) as exc:
        kernel_log = [repr(exc)]
    return {"python_stacks": py, "tasks": tasks, "kernel_log": kernel_log,
            "irq_before": irq_before, "irq_after_5s": irq_after}


def _series(ends, lat, samples, t0, t1, step):
    """Per-window throughput, latency and telemetry over the timed loop (for soak runs)."""
    out = []
    k = 0
    while t0 + k * step < t1 - step / 2:  # skip a trailing partial window shorter than half a step
        a, b = t0 + k * step, min(t0 + (k + 1) * step, t1)
        win = [ms for e, ms in zip(ends, lat) if a <= e < b]
        tel = summarize([s for s in samples if a <= s["t"] < b])
        oc = (tel["oc_events_max"] - tel["oc_events_min"]) if "oc_events_max" in tel else None
        out.append({"t_s": round(a - t0, 1), "frames": len(win), "fps": len(win) / (b - a) if b > a else 0.0,
                    "p50_ms": latency_stats(win)["p50"] if win else None, "oc_events": oc,
                    **{key: tel.get(key) for key in ("p_VDD_IN_mean", "temp_tj_max", "temp_gpu_max",
                                                     "temp_cpu_max", "gpu_mhz_mean", "gpu_mhz_min",
                                                     "cpu_mhz_mean", "emc_mhz_mean", "gpu_load_mean",
                                                     "mem_avail_mb_min", "rss_mb_max")}})
        k += 1
    return out


def run(pipe, frames, sensors, *, warmup_s=3.0, duration_s=10.0, min_frames=100, idle_s=2.0, workers=1,
        series_s=0, stall_s=0, on_stall=None, log=print):
    jpegs = [f["jpeg"] for f in frames]
    pipes = [pipe] + [pipe.clone() for _ in range(workers - 1)]
    try:
        return _run(pipes, frames, jpegs, sensors, warmup_s, duration_s, min_frames, idle_s, series_s,
                    stall_s, on_stall, log)
    finally:
        for p in pipes[1:]:
            p.close()


def _run(pipes, frames, jpegs, sensors, warmup_s, duration_s, min_frames, idle_s, series_s, stall_s, on_stall, log):
    pipe, workers = pipes[0], len(pipes)
    state = {"phase": "accuracy", "t0": None}
    total, ends, per_stage = [], [], {}
    lock = threading.Lock()
    throttle_before = sensors.throttle_events()
    sampler = Sampler(sensors, interval=0.2 if series_s else 0.05)  # soaks: 5 Hz keeps the sample list small

    def stalled(idle):
        """Watchdog thread: write what was measured, then end the process."""
        with lock:
            snap_total, snap_ends = list(total), list(ends)
        now = time.monotonic()
        row = _row(pipe, workers, frames, None, None, snap_total, snap_ends, {}, sampler, sensors, throttle_before,
                   state["t0"] or now, now, None, None, series_s)
        row.update(kind="pipeline-stalled", stalled_phase=state["phase"], stalled_after_s=round(idle, 1),
                   frames_done=len(snap_total), diagnostics=stall_diagnostics())
        log(f"\n  STALL: a worker has not finished a frame for {idle:.0f} s ({state['phase']}, "
            f"{len(snap_total)} timed frames so far)")
        for name, stack in row["diagnostics"]["python_stacks"].items():
            log(f"  -- {name}\n" + "".join(stack).rstrip())
        log("  tasks: " + ", ".join(f"{t['comm']}:{t['state']}:{t['wchan']}" for t in row["diagnostics"]["tasks"]))
        try:
            if on_stall:
                on_stall(row)
        finally:
            sys.stdout.flush()
            os._exit(3)

    with Watchdog(stall_s, stalled) if stall_s else contextlib.nullcontext() as dog:
        return _run_watched(pipes, frames, jpegs, sensors, warmup_s, duration_s, min_frames, idle_s, series_s, log,
                            state, total, ends, per_stage, lock, throttle_before, sampler, dog)


def _run_watched(pipes, frames, jpegs, sensors, warmup_s, duration_s, min_frames, idle_s, series_s, log,
                 state, total, ends, per_stage, lock, throttle_before, sampler, watch):
    pipe, workers = pipes[0], len(pipes)
    beat = watch.beat if watch else (lambda: None)
    t_acc = time.monotonic()
    if workers == 1:
        outputs = []
        for j in jpegs:
            outputs.append(pipe.process(j, {}))
            beat()
    else:
        outputs = [None] * len(jpegs)

        def keep(i, times, ms, t, out):
            outputs[i] = out
            beat()

        _pool(pipes, jpegs, lambda i: i < len(jpegs), keep, watch)
        beat()
    accuracy = pipe.score(frames, outputs)
    accuracy_pass_s = time.monotonic() - t_acc
    log("  accuracy: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in accuracy.items() if not isinstance(v, dict)))

    # per_stage: every key a pipeline records (stages, plus extras such as infer_gpu)

    def record(i, times, ms, t_end, out):
        with lock:
            total.append(ms)
            ends.append(t_end)
            for s in {*pipe.stages, *times}:
                per_stage.setdefault(s, []).append(times.get(s, 0.0))
        beat()

    with sampler:
        state["phase"] = "idle"
        t_idle = time.monotonic()
        time.sleep(idle_s)
        beat()
        state["phase"] = "warmup"
        t_warm = time.monotonic()
        if workers == 1:
            i = 0
            while time.monotonic() - t_warm < warmup_s:
                pipe.process(jpegs[i % len(jpegs)], {})
                beat()
                i += 1
            state["phase"], state["t0"] = "timed", time.monotonic()
            t0 = state["t0"]
            i = 0
            while time.monotonic() - t0 < duration_s or i < min_frames:
                times = {}
                h0 = time.perf_counter()
                pipe.process(jpegs[i % len(jpegs)], times)
                record(i, times, (time.perf_counter() - h0) * 1e3, time.monotonic(), None)
                i += 1
        else:
            _pool(pipes, jpegs, lambda i: time.monotonic() - t_warm < warmup_s, lambda *a: beat(), watch)
            beat()
            state["phase"], state["t0"] = "timed", time.monotonic()
            t0 = state["t0"]
            # Workers stop taking frames once both limits are met; frames in flight still finish.
            _pool(pipes, jpegs, lambda i: time.monotonic() - t0 < duration_s or i < min_frames, record, watch)
        t1 = time.monotonic()
        state["phase"] = "done"

    return _row(pipe, workers, frames, accuracy, accuracy_pass_s, total, ends, per_stage, sampler, sensors,
                throttle_before, t0, t1, t_idle, t_warm, series_s)


def _row(pipe, workers, frames, accuracy, accuracy_pass_s, total, ends, per_stage, sampler, sensors, throttle_before,
         t0, t1, t_idle, t_warm, series_s):
    idle = summarize(sampler.window(t_idle, t_warm)) if t_idle is not None else {}
    load = summarize(sampler.window(t0, t1))
    n, wall_s = len(total), t1 - t0
    energy = {}
    p_in, p_idle = load.get("p_VDD_IN_mean"), idle.get("p_VDD_IN_mean")
    if p_in is not None and n:
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
        "fps": n / wall_s if wall_s > 0 else 0.0,
        "total_ms": latency_stats(total) if total else None,
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
