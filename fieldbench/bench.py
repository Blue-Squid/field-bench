"""Run one benchmark configuration: idle baseline, warmup, timed loop, telemetry summary."""
import os
import time

from . import engine as trt_engine
from .catalog import MODELS
from .runner import TrtRunner
from .stats import latency_stats
from .telemetry import Sampler, summarize


def run(model, precision, sensors, *, warmup_s=3.0, duration_s=10.0, min_iters=200,
        idle_s=2.0, engine_dir="engines", log=print):
    spec = MODELS[model]
    engine_file, build_s = trt_engine.build(spec["onnx"], precision, engine_dir)
    log(f"  engine {'built in %.0fs' % build_s if build_s else 'cached'}: {engine_file.name}")

    runner = TrtRunner(engine_file)
    runner.fill_random()
    throttle_before = sensors.throttle_events()
    try:
        with Sampler(sensors) as sampler:
            t_idle = time.monotonic()
            time.sleep(idle_s)
            t_warm = time.monotonic()
            while time.monotonic() - t_warm < warmup_s:
                runner.infer()

            gpu, e2e = [], []
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration_s or len(gpu) < min_iters:
                g, e = runner.infer()
                gpu.append(g)
                e2e.append(e)
            t1 = time.monotonic()

        idle = summarize(sampler.window(t_idle, t_warm))
        load = summarize(sampler.window(t0, t1))
    finally:
        device_mem = runner.device_memory_bytes
        runner.close()

    n, wall_s = len(gpu), t1 - t0
    p_in = load.get("p_VDD_IN_mean")
    p_idle = idle.get("p_VDD_IN_mean")
    energy = {}
    if p_in is not None:
        # mW * s = mJ, spread over the inferences in the window.
        energy["mj_per_inf_total"] = p_in * wall_s / n
        if p_idle is not None:
            energy["mj_per_inf_dynamic"] = (p_in - p_idle) * wall_s / n
        energy["inf_per_joule"] = n / (p_in * wall_s / 1000)

    return {
        "model": model,
        "precision": precision,
        "int8_calibrated": False if precision == "int8" else None,
        "task": spec["task"],
        "input": spec["input"],
        "engine": engine_file.name,
        "engine_mb": os.path.getsize(engine_file) / 1e6,
        "engine_build_s": build_s,
        "device_memory_mb": device_mem / 1e6,
        "iterations": n,
        "wall_s": wall_s,
        "fps": n / wall_s,
        "gpu_ms": latency_stats(gpu),
        "e2e_ms": latency_stats(e2e),
        "energy": energy,
        "telemetry_idle": idle,
        "telemetry_load": load,
        "throttle_events": sensors.throttle_events() - throttle_before,
    }
