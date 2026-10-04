"""fieldbench: handheld-class computer-vision benchmarks on Jetson.

  python -m fieldbench info
  python -m fieldbench bench --models yolo11n mobilenetv3l --precisions fp16 int8
  python -m fieldbench bench --power-modes 15W 25W MAXN_SUPER
  python -m fieldbench power                 # list modes; --set 7W --reboot to change
"""
import argparse
import datetime as dt
import json
import socket
import sys
import time
from pathlib import Path

from . import power
from .catalog import MODELS
from .telemetry import Sensors, device_info

PRECISIONS = ["fp32", "fp16", "int8"]


def cmd_info(args):
    sensors = Sensors()
    print(json.dumps({"device": device_info(), "sample": sensors.read(),
                      "rails": list(sensors.rails), "thermal_zones": list(sensors.thermal)}, indent=2))


def _best_ref(model):
    refs = MODELS[model]["references"]
    return min(refs, key=lambda r: r["latency_ms"]) if refs else None


def _print_row(r):
    g, e, en = r["gpu_ms"], r["e2e_ms"], r["energy"]
    ref = _best_ref(r["model"])
    vs = f"{ref['latency_ms'] / g['p50']:5.2f}x" if ref else "   - "
    temp = r["telemetry_load"].get("temp_tj_max")
    print(f"{r['device']['power_mode'] or '?':<12}{r['model']:<14}{r['precision']:<6}{g['p50']:>8.2f}{g['p99']:>8.2f}{e['p50']:>8.2f}"
          f"{r['fps']:>8.0f}{r['telemetry_load'].get('p_VDD_IN_mean', 0) / 1000:>7.2f}"
          f"{en.get('mj_per_inf_total', 0):>8.1f}{temp or 0:>6.1f}{vs:>9}")


def _bench_mode(args, bench, sensors, f, rows):
    dev = device_info()
    print(f"{dev['model']} | power mode {dev['power_mode']} | GPU max {dev['gpu_max_mhz']:.0f} MHz"
          f" | CPUs {dev['cpus_online']} @ {dev['cpu_max_mhz']:.0f} MHz | EMC cap {dev['emc_cap_mhz']:.0f} MHz"
          f" | clocks locked: {dev['clocks_locked']}")
    for model in args.models:
        for precision in args.precisions:
            print(f"> {model} [{precision}] @ {dev['power_mode']}")
            try:
                r = bench.run(model, precision, sensors, warmup_s=args.warmup,
                              duration_s=args.duration, min_iters=args.min_iters)
            except Exception as exc:  # keep sweeping; one bad config shouldn't end the run
                print(f"  FAILED: {exc}", file=sys.stderr)
                continue
            r.update(timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                     host=socket.gethostname(), label=args.label, device=dev)
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
            if args.cooldown:
                time.sleep(args.cooldown)


def _power_plan(requested, original):
    """Power modes to sweep, lowest budget first; exits on unknown or reboot-only modes."""
    if not requested:
        return [original]
    available = {m["name"]: m for m in power.modes()}
    unknown = [m for m in requested if m not in available]
    blocked = [m for m in requested if m in available and not available[m]["live"]]
    if unknown:
        sys.exit(f"unknown power mode(s): {', '.join(unknown)}; available: {', '.join(available)}")
    if blocked:
        sys.exit(f"{', '.join(blocked)} need a reboot to switch to. Run `python -m fieldbench power --set "
                 f"{blocked[0]} --reboot`, then run without --power-modes (or with that mode only).")
    return sorted(requested, key=power.budget_rank)


def cmd_bench(args):
    from . import bench  # imports TensorRT/CUDA; keep `info` light

    sensors = Sensors()
    out = Path(args.out or f"results/{dt.datetime.now():%Y%m%d-%H%M%S}"
               f"{'-' + args.label if args.label else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    original = power.current()
    plan = _power_plan(args.power_modes, original)
    print(f"writing {out}\n")

    rows = []
    try:
        with out.open("a") as f:
            for mode in plan:
                if mode != power.current():
                    power.set_mode(mode, settle_s=args.settle)
                _bench_mode(args, bench, sensors, f, rows)
                print()
    finally:
        if args.power_modes and power.current() != original:
            print(f"restoring power mode {original}")
            power.set_mode(original, settle_s=0)

    print(f"\n{'mode':<12}{'model':<14}{'prec':<6}{'gpu p50':>8}{'gpu p99':>8}{'e2e p50':>8}"
          f"{'fps':>8}{'W':>7}{'mJ/inf':>8}{'Tj°C':>6}{'vs 6490':>9}")
    for r in rows:
        _print_row(r)
    print("\nvs 6490 = best published QCS6490 NPU latency / Jetson GPU p50 (>1 means Jetson is faster)")
    if any(r["precision"] == "int8" for r in rows):
        print("int8 engines are uncalibrated: latency is representative, accuracy is not.")


def cmd_power(args):
    if args.set:
        try:
            dev = power.set_mode(args.set, reboot=args.reboot, settle_s=0)
        except power.RebootRequired as exc:
            sys.exit(f"{exc}. Re-run with --reboot to switch and reboot now.")
        if dev:
            print(f"power mode now {dev['power_mode']}")
        return
    print(f"{'mode':<12}{'id':>3}  {'switch':<14}")
    for m in power.modes():
        print(f"{m['name']:<12}{m['id']:>3}  {'live' if m['live'] else 'needs reboot':<14}{'  <- current' if m['current'] else ''}")


def main():
    ap = argparse.ArgumentParser(prog="fieldbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="device facts and one telemetry sample").set_defaults(fn=cmd_info)

    b = sub.add_parser("bench", help="benchmark models across precisions")
    b.add_argument("--models", nargs="+", default=["yolo11n", "mobilenetv3l"], choices=list(MODELS))
    b.add_argument("--precisions", nargs="+", default=PRECISIONS, choices=PRECISIONS)
    b.add_argument("--duration", type=float, default=10.0, help="timed seconds per config")
    b.add_argument("--warmup", type=float, default=3.0, help="warmup seconds per config")
    b.add_argument("--min-iters", type=int, default=200)
    b.add_argument("--cooldown", type=float, default=5.0, help="seconds to rest between configs")
    b.add_argument("--power-modes", nargs="+", metavar="MODE",
                   help="nvpmodel modes to sweep (e.g. 15W 25W MAXN_SUPER); default: current mode only")
    b.add_argument("--settle", type=float, default=10.0, help="seconds to wait after a mode switch")
    b.add_argument("--label", help="tag stored with each result row")
    b.add_argument("--out", help="results JSONL path (appends)")
    b.set_defaults(fn=cmd_bench)

    p = sub.add_parser("power", help="list or set nvpmodel power modes (via jtop, no sudo)")
    p.add_argument("--set", metavar="MODE", help="switch to this mode")
    p.add_argument("--reboot", action="store_true", help="allow a reboot if the mode needs one")
    p.set_defaults(fn=cmd_power)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
