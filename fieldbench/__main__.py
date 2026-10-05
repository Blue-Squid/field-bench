"""fieldbench: handheld-class computer-vision benchmarks on Jetson.

  python -m fieldbench info
  python -m fieldbench bench --models yolo11n mobilenetv3l --precisions fp16 int8
  python -m fieldbench bench --power-modes 15W 25W MAXN_SUPER
  python -m fieldbench power                 # list modes; --set 7W --reboot to change
  python -m fieldbench pipeline barcode --modes zxing detect --sizes 640 1280 --precisions fp16 int8
  python -m fieldbench pipeline ocr --sizes 1280 1600 --precisions fp16 --jpeg cpu nvjpg --prep cpu gpu
  python -m fieldbench pipeline ocr --sizes 1280 --jpeg nvjpg --prep gpu --workers 1 2 3   # overlap frames
  python -m fieldbench pipeline assistant --order sequential concurrent   # barcode + OCR on every frame
  python -m fieldbench pipeline product      # embedding + kNN product recognition
  python -m fieldbench.gpuprep               # GPU preprocessing vs the CPU path: diffs and cost
"""
import argparse
import datetime as dt
import itertools
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


# Per-workload defaults for `pipeline`. Calibration images never come from the test set.
WORKLOADS = {
    "barcode": {"data": "data/barcodes/test", "calib": "data/barcodes/val/images", "sizes": [640, 1280],
                "precisions": ["fp16", "int8"], "calibrator": "minmax"},
    # OCR's detector stays FP16 by default: both calibrators cost it 28-53 points of exact-line
    # accuracy (docs/int8-calibration-in-fieldbench.md). INT8 is still available on request.
    "ocr": {"data": "data/ocr/test", "calib": "data/ocr/calib", "sizes": [1280, 1600],
            "precisions": ["fp16"], "calibrator": "entropy"},
    # Barcode 640 + OCR 1280 on every frame: Zebra's default barcode size and its mid OCR size.
    "assistant": {"data": "data/barcodes/test", "data_ocr": "data/ocr/test", "calib": "data/barcodes/val/images",
                  "sizes": [640], "ocr_size": 1280, "precisions": ["fp16"], "calibrator": "minmax"},
    "product": {"data": "data/products/test", "gallery": "data/products/gallery", "calib": None,
                "sizes": [224], "precisions": ["fp16"], "calibrator": None},
}


def _calib_batches(files, shape, prep):
    """Calibration inputs from JPEG files, preprocessed exactly like the pipeline does."""
    import cv2
    import numpy as np

    for f in files:
        buf = np.empty(shape, np.float32)
        prep(cv2.imread(str(f)), buf)
        yield buf


def _engine(onnx, precision, size, args, prep, tag):
    """Build or fetch an engine; INT8 is calibrated on args.calib. Returns the path, or None on failure."""
    from . import engine as trt_engine

    try:
        if precision == "int8":
            files = sorted(Path(args.calib).glob("*.jpg"))[:args.calib_images]
            if not files:
                raise RuntimeError(f"no calibration images in {args.calib}")
            batches = _calib_batches(files, (1, 3, size, size), prep)
            # The id records how many images were actually used, not how many were asked for.
            path, secs = trt_engine.build_calibrated(onnx, batches, f"{tag}{len(files)}", args.calibrator)
        else:
            path, secs = trt_engine.build(onnx, precision)
    except Exception as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        return None
    print(f"  engine {'built in %.0fs' % secs if secs else 'cached'}: {path.name}")
    return path


def _barcode_pipelines(args):
    """Yield ready pipelines for each requested configuration, building engines on the way."""
    from . import barcode, yolo
    from .jpeg import make_decoder
    from .runner import TrtRunner

    for mode in args.modes:
        if mode == "zxing":
            for jpeg in args.jpeg:
                print(f"> barcode zxing (full frame, CPU) [jpeg {jpeg}]")
                yield barcode.ZxingPipeline(jpeg=make_decoder(jpeg), jpeg_name=jpeg)
            continue
        for size in args.sizes:
            name = f"{args.bc_model}_{size}"
            if name not in MODELS:
                print(f"> no barcode detector at {size}; skipping")
                continue
            for precision in args.precisions:
                print(f"> barcode detect {size} [{precision}]")
                path = _engine(MODELS[name]["onnx"], precision, size, args, yolo.letterbox_into, "bc")
                for jpeg, prep in itertools.product(args.jpeg, args.prep) if path else []:
                    pipe = barcode.DetectPipeline(TrtRunner(path), name, precision, size,
                                                  int8_calibrated=True if precision == "int8" else None,
                                                  jpeg=make_decoder(jpeg), jpeg_name=jpeg, prep=prep)
                    pipe.config["calibrator"] = args.calibrator if precision == "int8" else None
                    yield pipe


def _ocr_pipelines(args):
    from . import ocr
    from . import engine as trt_engine
    from .jpeg import make_decoder
    from .runner import TrtRunner

    # The recognizer stays FP16: INT8 there would need calibration on line crops, and it is
    # not the stage that grows with input size.
    rec_path, secs = trt_engine.build(MODELS["ppocr5_rec_en"]["onnx"], "fp16")
    narrow = {}  # width bucket -> engine path; built on first use (several minutes each)
    for w in sorted({w for b in args.rec_buckets for w in b.split(",") if w and int(w) != 640}, key=int):
        print(f"> ocr rec width bucket {w} [fp16]")
        narrow[int(w)], _ = trt_engine.build(MODELS[f"ppocr5_rec_en_w{w}"]["onnx"], "fp16")
    charset = ocr.load_charset("models/ppocr5_rec_en.chars.txt")
    for size in args.sizes:
        name = f"ppocr5_det_{size}"
        for precision in args.precisions:
            print(f"> ocr det {size} [{precision}] + rec [fp16]")
            path = _engine(MODELS[name]["onnx"], precision, size, args, ocr.det_preprocess_into, "ocr")
            if path:
                for jpeg, prep, buckets in itertools.product(args.jpeg, args.prep, args.rec_buckets):
                    widths = [int(w) for w in buckets.split(",") if int(w) != 640]
                    pipe = ocr.OcrPipeline(TrtRunner(path), TrtRunner(rec_path), charset, name, size, precision,
                                           "fp16", jpeg=make_decoder(jpeg), jpeg_name=jpeg, prep=prep,
                                           rec_narrow=[TrtRunner(narrow[w]) for w in widths])
                    pipe.config["calibrator"] = args.calibrator if precision == "int8" else None
                    yield pipe


def _assistant_pipelines(args):
    from . import assistant, barcode, ocr, yolo
    from . import engine as trt_engine
    from .jpeg import make_decoder
    from .runner import TrtRunner

    rec_path, _ = trt_engine.build(MODELS["ppocr5_rec_en"]["onnx"], "fp16")
    oc_name = f"ppocr5_det_{args.ocr_size}"
    oc_path, _ = trt_engine.build(MODELS[oc_name]["onnx"], "fp16")
    charset = ocr.load_charset("models/ppocr5_rec_en.chars.txt")
    for size, precision in itertools.product(args.sizes, args.precisions):
        name = f"{args.bc_model}_{size}"
        print(f"> assistant: barcode {size} [{precision}] + ocr {args.ocr_size} [fp16]")
        bc_path = _engine(MODELS[name]["onnx"], precision, size, args, yolo.letterbox_into, "bc")
        for jpeg, prep, order in itertools.product(args.jpeg, args.prep, args.order) if bc_path else []:
            bc = barcode.DetectPipeline(TrtRunner(bc_path), name, precision, size,
                                        int8_calibrated=True if precision == "int8" else None,
                                        jpeg=make_decoder("cpu"), jpeg_name=jpeg, prep=prep)
            bc.config["calibrator"] = args.calibrator if precision == "int8" else None
            oc = ocr.OcrPipeline(TrtRunner(oc_path), TrtRunner(rec_path), charset, oc_name, args.ocr_size, "fp16",
                                 "fp16", jpeg=make_decoder("cpu"), jpeg_name=jpeg, prep=prep)
            yield assistant.AssistantPipeline(bc, oc, order, jpeg)


def _product_pipelines(args):
    from . import product
    from .jpeg import make_decoder
    from .runner import TrtRunner

    gallery = product.load_frames(args.gallery)
    for precision in args.precisions:
        print(f"> product embedding + gallery match [{precision}], gallery {len(gallery)} images")
        path = _engine(MODELS["mobilenetv3l_embed"]["onnx"], precision, 224, args, None, "prod")
        for jpeg in args.jpeg if path else []:
            yield product.ProductPipeline(TrtRunner(path), gallery, "mobilenetv3l_embed", precision,
                                          jpeg=make_decoder(jpeg), jpeg_name=jpeg, engine_path=path)


def _ref_ratio(r):
    """Zebra TC53 published pipeline time at this input size / our total p50 (>1: Jetson faster)."""
    refs = [next(iter(MODELS.get(d or "", {}).get("pipeline_references", [])), None)
            for d in (r.get("detector"), r.get("ocr_detector"))]
    refs = [x for x in refs if x]  # assistant: both Zebra jobs, run one after the other on a TC53
    return sum(x["latency_ms"] for x in refs) / r["total_ms"]["p50"] if refs else float("nan")


def _print_pipeline_table(workload, rows):
    stages = {"barcode": ["jpeg_decode", "preprocess", "infer", "postprocess", "decode"],
              "ocr": ["jpeg_decode", "det_pre", "det_infer", "det_post", "rec_pre", "rec_infer", "rec_post"],
              "assistant": ["jpeg_decode", "barcode", "ocr"],
              "product": ["jpeg_decode", "preprocess", "infer", "match"]}[workload]
    short = {"jpeg_decode": "jpeg", "preprocess": "prep", "postprocess": "post"}
    accs = {"barcode": [("decode_rate", "decoded"), ("detect_recall", "recall"), ("misreads", "miss")],
            "ocr": [("line_exact", "exact"), ("cer", "cer"), ("word_recall", "words"), ("detect_recall", "recall")],
            "assistant": [("decode_rate", "decoded"), ("line_exact", "exact"), ("misreads", "miss")],
            "product": [("top1", "top1"), ("top5", "top5"), ("top1_coarse", "coarse")]}[workload]
    print(f"\n{'config':<30}{'tot p50':>8}{'tot p90':>8}" + "".join(f"{short.get(s, s):>10}" for s in stages)
          + f"{'fps':>7}{'W':>7}{'mJ/fr':>8}" + "".join(f"{h:>8}" for _, h in accs) + f"{'vs TC53':>8}")
    multi_mode = len({r["device"].get("power_mode") for r in rows}) > 1
    for r in rows:
        st = {k: v["p50"] for k, v in r["stage_ms"].items()}
        what = ("zxing full frame" if r["mode"] == "zxing" else f"{r['input_size']} {r['precision']}"
                + (f"/{r['calibrator']}" if r.get("calibrator") else "")) + f" {r.get('jpeg', 'cpu')}" \
            + (" gpuprep" if r.get("prep") == "gpu" else "") + (f" x{r['workers']}" if r.get("workers", 1) > 1 else "") \
            + (f" {r['mode'][:3]}" if workload == "assistant" else "")
        if multi_mode:
            what = f"{r['device'].get('power_mode')} {what}"
        vals = [r["accuracy"].get(k, float("nan")) for k, _ in accs]
        print(f"{what:<30}{r['total_ms']['p50']:>8.1f}{r['total_ms']['p90']:>8.1f}"
              + "".join(f"{st.get(s, 0):>10.1f}" for s in stages)
              + f"{r['fps']:>7.1f}{r['telemetry_load'].get('p_VDD_IN_mean', 0) / 1000:>7.2f}"
              f"{r['energy'].get('mj_per_frame_total', 0):>8.0f}"
              + "".join(f"{v:>8}" if isinstance(v, int) else f"{v:>8.3f}" for v in vals) + f"{_ref_ratio(r):>8.2f}")
    print("\nstage columns are p50 ms (they don't sum to the total exactly: p50 of each stage, not of the sum)."
          "\nvs TC53 = Zebra's published time for the same job at that input size / our total p50 (>1: Jetson faster)")


def _stall_writer(f, args, dev, original_mode=None):
    """The watchdog's partial row goes to the same results file before the process exits, and the
    power mode is put back (the exit skips cmd_pipeline's finally)."""
    def write(row):
        row.update(timestamp=dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
                   label=args.label, device=dev, data=args.data, serialize=args.serialize)
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"  partial row written to {f.name}")
        if original_mode and power.current() != original_mode:
            print(f"  restoring power mode {original_mode}")
            power.set_mode(original_mode, settle_s=0)
    return write


def cmd_pipeline(args):
    from . import barcode, ocr
    from . import pipeline as pipe_run

    defaults = WORKLOADS[args.workload]
    args.data = args.data or defaults["data"]
    args.calib = args.calib or defaults["calib"]
    args.sizes = args.sizes or defaults["sizes"]
    args.calibrator = args.calibrator or defaults["calibrator"]
    args.precisions = args.precisions or defaults["precisions"]
    args.ocr_size = args.ocr_size or defaults.get("ocr_size")
    args.gallery = args.gallery or defaults.get("gallery")
    if args.serialize != "none":
        from .runner import TrtRunner
        TrtRunner.serialize = {"rec": ("ppocr5_rec",), "all": ("",)}[args.serialize]
    sensors = Sensors()
    if args.workload == "assistant":
        from . import assistant
        frames = assistant.load_frames(args.data, args.data_ocr or defaults["data_ocr"], args.every)
    elif args.workload == "product":
        from . import product
        frames = product.load_frames(args.data)
    else:
        frames = (barcode if args.workload == "barcode" else ocr).load_frames(args.data)
    items = sum(len(f.get("barcodes", f.get("lines", [1]))) for f in frames)
    out = Path(args.out or f"results/{dt.datetime.now():%Y%m%d-%H%M%S}-{args.workload}"
               f"{'-' + args.label if args.label else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    original = power.current()
    plan = _power_plan(args.power_modes, original)
    print(f"{len(frames)} frames from {args.data} ({items} items)\nwriting {out}\n")
    rows = []
    try:
        for mode in plan:
            if mode != power.current():
                power.set_mode(mode, settle_s=args.settle)
            dev = device_info()
            print(f"{dev['model']} | power mode {dev['power_mode']} | clocks locked: {dev['clocks_locked']}")
            pipes = {"barcode": _barcode_pipelines, "ocr": _ocr_pipelines, "assistant": _assistant_pipelines,
                     "product": _product_pipelines}[args.workload](args)
            with out.open("a") as f:
                for pipe in pipes:
                    try:
                        for workers in args.workers:
                            print(f"  workers {workers}")
                            r = pipe_run.run(pipe, frames, sensors, warmup_s=args.warmup, duration_s=args.duration,
                                             min_frames=args.min_frames, workers=workers, series_s=args.series,
                                             stall_s=args.stall_s, on_stall=_stall_writer(f, args, dev, original))
                            r.update(timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                                     host=socket.gethostname(), label=args.label, device=dev, data=args.data)
                            if args.serialize != "none":
                                r["serialize"] = args.serialize
                            f.write(json.dumps(r) + "\n")
                            f.flush()
                            rows.append(r)
                            print(f"  total p50 {r['total_ms']['p50']:.1f} ms, {r['fps']:.1f} fps")
                            if args.cooldown:
                                time.sleep(args.cooldown)
                    except Exception as exc:  # keep sweeping
                        print(f"  FAILED: {exc!r}", file=sys.stderr)
                    finally:
                        pipe.close()
    finally:
        if args.power_modes and power.current() != original:
            print(f"restoring power mode {original}")
            power.set_mode(original, settle_s=0)
    _print_pipeline_table(args.workload, rows)


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

    pl = sub.add_parser("pipeline", help="end-to-end handheld pipelines on real images, per-stage timing")
    pl.add_argument("workload", choices=list(WORKLOADS))
    pl.add_argument("--modes", nargs="+", default=["zxing", "detect"], choices=["zxing", "detect"],
                    help="barcode only: zxing = zxing-cpp on the whole frame; detect = detector + zxing-cpp on crops")
    pl.add_argument("--sizes", nargs="+", type=int, choices=[224, 640, 1280, 1600, 2560],
                    help="detector input sizes (barcode default 640 1280; ocr default 1280 1600)")
    pl.add_argument("--precisions", nargs="+", choices=PRECISIONS,
                    help="detector precision (default: barcode fp16 int8, ocr fp16); int8 is calibrated on --calib")
    pl.add_argument("--jpeg", nargs="+", default=["cpu"], choices=["cpu", "nvjpg"],
                    help="JPEG decoder: cpu = cv2.imdecode, nvjpg = Jetson NVJPG engine (fieldbench/jpeg.py)")
    pl.add_argument("--prep", nargs="+", default=["cpu"], choices=["cpu", "gpu"],
                    help="detector preprocessing: cpu = cv2 + numpy, gpu = fused CUDA kernel (fieldbench/gpuprep.py)")
    pl.add_argument("--data", help="folder with gt.jsonl + images/ (default per workload)")
    pl.add_argument("--calib", help="JPEGs for INT8 calibration (default per workload)")
    pl.add_argument("--calib-images", type=int, default=300)
    pl.add_argument("--calibrator", choices=["minmax", "entropy"],
                    help="INT8 calibration algorithm (default per workload: barcode minmax, ocr entropy)")
    pl.add_argument("--duration", type=float, default=20.0)
    pl.add_argument("--warmup", type=float, default=3.0)
    pl.add_argument("--min-frames", type=int, default=100)
    pl.add_argument("--cooldown", type=float, default=5.0)
    pl.add_argument("--power-modes", nargs="+", metavar="MODE",
                    help="nvpmodel modes to sweep, lowest budget first (e.g. 15W 25W MAXN_SUPER); default: current mode")
    pl.add_argument("--settle", type=float, default=10.0, help="seconds to wait after a mode switch")
    pl.add_argument("--workers", nargs="+", type=int, default=[1],
                    help="pipeline copies run as threads on overlapping frames (1 = one frame at a time); a list sweeps")
    pl.add_argument("--series", type=float, default=0,
                    help="also store throughput/latency/telemetry per window of this many seconds (soak runs)")
    pl.add_argument("--rec-buckets", nargs="+", default=["640"], metavar="W[,W...]",
                    help="ocr only: recognizer widths per configuration, e.g. 640 (pad every line to 640) and "
                         "320,480,640 (width buckets); a list sweeps")
    pl.add_argument("--bc-model", default="barcode_yolo11n", choices=["barcode_yolo11n", "barcode_real_yolo11n"],
                    help="barcode/assistant: detector weights (synthetic-trained, or fine-tuned on real photos)")
    pl.add_argument("--stall-s", type=float, default=120,
                    help="watchdog: if no frame finishes for this many seconds, write a partial row with every "
                         "thread's stack and exit (0 = off)")
    pl.add_argument("--serialize", default="none", choices=["none", "rec", "all"],
                    help="diagnostic: run the OCR recognizer's (rec) or every engine's (all) inferences one at a "
                         "time under a process-wide lock instead of overlapping on the GPU")
    pl.add_argument("--order", nargs="+", default=["sequential", "concurrent"], choices=["sequential", "concurrent"],
                    help="assistant only: barcode and OCR branches one after the other, or in parallel threads")
    pl.add_argument("--ocr-size", type=int, choices=[640, 1280, 1600, 2560], help="assistant only: OCR detector size")
    pl.add_argument("--data-ocr", help="assistant only: OCR test set (default data/ocr/test)")
    pl.add_argument("--every", type=int, default=2, help="assistant only: use every n-th frame of each test set")
    pl.add_argument("--gallery", help="product only: gallery folder (default data/products/gallery)")
    pl.add_argument("--label")
    pl.add_argument("--out")
    pl.set_defaults(fn=cmd_pipeline)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
