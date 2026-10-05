"""Render results/*.jsonl into a self-contained HTML report (report/index.html).

  python3 host/report.py                    # all results -> report/index.html
  python3 host/report.py results/x.jsonl -o report/x.html

The page is one HTML file with the data inlined as JSON and no network requests.
Every number in its titles, captions and annotations is computed in the browser
from that JSON, so re-running this script after new results re-tells the story.

Row selection
-------------
* Rows labelled "smoke" are skipped, and so is anything outside the top level of
  results/ (results/superseded/ is not globbed). So are the soak-hang diagnostics
  (labels "hang-*", and watchdog rows of kind "pipeline-stalled"): they are quoted in
  docs/results-phase3.md section 3, and their 10-minute rows would otherwise replace
  the Phase 3 chapter's rows under latest-row-wins.
* Phase 3 pipelines ("assistant", "product", PHASE3_PIPELINES) and rows measured on
  the BarBeR real-photo set (data "data/barber/...") only feed their own chapters;
  the Phase 1-2 chapters see neither (phase2_rows()).
* Appendix ("All measurements"): when the same configuration appears more than
  once, the most recent row wins. The configuration key is model_key() /
  pipeline_key() below (pipeline, mode, size, precision, calibrator, jpeg, prep,
  workers, power mode, locked clocks, data set).
* Chapters: with the default DVFS governors the same configuration moves 10-25%
  between sessions (docs/results-phase2.md, section 10), so each chapter compares
  rows from named sessions (run labels), the same sessions docs/results-phase*.md
  quote. Latest-row-wins is applied *within* a session only. Each chapter's payload
  carries a "sources" list that the page prints under its chart. If a named session
  is missing (e.g. a fresh checkout with other results), the chapter falls back to
  the latest row per configuration.

Adding a chapter (e.g. Phase 3)
-------------------------------
1. Write `chapter_<name>(rows)` here returning a JSON-able dict (or None to hide the
   chapter) and append `("<name>", chapter_<name>, scope)` to CHAPTERS, where scope
   filters the rows it sees (phase2_rows, or None for all rows).

2. In report_template.html, write `function renderName(data) { ... }` returning a
   chapter element (see renderOverlap for the pattern: chapter(), figure(), and one
   of the chart helpers) and append `{key: "<name>", render: renderName}` to
   CHAPTERS there. Chapters render in that order; a null payload hides the chapter.
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fieldbench.catalog import MODELS  # noqa: E402  (pure data, no TensorRT import)
from fieldbench.power import budget_rank  # noqa: E402  (jtop is imported lazily)

PHASE3_PIPELINES = {"assistant", "product"}
PRECISION_ORDER = {"fp32": 0, "fp16": 1, "int8": 2}
MODEL_NAMES = {"yolo11n": "YOLO11n", "mobilenetv3l": "MobileNetV3-L"}
JETPACK = {"36.4.7": "6.2.1", "36.4.4": "6.2.1", "36.4.3": "6.2", "36.4.0": "6.1", "36.3.0": "6.0"}

# Where each pipeline stage runs. "decode"/"prep" are the CPU stages chapter 4 moves off the CPU.
STAGE_CLASS = {
    "jpeg_decode": "decode",
    "preprocess": "prep", "det_pre": "prep",
    "infer": "net", "det_infer": "net", "rec_infer": "net",
    "postprocess": "other", "decode": "other", "det_post": "other", "rec_pre": "other", "rec_post": "other",
}
STAGE_NAMES = {
    "jpeg_decode": "JPEG decode", "preprocess": "Detector preprocessing", "det_pre": "Detector preprocessing",
    "infer": "Detector network", "det_infer": "Detector network", "rec_infer": "Recognizer network",
    "postprocess": "Box postprocessing", "decode": "zxing decode", "det_post": "Box postprocessing",
    "rec_pre": "Line crops", "rec_post": "CTC decode",
}


# ---------------------------------------------------------------------------- loading

def read_rows(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") == "smoke" or (r.get("label") or "").startswith("hang-") \
                    or r.get("kind") == "pipeline-stalled":
                continue
            if r.get("kind") == "pipeline":
                # Normalize switches that older rows omit or store as null.
                r["jpeg"] = r.get("jpeg") or "cpu"
                r["prep"] = r.get("prep") or "cpu"
                r["workers"] = r.get("workers") or 1
                if r.get("precision") == "int8" and not r.get("calibrator"):
                    r["calibrator"] = "minmax"  # the only calibrator before --calibrator existed
            r["_file"] = Path(path).name
            rows.append(r)
    return rows


def is_pipe(r):
    return r.get("kind") == "pipeline"


def is_barber(r):
    """Rows measured on the BarBeR real-photo set; they feed only the real-photo chapter."""
    return (r.get("data") or "").startswith("data/barber")


def data_set(r):
    return "barber" if is_barber(r) else "synthetic"


def locked(r):
    return bool(r["device"].get("clocks_locked"))


def model_key(r):
    return (r["model"], r["precision"], r["device"].get("power_mode"), locked(r))


def pipeline_key(r):
    return (r["pipeline"], r["mode"], r.get("input_size"), r.get("precision"), r.get("calibrator"),
            r["jpeg"], r["prep"], r["workers"], r["device"].get("power_mode"), locked(r), data_set(r))


def latest(rows):
    """Most recent row per configuration."""
    best = {}
    for r in rows:
        k = pipeline_key(r) if is_pipe(r) else model_key(r)
        if k not in best or r["timestamp"] > best[k]["timestamp"]:
            best[k] = r
    return list(best.values())


def session(rows, *labels):
    """Latest row per configuration within the named run labels; all rows if none of them exist."""
    picked = [r for r in rows if r.get("label") in labels]
    return latest(picked or rows), sorted({r.get("label") for r in picked}) or ["latest row per configuration"]


def where(rows, **conds):
    out = []
    for r in rows:
        dev = r["device"]
        ok = True
        for k, v in conds.items():
            have = locked(r) if k == "clocks_locked" else dev.get(k) if k == "power_mode" else r.get(k)
            if callable(v):
                ok = v(have)
            else:
                ok = have == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


# ---------------------------------------------------------------------------- slim rows

def best_npu_ref(model):
    refs = [x for x in MODELS.get(model, {}).get("references", []) if x.get("unit") == "NPU"]
    if not refs:
        return None
    x = min(refs, key=lambda x: x["latency_ms"])
    return {"ms": x["latency_ms"], "runtime": x["runtime"], "precision": x["precision"], "source": x["source"]}


def pipeline_ref(r):
    x = next(iter(MODELS.get(r.get("detector") or "", {}).get("pipeline_references", [])), None)
    return None if x is None else {"ms": x["latency_ms"], "device": x["device"], "source": x["source"]}


def slim_model(r):
    load, idle, en, dev = r["telemetry_load"], r["telemetry_idle"], r["energy"], r["device"]
    w = lambda v: None if v is None else v / 1000  # noqa: E731  mW -> W
    return {
        "model": r["model"], "name": MODEL_NAMES.get(r["model"], r["model"]), "precision": r["precision"],
        "power_mode": dev.get("power_mode"), "clocks_locked": locked(r),
        "int8_calibrated": r.get("int8_calibrated"),
        "p50": r["gpu_ms"]["p50"], "p99": r["gpu_ms"]["p99"], "e2e_p50": r["e2e_ms"]["p50"], "fps": r["fps"],
        "watts": w(load.get("p_VDD_IN_mean")), "idle_watts": w(idle.get("p_VDD_IN_mean")),
        "mj": en.get("mj_per_inf_total"), "mj_dynamic": en.get("mj_per_inf_dynamic"),
        "tj_max": load.get("temp_tj_max"), "gpu_mhz": load.get("gpu_mhz_mean"),
        "throttle": r.get("throttle_events"),
        "ref": best_npu_ref(r["model"]),
        "label": r.get("label"), "timestamp": r["timestamp"],
    }


def slim_pipe(r, stages=False):
    load, dev, acc = r["telemetry_load"], r["device"], r.get("accuracy") or {}
    out = {
        "pipeline": r["pipeline"], "mode": r["mode"], "size": r.get("input_size"), "precision": r.get("precision"),
        "calibrator": r.get("calibrator"), "jpeg": r["jpeg"], "prep": r["prep"], "workers": r["workers"],
        "power_mode": dev.get("power_mode"), "clocks_locked": locked(r),
        "p50": r["total_ms"]["p50"], "p90": r["total_ms"]["p90"], "mean": r["total_ms"]["mean"], "fps": r["fps"],
        "watts": None if load.get("p_VDD_IN_mean") is None else load["p_VDD_IN_mean"] / 1000,
        "mj": r["energy"].get("mj_per_frame_total"),
        "gpu_mhz": load.get("gpu_mhz_mean"), "emc_mhz": load.get("emc_mhz_mean"),
        "tj_max": load.get("temp_tj_max"), "throttle": r.get("throttle_events"),
        # Barcode: share of ground-truth barcodes decoded. OCR: share of lines read exactly.
        "accuracy": acc.get("decode_rate", acc.get("line_exact")),
        "misreads": acc.get("misreads"), "scored": acc.get("barcodes", acc.get("lines")),
        "data_set": data_set(r), "frames": r.get("frames_in_set"), "detect_recall": acc.get("detect_recall"),
        "boxes_per_frame": acc.get("boxes_per_frame", acc.get("lines_per_frame_pred")),
        "ref": pipeline_ref(r),
        "label": r.get("label"), "timestamp": r["timestamp"],
    }
    if stages:
        out["stages"] = [{"key": s, "name": STAGE_NAMES.get(s, s), "cls": STAGE_CLASS.get(s, "other"),
                          "mean": r["stage_ms"][s]["mean"], "p50": r["stage_ms"][s]["p50"]}
                         for s in r["stages"] if s in r["stage_ms"]]
    return out


def sort_pipes(ps):
    return sorted(ps, key=lambda p: (p["pipeline"], p["mode"] != "zxing", p["size"] or 0,
                                     PRECISION_ORDER.get(p["precision"], 9), p["calibrator"] or "",
                                     p["jpeg"], p["prep"], p["workers"], p["clocks_locked"]))


# ---------------------------------------------------------------------------- chapters

MAXN = "MAXN_SUPER"


def chapter_models(rows):
    """1. Bare networks at MAXN_SUPER by precision (docs/results-phase1.md section 1: run full-sweep)."""
    pool, sources = session(where(rows, kind=None, power_mode=MAXN, clocks_locked=False), "full-sweep")
    out = {}
    for m in MODEL_NAMES:
        ms = sorted((slim_model(r) for r in pool if r["model"] == m), key=lambda x: PRECISION_ORDER[x["precision"]])
        if ms:
            out[m] = ms
    return {"models": out, "sources": sources} if out else None


def chapter_power(rows):
    """2. Power-mode sweep (docs/results-phase1.md section 2: runs power-sweep + power-7w)."""
    pool, sources = session(where(rows, kind=None, clocks_locked=False), "power-sweep", "power-7w")
    modes = sorted({r["device"].get("power_mode") for r in pool if r["device"].get("power_mode")},
                   key=lambda m: (budget_rank(m), m))
    if len(modes) < 2:
        return None
    return {"modes": modes, "rows": [slim_model(r) for r in pool if r["model"] in MODEL_NAMES], "sources": sources}


def chapter_stages(rows):
    """3. Per-stage breakdown of the baseline pipelines (CPU decode, CPU prep, FP16; docs/results-phase2.md
    sections 6-7: run label phase2). Barcode 640, OCR 1280 and OCR 2560."""
    base = where(rows, kind="pipeline", power_mode=MAXN, clocks_locked=False, workers=1, jpeg="cpu", prep="cpu",
                 precision="fp16")
    pool, sources = session(base, "phase2")
    out = []
    for pipe, size in (("barcode", 640), ("ocr", 1280), ("ocr", 2560)):
        hit = [r for r in pool if r["pipeline"] == pipe and r.get("input_size") == size and r["mode"] != "zxing"]
        if hit:
            out.append(slim_pipe(hit[0], stages=True))
    return {"rows": out, "sources": sources} if out else None


def chapter_gpuprep(rows):
    """4. CPU vs GPU preprocessing, NVJPG decode, default governors, same session per pair
    (docs/results-phase2.md section 9, 'Default governors' table: runs ocr-gpuprep and bc-gpuprep).
    OCR stays FP16 (INT8 breaks its detector, chapter 5); barcode uses INT8, its best configuration.
    Note: the later 'overlap' session re-measured some of these configurations (workers=1) and would win
    latest-row-wins with different numbers (e.g. OCR 1280 135.9 -> 124.4 ms), hence the explicit labels."""
    base = where(rows, kind="pipeline", power_mode=MAXN, clocks_locked=False, workers=1, jpeg="nvjpg")
    out, sources = [], []
    for pipe, label, prec in (("barcode", "bc-gpuprep", "int8"), ("ocr", "ocr-gpuprep", "fp16")):
        pool, src = session([r for r in base if r["pipeline"] == pipe and r.get("precision") == prec], label)
        sources += src
        for size in sorted({r.get("input_size") for r in pool if r.get("input_size")}):
            pair = {p: next((r for r in pool if r.get("input_size") == size and r["prep"] == p), None)
                    for p in ("cpu", "gpu")}
            if all(pair.values()) and pair["cpu"].get("label") == pair["gpu"].get("label"):
                out.append({"pipeline": pipe, "size": size, "before": slim_pipe(pair["cpu"]),
                            "after": slim_pipe(pair["gpu"])})
    return {"pairs": out, "sources": sorted(set(sources))} if out else None


def chapter_accuracy(rows):
    """5. Accuracy by precision and calibrator (docs/results-phase2.md sections 6 and 8: runs phase2,
    phase2-int8, phase2-int8-entropy). Accuracy is deterministic per configuration, so any session agrees."""
    base = where(rows, kind="pipeline", power_mode=MAXN, clocks_locked=False, workers=1, jpeg="cpu", prep="cpu")
    pool, sources = session([r for r in base if r["mode"] != "zxing"], "phase2", "phase2-int8", "phase2-int8-entropy")
    keep = [slim_pipe(r) for r in pool if r.get("precision") in ("fp16", "int8")]
    return {"rows": sort_pipes(keep), "sources": sources} if keep else None


def chapter_dvfs(rows):
    """6. Default governors vs locked clocks, best configuration (NVJPG + GPU prep, FP16; docs/results-phase2.md
    section 10 table). Default rows come from the session the locked run was paired with (its label minus
    '-locked'), not from the later overlap session."""
    lock = where(rows, kind="pipeline", power_mode=MAXN, clocks_locked=True, workers=1, jpeg="nvjpg", prep="gpu",
                 precision="fp16")
    out, sources = [], set()
    for lk in sorted(latest(lock), key=lambda r: (r["pipeline"] != "ocr", r.get("input_size"))):
        dlabel = (lk.get("label") or "").removesuffix("-locked")
        cands = where(rows, kind="pipeline", pipeline=lk["pipeline"], mode=lk["mode"], input_size=lk.get("input_size"),
                      precision="fp16", jpeg="nvjpg", prep="gpu", workers=1, power_mode=MAXN, clocks_locked=False)
        same = [r for r in cands if r.get("label") == dlabel] or cands
        if not same:
            continue
        d = max(same, key=lambda r: r["timestamp"])
        sources |= {d.get("label"), lk.get("label")}
        out.append({"pipeline": lk["pipeline"], "size": lk.get("input_size"),
                    "default": slim_pipe(d), "locked": slim_pipe(lk)})
    return {"pairs": out, "sources": sorted(sources)} if out else None


def chapter_overlap(rows):
    """7. Stage overlap: 1..N pipeline workers on consecutive frames (docs/results-phase2.md section 11:
    run label overlap). One series per pipeline and size, NVJPG decode preferred."""
    multi = [r for r in rows if is_pipe(r) and r["workers"] > 1 and not locked(r)]
    if not multi:
        return None
    pool, sources = session(where(rows, kind="pipeline", clocks_locked=False, power_mode=MAXN), "overlap")
    series = []
    for pipe, size in sorted({(r["pipeline"], r.get("input_size")) for r in pool if r["workers"] > 1},
                             key=lambda x: (x[0] != "barcode", x[1] or 0)):
        by_jpeg = {}
        for r in pool:
            if r["pipeline"] == pipe and r.get("input_size") == size:
                by_jpeg.setdefault((r["jpeg"], r["prep"], r.get("precision")), []).append(r)
        # Pick the variant with the most worker counts; NVJPG + GPU prep wins ties.
        variant = max(by_jpeg, key=lambda k: (len({r["workers"] for r in by_jpeg[k]}), k[0] == "nvjpg", k[1] == "gpu"))
        pts = sorted((slim_pipe(r) for r in by_jpeg[variant]), key=lambda p: p["workers"])
        if len(pts) > 1 and pts[0]["workers"] == 1:
            series.append({"pipeline": pipe, "size": size, "points": pts})
    return {"series": series, "sources": sources} if series else None


def chapter_power_pipes(rows):
    """Pipelines across power modes (run label power-pipelines): barcode 640 and OCR 1280 FP16, NVJPG + GPU
    prep, workers 1, at each live-switchable mode. Compared within that session only."""
    pool = [r for r in latest([r for r in rows if r.get("label") == "power-pipelines"])
            if is_pipe(r) and r["workers"] == 1 and not locked(r)]
    series = []
    for pipe, size in sorted({(r["pipeline"], r.get("input_size")) for r in pool}, key=lambda x: (x[0] != "barcode", x[1])):
        pts = sorted((slim_pipe(r) for r in pool if r["pipeline"] == pipe and r.get("input_size") == size),
                     key=lambda p: budget_rank(p["power_mode"] or ""))
        if len(pts) > 1:
            series.append({"pipeline": pipe, "size": size, "points": pts})
    return {"series": series, "sources": ["power-pipelines"]} if series else None


# BarBeR real-photo check. The ground-truth-crop ceiling is a host measurement (zxing-cpp on each
# barcode's deskewed ground-truth crop, CPU), made when the subset was prepared (host/make_barber.py);
# it is not part of the Jetson rows, so it is carried here and printed as "(host)".
BARBER = {"citation": "Vezzali, Bolelli, Santi, Grana, “BarBeR: A Barcode Benchmarking Repository”, ICPR 2024",
          "url": "https://ditto.ing.unimore.it/barber/", "total_photos": 8748, "gt_crop_ceiling_host": 0.725}


def _jpeg_size(path):
    """(width, height) from a JPEG's SOF marker, without an image library."""
    with open(path, "rb") as f:
        d = f.read(1 << 18)
    i = 2
    while i + 9 < len(d):
        if d[i] != 0xFF:
            i += 1
            continue
        m, ln = d[i + 1], int.from_bytes(d[i + 2:i + 4], "big")
        if m in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(d[i + 7:i + 9], "big"), int.from_bytes(d[i + 5:i + 7], "big")
        i += 2 + ln
    return None


def chapter_barber(rows):
    """Real-photo check on a BarBeR subset (run label barber): whole-frame zxing and the synthetic-trained
    detector at 640/1280/1600 x FP16/INT8, CPU decode and prep. Accuracy only: photo sizes vary widely,
    so the latency of these rows is not compared with the TC53."""
    pool = [r for r in latest([r for r in rows if is_barber(r) and is_pipe(r)]) if r["workers"] == 1]
    if not pool:
        return None
    out = {"rows": sort_pipes(slim_pipe(r) for r in pool), "sources": sorted({r.get("label") for r in pool}), **BARBER}
    gt = ROOT / (pool[0].get("data") or "") / "gt.jsonl"
    if gt.exists():
        frames = [json.loads(x) for x in gt.read_text().splitlines() if x.strip()]
        out["datasets"] = len({f.get("source") for f in frames})
        mp = [s[0] * s[1] / 1e6 for f in frames if (s := _jpeg_size(gt.parent / f["image"]))]
        if mp:
            out["mp_range"] = [min(mp), max(mp)]
        mods = sorted(b["module_px"] for f in frames for b in f["barcodes"] if b.get("module_px", -1) > 0)
        if mods:
            out["module_px_median"] = mods[len(mods) // 2]
            out["module_lt2"] = sum(m < 2 for m in mods) / len(mods)

    return out


def chapter_assistant(rows):
    """Phase 3: barcode + OCR on every frame (pipeline assistant; docs/results-phase3.md section 1)."""
    pool = [r for r in latest([r for r in rows if r.get("pipeline") == "assistant" and r.get("label") != "soak"])
            if not locked(r)]
    out = []
    for r in pool:
        p = slim_pipe(r)
        sm, acc = r["stage_ms"], r.get("accuracy") or {}
        p.update(order=r["mode"], total_mean=r["total_ms"]["mean"],
                 bc_branch=sm.get("barcode", {}).get("mean"), ocr_branch=sm.get("ocr", {}).get("mean"),
                 bc_gpu=sm.get("bc.infer_gpu", {}).get("mean"), line_exact=acc.get("line_exact"),
                 decode_rate=acc.get("decode_rate"),
                 ref_ms=sum(x["latency_ms"] for x in (MODELS[f"barcode_yolo11n_{r.get('input_size')}"]["pipeline_references"]
                                                      + MODELS[f"ppocr5_det_{r.get('ocr_input_size')}"]["pipeline_references"])),
                 ocr_size=r.get("ocr_input_size"))
        out.append(p)
    out.sort(key=lambda p: (p["workers"], p["order"] != "sequential"))
    return {"rows": out, "sources": sorted({p["label"] for p in out})} if out else None


def chapter_product(rows):
    """Phase 3: product recognition (pipeline product; docs/results-phase3.md section 2)."""
    pool = [r for r in latest([r for r in rows if r.get("pipeline") == "product"]) if not locked(r)]
    if not pool:
        return None
    out = []
    for r in sorted(pool, key=lambda r: r["workers"]):
        p = slim_pipe(r)
        acc, sm = r.get("accuracy") or {}, r["stage_ms"]
        p.update(top1=acc.get("top1"), top5=acc.get("top5"), top1_coarse=acc.get("top1_coarse"), images=acc.get("images"),
                 infer_gpu=sm.get("infer_gpu", {}).get("mean"), mj_dynamic=r["energy"].get("mj_per_frame_dynamic"),
                 stage_means={k: sm[k]["mean"] for k in r["stages"] if k in sm})
        out.append(p)
    return {"rows": out, "npu_ref": best_npu_ref("mobilenetv3l"), "sources": sorted({p["label"] for p in out})}


def chapter_soak(rows):
    """Phase 3: sustained load. Renders the per-interval series of the latest 'soak' row; without one, the page
    prints the documented status of the soak run instead (docs/results-phase3.md section 3)."""
    soak = [r for r in rows if r.get("label") == "soak" and r.get("series")]
    if not soak:
        return {"missing": True, "doc": "docs/results-phase3.md#3-sustained-load-thermal-soak"}
    r = max(soak, key=lambda r: r["timestamp"])
    p = slim_pipe(r)
    # Only full windows: the last one often covers just the moments after the timed loop ended.
    step = r.get("series_s") or 30
    full = [w for w in r["series"] if w.get("t_s") is None or w["t_s"] + step <= (r.get("wall_s") or 0) + 0.5]
    p.update(order=r.get("mode"), series=full, wall_s=r.get("wall_s"))
    return {"row": p, "sources": [r.get("label")]}


def phase2_rows(rows):
    """Phase 1-2 rows: synthetic test sets, barcode and OCR pipelines (plus model rows)."""
    return [r for r in rows if not is_barber(r) and r.get("pipeline") not in PHASE3_PIPELINES]


# Render order is set by the template's CHAPTERS list; this list only decides which payload keys exist.
# The third element says which rows a chapter sees.
CHAPTERS = [
    ("models", chapter_models, phase2_rows),
    ("power", chapter_power, phase2_rows),
    ("stages", chapter_stages, phase2_rows),
    ("gpuprep", chapter_gpuprep, phase2_rows),
    ("accuracy", chapter_accuracy, phase2_rows),
    ("barber", chapter_barber, None),
    ("dvfs", chapter_dvfs, phase2_rows),
    ("powerpipes", chapter_power_pipes, phase2_rows),
    ("overlap", chapter_overlap, phase2_rows),
    ("assistant", chapter_assistant, None),
    ("product", chapter_product, None),
    ("soak", chapter_soak, None),
]


# ---------------------------------------------------------------------------- payload

def board_facts(rows):
    dev = next((r["device"] for r in rows if r.get("device")), {})
    model = dev.get("model") or ""
    short = re.sub(r"^NVIDIA\s+", "", model)
    short = re.sub(r"\s*Engineering Reference Developer Kit\s*", " ", short).strip()
    m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", dev.get("l4t") or "")
    l4t = f"{m.group(1)}.{m.group(2)}" if m else None
    return {"board": model, "board_short": short or model, "l4t": l4t, "jetpack": JETPACK.get(l4t or "")}


def build_payload(rows):
    newest = latest(rows)
    models = [slim_model(r) for r in newest if not is_pipe(r)]
    models.sort(key=lambda m: (list(MODEL_NAMES).index(m["model"]) if m["model"] in MODEL_NAMES else 9,
                               PRECISION_ORDER.get(m["precision"], 9), budget_rank(m["power_mode"] or ""),
                               m["clocks_locked"]))
    payload = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        **board_facts(rows),
        "appendix": {"models": models, "pipelines": sort_pipes(slim_pipe(r) for r in newest if is_pipe(r))},
    }
    for key, fn, scope in CHAPTERS:
        payload[key] = fn(scope(rows) if scope else rows)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="results JSONL files (default: results/*.jsonl)")
    ap.add_argument("-o", "--out", default=str(ROOT / "report" / "index.html"))
    args = ap.parse_args()

    files = args.files or sorted(str(p) for p in (ROOT / "results").glob("*.jsonl"))
    if not files:
        sys.exit("no results found; run a benchmark and `make pull` first")
    rows = read_rows(files)
    if not rows:
        sys.exit("no usable rows in the result files")
    payload = build_payload(rows)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (Path(__file__).parent / "report_template.html").read_text().replace("/*__DATA__*/null", data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    shown = [k for k, *_ in CHAPTERS if payload.get(k)]
    print(f"{len(payload['appendix']['models'])} model and {len(payload['appendix']['pipelines'])} pipeline"
          f" configurations from {len(files)} file(s); chapters: {', '.join(shown)} -> {out}")


if __name__ == "__main__":
    main()
