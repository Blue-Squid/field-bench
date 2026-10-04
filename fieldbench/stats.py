"""Latency summary statistics (no TensorRT import, so the host can use them too)."""
import statistics


def _percentile(sorted_vals, q):
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def latency_stats(vals):
    s = sorted(vals)
    return {
        "mean": statistics.fmean(s), "std": statistics.pstdev(s),
        "min": s[0], "p50": _percentile(s, 0.5), "p90": _percentile(s, 0.9),
        "p95": _percentile(s, 0.95), "p99": _percentile(s, 0.99), "max": s[-1],
    }
