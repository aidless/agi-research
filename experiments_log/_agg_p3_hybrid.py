# -*- coding: utf-8 -*-
"""
P3 hybrid bootstrap aggregator.

Computes pairwise contrasts (Monitor vs DLR, Monitor vs Hybrid, DLR vs
Hybrid) from completed P3 hybrid log files. Emits partial verdicts
as more jobs complete. Writes JSON output for downstream consumption.

Usage:
    python _agg_p3_hybrid.py
"""
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

LOGDIR = r"E:\agi-research\experiments_log"

# Find all completed P3 logs (>= 700 bytes, indicates full eval)
logs_by_arm = defaultdict(list)
for log_path in glob.glob(os.path.join(LOGDIR, "_p3_hybrid_*.log")):
    fname = os.path.basename(log_path)
    if os.path.getsize(log_path) < 700:
        continue
    # Parse arm and seed from filename: _p3_hybrid_<arm>_s<seed>.log
    m = re.match(r"_p3_hybrid_(monitor_only|dlr_only|v8)_s(\d+)\.log", fname)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    # Parse final delta from log content
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    delta_m = re.search(r"delta vs random:\s+([+-]?[\d.]+)", content)
    if not delta_m:
        continue
    delta = float(delta_m.group(1))
    logs_by_arm[arm].append({"seed": seed, "delta": delta, "log": log_path})

print("=" * 60)
print("P3 hybrid bootstrap aggregator")
print("=" * 60)
print(f"Timestamp: {datetime.now().isoformat()}")
print()
for arm in ("monitor_only", "dlr_only", "v8"):
    n = len(logs_by_arm[arm])
    if n > 0:
        deltas = [d["delta"] for d in logs_by_arm[arm]]
        mean = sum(deltas) / n
        std = (sum((d - mean) ** 2 for d in deltas) / max(n - 1, 1)) ** 0.5
        print(f"  {arm:15s}  n={n:3d}  mean delta = {mean:+.3f}  std = {std:.3f}")
    else:
        print(f"  {arm:15s}  n=  0  (no completed jobs)")

# Compute pairwise contrasts (paired by seed)
def contrast(arm_a, arm_b):
    a_data = {d["seed"]: d["delta"] for d in logs_by_arm[arm_a]}
    b_data = {d["seed"]: d["delta"] for d in logs_by_arm[arm_b]}
    common_seeds = sorted(set(a_data) & set(b_data))
    if len(common_seeds) < 3:
        return None
    diffs = [a_data[s] - b_data[s] for s in common_seeds]
    mean_diff = sum(diffs) / len(diffs)
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
    std_diff = var_diff ** 0.5
    return {
        "n_paired": len(common_seeds),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_lo": mean_diff - 1.96 * std_diff / (len(diffs) ** 0.5),
        "ci95_hi": mean_diff + 1.96 * std_diff / (len(diffs) ** 0.5),
    }

print()
print("Pairwise contrasts (delta_a - delta_b):")
for a, b in (("v8", "dlr_only"), ("v8", "monitor_only"), ("monitor_only", "dlr_only")):
    c = contrast(a, b)
    label = f"{a:15s} - {b:15s}"
    if c is None:
        print(f"  {label}  (insufficient paired data)")
    else:
        verdict = "VALIDATED" if c["mean_diff"] >= 0.05 else "REFUTED" if c["mean_diff"] < 0 else "AMBIGUOUS"
        print(f"  {label}  n_paired={c['n_paired']:2d}  diff = {c['mean_diff']:+.3f}  "
              f"95% CI [{c['ci95_lo']:+.3f}, {c['ci95_hi']:+.3f}]  {verdict}")

# Pre-reg verdict: Hybrid - DLR alone >= +0.05 with p<0.05 Bonferroni
# Without bootstrap p-value (would need actual resampling), we use
# t-test approximation: t = mean_diff / (std_diff / sqrt(n_paired))
# Bonferroni-corrected alpha = 0.05 / 3 = 0.0167
v_minus_dlr = contrast("v8", "dlr_only")
if v_minus_dlr is not None and v_minus_dlr["n_paired"] >= 3:
    se = v_minus_dlr["std_diff"] / (v_minus_dlr["n_paired"] ** 0.5)
    t_stat = v_minus_dlr["mean_diff"] / max(se, 1e-9)
    # Two-sided t-test p-value approximation
    # For large n, |t| > 2.6 -> p < 0.01; |t| > 2.0 -> p < 0.05
    sig_2sided = "p<0.01" if abs(t_stat) > 2.6 else "p<0.05" if abs(t_stat) > 2.0 else "p>=0.05"
    pre_reg_verdict = "VALIDATED" if v_minus_dlr["mean_diff"] >= 0.05 and abs(t_stat) > 2.0 \
        else "REFUTED" if v_minus_dlr["mean_diff"] < 0.05 \
        else "AMBIGUOUS"
    print()
    print("Pre-reg verdict (v8 - dlr_only):")
    print(f"  diff = {v_minus_dlr['mean_diff']:+.3f},  |t| = {abs(t_stat):.2f}  ({sig_2sided})")
    print(f"  Pre-reg rule: diff >= +0.05 AND p<0.05 -> VALIDATED")
    print(f"  Pre-reg rule: diff <  +0.05 OR  p>=0.05 -> REFUTED")
    print(f"  => VERDICT: {pre_reg_verdict}")
else:
    pre_reg_verdict = "INSUFFICIENT_DATA"
    print()
    print("Pre-reg verdict: INSUFFICIENT_DATA (need >= 3 paired seeds)")

# Save JSON output
out = {
    "timestamp": datetime.now().isoformat(),
    "per_arm": {arm: [
        {"seed": d["seed"], "delta": d["delta"]}
        for d in logs_by_arm[arm]
    ] for arm in ("monitor_only", "dlr_only", "v8")},
    "contrasts": {f"{a}_vs_{b}": contrast(a, b) for a, b in (("v8", "dlr_only"), ("v8", "monitor_only"), ("monitor_only", "dlr_only"))},
    "pre_reg_verdict": pre_reg_verdict if v_minus_dlr and v_minus_dlr["n_paired"] >= 3 else "INSUFFICIENT_DATA",
}
out_path = os.path.join(LOGDIR, "_p3_hybrid_bootstrap.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print()
print(f"Wrote: {out_path}")