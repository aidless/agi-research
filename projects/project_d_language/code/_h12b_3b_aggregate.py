"""Aggregate H12b-3B multi-seed results from log files."""
import sys, os, re
import numpy as np

log_dir = r"E:\agi-research\experiments_log"
seeds = [0, 1, 100, 200, 300, 400]

results = {}
for seed in seeds:
    log_path = os.path.join(log_dir, f"_h12b_3b_seed{seed}.log")
    if not os.path.exists(log_path):
        continue
    with open(log_path, "r", encoding="utf-16", errors="replace") as f:
        text = f.read()
    result = {"seed": seed}
    for arm in ["LM", "DLR", "Random"]:
        patterns = [
            rf"\s+{arm}\s+accuracy:\s+([0-9.]+)",
            rf"\s+{arm}-[^:]*:\s+([0-9.]+)$",
        ]
        m = None
        for p in patterns:
            mm = re.search(p, text, re.MULTILINE)
            if mm:
                m = mm
                break
        if m:
            result[arm] = float(m.group(1))
        else:
            result[arm] = float("nan")
    if "per-predicate" in text:
        m = re.search(r"per-predicate:\s+(\{[^}]+\})", text)
        if m:
            result["per_pred"] = m.group(1)
    m = re.search(r"failure rate: ([0-9.]+)", text)
    if m:
        result["failure_rate"] = float(m.group(1))
    results[seed] = result

print("=" * 70)
print("H12b-3B multi-seed aggregation")
print("=" * 70)
print(f"Seeds found: {sorted(results.keys())}")
print()
hdr = "{:<6}{:<14}{:<8}{:<8}{:<8}{:<10}".format("seed", "predicate", "LM", "DLR", "Random", "fail_rate")
print(hdr)
for seed in sorted(results.keys()):
    r = results[seed]
    pred = "?"
    if "per_pred" in r:
        m = re.search(r"'(\w+)':", r["per_pred"])
        if m:
            pred = m.group(1)
    f_rate = r.get("failure_rate", float("nan"))
    f_str = "%.2f" % f_rate if not np.isnan(f_rate) else "?"
    lm_v = r["LM"]; dlr_v = r["DLR"]; rand_v = r["Random"]
    line = "{:<6}{:<14}{:<8.3f}{:<8.3f}{:<8.3f}{:<10}".format(seed, pred, lm_v, dlr_v, rand_v, f_str)
    print(line)

lm_vals = [r["LM"] for r in results.values() if not np.isnan(r["LM"])]
dlr_vals = [r["DLR"] for r in results.values() if not np.isnan(r["DLR"])]
rand_vals = [r["Random"] for r in results.values() if not np.isnan(r["Random"])]
print()
print("Aggregate (n={}):".format(len(lm_vals)))
print("  LM:     mean={:.3f}  std={:.3f}".format(
    np.mean(lm_vals),
    np.std(lm_vals, ddof=1) if len(lm_vals) > 1 else 0.0))
print("  DLR:    mean={:.3f}  std={:.3f}".format(
    np.mean(dlr_vals),
    np.std(dlr_vals, ddof=1) if len(dlr_vals) > 1 else 0.0))
print("  Random: mean={:.3f}  std={:.3f}".format(
    np.mean(rand_vals),
    np.std(rand_vals, ddof=1) if len(rand_vals) > 1 else 0.0))
print()
print("Delta_F-D: {:+.3f}".format(np.mean(lm_vals) - np.mean(dlr_vals)))
print("Delta_F-R: {:+.3f}".format(np.mean(lm_vals) - np.mean(rand_vals)))
print()
f_mean = np.mean(lm_vals)
if f_mean >= 0.70 and len(lm_vals) >= 1:
    print("  H12b direction-consistent VALIDATED: LM >= 0.70 (mean {:.3f})".format(f_mean))
elif f_mean < 0.50:
    print("  H12b direction-consistent REFUTED: LM < 0.50 (mean {:.3f})".format(f_mean))
else:
    print("  H12b INCONCLUSIVE: LM = {:.3f} (in [0.50, 0.70))".format(f_mean))
print()
print("NOTE: n<5 seeds; full pre-reg H12b requires n=5 + 50 pairs/seed.")