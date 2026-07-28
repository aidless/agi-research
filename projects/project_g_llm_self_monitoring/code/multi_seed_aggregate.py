"""
Project G -- Multi-seed H10 pilot aggregator.

Reads per-seed pilot logs (one per H10_SEED) and aggregates the
3-arm AUROC across seeds. Computes Welch t-test on Frozen vs Joint
per the H10 pre-registered decision rule.

Usage:
    python multi_seed_aggregate.py <log_dir> <n_seeds>

Output:
    Prints aggregate stats (mean +/- std per arm, Welch t, verdict).
"""
import sys
import os
import re
import numpy as np


def parse_seed_log(path):
    """Parse a single-seed pilot log and return per-arm AUROC."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-16")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        text = f.read()
    result = {}
    for arm in ["Frozen", "Joint", "Random"]:
        # Match "  Frozen: 1.000" or "  Frozen: nan"
        pattern = r"\s+" + arm + r":\s+([0-9.]+|nan)"
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            result[arm] = float("nan") if val == "nan" else float(val)
        else:
            result[arm] = float("nan")
    return result


def welch_t(a, b):
    """Compute Welch t-statistic for two samples."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    # Drop NaNs
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, va = np.mean(a), np.var(a, ddof=1)
    mb, vb = np.mean(b), np.var(b, ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    return t, df


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    # Look for files matching _h10_real_pilot_simple_seed*.log
    seed_logs = []
    for i in range(n_seeds):
        path = os.path.join(log_dir, f"_h10_real_pilot_simple_seed{i}.log")
        if os.path.exists(path):
            seed_logs.append((i, path))
    if not seed_logs:
        print(f"ERROR: no seed logs found in {log_dir}")
        sys.exit(1)
    print(f"Aggregating {len(seed_logs)} seed logs:")
    for seed, path in seed_logs:
        print(f"  seed {seed}: {path}")
    print()
    # Parse all
    per_seed = []
    for seed, path in seed_logs:
        result = parse_seed_log(path)
        result["seed"] = seed
        per_seed.append(result)
    # Aggregate per arm
    print("Per-seed results:")
    print(f"  {'seed':<6}{'Frozen':<10}{'Joint':<10}{'Random':<10}")
    for r in per_seed:
        f = r["Frozen"]
        j = r["Joint"]
        ra = r["Random"]
        f_str = "%.3f" % f if not np.isnan(f) else "NaN"
        j_str = "%.3f" % j if not np.isnan(j) else "NaN"
        r_str = "%.3f" % ra if not np.isnan(ra) else "NaN"
        print(f"  {r['seed']:<6}{f_str:<10}{j_str:<10}{r_str:<10}")
    print()
    frozen_vals = [r["Frozen"] for r in per_seed]
    joint_vals = [r["Joint"] for r in per_seed]
    random_vals = [r["Random"] for r in per_seed]
    print("Aggregate (mean +/- std):")
    for name, vals in [("Frozen", frozen_vals), ("Joint", joint_vals), ("Random", random_vals)]:
        vals_clean = [v for v in vals if not np.isnan(v)]
        if len(vals_clean) > 0:
            print(f"  {name:<8}: {np.mean(vals_clean):.3f} +/- {np.std(vals_clean, ddof=1):.3f}  (n={len(vals_clean)})")
        else:
            print(f"  {name:<8}: NaN")
    print()
    # Welch t-test Frozen vs Joint
    t_fj, df_fj = welch_t(frozen_vals, joint_vals)
    print(f"Welch t-test (Frozen vs Joint): t={t_fj:.3f}, df={df_fj:.2f}")
    t_fr, df_fr = welch_t(frozen_vals, random_vals)
    print(f"Welch t-test (Frozen vs Random): t={t_fr:.3f}, df={df_fr:.2f}")
    print()
    # H10 verdict
    print("H10 pilot verdict (n=" + str(len(seed_logs)) + " seeds, NOT full pre-reg):")
    if np.isnan(t_fj):
        print("  Welch t undefined (insufficient non-NaN values)")
    elif np.mean(frozen_vals) > np.mean(joint_vals) + 0.05 and t_fj > 2.0:
        print("  CONSISTENT WITH H10 (Frozen > Joint, t > 2.0)")
    elif np.mean(joint_vals) > np.mean(frozen_vals) + 0.05:
        print("  CONTRADICTS H10 (Joint > Frozen)")
    else:
        print("  INCONCLUSIVE on direction (Frozen ~ Joint within 0.05)")
    print()
    print("NOTE: H10 pre-reg requires Welch t > 2.0 on 200 rollouts/seed.")
    print("      This pilot uses N=12/seed; statistical power is LIMITED.")
    print("      Verdict here is exploratory, NOT the pre-reg H10 result.")


if __name__ == "__main__":
    main()