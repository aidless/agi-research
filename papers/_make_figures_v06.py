# -*- coding: utf-8 -*-
"""
Generate the v0.6 figures for the GSM8K 200-token follow-up section.

Produces TWO figures:

  1. forest_h10_n20_gsm8k.png -- H10 n=20 GSM8K 200-token 3-arm forest plot.
  2. h10_shrinkage_timeline_v06.png -- Cross-task shrinkage timeline:
     F-J effect across all four pre-registered runs
     (n=5/n=20/n=100 simple arith + n=20 GSM8K 200-token).

The cross-task timeline reads ALL FOUR baseline JSONs to ensure
single source of truth -- no hardcoded numbers in this script.

Run from any cwd once 60 jobs have aggregated:
    python papers/_make_figures_v06.py

Reads (each is optional; missing ones are skipped with a warning):
    experiments_log/_h10_n5_stratified.json     (or *_n5_*)
    experiments_log/_h10_n20_bootstrap.json
    experiments_log/_h10_n100_bootstrap.json
    experiments_log/_h10_n20_gsm8k_bootstrap.json   (last to be created)

Writes:
    papers/figures_v2/forest_h10_n20_gsm8k.png
    papers/figures_v2/h10_shrinkage_timeline_v06.png
"""

import json
import os
import glob

import numpy as np

PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(PAPERS_DIR, "..", "experiments_log"))
OUT_DIR = os.path.join(PAPERS_DIR, "figures_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# Use non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_bootstrap(pattern):
    matches = sorted(glob.glob(os.path.join(EXP_DIR, pattern)))
    if not matches:
        return None
    return matches[-1]


def load_bootstrap(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_run_meta(report):
    if not report:
        return None
    co = report.get("contrasts", {}).get("F-J")
    if not co:
        return None
    return {
        "n": int(co.get("n", 0)),
        "mean": float(co.get("mean_diff", 0.0)),
        "ci_lo": float(co["ci95"][0]),
        "ci_hi": float(co["ci95"][1]),
        "d": float(co.get("cohens_d", 0.0)),
    }


def make_forest(gsm8k_report):
    contrasts = ["F-J", "F-R", "J-R"]
    labels = [
        "Frozen - Joint (decoupling effect)",
        "Frozen - Random (negative control)",
        "Joint - Random",
    ]
    means, los, his, ds = [], [], [], []
    for c in contrasts:
        co = gsm8k_report.get("contrasts", {}).get(c)
        if not co:
            print(f"  Contrast {c} not in bootstrap; skipping")
            return False
        means.append(co["mean_diff"])
        los.append(co["ci95"][0])
        his.append(co["ci95"][1])
        ds.append(co["cohens_d"])

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(contrasts))
    errors = np.array([
        [m - lo for m, lo in zip(means, los)],
        [hi - m for m, hi in zip(means, his)],
    ])
    ax.errorbar(means, y, xerr=errors, fmt="o", color="black",
                capsize=4, markersize=8, ecolor="black")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.05, color="red", linestyle=":", linewidth=0.8,
               label="pre-reg threshold +0.05")
    ax.axvline(0.10, color="darkred", linestyle=":", linewidth=0.8,
               label="kill-switch threshold +0.10")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUROC difference")
    ax.set_title("H10 n=20 GSM8K 200-token: paired seed-level contrast\n(60 jobs, 2000-rep bootstrap 95% CI)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", linestyle=":", linewidth=0.4)
    for i, (m, d) in enumerate(zip(means, ds)):
        ax.text(max(his) + 0.02, i, f"d={d:+.3f}", va="center", fontsize=9, color="dimgray")
    out_path = os.path.join(OUT_DIR, "forest_h10_n20_gsm8k.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  WROTE {out_path}")
    return True


def make_shrinkage_timeline(reports):
    rows = []
    for label, rep in reports.items():
        meta = get_run_meta(rep)
        if meta is None:
            print(f"  Skipping {label} (no F-J contrast)")
            continue
        rows.append((meta["n"], meta["mean"], meta["ci_lo"], meta["ci_hi"],
                     meta["d"], label))
    if not rows:
        print("  No usable runs; skipping timeline")
        return

    rows.sort(key=lambda r: r[0])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    los = [r[2] for r in rows]
    his = [r[3] for r in rows]  # ci_hi (was incorrectly r[4] which is Cohen d)
    errors = np.array([
        [y - lo for y, lo in zip(ys, los)],
        [hi - y for y, hi in zip(ys, his)],
    ])
    ax.errorbar(xs, ys, yerr=errors, fmt="o-", color="black", capsize=4, markersize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Sample size n (log scale)")
    ax.set_ylabel("Frozen - Joint AUROC difference (with 95% bootstrap CI)")
    ax.set_title("H10 cross-task F-J effect shrinkage across 4 pre-registered runs")
    ax.grid(True, linestyle=":", linewidth=0.4)
    for r in rows:
        n, m, _, _, d, label = r
        ax.annotate(label, xy=(n, m), xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color="dimgray", va="center")
        ax.annotate(f"d={d:+.3f}", xy=(n, m), xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color="gray", va="center")
    out_path = os.path.join(OUT_DIR, "h10_shrinkage_timeline_v06.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  WROTE {out_path}")


def main():
    print("=== Generating v0.6 figures ===")
    reports = {}

    n5_path = (find_bootstrap("_h10_n5_stratified.json")
               or find_bootstrap("_h10_n5_bootstrap.json")
               or find_bootstrap("*n5*.json"))
    n5_rep = load_bootstrap(n5_path)
    if n5_rep:
        reports["n=5 (simple arith)"] = n5_rep
    else:
        print("  WARN: no n=5 bootstrap found")

    n20_path = find_bootstrap("_h10_n20_bootstrap.json")
    n20_rep = load_bootstrap(n20_path)
    if n20_rep:
        reports["n=20 (simple arith)"] = n20_rep
    else:
        print("  WARN: no n=20 simple-arith bootstrap found")

    n100_path = find_bootstrap("_h10_n100_bootstrap.json")
    n100_rep = load_bootstrap(n100_path)
    if n100_rep:
        reports["n=100 (simple arith)"] = n100_rep
    else:
        print("  WARN: no n=100 bootstrap found")

    gms8k_path = find_bootstrap("_h10_n20_gsm8k_bootstrap.json")
    gms8k_rep = load_bootstrap(gms8k_path)
    if gms8k_rep:
        reports["n=20 (GSM8K 200-token)"] = gms8k_rep
    else:
        print("  WARN: no GSM8K bootstrap found -- run aggregator first")

    if gms8k_rep:
        make_forest(gms8k_rep)
    else:
        print("  Skipping forest plot (no GSM8K bootstrap)")

    make_shrinkage_timeline(reports)
    print("=== Done. ===")


if __name__ == "__main__":
    main()
