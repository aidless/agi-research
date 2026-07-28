"""
Project G -- Multi-arm smoke test (frozen / joint / random).

This is the smoke-test validation for the H10 architecture. It
trains all three arms (frozen Monitor, joint Monitor, random
Monitor negative control) and reports per-arm AUROC.

Per H10 pre-registration:
- Frozen Monitor AUROC should be higher than Joint Monitor AUROC
- Frozen Monitor AUROC should be higher than Random Monitor AUROC
- Joint Monitor may or may not be higher than Random Monitor

This file uses SYNTHETIC data (not real LLM rollouts) for the
smoke test. The real H10 experiment is in the real-LLM experiment
(future work).

NO_SELF_DECEPTION.md compliance:
- Smoke test result is NOT a "headline result". It is an architecture
  validation across 3 arms.
- The synthetic data has a known signal-to-noise ratio.
- The decision rule for H10 is unchanged regardless of smoke-test
  outcome.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from llm_monitor import LLMSlotMonitor
from frozen_rollout_collector import collect_synthetic_rollouts
from joint_monitor import train_frozen_monitor, train_joint_monitor


def compute_auroc(scores, labels):
    """Compute AUROC using the trapezoidal rule."""
    scores = scores.detach().numpy()
    labels = labels.detach().numpy()
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    for lab in labels_sorted:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            auc += tp / P * (fp - prev_fp) / N
            prev_fp = fp
    return auc


def random_monitor_auroc(eval_feats, eval_labels, seed=0):
    """Random Monitor: untrained, U[0,1] signal as failure probability."""
    rng = torch.Generator().manual_seed(seed)
    scores = torch.rand(eval_feats.size(0), generator=rng)
    return compute_auroc(scores, eval_labels)


def run_one_arm(arm_name, train_fn, monitor, optimizer, train_feats,
                train_labels, eval_feats, eval_labels, seed):
    """Train one arm and return AUROC."""
    if arm_name == "Random":
        return random_monitor_auroc(eval_feats, eval_labels, seed=seed)
    monitor, _ = train_fn(monitor, optimizer, train_feats, train_labels, seed=seed)
    monitor.eval()
    with torch.no_grad():
        pred = monitor(eval_feats)
    return compute_auroc(pred, eval_labels)


def main():
    print("=" * 70)
    print("Project G H10 multi-arm smoke test (3-arm architecture validation)")
    print("=" * 70)
    print("NOTE: synthetic data, NOT the real H10 experiment.")
    print()

    n_seeds = 5
    n_train = 160
    n_eval = 40

    results = {"Frozen": [], "Joint": [], "Random": []}
    per_seed_table = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        train_feats, train_labels = collect_synthetic_rollouts(
            n_rollouts=n_train, seed=seed * 1000
        )
        eval_feats, eval_labels = collect_synthetic_rollouts(
            n_rollouts=n_eval, seed=seed * 1000 + 999
        )

        # Frozen Monitor arm.
        torch.manual_seed(seed)
        frozen = LLMSlotMonitor()
        frozen_opt = torch.optim.Adam(frozen.parameters(), lr=1e-3)
        frozen_auroc = run_one_arm(
            "Frozen", train_frozen_monitor,
            frozen, frozen_opt, train_feats, train_labels,
            eval_feats, eval_labels, seed,
        )

        # Joint Monitor arm.
        torch.manual_seed(seed)
        joint = LLMSlotMonitor()
        joint_opt = torch.optim.Adam(joint.parameters(), lr=1e-3)
        joint_auroc = run_one_arm(
            "Joint", train_joint_monitor,
            joint, joint_opt, train_feats, train_labels,
            eval_feats, eval_labels, seed,
        )

        # Random Monitor arm (negative control).
        random_auroc = run_one_arm(
            "Random", None, None, None,
            train_feats, train_labels, eval_feats, eval_labels, seed,
        )

        results["Frozen"].append(frozen_auroc)
        results["Joint"].append(joint_auroc)
        results["Random"].append(random_auroc)

        per_seed_table.append(
            f"  Seed {seed}: Frozen={frozen_auroc:.3f}  "
            f"Joint={joint_auroc:.3f}  Random={random_auroc:.3f}  "
            f"Delta_F-J={frozen_auroc - joint_auroc:+.3f}"
        )

    print("Per-seed results:")
    for line in per_seed_table:
        print(line)
    print()

    print("Aggregate (mean +/- std, n=5):")
    for arm in ["Frozen", "Joint", "Random"]:
        vals = results[arm]
        print(f"  {arm:8s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    print()

    # Decision rule check (smoke-test version, NOT H10 pre-reg).
    f_mean = np.mean(results["Frozen"])
    j_mean = np.mean(results["Joint"])
    r_mean = np.mean(results["Random"])

    print("Smoke-test verdict:")
    if f_mean > j_mean + 0.05:
        print(f"  Frozen > Joint by {f_mean - j_mean:.3f} (consistent with H10)")
    elif j_mean > f_mean + 0.05:
        print(f"  Joint > Frozen by {j_mean - f_mean:.3f} (contradicts H10)")
    else:
        print(f"  Frozen ~ Joint (delta={f_mean - j_mean:+.3f}, inconclusive)")
    if f_mean > r_mean + 0.10:
        print(f"  Frozen > Random by {f_mean - r_mean:.3f} (negative control passes)")
    else:
        print(f"  Frozen ~ Random (delta={f_mean - r_mean:+.3f}, NEGATIVE CONTROL FAILS)")
    print()
    print("=" * 70)
    print("REMINDER: This is on SYNTHETIC data. Real H10 needs a frozen LM.")
    print("=" * 70)


if __name__ == "__main__":
    main()