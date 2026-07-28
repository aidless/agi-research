"""
Project G -- H10 smoke test (architecture validation only).

This file validates the architecture end-to-end on SYNTHETIC LLM
traces. It is NOT the H10 experiment. The H10 experiment (real LLM
rollouts) is defined in experiments_log/2026-07-28-PRE-REGISTERED-H10.md
and will be run AFTER the user has chosen a frozen LLM.

What this smoke test does:
1. Generate 200 synthetic LLM traces (with a known failure signal).
2. Train the LLMSlotMonitor for 50 epochs.
3. Evaluate on a 50-trace held-out set.
4. Report AUROC, accuracy.

Expected smoke-test AUROC: > 0.7 on the synthetic signal.
If AUROC is ~0.5 (random): the architecture has a bug.
If AUROC is > 0.9: the synthetic signal may be too easy.

NO_SELF_DECEPTION.md compliance:
- Smoke test result is NOT a "headline result". It is an architecture
  validation.
- Smoke test result does NOT count toward H10. The H10 verdict comes
  only from the real-LLM experiment.
- The decision rule for H10 is unchanged regardless of smoke-test
  outcome.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from llm_monitor import LLMSlotMonitor
from frozen_rollout_collector import collect_synthetic_rollouts


def compute_auroc(scores, labels):
    """Compute AUROC using the trapezoidal rule.

    scores: (N,) tensor of predicted failure probabilities
    labels: (N,) tensor of 0/1 failure labels
    """
    scores = scores.detach().numpy()
    labels = labels.detach().numpy()
    # Sort by score descending.
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
            # Trapezoid area: extend horizontally until the next label.
            auc += tp / P * (fp - prev_fp) / N
            prev_fp = fp
    return auc


def train_one_epoch(monitor, optimizer, feats, labels, batch_size=32):
    monitor.train()
    perm = torch.randperm(feats.size(0))
    total_loss = 0.0
    n_batches = 0
    for i in range(0, feats.size(0), batch_size):
        idx = perm[i:i + batch_size]
        x = feats[idx]
        y = labels[idx]
        optimizer.zero_grad()
        pred = monitor(x)
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(monitor, feats, labels):
    monitor.eval()
    with torch.no_grad():
        pred = monitor(feats)
        auroc = compute_auroc(pred, labels)
        acc = ((pred > 0.5).float() == labels).float().mean().item()
    return auroc, acc


def main():
    print("=" * 60)
    print("Project G H10 smoke test (architecture validation)")
    print("=" * 60)
    print("NOTE: synthetic data, NOT the real H10 experiment.")
    print()

    seed = 42
    torch.manual_seed(seed)

    # 200 traces, 80/20 train/eval split.
    all_feats, all_labels = collect_synthetic_rollouts(n_rollouts=200, seed=seed)
    n_train = 160
    train_feats = all_feats[:n_train]
    train_labels = all_labels[:n_train]
    eval_feats = all_feats[n_train:]
    eval_labels = all_labels[n_train:]

    monitor = LLMSlotMonitor()
    optimizer = torch.optim.Adam(monitor.parameters(), lr=1e-3)

    print(f"Train: {tuple(train_feats.shape)}, eval: {tuple(eval_feats.shape)}")
    print(f"Train failure rate: {train_labels.mean().item():.3f}")
    print(f"Eval failure rate:  {eval_labels.mean().item():.3f}")
    print()

    for epoch in range(1, 51):
        loss = train_one_epoch(monitor, optimizer, train_feats, train_labels)
        if epoch % 10 == 0:
            auroc, acc = evaluate(monitor, eval_feats, eval_labels)
            print(f"Epoch {epoch:3d}: loss={loss:.4f} eval_AUROC={auroc:.3f} eval_acc={acc:.3f}")

    final_auroc, final_acc = evaluate(monitor, eval_feats, eval_labels)
    print()
    print("=" * 60)
    print(f"Final: AUROC={final_auroc:.3f}, accuracy={final_acc:.3f}")
    print()
    if final_auroc < 0.55:
        print("VERDICT: Architecture has a bug -- AUROC near random.")
        print("  Action: debug slot attention / feature pipeline.")
    elif final_auroc > 0.95:
        print("VERDICT: Synthetic signal too easy -- AUROC suspiciously high.")
        print("  Action: confirm this is not data leakage before real H10.")
    else:
        print("VERDICT: Architecture validates on synthetic signal.")
        print("  Action: ready for real H10 (frozen LLM rollouts).")
    print("=" * 60)


if __name__ == "__main__":
    main()