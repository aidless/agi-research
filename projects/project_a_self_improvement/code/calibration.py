"""calibration.py - DEC-0011 v0.2 calibration utilities.

Provides:
  - compute_auroc(y_true, y_score): rank-based AUROC
  - platt_fit(scores, labels): fit 1-param logistic on raw scores
  - platt_apply(score, a, b): apply Platt scaling
  - find_threshold_for_fpr(scores, labels, target_fpr): pick threshold
    such that false-positive rate is at most target_fpr on a labeled set
  - count_unique_sa_pairs(episodes): for Q coverage guard
"""
from __future__ import annotations
import numpy as np


def compute_auroc(y_true, y_score) -> float:
    """Rank-based AUROC. y_true in {0, 1}, y_score real. Returns 0.5 for ties."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    # Concorde / Mann-Whitney U statistic
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and all_scores[order[j + 1]] == all_scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def platt_fit(scores, labels, max_iter: int = 200, lr: float = 0.1) -> tuple[float, float]:
    """Fit 1-param Platt logistic: p_cal = sigmoid(a * logit(p_raw) + b).

    Uses a tiny torch-free Newton-style update. Returns (a, b).
    Initial guess: a=1, b=0 (identity).
    """
    import math
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    eps = 1e-6
    s = np.clip(s, eps, 1.0 - eps)
    z = np.log(s / (1.0 - s))  # logits
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        z_cal = a * z + b
        p = 1.0 / (1.0 + np.exp(-z_cal))
        # gradient of NLL w.r.t. (a, b)
        r = p - y
        ga = float(np.mean(r * z))
        gb = float(np.mean(r))
        # Hessian diagonal (approximate, ignoring z^2 cross)
        w = p * (1.0 - p)
        haa = float(np.mean(w * z * z)) + 1e-6
        hbb = float(np.mean(w)) + 1e-6
        step_a = ga / haa
        step_b = gb / hbb
        a -= lr * step_a
        b -= lr * step_b
        if abs(ga) < 1e-6 and abs(gb) < 1e-6:
            break
    return float(a), float(b)


def platt_apply(scores, a, b):
    """Apply Platt scaling: return sigmoid(a*logit(score)+b)."""
    s = np.asarray(scores, dtype=np.float64)
    eps = 1e-6
    s = np.clip(s, eps, 1.0 - eps)
    z = np.log(s / (1.0 - s))
    z_cal = a * z + b
    return 1.0 / (1.0 + np.exp(-z_cal))


def find_threshold_for_fpr(scores, labels, target_fpr: float):
    """Find the smallest threshold th such that FPR(th) <= target_fpr
    on the (scores, labels) set.

    FPR = #{negatives scored >= th} / #{negatives}
    We return threshold with FPR closest to but <= target_fpr.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    neg = s[y == 0]
    if len(neg) == 0:
        return 1.1  # no negatives, never fire
    # candidate thresholds = unique scores
    cands = np.unique(s)
    best_th = float("inf")
    best_diff = float("inf")
    for th in cands:
        fpr = float((neg >= th).sum() / len(neg))
        diff = target_fpr - fpr
        if diff >= 0 and diff < best_diff:
            best_diff = diff
            best_th = float(th)
    if best_th == float("inf"):
        # even the lowest threshold exceeds target_fpr (too many positives)
        # fall back to max score + eps (never fire)
        best_th = float(s.max() + 1e-3)
    return best_th


def count_unique_sa_pairs(episodes, obs_dim: int, n_actions: int) -> int:
    """Count unique (state-discretized, action) pairs across all transitions.

    Discretization: round each obs coord to 2 decimals to bin similar states.
    """
    seen = set()
    for ep in episodes:
        for tr in ep.transitions:
            key = (tuple(np.round(tr.obs, 2).tolist()), int(tr.action))
            seen.add(key)
    return len(seen)
