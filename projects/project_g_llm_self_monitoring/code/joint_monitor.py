"""
Project G -- Joint Monitor training (H10 control arm).

This file implements the JOINT Monitor training procedure, which is
the control arm in the H10 pre-registered experiment. The joint
Monitor is trained simultaneously with the LLM update; the data
distribution drifts as the LLM improves.

Mirrors the H1 ablation in Project A: the classical-RL "joint Monitor"
was trained jointly with PPO; here the "joint Monitor" is trained
jointly with a simulated LLM update.

NO_SELF_DECEPTION.md compliance:
- The joint Monitor is the CONTROL arm, not the test arm. Per H10
  pre-registration, the test arm is the FROZEN Monitor.
- A random Monitor signal is the NEGATIVE CONTROL (handled in
  h10_multi_arm_smoke.py, not here).
- The training procedure is symmetric to the frozen-Monitor
  procedure except for the LLM-update simulation.

The LLM update simulation: in the real H10 experiment, the LLM
weights would be updated by gradient descent on the LLM loss. For
the smoke test, we simulate this by perturbing the trace features
with a small noise vector at every "LLM step". This is a proxy for
the real LLM update; it captures the essence of "data distribution
drift" without requiring a real LM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_monitor import LLMSlotMonitor


def simulated_llm_step(trace_features, rng, perturb_scale=0.05):
    """Simulate one LLM weight update by perturbing trace features.

    Args:
        trace_features: (batch, window, feat_dim) tensor
        rng: torch.Generator for reproducibility
        perturb_scale: scale of the perturbation (proxy for LLM lr)

    Returns:
        perturbed_features: (batch, window, feat_dim) tensor
    """
    noise = torch.randn(trace_features.shape, generator=rng) * perturb_scale
    return trace_features + noise


def train_joint_monitor(monitor, optimizer, feats, labels,
                       n_llm_steps=10, perturb_scale=0.05,
                       n_monitor_epochs_per_llm_step=5,
                       batch_size=32, seed=0):
    """Train a Joint Monitor with simulated LLM co-training.

    Args:
        monitor: LLMSlotMonitor instance
        optimizer: torch optimizer for the Monitor
        feats: (n_rollouts, window, feat_dim) training features
        labels: (n_rollouts,) failure labels
        n_llm_steps: number of simulated LLM weight updates
        perturb_scale: scale of feature perturbation per LLM step
        n_monitor_epochs_per_llm_step: monitor epochs between LLM steps
        batch_size: monitor training batch size

    Returns:
        monitor: the trained monitor (in-place modification)
        losses: list of average losses per epoch
    """
    rng = torch.Generator().manual_seed(seed)
    current_feats = feats.clone()
    losses = []
    monitor.train()
    for llm_step in range(n_llm_steps):
        # Simulate LLM weight update by perturbing trace features.
        current_feats = simulated_llm_step(current_feats, rng, perturb_scale)
        # Train Monitor for n_monitor_epochs_per_llm_step epochs on
        # the perturbed features.
        for epoch in range(n_monitor_epochs_per_llm_step):
            perm = torch.randperm(current_feats.size(0), generator=rng)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, current_feats.size(0), batch_size):
                idx = perm[i:i + batch_size]
                x = current_feats[idx]
                y = labels[idx]
                optimizer.zero_grad()
                pred = monitor(x)
                loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            losses.append(total_loss / max(n_batches, 1))
    return monitor, losses


def train_frozen_monitor(monitor, optimizer, feats, labels,
                        n_epochs=50, batch_size=32, seed=0):
    """Train a Frozen Monitor (no LLM co-training).

    This is the test arm in H10. The Monitor is trained on the
    *original* features (no perturbation). The training distribution
    is stationary, mirroring the frozen-policy setup in Project A H1.

    Args:
        monitor: LLMSlotMonitor instance
        optimizer: torch optimizer for the Monitor
        feats: (n_rollouts, window, feat_dim) training features
        labels: (n_rollouts,) failure labels
        n_epochs: number of training epochs
        batch_size: monitor training batch size

    Returns:
        monitor: the trained monitor (in-place modification)
        losses: list of average losses per epoch
    """
    rng = torch.Generator().manual_seed(seed)
    losses = []
    monitor.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(feats.size(0), generator=rng)
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
        losses.append(total_loss / max(n_batches, 1))
    return monitor, losses


if __name__ == "__main__":
    # Smoke test: train frozen vs joint side by side, compare losses.
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from frozen_rollout_collector import collect_synthetic_rollouts
    from h10_smoke import compute_auroc

    feats, labels = collect_synthetic_rollouts(n_rollouts=160, seed=42)
    eval_feats, eval_labels = collect_synthetic_rollouts(n_rollouts=40, seed=99)

    # Frozen Monitor (test arm).
    torch.manual_seed(42)
    frozen = LLMSlotMonitor()
    frozen_opt = torch.optim.Adam(frozen.parameters(), lr=1e-3)
    frozen, frozen_losses = train_frozen_monitor(
        frozen, frozen_opt, feats, labels, n_epochs=50, batch_size=32, seed=42
    )
    frozen.eval()
    with torch.no_grad():
        frozen_pred = frozen(eval_feats)
    frozen_auroc = compute_auroc(frozen_pred, eval_labels)

    # Joint Monitor (control arm).
    torch.manual_seed(42)
    joint = LLMSlotMonitor()
    joint_opt = torch.optim.Adam(joint.parameters(), lr=1e-3)
    joint, joint_losses = train_joint_monitor(
        joint, joint_opt, feats, labels,
        n_llm_steps=10, perturb_scale=0.05,
        n_monitor_epochs_per_llm_step=5, batch_size=32, seed=42
    )
    joint.eval()
    with torch.no_grad():
        joint_pred = joint(eval_feats)
    joint_auroc = compute_auroc(joint_pred, eval_labels)

    print(f"Frozen Monitor: final loss={frozen_losses[-1]:.4f}, eval AUROC={frozen_auroc:.3f}")
    print(f"Joint Monitor:  final loss={joint_losses[-1]:.4f}, eval AUROC={joint_auroc:.3f}")
    print(f"Delta (frozen - joint): {frozen_auroc - joint_auroc:+.3f}")
    if frozen_auroc > joint_auroc + 0.05:
        print("VERDICT: Synthetic data shows frozen > joint (consistent with H10)")
    elif joint_auroc > frozen_auroc + 0.05:
        print("VERDICT: Synthetic data shows joint > frozen (contradicts H10)")
    else:
        print("VERDICT: Synthetic data shows frozen ~ joint (inconclusive)")
    print("NOTE: This is on synthetic data, NOT the real H10 experiment.")