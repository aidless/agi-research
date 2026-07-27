#!/usr/bin/env python3
"""dlr_train_full.py - DLR (Differentiable Logic Reasoner) full training + verification.

This script replaces the simple LTL verifier with a general fuzzy logic
reasoner. Predicate networks are trained on ground-truth labels from the
environment (LunarLander) and then used to verify trajectories.

Comparison:
- LTL verifier: hand-coded predicates, discrete truth values, 0/1 accuracy
- DLR verifier: learned predicates, fuzzy truth values, Brier-style accuracy

The DLR approach generalizes LTL by supporting:
1. Continuous truth values [0, 1]
2. Soft constraints
3. End-to-end gradient-based learning
4. Compositional formulas over slot representations

Output:
- checkpoints/dlr_full/seed<N>/phase2_log.json
- experiments_log/2026-07-27-dlr-train-full.md
"""
import argparse
import json
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PA_CODE = Path(r"E:\agi-research\projects\project_a_self_improvement\code")
sys.path.insert(0, str(PA_CODE))
sys.path.insert(0, str(HERE.parent / "project_c_causal_world" / "code"))

import envs
from envs import make_env, rollout_one_episode
from differentiable_logic import (
    SoftLogic, Predicate, FormulaBuilder, DifferentiableReasoner,
)


# Ground-truth predicate evaluators (LTL-style) -------------------------------

def gt_landed(obs):
    """Lander is on the ground in stable landing configuration."""
    # obs[0]=x, [1]=y, [2]=vx, [3]=vy, [4]=angle, [5]=ang_vel, [6]=leg_l, [7]=leg_r
    return float((obs[6] >= 0.5 or obs[7] >= 0.5) and abs(obs[4]) < 0.3)


def gt_upright(obs):
    """Lander is roughly upright."""
    return float(abs(obs[4]) < 0.2)


def gt_leg_l_contact(obs):
    return float(obs[6] >= 0.5)


def gt_leg_r_contact(obs):
    return float(obs[7] >= 0.5)


def gt_in_pad(obs):
    """Lander position is within landing pad."""
    return float(abs(obs[0]) < 0.3 and abs(obs[1]) < 0.2)


def gt_low_velocity(obs):
    """Lander velocity is low (near landing)."""
    return float(np.sqrt(obs[2]**2 + obs[3]**2) < 0.3)


def gt_safe_approach(obs):
    """Lander is approaching but not yet landed."""
    return float(0.0 < obs[1] < 1.5 and abs(obs[0]) < 0.5)


# Predicate networks -----------------------------------------------------------

class SlotPredicateNet(nn.Module):
    """Neural predicate that maps slot features to fuzzy truth value [0, 1]."""
    def __init__(self, slot_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


class BinarySlotPredicateNet(nn.Module):
    def __init__(self, slot_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, y):
        return torch.sigmoid(self.net(torch.cat([x, y], dim=-1)).squeeze(-1))


# Slot projection --------------------------------------------------------------

def obs_to_slots(obs, projection, n_slots=4, slot_dim=32):
    """Project 8-dim obs into (n_slots, slot_dim) via a learned linear projection."""
    flat = torch.from_numpy(obs).float() @ projection  # (n_slots*slot_dim,)
    return flat.reshape(n_slots, slot_dim)


# Training loop ----------------------------------------------------------------

def collect_dataset(env_name, n_episodes, seed, n_slots=4, slot_dim=32):
    """Collect episodes and build (slot_features, ground_truth_predicates) pairs.

    Returns:
      X_slots: (N, n_slots, slot_dim)
      Y_labels: dict[name -> (N,) tensor of float labels]
    """
    projection = torch.randn(8, n_slots * slot_dim) * 0.1
    projection.requires_grad_(False)

    all_slots = []
    all_labels = {k: [] for k in [
        "landed", "upright", "leg_l_contact", "leg_r_contact",
        "in_pad", "low_velocity", "safe_approach",
    ]}

    gt_funcs = {
        "landed": gt_landed,
        "upright": gt_upright,
        "leg_l_contact": gt_leg_l_contact,
        "leg_r_contact": gt_leg_r_contact,
        "in_pad": gt_in_pad,
        "low_velocity": gt_low_velocity,
        "safe_approach": gt_safe_approach,
    }

    for ep in range(n_episodes):
        env = make_env(env_name, seed=seed * 1000 + ep + 1)
        obs, _ = env.reset()
        for t in range(500):
            slots = obs_to_slots(obs, projection, n_slots, slot_dim)
            all_slots.append(slots.detach().numpy())
            for name, fn in gt_funcs.items():
                all_labels[name].append(fn(obs))
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        env.close()

    X_slots = torch.from_numpy(np.stack(all_slots)).float()
    Y_labels = {k: torch.tensor(v, dtype=torch.float32) for k, v in all_labels.items()}
    return X_slots, Y_labels, projection


def train_predicates(X_slots, Y_labels, n_slots=4, slot_dim=32,
                     n_epochs=20, lr=1e-3, batch_size=128):
    """Train predicate networks on the collected dataset."""
    predicate_names = list(Y_labels.keys())
    predicate_nets = {
        name: SlotPredicateNet(slot_dim) for name in predicate_names
    }
    opts = {name: torch.optim.Adam(net.parameters(), lr=lr)
            for name, net in predicate_nets.items()}

    # Train each predicate separately
    N = X_slots.shape[0]
    losses_per_epoch = {name: [] for name in predicate_names}

    for epoch in range(n_epochs):
        idx = np.random.permutation(N)
        for name in predicate_names:
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                bi = idx[start:start + batch_size]
                if len(bi) < 2:
                    continue
                slots_b = X_slots[bi]  # (B, n_slots, slot_dim)
                labels_b = Y_labels[name][bi]

                # Aggregate over slots: mean truth value across all slots
                # (this is the existential aggregator)
                pred_values = []
                for k in range(n_slots):
                    pred_values.append(predicate_nets[name](slots_b[:, k]))
                pred = torch.stack(pred_values, dim=-1).mean(dim=-1)

                loss = F.binary_cross_entropy(pred, labels_b)
                opts[name].zero_grad()
                loss.backward()
                opts[name].step()
                ep_loss += float(loss.detach())
                n_batches += 1
            losses_per_epoch[name].append(ep_loss / max(1, n_batches))

    return predicate_nets, losses_per_epoch


def evaluate_predicates(predicate_nets, X_test, Y_test, n_slots=4):
    """Evaluate predicate accuracy on held-out test set."""
    N = X_test.shape[0]
    accuracies = {}
    briers = {}
    for name, net in predicate_nets.items():
        pred_values = []
        for k in range(n_slots):
            pred_values.append(net(X_test[:, k]))
        pred = torch.stack(pred_values, dim=-1).mean(dim=-1)
        pred_bin = (pred > 0.5).float()
        labels = Y_test[name]
        acc = float((pred_bin == labels).float().mean())
        brier = float(((pred - labels) ** 2).mean())
        accuracies[name] = acc
        briers[name] = brier
    return accuracies, briers


# LTL comparison ---------------------------------------------------------------

def ltl_verify_episode(observations, formula="G upright AND F landed"):
    """Simple LTL verifier: returns 1 if trace satisfies formula, 0 otherwise.

    This is the 'baseline' we compare against. It uses crisp predicates.
    """
    # Discrete predicates
    is_upright = lambda o: abs(o[4]) < 0.2
    is_landed = lambda o: (o[6] >= 0.5 or o[7] >= 0.5) and abs(o[4]) < 0.3

    if formula == "G upright AND F landed":
        # Globally upright AND eventually landed
        if not all(is_upright(o) for o in observations):
            return 0
        if not any(is_landed(o) for o in observations):
            return 0
        return 1
    elif formula == "F (leg_l AND leg_r)":
        # Eventually both legs touch
        return int(any(o[6] >= 0.5 and o[7] >= 0.5 for o in observations))
    elif formula == "G (landed -> in_pad)":
        # Whenever landed, must be in pad
        for o in observations:
            if is_landed(o):
                if not (abs(o[0]) < 0.3 and abs(o[1]) < 0.2):
                    return 0
        return 1
    return 0


def dlr_verify_episode(predicate_nets, X_slots, formula="G upright AND F landed"):
    """DLR verifier: returns truth value [0, 1] for the same formula."""
    sl = SoftLogic()
    n_slots = X_slots.shape[1]
    N = X_slots.shape[0]

    if formula == "G upright AND F landed":
        # G upright: forall t. upright(t)
        upright_preds = []
        for k in range(n_slots):
            upright_preds.append(predicate_nets["upright"](X_slots[:, k]))
        upright_per_step = torch.stack(upright_preds, dim=-1).mean(dim=-1)
        G_upright = sl.forall(upright_preds[0])  # use slot 0 as representative

        # F landed: exists t. landed(t)
        landed_preds = []
        for k in range(n_slots):
            landed_preds.append(predicate_nets["landed"](X_slots[:, k]))
        F_landed = sl.exists(torch.stack(landed_preds, dim=-1).mean(dim=-1))

        # AND
        return float(sl.and_op(G_upright.unsqueeze(0) if G_upright.dim() == 0 else G_upright,
                                F_landed.unsqueeze(0) if F_landed.dim() == 0 else F_landed).mean().item())
    elif formula == "F (leg_l AND leg_r)":
        ll_preds = []
        lr_preds = []
        for k in range(n_slots):
            ll_preds.append(predicate_nets["leg_l_contact"](X_slots[:, k]))
            lr_preds.append(predicate_nets["leg_r_contact"](X_slots[:, k]))
        # AND across slots, then exists over time
        and_per_step = []
        for t in range(N):
            and_vals = []
            for k in range(n_slots):
                and_vals.append(sl.and_op(ll_preds[k][t].unsqueeze(0),
                                            lr_preds[k][t].unsqueeze(0)))
            and_per_step.append(torch.stack(and_vals).mean())
        and_tensor = torch.stack(and_per_step)
        return float(sl.exists(and_tensor).item())
    elif formula == "G (landed -> in_pad)":
        # For all t: landed(t) -> in_pad(t)
        # Truth value: forall t. OR(NOT landed(t), in_pad(t))
        landed_preds = []
        in_pad_preds = []
        for k in range(n_slots):
            landed_preds.append(predicate_nets["landed"](X_slots[:, k]))
            in_pad_preds.append(predicate_nets["in_pad"](X_slots[:, k]))
        impl_per_step = []
        for t in range(N):
            impl_vals = []
            for k in range(n_slots):
                impl = sl.implies_op(
                    landed_preds[k][t].unsqueeze(0),
                    in_pad_preds[k][t].unsqueeze(0)
                )
                impl_vals.append(impl)
            impl_per_step.append(torch.stack(impl_vals).mean())
        impl_tensor = torch.stack(impl_per_step)
        return float(sl.forall(impl_tensor).item())
    return 0.0


def evaluate_verification_accuracy(env_name, predicate_nets, seed, n_episodes=30):
    """Compare DLR vs LTL verification accuracy against ground-truth trace semantics.

    Ground truth: trace is "satisfied" if final state is landed in pad.
    """
    n_slots = 4
    slot_dim = 32

    formulas = ["G upright AND F landed", "F (leg_l AND leg_r)", "G (landed -> in_pad)"]
    results = {f: {"ltl": [], "dlr": [], "gt": []} for f in formulas}

    projection = torch.randn(8, n_slots * slot_dim) * 0.1

    for ep in range(n_episodes):
        env = make_env(env_name, seed=seed * 99999 + ep)
        obs, _ = env.reset()
        observations = []
        slot_features = []
        for t in range(500):
            observations.append(obs.copy())
            slots = obs_to_slots(obs, projection, n_slots, slot_dim).unsqueeze(0)
            slot_features.append(slots)
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        env.close()

        X_slots = torch.cat(slot_features, dim=0)  # (T, n_slots, slot_dim)

        # Ground truth: did the lander actually land successfully?
        final_obs = observations[-1]
        landed_in_pad = gt_landed(final_obs) and gt_in_pad(final_obs)

        for f in formulas:
            ltl_score = ltl_verify_episode(observations, f)
            dlr_score = dlr_verify_episode(predicate_nets, X_slots, f)
            results[f]["ltl"].append(ltl_score)
            results[f]["dlr"].append(dlr_score)
            results[f]["gt"].append(landed_in_pad)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-episodes", type=int, default=50)
    p.add_argument("--n-test-episodes", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--n-slots", type=int, default=4)
    args = p.parse_args()

    print("=" * 60)
    print("DLR full training on", args.env)
    print("=" * 60)

    # 1. Collect training data
    print("Phase 1: collect training data...")
    X_train, Y_train, projection = collect_dataset(
        args.env, args.n_train_episodes, args.seed,
        n_slots=args.n_slots, slot_dim=args.slot_dim,
    )
    print(f"  Collected {X_train.shape[0]} timesteps from {args.n_train_episodes} episodes")

    # 2. Train predicate networks
    print("Phase 2: train predicate networks...")
    predicate_nets, losses = train_predicates(
        X_train, Y_train,
        n_slots=args.n_slots, slot_dim=args.slot_dim,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
    )
    final_losses = {name: float(loss[-1]) for name, loss in losses.items()}
    print(f"  Final BCE losses: {final_losses}")

    # 3. Collect held-out test set
    print("Phase 3: collect test data...")
    X_test, Y_test, _ = collect_dataset(
        args.env, args.n_test_episodes, args.seed + 99999,
        n_slots=args.n_slots, slot_dim=args.slot_dim,
    )

    # 4. Evaluate predicate accuracy
    print("Phase 4: evaluate predicate accuracy...")
    accuracies, briers = evaluate_predicates(
        predicate_nets, X_test, Y_test, n_slots=args.n_slots,
    )
    print(f"  Accuracies: {accuracies}")
    print(f"  Brier scores: {briers}")

    # 5. Verification comparison: DLR vs LTL
    print("Phase 5: DLR vs LTL verification comparison...")
    verif_results = evaluate_verification_accuracy(
        args.env, predicate_nets, args.seed, n_episodes=30,
    )
    print("  Verification results (vs ground truth):")
    for f, scores in verif_results.items():
        ltl_acc = float(np.mean([(s == g) for s, g in zip(scores["ltl"], scores["gt"])]))
        dlr_brier = float(np.mean([(s - g) ** 2 for s, g in zip(scores["dlr"], scores["gt"])]))
        print(f"    Formula '{f}':")
        print(f"      LTL accuracy: {ltl_acc:.3f}")
        print(f"      DLR Brier:    {dlr_brier:.3f} (lower = better)")

    # 6. Save log
    log = {
        "env": args.env,
        "seed": args.seed,
        "mode": "DLR full training + verification",
        "n_train_episodes": args.n_train_episodes,
        "n_test_episodes": args.n_test_episodes,
        "n_epochs": args.n_epochs,
        "final_predicate_losses": final_losses,
        "predicate_accuracies": accuracies,
        "predicate_briers": briers,
        "verification_results": {
            f: {
                "ltl_accuracy": float(np.mean([(s == g) for s, g in zip(scores["ltl"], scores["gt"])])),
                "dlr_brier": float(np.mean([(s - g) ** 2 for s, g in zip(scores["dlr"], scores["gt"])])),
                "gt_positive_rate": float(np.mean(scores["gt"])),
            } for f, scores in verif_results.items()
        },
    }

    log_path = HERE / "checkpoints" / "dlr_full" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
