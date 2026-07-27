"""dlr_cross_env.py - DLR predicates cross-environment test.

Tests whether the DLR-attention fix (95.5% accuracy on LunarLander)
generalizes to other environments.

Y1 Q1 priority: validate DLR predicates on CartPole-v1.

LunarLander predicates (7):
- landed, upright, leg_l_contact, leg_r_contact
- in_pad, low_velocity, safe_approach

CartPole predicates (4):
- upright (|angle| < 0.2)
- centered (|position| < 1.0)
- low_velocity (|velocity| < 1.0)
- low_ang_vel (|angular_velocity| < 1.0)

If DLR works on CartPole, it generalizes.
If not, it's LunarLander-specific.
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
sys.path.insert(0, str(HERE.parent.parent / "project_a_self_improvement" / "code"))

import envs
from envs import make_env
from differentiable_logic import SoftLogic
from dlr_attention import ObsToSlots, AttnSlotPredicateNet


# Environment-specific predicate definitions
def get_predicates(env_name):
    """Return ground-truth predicate functions for env."""
    if env_name == "CartPole-v1":
        obs_dim = 4
        n_actions = 2
        return {
            "upright": lambda obs: float(abs(obs[2]) < 0.2),  # angle
            "centered": lambda obs: float(abs(obs[0]) < 1.0),  # position
            "low_velocity": lambda obs: float(abs(obs[1]) < 1.0),  # cart velocity
            "low_ang_vel": lambda obs: float(abs(obs[3]) < 1.0),  # angular velocity
        }, obs_dim, n_actions
    elif env_name == "LunarLander-v3":
        obs_dim = 8
        n_actions = 4
        return {
            "landed": lambda obs: float((obs[6] >= 0.5 or obs[7] >= 0.5) and abs(obs[4]) < 0.3),
            "upright": lambda obs: float(abs(obs[4]) < 0.2),
            "leg_l_contact": lambda obs: float(obs[6] >= 0.5),
            "leg_r_contact": lambda obs: float(obs[7] >= 0.5),
            "in_pad": lambda obs: float(abs(obs[0]) < 0.3 and abs(obs[1]) < 0.2),
            "low_velocity": lambda obs: float(np.sqrt(obs[2]**2 + obs[3]**2) < 0.3),
            "safe_approach": lambda obs: float(0.0 < obs[1] < 1.5 and abs(obs[0]) < 0.5),
        }, obs_dim, n_actions
    elif env_name == "Acrobot-v1":
        obs_dim = 6
        n_actions = 3
        return {
            "joint1_up": lambda obs: float(obs[0] > 0),  # cos(theta1) > 0
            "joint2_up": lambda obs: float(obs[2] > 0),  # cos(theta2) > 0
            "low_ang_vel1": lambda obs: float(abs(obs[4]) < 1.0),
            "low_ang_vel2": lambda obs: float(abs(obs[5]) < 1.0),
            "near_goal": lambda obs: float(obs[0] > -0.5 and obs[2] > 0),
        }, obs_dim, n_actions
    else:
        raise ValueError(f"Unknown env: {env_name}")


def collect_dataset(env_name, n_episodes, seed):
    """Collect (obs, predicate labels) tuples."""
    predicates, obs_dim, n_actions = get_predicates(env_name)
    all_obs = []
    all_labels = {k: [] for k in predicates.keys()}

    for ep in range(n_episodes):
        env = make_env(env_name, seed=seed * 1000 + ep + 1)
        obs, _ = env.reset()
        for t in range(500):
            all_obs.append(obs.copy())
            for name, fn in predicates.items():
                all_labels[name].append(fn(obs))
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        env.close()

    X = torch.from_numpy(np.stack(all_obs)).float()
    Y = {k: torch.tensor(v, dtype=torch.float32) for k, v in all_labels.items()}
    return X, Y, obs_dim, n_actions, list(predicates.keys())


def train_dlr_jointly(X_train, Y_train, n_epochs=30, lr=1e-3, batch_size=128,
                     n_slots=4, slot_dim=32, hidden=64):
    """Train ObsToSlots + predicate nets jointly."""
    obs_proj = ObsToSlots(obs_dim=X_train.shape[1], n_slots=n_slots,
                           slot_dim=slot_dim, hidden=hidden)
    predicate_names = list(Y_train.keys())
    predicate_nets = nn.ModuleDict({
        name: AttnSlotPredicateNet(slot_dim) for name in predicate_names
    })
    params = list(obs_proj.parameters()) + list(predicate_nets.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    N = X_train.shape[0]
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
                obs_b = X_train[bi]
                labels_b = Y_train[name][bi]

                slots = obs_proj(obs_b)
                pred = predicate_nets[name](slots)
                loss = F.binary_cross_entropy(pred, labels_b)

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach())
                n_batches += 1
            losses_per_epoch[name].append(ep_loss / max(1, n_batches))

    return obs_proj, predicate_nets, losses_per_epoch


def evaluate(obs_proj, predicate_nets, X_test, Y_test):
    """Evaluate accuracy + Brier on each predicate."""
    with torch.no_grad():
        slots = obs_proj(X_test)
    accuracies = {}
    briers = {}
    for name, net in predicate_nets.items():
        pred = net(slots)
        pred_bin = (pred > 0.5).float()
        labels = Y_test[name]
        acc = float((pred_bin == labels).float().mean())
        brier = float(((pred - labels) ** 2).mean())
        accuracies[name] = acc
        briers[name] = brier
    return accuracies, briers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-episodes", type=int, default=30)
    p.add_argument("--n-test-episodes", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    args = p.parse_args()

    print("=" * 60)
    print(f"DLR cross-env: {args.env} seed {args.seed}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Collect train + test data
    print(f"\n[Phase 1] Collect {args.n_train_episodes} train episodes...")
    X_train, Y_train, obs_dim, n_actions, predicate_names = collect_dataset(
        args.env, args.n_train_episodes, args.seed,
    )
    print(f"  obs_dim={obs_dim}, n_actions={n_actions}, "
          f"predicates={len(predicate_names)}, samples={X_train.shape[0]}")

    print(f"\n[Phase 2] Collect {args.n_test_episodes} test episodes...")
    X_test, Y_test, _, _, _ = collect_dataset(
        args.env, args.n_test_episodes, args.seed + 99999,
    )

    # 2. Train
    print(f"\n[Phase 3] Train DLR (joint obs_proj + predicate nets)...")
    obs_proj, predicate_nets, losses = train_dlr_jointly(
        X_train, Y_train,
        n_epochs=args.n_epochs, lr=1e-3, batch_size=args.batch_size,
        n_slots=args.n_slots, slot_dim=args.slot_dim, hidden=64,
    )
    final_losses = {name: float(loss[-1]) for name, loss in losses.items()}
    print(f"  Final BCE losses: {final_losses}")

    # 3. Evaluate
    print(f"\n[Phase 4] Evaluate on test set...")
    accuracies, briers = evaluate(obs_proj, predicate_nets, X_test, Y_test)
    print(f"  Accuracies: {accuracies}")
    print(f"  Brier scores: {briers}")

    # 4. Summary
    print()
    print("=" * 60)
    print(f"DLR CROSS-ENV SUMMARY: {args.env} seed {args.seed}")
    print("=" * 60)
    print(f"  Mean accuracy: {np.mean(list(accuracies.values())):.3f}")
    print(f"  Mean Brier:    {np.mean(list(briers.values())):.3f}")
    print(f"  Per-predicate accuracy:")
    for name, acc in accuracies.items():
        print(f"    {name}: {acc:.3f}")

    # Save log
    log_path = HERE / "checkpoints" / "dlr_cross_env" / f"{args.env}_seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "DLR cross-env (joint obs_proj + predicates)",
        "n_train_episodes": args.n_train_episodes,
        "n_test_episodes": args.n_test_episodes,
        "n_epochs": args.n_epochs,
        "obs_dim": obs_dim, "n_actions": n_actions,
        "n_predicates": len(predicate_names),
        "final_losses": final_losses,
        "accuracies": accuracies,
        "briers": briers,
        "mean_accuracy": float(np.mean(list(accuracies.values()))),
        "mean_brier": float(np.mean(list(briers.values()))),
    }, indent=2))
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
