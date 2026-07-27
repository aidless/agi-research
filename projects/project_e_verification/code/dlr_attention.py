"""dlr_attention.py - DLR with attention-based slot aggregation.

This is a fix for the dlr_train_full.py failure mode: the random projection
from observation to slot features loses angular information, so the `upright`
predicate (which depends on angle) cannot be learned from the random projection.

Solution: instead of using a FIXED random projection, learn a projection that
is jointly optimized with the predicate networks. Additionally, use
attention-based aggregation over slots instead of mean.

Architecture:
  - obs_proj: small MLP that maps 8-dim obs to (n_slots, slot_dim)
  - predicate_net: per-predicate MLP with attention over slots

For each predicate, the slot-level truth values are computed and then
aggregated via learned attention weights.
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
from envs import make_env
from differentiable_logic import SoftLogic


# Ground-truth predicate evaluators (same as dlr_train_full.py) ---------------

def gt_landed(obs):
    return float((obs[6] >= 0.5 or obs[7] >= 0.5) and abs(obs[4]) < 0.3)

def gt_upright(obs):
    return float(abs(obs[4]) < 0.2)

def gt_leg_l_contact(obs):
    return float(obs[6] >= 0.5)

def gt_leg_r_contact(obs):
    return float(obs[7] >= 0.5)

def gt_in_pad(obs):
    return float(abs(obs[0]) < 0.3 and abs(obs[1]) < 0.2)

def gt_low_velocity(obs):
    return float(np.sqrt(obs[2]**2 + obs[3]**2) < 0.3)

def gt_safe_approach(obs):
    return float(0.0 < obs[1] < 1.5 and abs(obs[0]) < 0.5)


# Learned obs -> slots projection ---------------------------------------------

class ObsToSlots(nn.Module):
    """Learned projection from 8-dim obs to (n_slots, slot_dim).

    Each slot gets a different linear combination of obs features. This is
    much more expressive than a single random projection.
    """
    def __init__(self, obs_dim=8, n_slots=4, slot_dim=32, hidden=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_slots * slot_dim),
        )
        # Initialize to give each slot a distinct linear projection
        with torch.no_grad():
            # Make the linear projection roughly diagonal-ish: each slot focuses
            # on a different obs feature first
            self.shared[2].weight.data = torch.zeros(n_slots * slot_dim, hidden)
            for k in range(n_slots):
                for d in range(slot_dim):
                    obs_idx = (k * slot_dim + d) % obs_dim
                    self.shared[2].weight.data[k * slot_dim + d, obs_idx % hidden] = 0.1
            self.shared[2].bias.data.zero_()

    def forward(self, obs):
        flat = self.shared(obs)
        return flat.reshape(-1, self.n_slots, self.slot_dim)


# Predicate network with attention over slots ---------------------------------

class AttnSlotPredicateNet(nn.Module):
    """Per-predicate net with attention-based slot aggregation.

    For each predicate:
      1. Compute per-slot truth value: slot -> MLP -> scalar in [0, 1]
      2. Compute attention weights over slots (via learned query)
      3. Aggregate: weighted average of per-slot truth values
    """
    def __init__(self, slot_dim, hidden=32):
        super().__init__()
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Attention query (learned per predicate)
        self.attn_query = nn.Parameter(torch.randn(slot_dim) * 0.1)

    def forward(self, slots):
        # slots: (B, n_slots, slot_dim)
        per_slot = torch.sigmoid(self.slot_mlp(slots).squeeze(-1))  # (B, n_slots)
        # Attention: dot product of slots with learned query
        attn_logits = (slots * self.attn_query.view(1, 1, -1)).sum(dim=-1)  # (B, n_slots)
        attn = F.softmax(attn_logits, dim=-1)
        # Weighted aggregation
        out = (per_slot * attn).sum(dim=-1)
        return out


# Training loop ---------------------------------------------------------------

def collect_dataset(env_name, n_episodes, seed):
    """Collect obs vectors and ground-truth labels."""
    all_obs = []
    all_labels = {k: [] for k in [
        "landed", "upright", "leg_l_contact", "leg_r_contact",
        "in_pad", "low_velocity", "safe_approach",
    ]}
    gt_funcs = {
        "landed": gt_landed, "upright": gt_upright,
        "leg_l_contact": gt_leg_l_contact, "leg_r_contact": gt_leg_r_contact,
        "in_pad": gt_in_pad, "low_velocity": gt_low_velocity,
        "safe_approach": gt_safe_approach,
    }
    for ep in range(n_episodes):
        env = make_env(env_name, seed=seed * 1000 + ep + 1)
        obs, _ = env.reset()
        for t in range(500):
            all_obs.append(obs.copy())
            for name, fn in gt_funcs.items():
                all_labels[name].append(fn(obs))
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        env.close()
    X = torch.from_numpy(np.stack(all_obs)).float()
    Y = {k: torch.tensor(v, dtype=torch.float32) for k, v in all_labels.items()}
    return X, Y


def train_jointly(X_train, Y_train, n_epochs=30, lr=1e-3, batch_size=128,
                  n_slots=4, slot_dim=32, hidden=64):
    """Train projection + predicate nets jointly with BCE loss."""
    obs_proj = ObsToSlots(obs_dim=8, n_slots=n_slots, slot_dim=slot_dim, hidden=hidden)
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


def evaluate_attn(obs_proj, predicate_nets, X_test, Y_test, n_slots=4):
    """Evaluate attention-DLR predicate accuracy."""
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
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-episodes", type=int, default=30)
    p.add_argument("--n-test-episodes", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    print("=" * 60)
    print("DLR with Attention-based Slot Aggregation")
    print("=" * 60)

    print("Phase 1: collect train data...")
    X_train, Y_train = collect_dataset(args.env, args.n_train_episodes, args.seed)
    print(f"  {X_train.shape[0]} timesteps from {args.n_train_episodes} episodes")

    print("Phase 2: train jointly (projection + predicates)...")
    obs_proj, predicate_nets, losses = train_jointly(
        X_train, Y_train,
        n_epochs=args.n_epochs, lr=1e-3, batch_size=args.batch_size,
        n_slots=args.n_slots, slot_dim=args.slot_dim, hidden=args.hidden,
    )
    final_losses = {name: float(loss[-1]) for name, loss in losses.items()}
    print(f"  Final BCE losses: {final_losses}")

    print("Phase 3: collect test data...")
    X_test, Y_test = collect_dataset(args.env, args.n_test_episodes, args.seed + 99999)

    print("Phase 4: evaluate...")
    accuracies, briers = evaluate_attn(obs_proj, predicate_nets, X_test, Y_test)
    print(f"  Accuracies: {accuracies}")
    print(f"  Briers: {briers}")

    # Save log
    log = {
        "env": args.env, "seed": args.seed,
        "mode": "DLR with attention-based slot aggregation",
        "n_train_episodes": args.n_train_episodes,
        "n_test_episodes": args.n_test_episodes,
        "n_epochs": args.n_epochs,
        "final_predicate_losses": final_losses,
        "predicate_accuracies": accuracies,
        "predicate_briers": briers,
    }
    log_path = HERE / "checkpoints" / "dlr_attention" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
