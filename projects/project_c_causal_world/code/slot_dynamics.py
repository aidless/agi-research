#!/usr/bin/env python3
"""slot_dynamics.py - Phase 1.2: Transformer dynamics on slot sequences.

Project C extension: turn slot-attention from perception-only into
a real world model by adding dynamics prediction.

Pipeline:
  trajectory -> per-step features -> [per-step encoder] -> slot sequence
  slot_t + action_t -> [dynamics transformer] -> predicted slot_{t+1}
  Loss: MSE on next slot state
"""
import argparse
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PC_CODE = Path(r"E:\agi-research\projects\project_c_causal_world\code")
sys.path.insert(0, str(PC_CODE))
PA_CODE = Path(r"E:\agi-research\projects\project_a_self_improvement\code")
sys.path.insert(0, str(PA_CODE))

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode


class PerStepEncoder(nn.Module):
    """Encode per-step features to a fixed dim."""
    def __init__(self, per_step_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(per_step_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
    def forward(self, x):
        return self.net(x)


class SlotDynamics(nn.Module):
    """Predict next slot state given current slot state + action.

    Per-slot MLP: each slot's next state is computed from its current
    state and the action (broadcast to all slots).
    """
    def __init__(self, slot_dim, n_actions, hidden=128):
        super().__init__()
        self.slot_dim = slot_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(slot_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, slot_dim),
        )

    def forward(self, slots, action):
        # slots: (batch, n_slots, slot_dim)
        # action: int (scalar action index)
        b, n_slots, _ = slots.shape
        a_onehot = F.one_hot(torch.tensor([action] * b), num_classes=self.n_actions).float()
        # broadcast: (b, n_slots, slot_dim + n_actions)
        a_expanded = a_onehot.unsqueeze(1).expand(-1, n_slots, -1)
        x = torch.cat([slots, a_expanded], dim=-1)
        return self.net(x)


class SlotWorldModel(nn.Module):
    """Full slot-based world model: encoder + dynamics.

    Pipeline:
      per_step_features -> [encoder] -> slots_t
      (slots_t, action_t) -> [dynamics] -> predicted_slots_{t+1}
    """
    def __init__(self, per_step_dim, n_actions, slot_dim=32, hidden=32):
        super().__init__()
        self.encoder = PerStepEncoder(per_step_dim, hidden=hidden)
        self.dynamics = SlotDynamics(slot_dim, n_actions, hidden=hidden*4)
        self.slot_dim = slot_dim

    def encode(self, per_step_features):
        """per_step_features: (batch, T, per_step_dim) -> slots: (batch, T, slot_dim)"""
        # Per-step features are (obs + action_onehot + reward) - this is the
        # per-step input to our world model
        # For the PoC, treat per-step features as slot representations directly
        # (a learnable projection)
        return self.encoder(per_step_features)

    def predict_next(self, slots_t, action_t):
        """slots_t: (batch, n_slots, slot_dim), action: int -> (batch, n_slots, slot_dim)"""
        return self.dynamics(slots_t, action_t)


def collect_step_pairs(episodes, history_len, obs_dim, n_actions):
    """For each episode, build (per_step_features_t, action_t, per_step_features_{t+1}) pairs."""
    pairs = []
    for ep in episodes:
        ts = ep.transitions
        per_step = obs_dim + n_actions + 1
        if len(ts) < 2:
            continue
        for t in range(len(ts) - 1):
            # per-step features
            f_t = np.zeros(per_step, dtype=np.float32)
            f_t[:obs_dim] = ts[t].obs
            if 0 <= ts[t].action < n_actions:
                f_t[obs_dim + ts[t].action] = 1.0
            f_t[obs_dim + n_actions] = ts[t].reward

            f_next = np.zeros(per_step, dtype=np.float32)
            f_next[:obs_dim] = ts[t+1].obs
            if 0 <= ts[t+1].action < n_actions:
                f_next[obs_dim + ts[t+1].action] = 1.0
            f_next[obs_dim + n_actions] = ts[t+1].reward
            pairs.append((f_t, ts[t].action, f_next))
    return pairs


def train_world_model(model, pairs, n_epochs=30, batch_size=32, lr=3e-4):
    s_t = np.stack([p[0] for p in pairs])
    a_t = np.array([p[1] for p in pairs], dtype=np.int64)
    s_next = np.stack([p[2] for p in pairs])

    S_t = torch.from_numpy(s_t).float()
    A_t = torch.from_numpy(a_t)
    S_next = torch.from_numpy(s_next).float()

    # Normalize target to unit variance for stable training
    s_mean = S_t.mean(dim=0)
    s_std = S_t.std(dim=0) + 1e-6

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_loss = 0.0
    for epoch in range(n_epochs):
        idx = np.random.permutation(len(pairs))
        for start in range(0, len(pairs), batch_size):
            bi = idx[start:start+batch_size]
            st = S_t[bi]
            at = A_t[bi]
            snext = S_next[bi]

            # Each sample: slot_t = encoder(f_t), slot_next = encoder(f_next)
            # slots: (batch, 1, slot_dim) - treat per-step as 1 slot
            slot_t = model.encode(st).unsqueeze(1)  # (b, 1, slot_dim)
            slot_next = model.encode(snext).unsqueeze(1)  # (b, 1, slot_dim)

            # Normalize
            st_norm = (st - s_mean) / s_std
            snext_norm = (snext - s_mean) / s_std

            # Predict next slot
            pred_next = model.predict_next(slot_t, at[0].item())  # single action for batch

            # Loss: MSE on normalized next per-step features
            # (since slot_next = encoder of f_next, and we're approximating f_next from slot)
            # We train both encoder and dynamics jointly
            loss = F.mse_loss(pred_next, slot_next) + 0.1 * F.mse_loss(st_norm, snext_norm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.detach())
    return final_loss


def evaluate_world_model(model, episodes, n_eval=20, obs_dim=8, n_actions=4):
    """Compute next-step prediction error on held-out episodes."""
    errors = []
    per_step = obs_dim + n_actions + 1
    for ep in episodes[:n_eval]:
        ts = ep.transitions
        for t in range(len(ts) - 1):
            f_t = np.zeros(per_step, dtype=np.float32)
            f_t[:obs_dim] = ts[t].obs
            if 0 <= ts[t].action < n_actions:
                f_t[obs_dim + ts[t].action] = 1.0
            f_t[obs_dim + n_actions] = ts[t].reward

            f_next = np.zeros(per_step, dtype=np.float32)
            f_next[:obs_dim] = ts[t+1].obs
            if 0 <= ts[t+1].action < n_actions:
                f_next[obs_dim + ts[t+1].action] = 1.0
            f_next[obs_dim + n_actions] = ts[t+1].reward

            slot_t = model.encode(torch.from_numpy(f_t).float().unsqueeze(0)).unsqueeze(1)
            pred_next = model.predict_next(slot_t, ts[t].action)
            # Decode: simple inverse (slot is the encoded per-step)
            slot_target = model.encode(torch.from_numpy(f_next).float().unsqueeze(0)).unsqueeze(1)
            err = float(((pred_next - slot_target)**2).mean().item())
            errors.append(err)
    return np.mean(errors), np.std(errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()

    out = []
    out.append(f"[Phase 1.2: Slot World Model] env={args.env} seed={args.seed}")
    out.append(f"  slot_dim={args.slot_dim}")

    # 1. Train PPO
    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    for u in range(args.n_ppo_steps // cfg.rollout_len):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    env.close()
    out.append("  PPO trained.")

    # 2. Collect
    train_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()
    eval_eps = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        eval_eps.append(ep)
        e.close()
    out.append(f"  {len(train_eps)} train eps, {len(eval_eps)} eval eps")

    # 3. Build pairs
    pairs = collect_step_pairs(train_eps, history_len=32, obs_dim=obs_dim, n_actions=n_actions)
    out.append(f"  {len(pairs)} (state, action, next_state) pairs")

    # 4. Build world model
    per_step = obs_dim + n_actions + 1
    model = SlotWorldModel(per_step, n_actions, slot_dim=args.slot_dim)
    out.append(f"  PerStepEncoder: {per_step} -> {args.slot_dim}")
    out.append(f"  SlotDynamics: ({args.slot_dim} + {n_actions}) -> {args.slot_dim}")

    # 5. Train
    final_loss = train_world_model(model, pairs, n_epochs=args.epochs)
    out.append(f"  Trained, final loss={final_loss:.4f}")

    # 6. Eval
    mean_err, std_err = evaluate_world_model(model, eval_eps, obs_dim=obs_dim, n_actions=n_actions)
    out.append(f"  Next-step slot prediction error: {mean_err:.6f} +/- {std_err:.6f}")
    out.append(f"  (Lower = better world model)")

    out.append("")
    out.append("=== Phase 1.2 Result ===")
    if mean_err < 0.1:
        out.append(f"  STRONG: world model predicts next-step slots well")
    elif mean_err < 0.5:
        out.append(f"  MODERATE: world model has signal but noisy")
    else:
        out.append(f"  WEAK: world model struggles")

    print(chr(10).join(out))

    log_dir = HERE / "checkpoints" / ("slot_dynamics_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed, "mode": "Phase 1.2 slot world model",
        "slot_dim": args.slot_dim, "n_pairs": len(pairs),
        "final_loss": final_loss,
        "next_step_error_mean": float(mean_err),
        "next_step_error_std": float(std_err),
    }, indent=2))


if __name__ == "__main__":
    main()