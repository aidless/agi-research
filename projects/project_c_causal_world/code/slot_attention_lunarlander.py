#!/usr/bin/env python3
"""slot_attention_lunarlander.py - Slot attention on real LunarLander trajectories.

Project C PoC: do slots learn to specialize on interpretable features
(position, velocity, angle, leg contact) when trained on real env states?

Setup:
- Collect 200 LunarLander trajectories from frozen PPO
- Each trajectory padded/truncated to length 64
- Slot attention: input (batch, 64, 8) -> (batch, n_slots, slot_dim)
- Train to reconstruct next state from current state + slot representations
- Analyze per-slot attention weights and specialization
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
# Borrow envs/ppo from Project A
PA_CODE = Path(r"E:\agi-research\projects\project_a_self_improvement\code")
sys.path.insert(0, str(PA_CODE))
sys.path.insert(0, str(HERE))

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode, Transition

# Reuse slot_attention module
from slot_attention import SlotAttention, slot_diversity_loss


def pad_trajectory(transitions, max_len, obs_dim):
    """Pad trajectory to max_len with zeros. Returns (max_len, obs_dim)."""
    out = np.zeros((max_len, obs_dim), dtype=np.float32)
    for i, t in enumerate(transitions[:max_len]):
        out[i] = t.obs
    return out


def collect_lunarlander_trajectories(n_episodes, seed):
    """Train PPO briefly then collect trajectories."""
    env = envs.make_env("LunarLander-v3", seed=seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    n_updates = 50000 // cfg.rollout_len  # 50K PPO
    for u in range(n_updates):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    env.close()

    trajectories = []
    for i in range(n_episodes):
        e = envs.make_env("LunarLander-v3", seed=seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        trajectories.append(ep)
        e.close()
    return trajectories, obs_dim, n_actions, agent


class SlotDynamicsModel(nn.Module):
    """Given current state + slots, predict next state."""
    def __init__(self, obs_dim, slot_dim, n_slots, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_slots * slot_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, state, slots_flat):
        x = torch.cat([state, slots_flat], dim=-1)
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    out = []
    out.append(f"[Slot-WM LunarLander] n_episodes={args.n_episodes} max_len={args.max_len} n_slots={args.n_slots}")

    # 1. Collect trajectories
    out.append("[Stage 1] Training PPO and collecting trajectories...")
    trajectories, obs_dim, n_actions, agent = collect_lunarlander_trajectories(args.n_episodes, args.seed)
    out.append(f"  collected {len(trajectories)} episodes, obs_dim={obs_dim}")

    # Stats on trajectories
    ep_lens = [len(ep.transitions) for ep in trajectories]
    ep_returns = [ep.total_reward for ep in trajectories]
    out.append(f"  episode lengths: min={min(ep_lens)}, max={max(ep_lens)}, mean={np.mean(ep_lens):.0f}")
    out.append(f"  returns: min={min(ep_returns):.1f}, max={max(ep_returns):.1f}, mean={np.mean(ep_returns):.1f}")

    # 2. Pad trajectories
    padded = np.stack([pad_trajectory(ep.transitions, args.max_len, obs_dim) for ep in trajectories])
    out.append(f"  padded shape: {padded.shape}")

    # 3. Train slot attention
    out.append("[Stage 2] Training slot attention + dynamics...")
    slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                          n_iters=3, hidden_dim=64, input_dim=obs_dim)
    dyn = SlotDynamicsModel(obs_dim, args.slot_dim, args.n_slots)
    opt = torch.optim.Adam(list(slot.parameters()) + list(dyn.parameters()), lr=3e-4)
    X = torch.from_numpy(padded)  # (n_eps, max_len, obs_dim)

    for epoch in range(args.n_epochs):
        # Mini-batch
        idx = np.random.choice(len(trajectories), args.batch_size, replace=False)
        batch = X[idx]  # (B, max_len, obs_dim)
        slots = slot(batch)  # (B, n_slots, slot_dim)

        # Reconstruction loss: predict next state from current state + slots
        states = batch[:, :-1, :]  # (B, max_len-1, obs_dim)
        next_states = batch[:, 1:, :]  # (B, max_len-1, obs_dim)
        slots_repeat = slots.unsqueeze(1).expand(-1, args.max_len - 1, -1, -1)
        slots_flat = slots_repeat.reshape(args.batch_size, args.max_len - 1, -1)

        pred = dyn(states.reshape(-1, obs_dim), slots_flat.reshape(-1, args.n_slots * args.slot_dim))
        pred = pred.reshape(args.batch_size, args.max_len - 1, obs_dim)
        recon = F.mse_loss(pred, next_states)

        # Diversity loss
        div = slot_diversity_loss(slots)

        loss = recon + 0.1 * div
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            out.append(f"  epoch {epoch+1}/{args.n_epochs}: recon={recon.item():.4f} div={div.item():.4f}")
    out.append(f"  final: recon={recon.item():.4f} div={div.item():.4f}")

    # 4. Analyze slot specialization
    out.append("[Stage 3] Analyzing slot specialization...")
    slot.eval()
    with torch.no_grad():
        slots_all = slot(X)  # (n_eps, n_slots, slot_dim)

        # For each slot, compute average cosine similarity to each feature dimension
        # (positive = slot binds to that feature, negative = ignores)
        X_flat = X.reshape(-1, obs_dim)
        # Normalize slots
        slots_flat_all = slots_all.reshape(-1, args.slot_dim)
        slot_norms = F.normalize(slots_flat_all, dim=-1)

        # For each (slot, feature) compute correlation
        feat_corr = np.zeros((args.n_slots, obs_dim))
        for s in range(args.n_slots):
            # For each episode, summarize slot as its L2 norm, summarize feature
            # as mean over timesteps. Then correlate these two (n_eps,) vectors.
            slot_norm_per_ep = slots_all[:, s, :].norm(dim=-1)  # (n_eps,)
            slot_v = (slot_norm_per_ep - slot_norm_per_ep.mean()) / (slot_norm_per_ep.std() + 1e-6)
            for d in range(obs_dim):
                # Per-episode mean of feature d
                feat_per_ep = X[:, :, d].mean(dim=-1)  # (n_eps,)
                feat_v = (feat_per_ep - feat_per_ep.mean()) / (feat_per_ep.std() + 1e-6)
                corr = float(torch.dot(slot_v, feat_v) / len(slot_v))
                feat_corr[s, d] = corr

        # Per-slot variance of attention (high variance = specialized)
        slot_var = slots_all.var(dim=0).mean(dim=-1).numpy()  # (n_slots,)
        slot_mean_norm = slots_all.norm(dim=-1).mean(dim=0).numpy()  # (n_slots,)

    feature_names = ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "ang_vel", "leg_l", "leg_r"]
    out.append("  Per-slot feature correlations (top 3 features per slot):")
    for s in range(args.n_slots):
        top_idx = np.argsort(np.abs(feat_corr[s]))[::-1][:3]
        top_feats = [(feature_names[i], f"{feat_corr[s, i]:+.3f}") for i in top_idx]
        out.append(f"    Slot {s}: " + ", ".join(f"{n}={v}" for n, v in top_feats) +
                   f" | var={slot_var[s]:.3f} norm={slot_mean_norm[s]:.2f}")

    # 5. Save results
    log_path = HERE / "checkpoints" / "slot_wm_lunarlander" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "LunarLander-v3", "n_episodes": args.n_episodes, "max_len": args.max_len,
        "n_slots": args.n_slots, "slot_dim": args.slot_dim,
        "n_epochs": args.n_epochs, "seed": args.seed,
        "final_recon_loss": float(recon.item()),
        "final_diversity_loss": float(div.item()),
        "feature_correlations": feat_corr.tolist(),
        "slot_variance": slot_var.tolist(),
        "slot_mean_norm": slot_mean_norm.tolist(),
        "feature_names": feature_names,
    }, indent=2))

    out.append("")
    out.append("=== Slot-WM PoC Result ===")
    out.append(f"  Reconstruction loss: {recon.item():.4f} (lower=better)")
    out.append(f"  Diversity loss: {div.item():.4f} (lower=more diverse)")
    out.append(f"  Slot specialization: see per-slot top features above")
    if div.item() < 0.15:
        out.append(f"  STRONG specialization: slots have distinct feature preferences")
    elif div.item() < 0.30:
        out.append(f"  MODERATE specialization: some slot differentiation")
    else:
        out.append(f"  WEAK specialization: slots are similar to each other")

    print(chr(10).join(out))


if __name__ == "__main__":
    main()