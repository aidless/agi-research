#!/usr/bin/env python3
"""slot_monitor.py - A+C integration: Monitor using slot-attention output.

Instead of feeding raw history vector (obs + action + reward 脳 H) to
Monitor, we first run slot-attention over the trajectory to get
structured slot representations, then feed those to Monitor.

Hypothesis: slot-attention's object-centric decomposition provides
better features for failure prediction than raw flattened history.
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
# Borrow slot_attention from Project C
PC_CODE = Path(r"E:\agi-research\projects\project_c_causal_world\code")
sys.path.insert(0, str(PC_CODE))

import envs
from ppo import PPOAgent, PPOConfig
from monitor import FailureMonitor, MonitorConfig, _quick_auroc
from envs import rollout_one_episode
from slot_attention import SlotAttention

# Borrow / import existing pieces
from env_state_cloner import EnvStateCloner


def build_slot_input(transitions, history_len, obs_dim, n_actions):
    """Build slot-attention input from a trajectory.

    Each timestep contributes (obs + onehot(action) + reward) = obs_dim + n_actions + 1
    Pad to history_len if shorter.
    """
    per_step = obs_dim + n_actions + 1
    arr = np.zeros((history_len, per_step), dtype=np.float32)
    for i, t in enumerate(transitions[-history_len:]):
        arr[i, :obs_dim] = t.obs
        if 0 <= t.action < n_actions:
            arr[i, obs_dim + t.action] = 1.0
        arr[i, obs_dim + n_actions] = t.reward
    return arr


class SlotMonitor(nn.Module):
    """Monitor that takes slot-attention output as input.

    Pipeline: trajectory -> slot-attention -> per-slot features -> aggregate -> MLP -> failure prob
    """
    def __init__(self, slot_attention, n_slots, slot_dim, hidden=64):
        super().__init__()
        self.slot_attention = slot_attention
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        # MLP on flattened slot representation
        self.head = nn.Sequential(
            nn.Linear(n_slots * slot_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (batch, history_len, per_step_dim)
        # Run slot attention
        slots = self.slot_attention(x)  # (batch, n_slots, slot_dim)
        flat = slots.reshape(slots.size(0), -1)  # (batch, n_slots * slot_dim)
        return torch.sigmoid(self.head(flat)).squeeze(-1)


def train_slot_monitor(monitor, slot_monitor_opt, episodes, threshold,
                       history_len, obs_dim, n_actions, n_epochs=20,
                       batch_size=16, balance=True):
    """Train SlotMonitor via BCE on failure labels."""
    inputs = []
    labels = []
    for ep in episodes:
        if len(ep.transitions) < 1:
            continue
        x = build_slot_input(ep.transitions, history_len, obs_dim, n_actions)
        inputs.append(x)
        labels.append(1.0 if ep.total_reward < threshold else 0.0)

    if balance and len(set(labels)) == 2:
        pos_idx = [i for i, l in enumerate(labels) if l == 1.0]
        neg_idx = [i for i, l in enumerate(labels) if l == 0.0]
        if len(pos_idx) > 0 and len(neg_idx) > len(pos_idx) * 2:
            np.random.seed(42)
            chosen = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 4), replace=False)
            keep = np.concatenate([np.array(pos_idx), chosen])
            np.random.shuffle(keep)
            inputs = [inputs[i] for i in keep]
            labels = [labels[i] for i in keep]

    X = torch.from_numpy(np.stack(inputs)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.float32))

    final_loss = 0.0
    n_pos = int(y.sum().item())
    n_neg = len(y) - n_pos
    print(f"  SlotMonitor training: {len(y)} samples ({n_pos} pos, {n_neg} neg)")
    for epoch in range(n_epochs):
        idx = np.random.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            preds = monitor(xb)
            loss = F.binary_cross_entropy(preds, yb)
            slot_monitor_opt.zero_grad()
            loss.backward()
            slot_monitor_opt.step()
            final_loss = float(loss.detach())
    return final_loss


def per_episode_slot_monitor_probs(slot_monitor, episodes, history_len, obs_dim, n_actions):
    means = []
    for ep in episodes:
        x = build_slot_input(ep.transitions, history_len, obs_dim, n_actions)
        x_t = torch.from_numpy(x).float().unsqueeze(0)
        with torch.no_grad():
            p = float(slot_monitor(x_t).item())
        means.append(p)
    return np.array(means)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=100)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--slot-iters", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--threshold-floor", type=float, default=0.0)
    p.add_argument("--percentile", type=float, default=10.0)
    args = p.parse_args()

    out = []
    out.append(f"[A+C Integration: SlotMonitor] env={args.env} seed={args.seed}")
    out.append(f"  n_slots={args.n_slots}, slot_dim={args.slot_dim}, history_len={args.history_len}")

    # 1. Train PPO
    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    out.append(f"  obs_dim={obs_dim} n_actions={n_actions}")
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    n_updates = args.n_ppo_steps // cfg.rollout_len
    for u in range(n_updates):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
        if (u + 1) % 10 == 0:
            out.append(f"  PPO {(u+1)*cfg.rollout_len}/{args.n_ppo_steps}")
    env.close()

    # 2. Collect frozen rollouts
    out.append("[Stage 2] Collecting frozen rollouts...")
    train_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()
    train_returns = [e.total_reward for e in train_eps]
    threshold = max(args.threshold_floor,
                    envs.percentile_failure_threshold(train_returns, args.percentile))
    n_fail = sum(1 for r in train_returns if r < threshold)
    out.append(f"  {len(train_eps)} episodes, mean={np.mean(train_returns):.1f}, fail={n_fail}, threshold={threshold:.1f}")

    # 3. Build SlotMonitor
    out.append("[Stage 3] Building SlotMonitor...")
    per_step = obs_dim + n_actions + 1
    slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                          n_iters=args.slot_iters, hidden_dim=64,
                          input_dim=per_step)
    slot_monitor = SlotMonitor(slot, args.n_slots, args.slot_dim, hidden=64)
    opt = torch.optim.Adam(slot_monitor.parameters(), lr=3e-4)

    # 4. Train SlotMonitor
    final_loss = train_slot_monitor(slot_monitor, opt, train_eps, threshold,
                                     args.history_len, obs_dim, n_actions,
                                     n_epochs=args.epochs)
    out.append(f"  SlotMonitor trained, final loss={final_loss:.4f}")

    # 5. Eval
    out.append("[Stage 4] Evaluating...")
    eval_eps = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        eval_eps.append(ep)
        e.close()
    eval_returns = [e.total_reward for e in eval_eps]
    eval_probs = per_episode_slot_monitor_probs(
        slot_monitor, eval_eps, args.history_len, obs_dim, n_actions
    )
    fail_labels = np.array([1.0 if r < threshold else 0.0 for r in eval_returns])
    if fail_labels.std() > 1e-9:
        auroc = _quick_auroc(fail_labels, eval_probs)
        pearson = float(np.corrcoef(eval_probs, eval_returns)[0, 1]) if eval_probs.std() > 1e-9 else float("nan")
    else:
        auroc = float("nan")
        pearson = float("nan")
    out.append(f"  Eval: n={len(eval_eps)} fail_rate={fail_labels.mean():.3f}")
    out.append(f"  AUROC={auroc:.4f}  Pearson={pearson:.4f}")

    # Compare to baseline raw-history Monitor
    out.append("")
    out.append("=== A+C Integration Result ===")
    out.append(f"  SlotMonitor (slot-attention -> MLP) AUROC: {auroc:.4f}")
    out.append(f"  Reference (frozen Monitor, raw history) AUROC: 0.796 (from CHANGELOG v1.8)")

    print(chr(10).join(out))

    # Save
    log_dir = HERE / "checkpoints" / ("slot_monitor_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed, "mode": "A+C integration (slot-Monitor)",
        "n_slots": args.n_slots, "slot_dim": args.slot_dim, "slot_iters": args.slot_iters,
        "n_ppo_steps": args.n_ppo_steps, "n_train_episodes": args.n_train_episodes,
        "threshold": float(threshold),
        "n_eval_episodes": len(eval_eps),
        "fail_rate": float(fail_labels.mean()),
        "auroc": float(auroc) if auroc == auroc else None,
        "pearson_prob_reward": float(pearson) if pearson == pearson else None,
        "final_loss": final_loss,
        "reference_auroc_raw_history": 0.796,
        "delta_vs_reference": float(auroc - 0.796) if auroc == auroc else None,
        "note": "Slot-attention output replaces raw history vector as Monitor input",
    }, indent=2))


if __name__ == "__main__":
    main()