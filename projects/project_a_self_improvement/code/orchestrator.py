#!/usr/bin/env python3
"""orchestrator.py - Self-aware agent with Monitor-guided action gating.

Phase 2 of the AGI roadmap: agent uses its own Monitor prediction to
modify its action selection. This is the simplest end-to-end
self-awareness demo:
- At each step, compute Monitor(s) failure probability
- If high, override PPO's action with action 0 (safe fallback)
- Otherwise use PPO's action
- Compare gated vs ungated performance

This is "self-aware" in the literal sense: the agent uses its own
self-prediction to gate its own actions.
"""
import argparse
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PC_CODE = Path(r"E:\agi-research\projects\project_c_causal_world\code")
sys.path.insert(0, str(PC_CODE))

import envs
from ppo import PPOAgent, PPOConfig
from monitor import FailureMonitor, MonitorConfig, _quick_auroc
from envs import rollout_one_episode
from slot_attention import SlotAttention


class SlotMonitor(torch.nn.Module):
    def __init__(self, slot_attention, n_slots, slot_dim, hidden=64):
        super().__init__()
        self.slot_attention = slot_attention
        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_slots * slot_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        slots = self.slot_attention(x)
        flat = slots.reshape(slots.size(0), -1)
        return torch.sigmoid(self.head(flat)).squeeze(-1)


def build_slot_input(transitions, history_len, obs_dim, n_actions):
    per_step = obs_dim + n_actions + 1
    arr = np.zeros((history_len, per_step), dtype=np.float32)
    for i, t in enumerate(transitions[-history_len:]):
        arr[i, :obs_dim] = t.obs
        if 0 <= t.action < n_actions:
            arr[i, obs_dim + t.action] = 1.0
        arr[i, obs_dim + n_actions] = t.reward
    return arr


def train_slot_monitor(slot_monitor, episodes, threshold, history_len, obs_dim, n_actions, n_epochs=20):
    inputs = []
    labels = []
    for ep in episodes:
        if len(ep.transitions) < 1:
            continue
        inputs.append(build_slot_input(ep.transitions, history_len, obs_dim, n_actions))
        labels.append(1.0 if ep.total_reward < threshold else 0.0)
    # Balance
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
    opt = torch.optim.Adam(slot_monitor.parameters(), lr=3e-4)
    for epoch in range(n_epochs):
        for start in range(0, len(y), 16):
            xb = X[start:start+16]
            yb = y[start:start+16]
            if len(xb) == 0:
                continue
            preds = slot_monitor(xb)
            loss = torch.nn.functional.binary_cross_entropy(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.detach())


def gated_rollout(env, agent, slot_monitor, threshold, history_len,
                  obs_dim, n_actions, max_steps=500):
    """Rollout where Monitor can override PPO's action when it predicts failure."""
    obs, _ = env.reset()
    transitions = []
    gate_count = 0
    for t in range(max_steps):
        # Get PPO action
        ppo_action = agent.select_action(obs)

        # If we have enough history, check Monitor
        if len(transitions) >= 5:
            x = build_slot_input(transitions, history_len, obs_dim, n_actions)
            x_t = torch.from_numpy(x).float().unsqueeze(0)
            with torch.no_grad():
                fail_prob = float(slot_monitor(x_t).item())
            if fail_prob > threshold:
                chosen = 0  # safe action (do nothing)
                gate_count += 1
            else:
                chosen = ppo_action
        else:
            chosen = ppo_action

        next_obs, reward, term, trunc, _ = env.step(chosen)
        transitions.append(envs.Transition(obs=obs.copy(), action=chosen, reward=reward))
        obs = next_obs
        if term or trunc:
            break
    return transitions, gate_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--gate-threshold", type=float, default=0.5)
    args = p.parse_args()

    out = []
    out.append(f"[Phase 2 Orchestrator: Self-aware Agent] env={args.env} seed={args.seed}")
    out.append(f"  Gate threshold: Monitor prob > {args.gate_threshold} -> action 0 (safe)")

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

    # 2. Train SlotMonitor
    train_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()
    train_returns = [e.total_reward for e in train_eps]
    threshold = max(0.0, float(np.percentile(train_returns, 10.0)))
    out.append(f"  Threshold: {threshold:.1f}")

    per_step = obs_dim + n_actions + 1
    slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                          n_iters=3, hidden_dim=64, input_dim=per_step)
    slot_monitor = SlotMonitor(slot, args.n_slots, args.slot_dim, hidden=64)
    final_loss = train_slot_monitor(slot_monitor, train_eps, threshold,
                                      args.history_len, obs_dim, n_actions)
    out.append(f"  SlotMonitor trained, final loss={final_loss:.4f}")

    # 3. Eval ungated (baseline PPO)
    ungated_returns = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        ungated_returns.append(ep.total_reward)
        e.close()
    out.append(f"  Ungated PPO: mean={np.mean(ungated_returns):.1f} +/- {np.std(ungated_returns):.1f}")

    # 4. Eval gated (Monitor-guided)
    gated_returns = []
    gate_counts = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i + 50000)
        ts, gc = gated_rollout(e, agent, slot_monitor, args.gate_threshold,
                                args.history_len, obs_dim, n_actions, max_steps=500)
        gated_returns.append(sum(t.reward for t in ts))
        gate_counts.append(gc)
        e.close()
    out.append(f"  Gated (Monitor-guided): mean={np.mean(gated_returns):.1f} +/- {np.std(gated_returns):.1f}")
    out.append(f"  Gates triggered: total={sum(gate_counts)}, avg per ep={np.mean(gate_counts):.1f}")

    delta = float(np.mean(gated_returns) - np.mean(ungated_returns))
    out.append(f"  Delta (gated - ungated): {delta:+.1f}")
    out.append("")
    out.append("=== Phase 2 Orchestrator Result ===")
    if delta > 5:
        out.append(f"  POSITIVE: Self-aware agent beats baseline by {delta:.1f}")
    elif delta < -5:
        out.append(f"  NEGATIVE: Self-aware gating hurts by {-delta:.1f}")
    else:
        out.append(f"  NEUTRAL: ~ baseline (delta={delta:.1f})")

    print(chr(10).join(out))

    log_dir = HERE / "checkpoints" / ("orchestrator_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed, "mode": "Phase 2 orchestrator (gated)",
        "gate_threshold": args.gate_threshold,
        "n_ppo_steps": args.n_ppo_steps,
        "ungated_mean": float(np.mean(ungated_returns)),
        "ungated_std": float(np.std(ungated_returns)),
        "gated_mean": float(np.mean(gated_returns)),
        "gated_std": float(np.std(gated_returns)),
        "delta": delta,
        "total_gates": int(sum(gate_counts)),
        "avg_gates_per_ep": float(np.mean(gate_counts)),
        "slot_monitor_loss": final_loss,
    }, indent=2))


if __name__ == "__main__":
    main()