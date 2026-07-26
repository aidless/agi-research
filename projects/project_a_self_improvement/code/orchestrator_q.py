#!/usr/bin/env python3
"""orchestrator_q.py - Self-aware agent with Monitor + Q-BoN gating.

Phase 2.5: instead of gating to action 0 (do nothing), use Q-function
to pick a SAFER alternative when Monitor predicts failure. This
combines:
- A: SlotMonitor for failure detection
- Q-BoN: Q-function for safer action selection
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

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode, Transition
from slot_attention import SlotAttention


class SlotMonitor(nn.Module):
    def __init__(self, slot_attention, n_slots, slot_dim, hidden=64):
        super().__init__()
        self.slot_attention = slot_attention
        self.head = nn.Sequential(
            nn.Linear(n_slots * slot_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        slots = self.slot_attention(x)
        flat = slots.reshape(slots.size(0), -1)
        return torch.sigmoid(self.head(flat)).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        a_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        x = torch.cat([state, a_onehot], dim=-1)
        return self.net(x).squeeze(-1)


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
    inputs, labels = [], []
    for ep in episodes:
        if len(ep.transitions) < 1: continue
        inputs.append(build_slot_input(ep.transitions, history_len, obs_dim, n_actions))
        labels.append(1.0 if ep.total_reward < threshold else 0.0)
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
            if len(xb) == 0: continue
            preds = slot_monitor(xb)
            loss = F.binary_cross_entropy(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def train_q_with_cql(q_net, episodes, n_epochs=20, lr=3e-4, gamma=0.99, cql_alpha=1.0):
    transitions = []
    for ep in episodes:
        ts = ep.transitions
        for i in range(len(ts) - 1):
            transitions.append((ts[i].obs.copy(), ts[i].action, ts[i].reward,
                                  ts[i+1].obs.copy(), False))
        if len(ts) > 0:
            transitions.append((ts[-1].obs.copy(), ts[-1].action, ts[-1].reward,
                                  ts[-1].obs.copy(), True))
    s_all = np.stack([t[0] for t in transitions])
    a_all = np.array([t[1] for t in transitions], dtype=np.int64)
    r_all = np.array([t[2] for t in transitions], dtype=np.float32)
    s_next_all = np.stack([t[3] for t in transitions])
    done_all = np.array([t[4] for t in transitions], dtype=np.float32)
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    for epoch in range(n_epochs):
        idx = np.random.permutation(len(transitions))
        for start in range(0, len(transitions), 32):
            bi = idx[start:start+32]
            s = torch.from_numpy(s_all[bi]).float()
            a = torch.from_numpy(a_all[bi])
            r = torch.from_numpy(r_all[bi])
            s_next = torch.from_numpy(s_next_all[bi]).float()
            done = torch.from_numpy(done_all[bi])
            q_pred = q_net(s, a)
            with torch.no_grad():
                q_next_all = torch.stack([
                    q_net(s_next, torch.full((s_next.size(0),), ai, dtype=torch.long))
                    for ai in range(q_net.n_actions)
                ], dim=-1)
                q_next_max = q_next_all.max(dim=-1).values
                td_target = r + gamma * q_next_max * (1.0 - done)
            td_loss = F.mse_loss(q_pred, td_target)
            if cql_alpha > 0:
                q_all_a = torch.stack([
                    q_net(s, torch.full((s.size(0),), ai, dtype=torch.long))
                    for ai in range(q_net.n_actions)
                ], dim=-1)
                cql_loss = (torch.logsumexp(q_all_a, dim=-1) - q_pred).mean()
                loss = td_loss + cql_alpha * cql_loss
            else:
                loss = td_loss
            opt.zero_grad(); loss.backward(); opt.step()
    return q_net


def gated_rollout_q(env, agent, slot_monitor, q_net, threshold,
                     history_len, obs_dim, n_actions, gate_thresh=0.5, max_steps=500):
    """Rollout where Monitor triggers Q-based action replacement."""
    obs, _ = env.reset()
    transitions = []
    gate_count = 0
    for t in range(max_steps):
        ppo_action = agent.select_action(obs)

        if len(transitions) >= 5:
            x = build_slot_input(transitions, history_len, obs_dim, n_actions)
            x_t = torch.from_numpy(x).float().unsqueeze(0)
            with torch.no_grad():
                fail_prob = float(slot_monitor(x_t).item())
            if fail_prob > gate_thresh:
                # Pick argmax Q as safer action
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                    q_vals = torch.stack([
                        q_net(obs_t, torch.tensor([a], dtype=torch.long))
                        for a in range(n_actions)
                    ]).squeeze(-1)
                    chosen = int(torch.argmax(q_vals).item())
                gate_count += 1
            else:
                chosen = ppo_action
        else:
            chosen = ppo_action

        next_obs, reward, term, trunc, _ = env.step(chosen)
        transitions.append(Transition(obs=obs.copy(), action=chosen, reward=reward))
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
    out.append(f"[Phase 2.5: Monitor+Q Smart Gating] env={args.env} seed={args.seed}")

    # 1. PPO
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

    # 2. Collect rollouts
    train_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()
    train_returns = [e.total_reward for e in train_eps]
    threshold = max(0.0, float(np.percentile(train_returns, 10.0)))
    out.append(f"  Threshold: {threshold:.1f}, train mean: {np.mean(train_returns):.1f}")

    # 3. SlotMonitor
    per_step = obs_dim + n_actions + 1
    slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                          n_iters=3, hidden_dim=64, input_dim=per_step)
    slot_monitor = SlotMonitor(slot, args.n_slots, args.slot_dim, hidden=64)
    train_slot_monitor(slot_monitor, train_eps, threshold,
                       args.history_len, obs_dim, n_actions)

    # 4. Q + CQL
    q_net = QNetwork(obs_dim, n_actions, hidden=64)
    train_q_with_cql(q_net, train_eps, n_epochs=20, cql_alpha=1.0)

    # 5. Eval
    ungated_returns = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        ungated_returns.append(ep.total_reward)
        e.close()
    out.append(f"  Ungated PPO: mean={np.mean(ungated_returns):.1f}")

    gated_returns = []
    gate_counts = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 9999 + i + 50000)
        ts, gc = gated_rollout_q(e, agent, slot_monitor, q_net, threshold,
                                   args.history_len, obs_dim, n_actions,
                                   gate_thresh=args.gate_threshold)
        gated_returns.append(sum(t.reward for t in ts))
        gate_counts.append(gc)
        e.close()
    out.append(f"  Gated (Monitor+Q): mean={np.mean(gated_returns):.1f}")
    out.append(f"  Gates: total={sum(gate_counts)}, avg/ep={np.mean(gate_counts):.1f}")

    delta = float(np.mean(gated_returns) - np.mean(ungated_returns))
    out.append(f"  Delta (gated - ungated): {delta:+.1f}")
    out.append("")
    out.append("=== Phase 2.5 Result ===")
    if delta > 5: out.append(f"  POSITIVE: smart gating wins by {delta:.1f}")
    elif delta < -5: out.append(f"  NEGATIVE: smart gating loses by {-delta:.1f}")
    else: out.append(f"  NEUTRAL: delta={delta:.1f}")

    print(chr(10).join(out))

    log_dir = HERE / "checkpoints" / ("orchestrator_q_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "Phase 2.5 Monitor+Q smart gating",
        "ungated_mean": float(np.mean(ungated_returns)),
        "gated_mean": float(np.mean(gated_returns)),
        "delta": delta, "total_gates": int(sum(gate_counts)),
    }, indent=2))


if __name__ == "__main__":
    main()