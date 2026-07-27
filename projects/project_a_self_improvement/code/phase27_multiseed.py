"""phase27_multiseed.py - Multi-seed validation of threshold sweep.

Trains PPO + Monitor + Q on multiple seeds, then sweeps thresholds.
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
PC_CODE = Path(r"E:\agi-research\projects\project_c_causal_world\code")
sys.path.insert(0, str(PA_CODE))
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


def train_slot_monitor(slot_monitor, episodes, threshold, history_len, obs_dim, n_actions, n_epochs=15):
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
            xb = X[start:start+16]; yb = y[start:start+16]
            if len(xb) == 0: continue
            preds = slot_monitor(xb)
            loss = F.binary_cross_entropy(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return slot_monitor


def train_q_cql(q_net, episodes, n_epochs=15, cql_alpha=1.0, gamma=0.99):
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
    opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
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
            q_all_a = torch.stack([
                q_net(s, torch.full((s.size(0),), ai, dtype=torch.long))
                for ai in range(q_net.n_actions)
            ], dim=-1)
            cql_loss = (torch.logsumexp(q_all_a, dim=-1) - q_pred).mean()
            loss = td_loss + cql_alpha * cql_loss
            opt.zero_grad(); loss.backward(); opt.step()
    return q_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    out = []
    out.append(f"[Phase 2.7 Multi-seed] env={args.env} seeds={seeds} thresholds={thresholds}")
    out.append("=" * 70)

    all_results = []

    for seed in seeds:
        out.append(f"\n--- SEED {seed} ---")
        # 1. Train PPO
        env = envs.make_env(args.env, seed=seed + 1)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=seed)
        agent = PPOAgent(cfg)
        obs, _ = env.reset()
        n_updates = args.n_ppo_steps // cfg.rollout_len
        for u in range(n_updates):
            batch = agent.collect_rollout(env, obs)
            agent.update(batch)
            obs = batch["final_obs"]
        env.close()
        out.append(f"  [A] PPO trained ({args.n_ppo_steps} steps)")

        # 2. Collect
        train_eps = []
        for i in range(args.n_train_episodes):
            e = envs.make_env(args.env, seed=seed * 1000 + i + 7777)
            ep = rollout_one_episode(e, agent.select_action, max_steps=500)
            train_eps.append(ep)
            e.close()
        threshold = max(0.0, float(np.percentile([e.total_reward for e in train_eps], 10.0)))

        # 3. Monitor
        per_step = obs_dim + n_actions + 1
        slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                              n_iters=3, hidden_dim=64, input_dim=per_step)
        slot_monitor = SlotMonitor(slot, args.n_slots, args.slot_dim, hidden=64)
        train_slot_monitor(slot_monitor, train_eps, threshold,
                           args.history_len, obs_dim, n_actions)

        # 4. Q
        q_net = QNetwork(obs_dim, n_actions, hidden=64)
        train_q_cql(q_net, train_eps, n_epochs=15, cql_alpha=1.0)

        # 5. Eval all thresholds
        seed_results = {"seed": seed, "thresholds": {}}
        for g_thresh in thresholds:
            returns_gated = []
            for ep_idx in range(args.n_eval_episodes):
                e = envs.make_env(args.env, seed=seed * 1000 + 9999 + ep_idx)
                obs, _ = e.reset()
                ep_transitions = []
                ep_reward = 0.0
                for t in range(500):
                    ppo_action = agent.select_action(obs)
                    monitor_prob = 0.0
                    if len(ep_transitions) >= 5:
                        x = build_slot_input(ep_transitions, args.history_len, obs_dim, n_actions)
                        x_t = torch.from_numpy(x).float().unsqueeze(0)
                        with torch.no_grad():
                            monitor_prob = float(slot_monitor(x_t).item())
                    if monitor_prob > g_thresh:
                        with torch.no_grad():
                            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                            q_vals = torch.stack([
                                q_net(obs_t, torch.tensor([a], dtype=torch.long))
                                for a in range(n_actions)
                            ]).squeeze(-1)
                            chosen = int(torch.argmax(q_vals).item())
                    else:
                        chosen = ppo_action
                    next_obs, reward, term, trunc, _ = e.step(chosen)
                    ep_reward += reward
                    ep_transitions.append(Transition(obs=obs.copy(), action=chosen, reward=reward))
                    obs = next_obs
                    if term or trunc:
                        break
                e.close()
                returns_gated.append(ep_reward)
            mean = float(np.mean(returns_gated))
            seed_results["thresholds"][str(g_thresh)] = mean
            out.append(f"    thresh={g_thresh}: mean={mean:.1f}")
        all_results.append(seed_results)

    # Summary
    out.append("\n" + "=" * 70)
    out.append("MULTI-SEED SUMMARY")
    out.append("=" * 70)
    out.append(f"  Seeds: {seeds}")
    for g_thresh in thresholds:
        key = str(g_thresh)
        vals = [r["thresholds"][key] for r in all_results]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        out.append(f"  thresh={g_thresh}: mean across {len(seeds)} seeds = {mean_val:+.1f} +/- {std_val:.1f}")

    print(chr(10).join(out))

    log_dir = HERE / "checkpoints" / ("phase27_multiseed_" + args.env)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seeds": seeds, "thresholds": thresholds,
        "n_ppo_steps": args.n_ppo_steps,
        "all_results": all_results,
    }, indent=2))


if __name__ == "__main__":
    main()