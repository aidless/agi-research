#!/usr/bin/env python3
"""q_bon.py - Q-function BoN TTC (replacement for Monitor-BoN).

ADR 0011 alternative: train Q(s, a) on frozen rollouts, use Q to rank
candidate actions. This is the standard model-based RL approach.

Key difference from Monitor-BoN:
- Monitor predicts failure probability (binary)
- Q-function predicts expected return (continuous)
- BoN should pick HIGHEST Q, not lowest failure probability
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

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode, Transition
from env_state_cloner import EnvStateCloner


class QNetwork(nn.Module):
    """Q(s, a) network: input = state + onehot action, output = scalar Q."""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        a_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        x = torch.cat([state, a_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def collect_q_training_data(episodes):
    """Extract (s, a, r, s', done) tuples from episodes for Q-learning."""
    transitions = []
    for ep in episodes:
        ts = ep.transitions
        for i in range(len(ts) - 1):
            transitions.append({
                's': ts[i].obs.copy(),
                'a': ts[i].action,
                'r': ts[i].reward,
                's_next': ts[i + 1].obs.copy(),
                'done': False,
            })
        # Last transition: episode done, s_next = s (terminal)
        if len(ts) > 0:
            transitions.append({
                's': ts[-1].obs.copy(),
                'a': ts[-1].action,
                'r': ts[-1].reward,
                's_next': ts[-1].obs.copy(),  # terminal: pad with same
                'done': True,
            })
    return transitions


def train_q_network(q_net, transitions, n_epochs=20, lr=3e-4, gamma=0.99,
                       batch_size=32, cql_alpha=0.0):
    """Train Q via TD(0) + optional CQL regularization.

    CQL: L = TD_loss + alpha * (logsumexp_a Q(s, a) - Q(s, a_data))
    This penalizes Q-values for OOD actions, keeping Q conservative.
    """
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    n = len(transitions)
    s_all = np.stack([t['s'] for t in transitions])
    a_all = np.array([t['a'] for t in transitions], dtype=np.int64)
    r_all = np.array([t['r'] for t in transitions], dtype=np.float32)
    s_next_all = np.stack([t['s_next'] for t in transitions])
    done_all = np.array([t['done'] for t in transitions], dtype=np.float32)

    for ep in range(n_epochs):
        idx = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            s = torch.from_numpy(s_all[batch_idx]).float()
            a = torch.from_numpy(a_all[batch_idx])
            r = torch.from_numpy(r_all[batch_idx])
            s_next = torch.from_numpy(s_next_all[batch_idx]).float()
            done = torch.from_numpy(done_all[batch_idx])

            # Current Q
            q_pred = q_net(s, a)

            # Target: r + gamma * max_a' Q(s', a') * (1 - done)
            with torch.no_grad():
                q_next_all_actions = torch.stack([
                    q_net(s_next, torch.full((s_next.size(0),), a_idx, dtype=torch.long))
                    for a_idx in range(q_net.n_actions)
                ], dim=-1)
                q_next_max = q_next_all_actions.max(dim=-1).values
                target = r + gamma * q_next_max * (1.0 - done)

            td_loss = F.mse_loss(q_pred, target)

            if cql_alpha > 0:
                # CQL: penalize Q for all actions vs data actions
                q_all_actions = torch.stack([
                    q_net(s, torch.full((s.size(0),), a_idx, dtype=torch.long))
                    for a_idx in range(q_net.n_actions)
                ], dim=-1)
                # logsumexp - mean Q on data = CQL penalty
                cql_logsumexp = torch.logsumexp(q_all_actions, dim=-1)
                cql_data_q = q_pred  # Q on data actions (already computed)
                cql_loss = (cql_logsumexp - cql_data_q).mean()
                loss = td_loss + cql_alpha * cql_loss
            else:
                loss = td_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        if (ep + 1) % 5 == 0:
            print(f"  Q epoch {ep+1}/{n_epochs}: loss={avg_loss:.4f}")
    return avg_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bon-n", type=int, default=4)
    p.add_argument("--q-hidden", type=int, default=64)
    p.add_argument("--q-epochs", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--cql-alpha", type=float, default=1.0,
                   help="CQL regularization weight (0 = vanilla TD, 1 = standard CQL)")
    args = p.parse_args()

    out = []
    out.append(f"[Q-BoN TTC] env={args.env} seed={args.seed} N={args.bon_n}")

    # 1. Train PPO
    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    n_updates = args.n_ppo_steps // cfg.rollout_len
    for u in range(n_updates):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
        if (u + 1) % 10 == 0:
            out.append(f"  PPO step {(u+1)*cfg.rollout_len}/{args.n_ppo_steps}")
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
    out.append(f"  collected {len(train_eps)} episodes, mean reward={np.mean(train_returns):.1f}")

    # 3. Train Q-network
    out.append(f"[Stage 3] Training Q-network via TD(0) + CQL(alpha={args.cql_alpha})...")
    q_net = QNetwork(obs_dim, n_actions, hidden=args.q_hidden)
    transitions = collect_q_training_data(train_eps)
    out.append(f"  {len(transitions)} (s, a, r, s') tuples")
    final_loss = train_q_network(q_net, transitions, n_epochs=args.q_epochs,
                                   gamma=args.gamma, cql_alpha=args.cql_alpha)

    # Sanity: compare Q values across actions for some sample states
    sample_states = torch.from_numpy(np.stack([t['s'] for t in transitions[:8]])).float()
    with torch.no_grad():
        q_values = torch.stack([
            q_net(sample_states, torch.full((8,), a_idx, dtype=torch.long))
            for a_idx in range(n_actions)
        ], dim=-1)
    out.append(f"  Sample Q values (8 states x {n_actions} actions):")
    q_np = q_values.numpy()
    for i in range(8):
        top_action = int(np.argmax(q_np[i]))
        out.append(f"    state {i}: Q = {q_np[i].round(2).tolist()} -> argmax = {top_action}")
    out.append(f"  Final Q training loss: {final_loss:.4f}")

    # 4. Eval vanilla PPO
    out.append("[Stage 4] Vanilla PPO eval...")
    ppo_returns = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        ppo_returns.append(ep.total_reward)
        e.close()
    ppo_mean = float(np.mean(ppo_returns))
    out.append(f"  PPO: mean={ppo_mean:.1f} +/- {np.std(ppo_returns):.1f}")

    # 5. Eval Q-BoN
    out.append(f"[Stage 5] Q-BoN eval (N={args.bon_n})...")
    qbon_returns = []
    qbon_action_counts = {a: 0 for a in range(n_actions)}
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i + 50000)
        cloner = EnvStateCloner(e)
        obs, _ = e.reset()
        ep_reward = 0.0
        for t in range(500):
            # Sample N candidates
            candidates = [int(agent.select_action(obs)) for _ in range(args.bon_n)]
            # Rank by Q(s, a)
            with torch.no_grad():
                q_vals = torch.tensor([
                    float(q_net(torch.from_numpy(obs).float().unsqueeze(0),
                                torch.tensor([a], dtype=torch.long)))
                    for a in candidates
                ])
            best = int(torch.argmax(q_vals).item())
            chosen = candidates[best]
            qbon_action_counts[chosen] += 1
            obs, reward, term, trunc, _ = e.step(chosen)
            ep_reward += reward
            if term or trunc:
                break
        qbon_returns.append(ep_reward)
        e.close()
    qbon_mean = float(np.mean(qbon_returns))
    delta = qbon_mean - ppo_mean
    out.append(f"  Q-BoN: mean={qbon_mean:.1f} +/- {np.std(qbon_returns):.1f}")
    out.append(f"  Delta (Q-BoN - PPO): {delta:+.1f}")
    out.append(f"  Action distribution: {dict(qbon_action_counts)}")

    # Summary
    out.append("")
    out.append("=== Q-BoN TTC Result ===")
    if delta > 5:
        out.append(f"  POSITIVE: Q-BoN beats PPO by {delta:.1f}")
    elif delta < -5:
        out.append(f"  NEGATIVE: Q-BoN underperforms PPO by {-delta:.1f}")
    else:
        out.append(f"  NEUTRAL: Q-BoN ~ PPO (delta={delta:.1f})")
    out.append(f"  BoN overhead = {args.bon_n}x step cost")

    print(chr(10).join(out))

    # Save log
    log_dir = HERE / "checkpoints" / ("qbon_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed, "bon_n": args.bon_n,
        "q_hidden": args.q_hidden, "q_epochs": args.q_epochs, "gamma": args.gamma,
        "n_ppo_steps": args.n_ppo_steps, "n_eval_episodes": args.n_eval_episodes,
        "ppo_mean": ppo_mean, "qbon_mean": qbon_mean, "delta": delta,
        "action_distribution": qbon_action_counts,
        "final_q_loss": final_loss,
        "note": "Q-BoN: train Q on frozen rollouts via TD(0), BoN pick argmax_Q",
    }, indent=2))


if __name__ == "__main__":
    main()