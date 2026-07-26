#!/usr/bin/env python3
"""ttc_bon_monitor.py - Test-Time Compute (BoN+Monitor) PoC for Project A.

ADR 0011 implementation. Tests whether sampling N candidate actions,
scoring each with Monitor, and picking the lowest-failure-probability
candidate gives a meaningful TTC gain over vanilla PPO.

Variant: at each step, sample N candidate next actions from PPO.
For each, roll out K steps with PPO. Score each K-step rollout
with Monitor. Take the action whose rollout got the lowest Monitor
score.

This is the policy-level analog of Lightman 2023 Best-of-N + PRM
and Snell 2024 TTC scaling.
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

import envs
from ppo import PPOAgent, PPOConfig
from monitor import FailureMonitor, MonitorConfig, _quick_auroc
from envs import rollout_one_episode, EpisodeLog, Transition


def build_history_vector(transitions, history_len, obs_dim, n_actions):
    per_step = obs_dim + n_actions + 1
    vec = np.zeros(history_len * per_step, dtype=np.float32)
    for k, h in enumerate(transitions[-history_len:]):
        base = k * per_step
        vec[base: base + obs_dim] = h.obs
        vec[base + obs_dim + h.action] = 1.0
        vec[base + obs_dim + n_actions] = h.reward
    return vec


def bon_monitor_action(args, env_id, agent, monitor, obs, history_len, obs_dim,
                       n_actions, n_candidates, rollout_steps, rng):
    """Sample N candidate actions. For each, rollout K future steps.
    Score each rollout with Monitor. Pick action with lowest score.
    """
    candidates = []
    for _ in range(n_candidates):
        a = agent.select_action(obs)
        candidates.append(a)

    candidate_scores = []
    for a in candidates:
        sim_env = envs.make_env(env_id, seed=int(rng.integers(0, 2**31)))
        sim_obs, _ = sim_env.reset()
        future_transitions = [Transition(
            obs=sim_obs.copy(), action=a, reward=0.0
        )]
        cur_obs = sim_obs
        for _ in range(rollout_steps - 1):
            sim_obs, _, term, trunc, _ = sim_env.step(a)
            future_transitions.append(Transition(
                obs=cur_obs.copy(), action=a, reward=0.0
            ))
            cur_obs = sim_obs
            if term or trunc:
                break
        v = build_history_vector(future_transitions, history_len, obs_dim, n_actions)
        with torch.no_grad():
            score = float(monitor.predict(v))
        candidate_scores.append(score)
        sim_env.close()

    best_idx = int(np.argmin(candidate_scores))
    return candidates[best_idx], candidate_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--monitor-epochs", type=int, default=5)
    p.add_argument("--bon-n", type=int, default=4, help="BoN: number of candidate actions per step")
    p.add_argument("--bon-rollout", type=int, default=10, help="Number of future steps to rollout per candidate")
    args = p.parse_args()

    out = []
    out.append(f"[TTC BoN+Monitor PoC] env={args.env} seed={args.seed} N={args.bon_n} K={args.bon_rollout}")

    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    out.append(f"  obs_dim={obs_dim} n_actions={n_actions}")

    # 1. Train PPO
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
    for i in range(150):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()

    # 3. Train Monitor
    train_returns = [e.total_reward for e in train_eps]
    threshold = 0.0
    if train_returns:
        threshold = max(0.0, float(np.percentile(train_returns, 10.0)))
    out.append(f"  threshold={threshold:.1f} train mean={np.mean(train_returns):.1f} fail={sum(1 for r in train_returns if r < threshold)}/{len(train_eps)}")

    history_dim = args.history_len * (obs_dim + n_actions + 1)
    mcfg = MonitorConfig(history_dim=history_dim, seed=args.seed, epochs=args.monitor_epochs)
    monitor = FailureMonitor(mcfg)

    # Train Monitor
    optimizer = torch.optim.Adam(monitor.parameters(), lr=mcfg.lr)
    for ep in train_eps:
        if len(ep.transitions) < 1:
            continue
    X_list, y_list = [], []
    for ep in train_eps:
        if len(ep.transitions) < 1:
            continue
        v = ep.history_vector(history_len=args.history_len, n_actions=n_actions)
        X_list.append(v)
        y_list.append(1.0 if ep.total_reward < threshold else 0.0)
    X = torch.from_numpy(np.stack(X_list))
    y = torch.from_numpy(np.array(y_list, dtype=np.float32))
    for ep in range(args.monitor_epochs):
        optimizer.zero_grad()
        preds = monitor(X)
        loss = torch.nn.functional.binary_cross_entropy(preds, y)
        loss.backward()
        optimizer.step()
    out.append(f"  Monitor trained. final loss={float(loss):.4f}")

    # 4. Evaluate vanilla PPO (baseline)
    out.append("[Stage 4] Vanilla PPO eval (baseline)...")
    ppo_returns = []
    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        ppo_returns.append(ep.total_reward)
        e.close()
    ppo_mean = float(np.mean(ppo_returns))
    ppo_std = float(np.std(ppo_returns))
    out.append(f"  Vanilla PPO: mean={ppo_mean:.1f} +/- {ppo_std:.1f}")

    # 5. Evaluate BoN+Monitor
    out.append(f"[Stage 5] BoN+Monitor eval (N={args.bon_n}, K={args.bon_rollout})...")
    rng = np.random.default_rng(args.seed)
    bon_returns = []
    bon_action_counts = {a: 0 for a in range(n_actions)}

    for i in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + i + 50000)
        obs, _ = e.reset()
        ep_reward = 0.0
        for t in range(500):
            chosen, scores = bon_monitor_action(
                args, args.env, agent, monitor, obs, args.history_len, obs_dim,
                n_actions, args.bon_n, args.bon_rollout, rng
            )
            bon_action_counts[chosen] += 1
            obs, reward, term, trunc, _ = e.step(chosen)
            ep_reward += reward
            if term or trunc:
                break
        bon_returns.append(ep_reward)
        e.close()
    bon_mean = float(np.mean(bon_returns))
    bon_std = float(np.std(bon_returns))
    delta = bon_mean - ppo_mean
    out.append(f"  BoN+Monitor: mean={bon_mean:.1f} +/- {bon_std:.1f}")
    out.append(f"  Delta (BoN - PPO): {delta:+.1f}")
    out.append(f"  Action distribution: {dict(bon_action_counts)}")

    # Summary
    out.append("")
    out.append("=== TTC BoN+Monitor Result ===")
    if delta > 5:
        out.append(f"  POSITIVE: BoN+Monitor beats PPO by {delta:.1f} points")
    elif delta < -5:
        out.append(f"  NEGATIVE: BoN+Monitor underperforms PPO by {-delta:.1f} points")
    else:
        out.append(f"  NEUTRAL: BoN+Monitor ~ PPO (delta={delta:.1f})")
    out.append(f"  Computed at matched PPO budget; BoN overhead = {args.bon_n}x step cost")

    print(chr(10).join(out))

    # Save log
    log_dir = HERE / "checkpoints" / ("ttc_bon_monitor_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "bon_n": args.bon_n, "bon_rollout": args.bon_rollout,
        "n_ppo_steps": args.n_ppo_steps, "n_eval_episodes": args.n_eval_episodes,
        "ppo_mean": ppo_mean, "ppo_std": ppo_std,
        "bon_mean": bon_mean, "bon_std": bon_std,
        "delta": delta, "action_distribution": bon_action_counts,
        "note": "TTC BoN+Monitor PoC: candidate actions ranked by Monitor future-rollout score",
    }, indent=2))


if __name__ == "__main__":
    main()