#!/usr/bin/env python3
"""joint_phase2.py - TRUE JOINT Monitor ablation for Project A.

H1 ablation: Monitor trained JOINTLY with PPO (Monitor sees fresh rollouts
from currently-updating PPO), vs frozen (Monitor trained only after PPO
convergence on frozen rollouts).

Key difference vs frozen Monitor:
  - In frozen: PPO runs all 256K steps -> snapshot PPO -> collect rollouts
    with frozen PPO -> train Monitor on those frozen rollouts
  - In joint: every K PPO updates, collect fresh rollouts from CURRENT
    (still-updating) PPO -> train Monitor on those -> repeat
"""
import argparse
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import envs
from ppo import PPOAgent, PPOConfig
from monitor import FailureMonitor, MonitorConfig, _quick_auroc
from envs import rollout_one_episode


def make_env_factory(env_name):
    def factory(seed):
        return envs.make_env(env_name, seed=seed)
    return factory


def collect_rollouts(env_factory, agent, n_episodes, seed_offset):
    out = []
    for i in range(n_episodes):
        env = env_factory(seed_offset + i + 7777)
        ep = rollout_one_episode(env, agent.select_action, max_steps=500)
        ep.env_name = "lunarlander"
        out.append(ep)
        env.close()
    return out


def per_episode_monitor_probs(monitor, episodes, obs_dim, n_actions, history_len):
    means = []
    for ep in episodes:
        probs = []
        for t, _ in enumerate(ep.transitions):
            recent = ep.transitions[: t + 1][-history_len:]
            v = ep.history_vector.__wrapped__(recent) if False else None
            v = ep.history_vector(history_len=history_len, n_actions=n_actions)
            probs.append(monitor.predict(v))
        if probs:
            means.append(float(np.mean(probs)))
        else:
            means.append(0.5)
    return np.array(means)


def joint_monitor_step(monitor, episodes, threshold, optimizer, n_epochs, history_len, n_actions):
    inputs = []
    labels = []
    for ep in episodes:
        if len(ep.transitions) < 1:
            continue
        v = ep.history_vector(history_len=history_len, n_actions=n_actions)
        inputs.append(v)
        labels.append(1.0 if ep.total_reward < threshold else 0.0)
    if not inputs:
        return 0.0, 0
    X = torch.from_numpy(np.stack(inputs))
    y = torch.from_numpy(np.array(labels, dtype=np.float32))
    last_loss = 0.0
    for _ in range(n_epochs):
        optimizer.zero_grad()
        preds = monitor(X)
        loss = F.binary_cross_entropy(preds, y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return last_loss, len(inputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=100)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--percentile", type=float, default=10.0)
    p.add_argument("--threshold-floor", type=float, default=0.0)
    p.add_argument("--monitor-interval", type=int, default=4)
    p.add_argument("--monitor-epochs-per-step", type=int, default=2)
    p.add_argument("--n-monitor-rollouts", type=int, default=20)
    args = p.parse_args()

    out = []
    out.append("[Project A Phase 2 JOINT ABLATION] " + args.env + " seed=" + str(args.seed))
    out.append("  PPO=" + str(args.n_ppo_steps) + " train=" + str(args.n_train_episodes) +
               " eval=" + str(args.n_eval_episodes) + " joint_interval=" + str(args.monitor_interval) +
               " monitor_epochs_per_step=" + str(args.monitor_epochs_per_step))

    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    out.append("  obs_dim=" + str(obs_dim) + " n_actions=" + str(n_actions))

    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    history_dim = args.history_len * (obs_dim + n_actions + 1)
    mcfg = MonitorConfig(history_dim=history_dim, seed=args.seed, epochs=2)
    monitor = FailureMonitor(mcfg)
    optimizer_monitor = torch.optim.Adam(monitor.parameters(), lr=mcfg.lr)
    out.append("  Monitor history_dim=" + str(history_dim))

    obs, _ = env.reset()
    n_updates = args.n_ppo_steps // cfg.rollout_len
    all_returns = []
    monitor_losses = []
    factory = make_env_factory(args.env)
    running_threshold = -50.0

    for u in range(n_updates):
        batch = agent.collect_rollout(env, obs)
        info = agent.update(batch)
        obs = batch["final_obs"]
        all_returns.extend(batch["ep_returns"])

        if len(all_returns) >= 50:
            running_threshold = float(np.percentile(all_returns, args.percentile))

        if (u + 1) % args.monitor_interval == 0:
            fresh_rollouts = collect_rollouts(factory, agent, args.n_monitor_rollouts,
                                              args.seed * 1000 + u * 100)
            loss, n = joint_monitor_step(monitor, fresh_rollouts, running_threshold,
                                          optimizer_monitor, args.monitor_epochs_per_step,
                                          args.history_len, n_actions)
            monitor_losses.append(loss)
            if (u + 1) % 20 == 0:
                out.append("  u=" + str(u + 1) + "/" + str(n_updates) +
                           " mean_r(last200)=" + str(round(np.mean(all_returns[-200:]) if all_returns else 0.0, 1)) +
                           " monitor_loss(last)=" + str(round(loss, 4)) +
                           " thresh=" + str(round(running_threshold, 1)))
    env.close()

    threshold = max(args.threshold_floor,
                    envs.percentile_failure_threshold(all_returns, args.percentile))
    out.append("  PPO done. mean=" + str(round(np.mean(all_returns), 2)) +
               " p=" + str(args.percentile) + "=" + str(round(threshold, 2)) +
               " n_monitor_steps=" + str(len(monitor_losses)))

    ckpt_dir = HERE / "checkpoints" / ("joint_" + args.env + "_seed" + str(args.seed))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"monitor": monitor.state_dict(), "history_dim": history_dim,
                "config": vars(mcfg), "n_monitor_steps": len(monitor_losses)},
               str(ckpt_dir / "monitor.pt"))

    train_eps = collect_rollouts(factory, agent, args.n_train_episodes, args.seed * 1000)
    train_returns = [e.total_reward for e in train_eps]
    fail_n = sum(1 for r in train_returns if r < threshold)
    out.append("  collected final train. mean=" + str(round(np.mean(train_returns), 2)) +
               " fail=" + str(fail_n) + "/" + str(len(train_eps)))

    eval_eps = collect_rollouts(factory, agent, args.n_eval_episodes, args.seed * 1000 + 999)
    eval_returns = [e.total_reward for e in eval_eps]
    eval_probs = per_episode_monitor_probs(monitor, eval_eps, obs_dim, n_actions, args.history_len)
    fail_labels = np.array([1.0 if r < threshold else 0.0 for r in eval_returns])
    if fail_labels.std() > 1e-9:
        auroc = _quick_auroc(fail_labels, eval_probs)
        pearson = float(np.corrcoef(eval_probs, eval_returns)[0, 1]) if eval_probs.std() > 1e-9 else float("nan")
    else:
        auroc = float("nan")
        pearson = float("nan")

    out.append("=== Phase 2 JOINT (seed=" + str(args.seed) + ") ===")
    out.append("  Eval: " + str(len(eval_eps)) + "  fail_rate: " + str(round(fail_labels.mean(), 3)))
    out.append("  AUROC: " + str(auroc) + "  Pearson: " + str(pearson))
    if isinstance(auroc, float) and auroc == auroc:
        if auroc > 0.55:
            out.append("  RESULT: JOINT AUROC > 0.55. H1 NOT supported on this seed (joint not worse).")
        else:
            out.append("  RESULT: JOINT AUROC <= 0.55. H1 SUPPORTED on this seed (joint degraded).")
    else:
        out.append("  RESULT: AUROC=NaN. Inconclusive.")

    print(chr(10).join(out))
    log_path = ckpt_dir / "phase2_log.json"
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed, "mode": "JOINT ABLATION (true interleaved)",
        "n_ppo_steps": args.n_ppo_steps, "threshold_percentile": args.percentile,
        "threshold": float(threshold),
        "n_train_eps": len(train_eps), "n_eval_eps": len(eval_eps),
        "fail_rate": float(fail_labels.mean()),
        "eval_reward_mean": float(np.mean(eval_returns)),
        "eval_reward_std": float(np.std(eval_returns)),
        "eval_prob_mean": float(eval_probs.mean()),
        "eval_prob_std": float(eval_probs.std()),
        "auroc": float(auroc) if auroc == auroc else None,
        "pearson_prob_reward": float(pearson) if pearson == pearson else None,
        "n_monitor_steps": len(monitor_losses),
        "monitor_interval": args.monitor_interval,
        "monitor_epochs_per_step": args.monitor_epochs_per_step,
        "mean_monitor_loss": float(np.mean(monitor_losses)) if monitor_losses else 0.0,
        "note": "JOINT: PPO updated continuously; Monitor trained every K PPO updates on FRESH rollouts from CURRENT (not frozen) policy",
    }, indent=2))
    print("  saved " + str(log_path))


if __name__ == "__main__":
    main()