#!/usr/bin/env python3
"""
procgen_phase2.py - Project A Phase 2: decoupled Failure Monitor on Procgen.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import envs
from encoders import ProcgenEncoder
from ppo import PPOAgent, PPOConfig
from monitor import FailureMonitor, MonitorConfig, train_monitor, _quick_auroc
from envs import rollout_one_episode


def collect_rollouts(env_factory, agent, n_episodes, history_len, seed):
    episodes = []
    for i in range(n_episodes):
        env = env_factory(seed * 1000 + i + 42)
        ep_log = rollout_one_episode(env, agent.select_action, max_steps=200)
        ep_log.env_name = "procgen"
        if len(ep_log.transitions) > history_len:
            ep_log.transitions = ep_log.transitions[-history_len:]
        episodes.append(ep_log)
        env.close()
    return episodes


def per_episode_monitor_probs(monitor, episodes, obs_dim, n_actions, history_len):
    means = []
    per_step_dim = obs_dim + n_actions + 1
    for ep in episodes:
        per_step = []
        for t, tr in enumerate(ep.transitions):
            history = ep.transitions[: t + 1][-history_len:]
            vec = np.zeros(history_len * per_step_dim, dtype=np.float32)
            for k, h_tr in enumerate(history):
                base = k * per_step_dim
                vec[base : base + obs_dim] = h_tr.obs
                vec[base + obs_dim + h_tr.action] = 1.0
                vec[base + obs_dim + n_actions] = h_tr.reward
            prob = monitor.predict(vec)
            per_step.append(prob)
        if per_step:
            means.append(float(np.mean(per_step)))
        else:
            means.append(0.5)
    return np.array(means)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="coinrun", choices=envs.PROCGEN_TRAIN_GAMES)
    parser.add_argument("--n-ppo-steps", type=int, default=50000)
    parser.add_argument("--n-train-episodes", type=int, default=100)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor-epochs", type=int, default=5)
    args = parser.parse_args()

    print(f"\n[Project A Phase 2] game={args.game}  seed={args.seed}")
    print(f"  PPO steps={args.n_ppo_steps}  train_eps={args.n_train_episodes}  "
          f"eval_eps={args.n_eval_episodes}  history_len={args.history_len}\n")

    encoder = ProcgenEncoder(target_size=32)
    obs_dim = encoder.obs_dim
    n_actions = 15

    def make_env(seed_offset):
        return envs.create_procgen_env(
            args.game, seed=seed_offset, obs_encoder=encoder,
            num_levels=200, start_level=seed_offset % 200,
        )

    print("[Stage 1] Training PPO baseline...")
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    train_env = make_env(args.seed + 1)
    obs, _ = train_env.reset()
    n_updates = args.n_ppo_steps // cfg.rollout_len
    all_returns = []
    for u in range(n_updates):
        batch = agent.collect_rollout(train_env, obs)
        info = agent.update(batch)
        obs = batch["final_obs"]
        all_returns.extend(batch["ep_returns"])
        if (u + 1) % 5 == 0:
            mean_r = np.mean(all_returns[-100:]) if all_returns else 0.0
            print(f"  update={u+1}/{n_updates}  mean_return(last100)={mean_r:.2f}")
    train_env.close()

    if all_returns:
        p30 = envs.percentile_failure_threshold(all_returns, 30.0)
        print(f"\n  PPO complete. mean_return={np.mean(all_returns):.2f}  p30={p30:.2f}  "
              f"n_episodes={len(all_returns)}")
    else:
        p30 = 0.0

    print(f"\n[Stage 2] Collecting {args.n_train_episodes} frozen-policy rollouts for Monitor...")
    train_eps = collect_rollouts(
        lambda offset: make_env(args.seed * 1000 + offset + 7777),
        agent, args.n_train_episodes, args.history_len, args.seed,
    )
    train_returns = [e.total_reward for e in train_eps]
    print(f"  collected {len(train_eps)} train episodes, "
          f"returns mean={np.mean(train_returns):.2f}")

    print(f"\n[Stage 3] Training decoupled FailureMonitor...")
    history_dim = args.history_len * (obs_dim + n_actions + 1)
    mcfg = MonitorConfig(history_dim=history_dim, seed=args.seed, epochs=args.monitor_epochs)
    monitor, mmetrics = train_monitor(mcfg, train_eps, history_len=args.history_len, verbose=True)

    ckpt_dir = HERE / "checkpoints" / f"procgen_{args.game}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"monitor": monitor.state_dict(),
                "history_dim": history_dim,
                "config": vars(mcfg)},
               ckpt_dir / "monitor.pt")
    print(f"\n  saved {ckpt_dir / 'monitor.pt'}")

    print(f"\n[Stage 4] Evaluating Monitor on {args.n_eval_episodes} held-out episodes...")
    eval_eps = collect_rollouts(
        lambda offset: make_env(args.seed * 1000 + offset + 9999),
        agent, args.n_eval_episodes, args.history_len, args.seed + 1,
    )
    eval_returns = [e.total_reward for e in eval_eps]
    eval_probs = per_episode_monitor_probs(
        monitor, eval_eps, obs_dim, n_actions, args.history_len,
    )

    threshold = p30 if p30 > 0 else 0.0
    fail_labels = np.array([1.0 if r < threshold else 0.0 for r in eval_returns])

    auroc_mean = _quick_auroc(fail_labels, eval_probs)

    if eval_probs.std() > 1e-9:
        pearson_p = float(np.corrcoef(eval_probs, eval_returns)[0, 1])
    else:
        pearson_p = float("nan")
    if eval_probs.std() > 1e-9 and fail_labels.std() > 1e-9:
        pearson_f = float(np.corrcoef(eval_probs, fail_labels)[0, 1])
    else:
        pearson_f = float("nan")

    print(f"\n=== Phase 2 results ({args.game} seed={args.seed}) ===")
    print(f"  Eval episodes            : {len(eval_eps)}")
    print(f"  Failure-rate (label rate): {fail_labels.mean():.3f}")
    print(f"  Reward mean +/- std      : {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")
    print(f"  Monitor mean prob mean+/-std: {eval_probs.mean():.3f} +/- {eval_probs.std():.3f}")
    print(f"  AUROC (mean prob -> fail): {auroc_mean:.3f}")
    print(f"  Pearson (prob, reward)   : {pearson_p:.3f}")
    print(f"  Pearson (prob, fail)     : {pearson_f:.3f}")
    print()
    if auroc_mean > 0.55 and fail_labels.mean() > 0:
        print("  RESULT: Monitor predicts failure above chance. Decoupling signal SUPPORTED.")
    elif fail_labels.mean() == 0:
        print("  RESULT: No failure cases found at p30=0 (policy too uniform).")
    else:
        print("  RESULT: Monitor did NOT show signal. Need more / better data.")

    out = {
        "game": args.game,
        "seed": args.seed,
        "n_ppo_steps": args.n_ppo_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_eval_episodes": args.n_eval_episodes,
        "history_len": args.history_len,
        "p30_threshold": float(p30),
        "n_episodes_training": len(all_returns),
        "mean_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "fail_rate": float(fail_labels.mean()),
        "eval_reward_mean": float(np.mean(eval_returns)),
        "eval_reward_std": float(np.std(eval_returns)),
        "eval_prob_mean": float(eval_probs.mean()),
        "eval_prob_std": float(eval_probs.std()),
        "auroc_mean": float(auroc_mean),
        "pearson_prob_reward": float(pearson_p) if not np.isnan(pearson_p) else None,
        "pearson_prob_fail": float(pearson_f) if not np.isnan(pearson_f) else None,
        "monitor_train_metrics": mmetrics,
    }

    log_path = ckpt_dir / "phase2_log.json"
    log_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {log_path}")


if __name__ == "__main__":
    main()
