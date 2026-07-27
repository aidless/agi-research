"""ppo_only_baseline.py - Y1.3 control condition: PPO only, no Monitor.

This is the baseline against which y13_monitor_regularizer.py is compared.
Same PPO budget (100K steps), same eval protocol (50 episodes), same seeds.
The only difference: NO Monitor is ever trained or used.

If y13 > baseline: Monitor signal is useful for training.
If y13 = baseline: Monitor is noise (averages out in training).
If y13 < baseline: Monitor signal is anti-correlated (unlikely given
Monitor AUROC 0.99).
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
from envs import rollout_one_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", default="ppobase")
    args = p.parse_args()

    out = []
    out.append(f"[PPO-Only Baseline] env={args.env} seed={args.seed}")
    out.append(f"  n_ppo={args.n_ppo_steps}  n_eval={args.n_eval_episodes}")
    out.append("=" * 70)

    # PPO training (no Monitor, no shaping)
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
    env.close()
    out.append(f"[PPO] trained: {args.n_ppo_steps} steps, {n_updates} updates")

    # Evaluation
    eval_returns = []
    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        eval_returns.append(ep.total_reward)
        e.close()

    out.append("")
    out.append("=" * 70)
    out.append("PPO-ONLY BASELINE EVALUATION")
    out.append("=" * 70)
    out.append(f"  Episodes:    {args.n_eval_episodes}")
    out.append(f"  Mean return: {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")
    out.append(f"  Median:      {np.median(eval_returns):.2f}")

    print(chr(10).join(out))

    log_dir = HERE / "checkpoints" / ("full_integration_" + args.out_tag + "_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "Y1.3 PPO-only baseline (no Monitor)",
        "n_ppo_steps": args.n_ppo_steps,
        "n_eval_episodes": args.n_eval_episodes,
        "eval_mean": float(np.mean(eval_returns)),
        "eval_std": float(np.std(eval_returns)),
        "per_episode_eval": [float(x) for x in eval_returns],
    }, indent=2))


if __name__ == "__main__":
    main()
