"""
main.py — Entry point for Project A v1.

Pipeline:
    1. Train PPO policy on chosen env (default CartPole-v1).
    2. After policy is "good enough", collect evaluation rollouts.
    3. Train the FailureMonitor on those rollouts.
    4. Save policy + monitor to ./checkpoints/

Usage (from inside code/):

    python main.py --env CartPole-v1 --total-steps 60000 --eval-episodes 80

REVIEW-ME:
- Default ``total_steps=60_000`` is chosen so v1 finishes on a laptop CPU
  in about 3-5 minutes for CartPole-v1. Bump it for harder envs.
- After v1 works end-to-end we'll add a CLI flag for ``--disable-monitor``
  to A/B test baseline only.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from envs import EpisodeLog, make_env, rollout_one_episode
from ppo import PPOAgent, PPOConfig
from monitor import MonitorConfig, train_monitor


HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True, parents=True)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--disable-monitor", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    print(f"\n[Project A] v1 on {args.env}  seed={args.seed}  total_steps={args.total_steps}\n")
    env = make_env(args.env, seed=args.seed)
    obs0, _ = env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, seed=args.seed)
    agent = PPOAgent(cfg)

    # ---------- Stage 1: Train policy ----------
    print("=== Stage 1: Train policy via PPO ===")
    total = 0
    iteration = 0
    ep_returns_window = []
    while total < args.total_steps:
        batch = agent.collect_rollout(env, obs0)
        info = agent.update(batch)
        obs0 = batch["final_obs"]
        total += cfg.rollout_len
        iteration += 1
        ep_returns_window.extend(batch["ep_returns"])
        ep_returns_window = ep_returns_window[-30:]
        if iteration % 5 == 0:
            avg = float(np.mean(ep_returns_window)) if ep_returns_window else float("nan")
            print(f"  iter={iteration}  steps={total}  "
                  f"avg_return(last_30ep)={avg:.1f}  "
                  f"pol_loss={info['policy_loss']:.3f}  "
                  f"v_loss={info['value_loss']:.3f}")

    # save policy
    torch.save({
        "policy": agent.policy.state_dict(),
        "value": agent.value.state_dict(),
        "env": args.env,
        "seed": args.seed,
    }, CKPT_DIR / "policy.pt")
    print(f"\n[saved] {CKPT_DIR / 'policy.pt'}")

    # ---------- Stage 2: Collect rollouts for Monitor ----------
    print("\n=== Stage 2: Collect rollouts for Monitor training ===")
    eval_episodes: list[EpisodeLog] = []
    for i in range(args.eval_episodes):
        # set seed per episode so behaviour is reproducible
        e = rollout_one_episode(env, agent.select_action)
        e.env_name = args.env
        eval_episodes.append(e)
    rets = [e.total_reward for e in eval_episodes]
    print(f"  collected {len(eval_episodes)} eval episodes")
    print(f"  reward mean={np.mean(rets):.1f}  std={np.std(rets):.1f}  "
          f"min={np.min(rets):.1f}  max={np.max(rets):.1f}")

    # ---------- Stage 3: Train Monitor ----------
    if args.disable_monitor:
        print("\n[disabled] monitor training skipped")
        monitor = None
        monitor_metrics = {"auroc": float("nan"), "final_loss": float("nan")}
    else:
        print("\n=== Stage 3: Train Failure Monitor (frozen-policy decoupling) ===")
        history_dim = args.history_len * (obs_dim + 2)
        mcfg = MonitorConfig(history_dim=history_dim, seed=args.seed)
        monitor, monitor_metrics = train_monitor(
            mcfg, eval_episodes, history_len=args.history_len, verbose=True
        )
        torch.save({
            "monitor": monitor.state_dict(),
            "history_dim": history_dim,
            "config": vars(mcfg),
        }, CKPT_DIR / "monitor.pt")
        print(f"\n[saved] {CKPT_DIR / 'monitor.pt'}")

    # ---------- Summary ----------
    print("\n=== Run Summary ===")
    print(f"  policy: stage1 final avg ep return = {np.mean(rets):.2f}")
    print(f"  monitor: auroc = {monitor_metrics['auroc']:.3f}, "
          f"final_loss = {monitor_metrics['final_loss']:.4f}")

    # machine-readable run log so evaluate.py + the user can see what we did
    import json, time
    log = {
        "env": args.env,
        "seed": args.seed,
        "total_steps": total,
        "policy_iterations": iteration,
        "eval_episode_count": len(eval_episodes),
        "eval_reward_mean": float(np.mean(rets)),
        "eval_reward_std": float(np.std(rets)),
        "monitor_metrics": monitor_metrics,
        "timestamp": time.time(),
    }
    (CKPT_DIR / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"  run log: {CKPT_DIR / 'run_log.json'}")


if __name__ == "__main__":
    main()
