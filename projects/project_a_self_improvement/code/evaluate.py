"""
evaluate.py — Measure the Monitor's predictive performance on a fresh
set of evaluation episodes (these should NOT overlap with the episodes
used to train the Monitor).

What we report:
    1. Average episode reward (just to confirm policy quality)
    2. Monitor AUROC for "predicts the eventual outcome is failure"
    3. Pearson correlation between Monitor-predicted-failure-probability
       (mean over the episode) and total episode reward.

This is the main script we use to justify the paper's Figure 2.

Usage:
    python evaluate.py --env CartPole-v1 --n-episodes 100
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from envs import EpisodeLog, is_failure_episode, make_env, rollout_one_episode
from monitor import FailureMonitor, MonitorConfig
from ppo import Policy


HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)  # different from training seed!
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold for failure classification")
    return p.parse_args()


def load_policy(env, ckpt_path: Path, device: str = "cpu") -> Policy:
    pkg = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = pkg["policy"]
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    p = Policy(obs_dim, n_actions, hidden=64)
    p.load_state_dict(cfg)
    p.eval()
    return p


def load_monitor(ckpt_path: Path, device: str = "cpu") -> tuple[FailureMonitor, int]:
    pkg = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = pkg["config"]
    cfg = MonitorConfig(**cfg_dict)
    m = FailureMonitor(cfg)
    m.load_state_dict(pkg["monitor"])
    m.eval()
    return m, int(pkg["history_dim"])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. NaN if either has zero variance."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = get_args()
    env = make_env(args.env, seed=args.seed)
    obs_dim = env.observation_space.shape[0]

    print(f"\n[Evaluate] env={args.env}  episodes={args.n_episodes}  seed={args.seed}\n")

    policy = load_policy(env, CKPT_DIR / "policy.pt")
    monitor, _ = load_monitor(CKPT_DIR / "monitor.pt")

    # Roll out n episodes, build EpisodeLog AND track per-episode monitor predictions
    episodes: list[EpisodeLog] = []
    mean_monitor_p: list[float] = []
    final_monitor_p: list[float] = []

    for i in range(args.n_episodes):
        ep_probs = []
        obs, _ = env.reset(seed=args.seed + i)
        done = False
        steps = 0
        ep_log = EpisodeLog(env_name=args.env)

        while not done and steps < 1000:
            a = int(policy.act(obs))
            from envs import Transition
            ep_log.transitions.append(Transition(obs=obs, action=a, reward=0.0))
            new_obs, r, term, trunc, _ = env.step(a)
            ep_log.transitions[-1].reward = float(r)
            ep_log.total_reward += float(r)

            # query monitor every 4 steps to save compute (REVIEW-ME)
            if steps % 4 == 0:
                vec = ep_log.history_vector(history_len=args.history_len)
                p = monitor.predict(vec)
                ep_probs.append(p)

            obs = new_obs
            done = term or trunc
            steps += 1
        episodes.append(ep_log)
        mean_monitor_p.append(float(np.mean(ep_probs)) if ep_probs else 0.5)
        final_monitor_p.append(float(ep_probs[-1]) if ep_probs else 0.5)

    rewards = np.array([e.total_reward for e in episodes])
    fail_labels = np.array([1.0 if is_failure_episode(e) else 0.0 for e in episodes])
    mean_p = np.array(mean_monitor_p)
    final_p = np.array(final_monitor_p)

    # --- AUROC: does Monitor-mean-p predict eventual failure? ---
    from monitor import _quick_auroc
    auroc_mean = _quick_auroc(fail_labels, mean_p)
    auroc_final = _quick_auroc(fail_labels, final_p)

    corr_mean_r = pearson(mean_p, rewards)
    corr_mean_f = pearson(mean_p, fail_labels)

    print("=== Results ===")
    print(f"  Eval episodes            : {args.n_episodes}")
    print(f"  Failure-rate (label rate): {fail_labels.mean():.3f}")
    print(f"  Reward  mean ± std       : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"  Monitor mean prob  mean±std: {mean_p.mean():.3f} ± {mean_p.std():.3f}")
    print(f"  AUROC (mean prob → failure): {auroc_mean:.3f}")
    print(f"  AUROC (final prob → failure): {auroc_final:.3f}")
    print(f"  Pearson(mean_p, reward)    : {corr_mean_r:.3f}")
    print(f"  Pearson(mean_p, fail)      : {corr_mean_f:.3f}")
    print()
    print("Interpretation guide:")
    print("  AUROC > 0.5  → Monitor has signal.")
    print("  AUROC ~ 0.5  → Monitor is noise.")
    print("  AUROC < 0.5  → Monitor is INVERTED (we trained it wrong).")
    print("  Pearson(mean_p, fail) > 0 → higher monitor prob correlates with failure.")
    if auroc_mean > 0.6 and fail_labels.mean() > 0.05:
        print("\n  RESULT: Monitor has predictive signal! Paper v1 main claim SUPPORTED.")
    else:
        print("\n  RESULT: Monitor did NOT show strong signal. Probably needs more")
        print("          eval episodes, different env, or different failure threshold.")

    out = {
        "n_episodes": args.n_episodes,
        "fail_rate": float(fail_labels.mean()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "mean_monitor_p": float(mean_p.mean()),
        "std_monitor_p": float(mean_p.std()),
        "auroc_mean": float(auroc_mean),
        "auroc_final": float(auroc_final),
        "pearson_monitor_vs_reward": float(corr_mean_r) if not np.isnan(corr_mean_r) else None,
        "pearson_monitor_vs_failure": float(corr_mean_f) if not np.isnan(corr_mean_f) else None,
    }
    (CKPT_DIR / "eval_log.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {CKPT_DIR / 'eval_log.json'}")


if __name__ == "__main__":
    main()
