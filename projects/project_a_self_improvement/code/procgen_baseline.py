#!/usr/bin/env python3
"""
procgen_baseline.py — Procgen CPU-friendly experiment orchestration for Project A.

Phases:
  1. Policy-only baseline: 16 games × 5 seeds, each 25M env steps.
     Collects per-game reward series for Monitor threshold calibration.
  2. (Future) Self-critic intervention training.
  3. (Future) Final evaluation: 16 games × 5 seeds per condition.

Usage:
    python code/procgen_baseline.py --phase 1 --n-seeds 2 --n-timesteps 1_000_000
    python code/procgen_baseline.py --phase 1 --n-seeds 5 --n-timesteps 25_000_000 --full
"""

from __future__ import annotations
import argparse
import json
import os
import time
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envs
from encoders import ProcgenEncoder
from ppo import PPOConfig, PPOAgent


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "procgen"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Phase 1: Policy-only baseline ────────────────────────────────────────────

def run_single_game(
    game: str,
    seed: int,
    n_timesteps: int,
    encoder,
    rollout_len: int = 2048,
    lr: float = 3e-4,
    hidden: int = 64,
    log_interval: int = 10,
) -> dict:
    """Train PPO on one procgen game for `n_timesteps` env steps.

    Returns summary dict with per-update returns for threshold calibration.
    """
    cfg = PPOConfig(
        obs_dim=encoder.obs_dim,
        n_actions=15,  # procgen action space is 15
        lr=lr,
        hidden=hidden,
        rollout_len=rollout_len,
        seed=seed,
    )
    agent = PPOAgent(cfg, device="cpu")
    env = envs.create_procgen_env(game, seed=seed + 42, obs_encoder=encoder,
                                  num_levels=0, start_level=seed * 1000)

    obs, _ = env.reset()
    updates = n_timesteps // rollout_len
    all_returns: list[float] = []
    episode_returns: list[float] = []

    for update_i in range(updates):
        batch = agent.collect_rollout(env, obs)
        info = agent.update(batch)
        obs = batch["final_obs"]
        all_returns.extend(batch["ep_returns"])
        episode_returns.extend(batch["ep_returns"])

        if update_i % log_interval == 0:
            mean_ret = np.mean(batch["ep_returns"]) if batch["ep_returns"] else 0.0
            print(
                f"[{game}] seed={seed}  update={update_i}/{updates}  "
                f"ep_ret={mean_ret:.1f}  "
                f"pl={info['policy_loss']:.3f}  vl={info['value_loss']:.3f}  "
                f"ent={info['entropy']:.3f}"
            )

    env.close()
    return {
        "game": game,
        "seed": seed,
        "n_timesteps": n_timesteps,
        "updates": updates,
        "episode_returns": episode_returns,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "median_return": float(np.median(episode_returns)) if episode_returns else 0.0,
        "std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "n_episodes": len(episode_returns),
    }


def phase1(n_seeds: int, n_timesteps: int, games: list[str] | None = None):
    """Run policy-only baseline on all (or selected) procgen games."""
    target_games = games or envs.ALL_PROCGEN_GAMES
    encoder = ProcgenEncoder(target_size=32)
    all_results: dict[str, list[dict]] = {}

    for game in target_games:
        print(f"\n{'='*60}")
        print(f"Phase 1 — {game}  ({n_seeds} seeds × {n_timesteps:,} steps)")
        print(f"{'='*60}")
        game_results: list[dict] = []
        for seed in range(n_seeds):
            t0 = time.time()
            result = run_single_game(game, seed, n_timesteps, encoder)
            elapsed = time.time() - t0
            result["elapsed_sec"] = round(elapsed, 1)
            print(f"  → done in {elapsed:.0f}s  mean_ret={result['mean_return']:.2f}")
            game_results.append(result)

        # Compute adaptive threshold
        all_returns = []
        for r in game_results:
            all_returns.extend(r["episode_returns"])
        threshold = envs.percentile_failure_threshold(all_returns, percentile=30.0)
        print(f"  ★ p30 threshold for {game}: {threshold:.2f}")

        all_results[game] = {
            "seeds": game_results,
            "threshold_p30": threshold,
            "group": envs.PROCGEN_GAME_GROUPS.get(game, "?"),
        }

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"phase1_{timestamp}.json"
    with open(path, "w") as f:
        json.dump({"config": {"n_seeds": n_seeds, "n_timesteps": n_timesteps},
                    "games": all_results}, f, indent=2)
    print(f"\nSaved → {path}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary  (mean return across seeds)")
    print(f"{'='*60}")
    print(f"{'Game':<14} {'Group':<6} {'MeanRet':<10} {'p30':<10} {'N_ep':<8}")
    print("-" * 50)
    for game in target_games:
        r = all_results[game]
        seeds = r["seeds"]
        avg = np.mean([s["mean_return"] for s in seeds])
        print(f"{game:<14} {r['group']:<6} {avg:<10.1f} {r['threshold_p30']:<10.2f} "
              f"{sum(s['n_episodes'] for s in seeds):<8}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Procgen baseline experiments")
    p.add_argument("--phase", type=int, default=1, choices=[1],
                   help="Experiment phase (default: 1)")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-timesteps", type=int, default=25_000_000)
    p.add_argument("--games", nargs="*", help="Subset of games (default: all 16)")
    p.add_argument("--full", action="store_true",
                   help="Run all 16 games (default: train games only)")
    args = p.parse_args()

    games = args.games
    if games is None:
        games = list(envs.ALL_PROCGEN_GAMES if args.full else envs.PROCGEN_TRAIN_GAMES)

    if args.phase == 1:
        phase1(args.n_seeds, args.n_timesteps, games)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()