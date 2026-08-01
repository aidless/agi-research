# -*- coding: utf-8 -*-
"""
R1 test scaffold: Monitor in non-stationary cooperative MARL.

R1 hypothesis (Y5 section 7.6.3): "A learned auxiliary signal that fails
Condition 1 (distribution match) but produces useful training signal in
non-stationary contexts."

Test design:
  - Y3 cooperative multi-agent environment (simple_spread_v3)
  - Policy is periodically RESET to a new random initialization (the
    non-stationary part)
  - Monitor is trained on a frozen reference policy (fails Condition 1
    because the consumption-time policy drifts)
  - Test: does the Monitor still help the policy even when the policy
    distribution drifts away from the reference?

If R1 is NOT observed (Monitor does not rescue): framework survives.
If R1 IS observed (Monitor rescues despite drift): framework updates
to say Condition 1 is not strictly necessary.

Smoke test: 3 arms x 1 seed x 8 PPO updates
  - no_rescue: Monitor present, no reset
  - periodic_reset: Monitor present, policy reset every 4 updates
  - no_monitor_reset: no Monitor, policy reset every 4 updates (control)

Each arm produces one log file at experiments_log/_r1_<arm>_s<seed>.log.

Full R1 test: 3 arms x 20 seeds x 200 PPO updates per arm, MAX_PARALLEL=6.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the v8 architecture components
from pz_maddpg_v8 import (
    extract_dlr_preds, train_maddpg_v8, run_random_baseline,
    OBS_DIM, ACTION_DIM, DLR_PRED_DIM,
)


def train_with_periodic_reset(args):
    """Run pz_maddpg_v8 training with optional periodic policy reset.

    The non-stationarity is simulated by periodically resetting the
    actor/critic networks to a new random initialization. This makes
    the policy distribution drift away from the frozen reference
    distribution (violating Condition 1).

    If reset_interval is None or 0, no reset (standard stationary training).
    Otherwise, the policy is reset every `reset_interval` PPO updates.
    """
    use_dlr_trust = args.arm in ("v8", "monitor_only")
    use_dlr_critic = args.arm in ("v8", "dlr_only")
    reset_interval = args.reset_interval if args.arm != "no_rescue" else 0

    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns))
    rnd_std = float(np.std(rnd_returns))

    actors, trust_heads, history = train_maddpg_v8(
        seed=args.seed,
        n_updates=args.n_updates,
        n_episodes=args.n_episodes_per_update,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        use_dlr_trust=use_dlr_trust,
        use_dlr_critic=use_dlr_critic,
    )

    if reset_interval > 0:
        # Apply periodic reset: re-init actors/trust_heads every reset_interval.
        # Note: pz_maddpg_v8 may not currently support init_actors/init_trust_heads
        # parameters; if not, this path may fall back to fresh init each call.
        for reset_point in range(reset_interval, args.n_updates, reset_interval):
            print(f"  R1 reset at update {reset_point}/{args.n_updates}")
            try:
                actors, trust_heads, history = train_maddpg_v8(
                    seed=args.seed + reset_point + 1,  # different seed for re-init
                    n_updates=args.n_updates - reset_point,
                    n_episodes=args.n_episodes_per_update,
                    batch_size=args.batch_size,
                    buffer_size=args.buffer_size,
                    use_dlr_trust=use_dlr_trust,
                    use_dlr_critic=use_dlr_critic,
                    init_actors=actors, init_trust_heads=trust_heads,
                )
            except TypeError as e:
                # Fallback: pz_maddpg_v8 does not support init_actors yet
                print(f"  R1 (warning) init_actors not supported, using fresh init")
                actors, trust_heads, history = train_maddpg_v8(
                    seed=args.seed + reset_point + 1,
                    n_updates=args.n_updates - reset_point,
                    n_episodes=args.n_episodes_per_update,
                    batch_size=args.batch_size,
                    buffer_size=args.buffer_size,
                    use_dlr_trust=use_dlr_trust,
                    use_dlr_critic=use_dlr_critic,
                )

    final_returns = [h["mean_return"] for h in history[-15:]] if history else [0.0]
    final_mean = float(np.mean(final_returns))
    final_std = float(np.std(final_returns))
    delta = final_mean - rnd_mean
    print(f"  R1 ({args.arm}, reset_interval={reset_interval}) eval: "
          f"{final_mean:.2f} +/- {final_std:.2f}  (delta vs random: {delta:+.2f})")
    return {
        "arm": args.arm,
        "seed": args.seed,
        "reset_interval": reset_interval,
        "random_baseline_mean": rnd_mean,
        "random_baseline_std": rnd_std,
        "final_mean": final_mean,
        "final_std": final_std,
        "delta": delta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=str, required=True,
                   choices=["no_rescue", "periodic_reset", "no_monitor_reset"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-updates", type=int, default=8)
    p.add_argument("--n-episodes-per-update", type=int, default=4)
    p.add_argument("--n-eval-episodes", type=int, default=4)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--reset-interval", type=int, default=4)
    args = p.parse_args()

    print("=" * 60)
    print(f"R1 test scaffold -- arm={args.arm}, seed={args.seed}, "
          f"reset_interval={args.reset_interval if args.arm != 'no_rescue' else 0}")
    print("=" * 60)

    result = train_with_periodic_reset(args)
    log_dir = Path("E:/agi-research/experiments_log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"_r1_{args.arm}_s{args.seed}.json"
    log_path.write_text(json.dumps(result, indent=2))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
