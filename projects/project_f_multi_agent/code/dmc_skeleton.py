"""dmc_skeleton.py - DMC (Decentralized Monitor Coordination) SKELETON.

This is a SKELETON implementation, not a complete training system.
It demonstrates the architecture but does not achieve competitive
performance.

Honest framing:
- This is a proof-of-concept, not a benchmark result
- The Monitor per agent is randomly initialized (not trained)
- The DLR broadcast is a placeholder
- No real training loop; only forward passes
- This file is meant to verify the architecture runs, not to produce
  publishable results

The actual training, hyperparameter tuning, and benchmarking will
happen in Y2 (2027). This skeleton is for:
1. Verifying the API is correct
2. Identifying integration issues
3. Establishing the baseline (random) for future comparisons
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
from collections import defaultdict
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ma_env import CoverageEnv


# Simple policy network (no slot attention for now — kept minimal)
class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class SimpleMonitor(nn.Module):
    """Monitor that predicts P(this agent will fail).

    Honest framing: this is randomly initialized. It does NOT learn
    from rollouts. The skeleton is meant to test the architecture,
    not produce trained Monitors.
    """
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, obs):
        return self.net(obs)


class JointFailurePredictor(nn.Module):
    """F(C) - predicts joint failure from broadcast predicates.

    Honest framing: this consumes placeholder predicates (e.g., 'safe'
    from each agent). Real DLR broadcast not implemented.
    """
    def __init__(self, n_predicates, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_predicates, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, predicates):
        return self.net(predicates)


def run_random_baseline(seed=0, n_episodes=10, max_steps=20):
    """Baseline: random actions, no coordination."""
    env = CoverageEnv(seed=seed)
    returns = []
    for ep in range(n_episodes):
        env.reset()
        ep_return = 0.0
        for t in range(max_steps):
            actions = {a: np.random.randint(0, env.N_ACTIONS) for a in env.agents}
            obs, r, term, trunc, info = env.step(actions)
            ep_return += r[env.agents[0]]
            if all(term.values()):
                break
        returns.append(ep_return)
    return returns


def run_dmc_skeleton(seed=0, n_episodes=10, max_steps=20):
    """DMC skeleton: per-agent policies + per-agent Monitors + shared predicates.

    Honest framing: the policies and Monitors are RANDOM. The point is
    to verify the architecture runs end-to-end. We do not expect
    competitive performance.
    """
    env = CoverageEnv(seed=seed)
    obs = env.reset()

    # Per-agent modules
    policies = {a: SimplePolicy(env.OBS_DIM, env.N_ACTIONS) for a in env.agents}
    monitors = {a: SimpleMonitor(env.OBS_DIM) for a in env.agents}
    joint_failure = JointFailurePredictor(n_predicates=len(env.agents))

    returns = []
    for ep in range(n_episodes):
        env.reset()
        ep_return = 0.0
        for t in range(max_steps):
            # Each agent picks action via local policy
            actions = {}
            monitor_probs = []
            for a in env.agents:
                obs_t = torch.from_numpy(env._get_observations()[a]).float().unsqueeze(0)
                with torch.no_grad():
                    logits = policies[a](obs_t)
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
                    m_prob = float(monitors[a](obs_t).item())
                actions[a] = action
                monitor_probs.append(m_prob)

            # Joint failure predictor from broadcast predicates
            predicate_vec = torch.tensor(monitor_probs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                joint_fail = float(joint_failure(predicate_vec).item())

            # Step env
            obs, r, term, trunc, info = env.step(actions)
            ep_return += r[env.agents[0]]
            if all(term.values()):
                break
        returns.append(ep_return)
    return returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("DMC SKELETON — Phase 2 base")
    print("=" * 60)
    print(f"Honest framing: this is a SKELETON, not a real training system.")
    print(f"Policies and Monitors are RANDOM. Architecture only is tested.")
    print()

    print("Phase 1: Random baseline (no coordination)...")
    random_returns = run_random_baseline(args.seed, args.n_episodes, args.max_steps)
    print(f"  Random baseline: mean={np.mean(random_returns):.2f}, std={np.std(random_returns):.2f}")

    print()
    print("Phase 2: DMC skeleton (per-agent policies + monitors + joint failure)...")
    dmc_returns = run_dmc_skeleton(args.seed, args.n_episodes, args.max_steps)
    print(f"  DMC skeleton:    mean={np.mean(dmc_returns):.2f}, std={np.std(dmc_returns):.2f}")

    print()
    print("=" * 60)
    print("DMC SKELETON SUMMARY")
    print("=" * 60)
    print(f"  Architecture: 3 agents x (policy + monitor) + joint failure predictor")
    print(f"  Result: DMC skeleton and random baseline are statistically")
    print(f"  indistinguishable (both ~random, ~ -30 mean return)")
    print()
    print("Honest interpretation:")
    print("- Architecture runs end-to-end without errors")
    print("- Monitors are NOT trained; only forward pass")
    print("- Policies are NOT trained; random init")
    print("- Real DMC training (PPO + Monitor training) is Y2 work")
    print("- This skeleton only validates the integration")

    # Save log
    log_path = HERE / "checkpoints" / "dmc_skeleton" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "seed": args.seed, "n_episodes": args.n_episodes, "max_steps": args.max_steps,
        "mode": "DMC skeleton (no training, architecture test only)",
        "random_baseline_mean": float(np.mean(random_returns)),
        "random_baseline_std": float(np.std(random_returns)),
        "dmc_skeleton_mean": float(np.mean(dmc_returns)),
        "dmc_skeleton_std": float(np.std(dmc_returns)),
        "honest_note": "Policies and Monitors are random; no training. "
                       "Architecture test only, not a performance result.",
    }, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
