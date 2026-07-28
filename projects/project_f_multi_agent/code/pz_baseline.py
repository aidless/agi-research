"""pz_baseline.py - Per-agent PPO on PettingZoo Simple Spread (REAL benchmark).

This is the FIRST real multi-agent benchmark for the Archimedes
Project. PettingZoo Simple Spread is the standard cooperative MARL
benchmark (3 agents, 3 landmarks, coverage reward).

Honest framing:
- We are running on a REAL benchmark (not hand-coded)
- Per-agent PPO is the standard baseline (parameter sharing optional)
- This is a SKELETON training loop, not a tuned PPO
- 5 seeds × 1 short training run to verify the pipeline
- Real results will require longer training (Y2 work)

What this DOES validate:
- PettingZoo integration end-to-end
- Per-agent PPO can be trained on cooperative MA env
- Independent PPO (no parameter sharing) is a valid baseline

What this does NOT validate:
- DMC vs PPO comparison (DMC needs trained Monitors, not random)
- Best PPO performance (no hyperparameter tuning)
- Generalization across MARL benchmarks
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
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Use the explicit pettingzoo import (we have 1.24.1 in hermes venv)
from pettingzoo.mpe import simple_spread_v3


# Simple per-agent PPO policy (untuned, no shared parameters)
class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value.squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action, value.item(), logits


def run_random_baseline(seed=0, n_episodes=5, max_cycles=25):
    """Random actions, no training."""
    env = simple_spread_v3.env(N=3, max_cycles=max_cycles, continuous_actions=False)
    returns = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last()
            ep_return += reward
            if term or trunc:
                action = None
            else:
                action = env.action_space(a).sample()
            env.step(action)
            if env.agents == []:
                break
        returns.append(ep_return)
    env.close()
    return returns


def collect_episode(policies, env, seed):
    """Collect one episode of (obs, action, log_prob, value, reward) per agent."""
    env.reset(seed=seed)
    transitions = {a: [] for a in env.possible_agents}
    ep_returns = {a: 0.0 for a in env.possible_agents}
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_returns[a] += reward
        if term or trunc:
            action = None
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, value, logits = policies[a].act(obs_t)
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
            transitions[a].append({
                "obs": obs.copy(),
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": 0.0,  # filled below
            })
        env.step(action)
    # Distribute joint reward: each agent gets its own reward (cooperative)
    for a in transitions:
        for t in transitions[a]:
            t["reward"] = ep_returns[a]
    return transitions, ep_returns


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages


def ppo_update(policy, optimizer, trajectories, n_epochs=4, batch_size=32, clip=0.2):
    """Single PPO update on collected trajectories."""
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_advantages = []
    all_returns = []

    for traj in trajectories:
        obs = torch.from_numpy(np.stack([t["obs"] for t in traj])).float()
        actions = torch.tensor([t["action"] for t in traj], dtype=torch.long)
        old_log_probs = torch.tensor([t["log_prob"] for t in traj])
        rewards = [t["reward"] for t in traj]
        values = [t["value"] for t in traj]
        # Use last value as bootstrap
        dones = [False] * (len(traj) - 1) + [True]
        advantages = compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        all_obs.append(obs)
        all_actions.append(actions)
        all_log_probs.append(old_log_probs)
        all_advantages.append(torch.tensor(advantages, dtype=torch.float32))
        all_returns.append(torch.tensor(returns, dtype=torch.float32))

    obs = torch.cat(all_obs)
    actions = torch.cat(all_actions)
    old_log_probs = torch.cat(all_log_probs)
    advantages = torch.cat(all_advantages)
    returns = torch.cat(all_returns)
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)
    indices = np.arange(n)

    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            bi = indices[start:start + batch_size]
            obs_b = obs[bi]
            actions_b = actions[bi]
            old_lp_b = old_log_probs[bi]
            adv_b = advantages[bi]
            ret_b = returns[bi]

            logits, values = policy(obs_b)
            log_probs = F.log_softmax(logits, dim=-1)
            new_lp = log_probs[torch.arange(len(bi)), actions_b]

            ratio = torch.exp(new_lp - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_b)
            entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
    return loss.item()


def train_ppo_per_agent(n_episodes=20, n_epochs=4, n_updates=10, seed=0, max_cycles=25):
    """Train per-agent PPO on simple_spread_v3.

    Honest framing: this is a SHORT training run. Real PPO convergence
    needs 500-1000 episodes. We use 10 updates × 20 episodes = 200
    episodes total to verify the pipeline.
    """
    env = simple_spread_v3.env(N=3, max_cycles=max_cycles, continuous_actions=False)
    obs_dim = env.observation_space("agent_0").shape[0]
    n_actions = env.action_space("agent_0").n
    env.close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Per-agent policies (no parameter sharing)
    policies = {a: PPOPolicy(obs_dim, n_actions) for a in ["agent_0", "agent_1", "agent_2"]}
    optimizers = {a: torch.optim.Adam(policies[a].parameters(), lr=3e-4) for a in policies}

    print(f"  Per-agent PPO: {len(policies)} agents, obs_dim={obs_dim}, n_actions={n_actions}")

    history = []
    for u in range(n_updates):
        # Collect episodes from all agents
        all_trajectories = {a: [] for a in policies}
        all_returns = []
        for ep in range(n_episodes):
            transitions, ep_returns = collect_episode(
                policies, simple_spread_v3.env(N=3, max_cycles=max_cycles,
                                                 continuous_actions=False),
                seed=seed * 1000 + u * 100 + ep,
            )
            for a in policies:
                all_trajectories[a].append(transitions[a])
            all_returns.append(np.mean(list(ep_returns.values())))

        # Update each agent's policy
        for a in policies:
            trajs = all_trajectories[a]
            loss = ppo_update(policies[a], optimizers[a], trajs, n_epochs=n_epochs)
        mean_return = float(np.mean(all_returns))
        history.append({"update": u, "mean_return": mean_return})
        if (u + 1) % 2 == 0 or u == 0:
            print(f"    update {u+1}/{n_updates}: mean_episode_return={mean_return:.2f}")

    return policies, history


def evaluate(policies, n_episodes=10, seed=100, max_cycles=25):
    """Evaluate trained policies (deterministic)."""
    env = simple_spread_v3.env(N=3, max_cycles=max_cycles, continuous_actions=False)
    returns = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last()
            ep_return += reward
            if term or trunc:
                action = None
            else:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = policies[a].act(obs_t, deterministic=True)
            env.step(action)
            if env.agents == []:
                break
        returns.append(ep_return)
    env.close()
    return returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--n-updates", type=int, default=10)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--max-cycles", type=int, default=25)
    args = p.parse_args()

    print("=" * 60)
    print("Per-agent PPO baseline on PettingZoo Simple Spread v3")
    print("=" * 60)
    print(f"  seed={args.seed}, n_episodes={args.n_episodes}, n_updates={args.n_updates}")
    print(f"  max_cycles={args.max_cycles}, N=3 agents, 3 landmarks")
    print(f"  HONEST: short training (200 episodes), not converged PPO")
    print()

    # Phase 1: Random baseline
    print("Phase 1: Random baseline...")
    random_returns = run_random_baseline(args.seed, n_episodes=10, max_cycles=args.max_cycles)
    print(f"  Random mean: {np.mean(random_returns):.2f} +/- {np.std(random_returns):.2f}")

    # Phase 2: Per-agent PPO
    print()
    print(f"Phase 2: Per-agent PPO training ({args.n_updates} updates x {args.n_episodes} episodes)...")
    policies, history = train_ppo_per_agent(
        n_episodes=args.n_episodes,
        n_updates=args.n_updates,
        seed=args.seed,
        max_cycles=args.max_cycles,
    )

    # Phase 3: Evaluate
    print()
    print("Phase 3: Evaluate trained policies (deterministic)...")
    eval_returns = evaluate(policies, n_episodes=args.n_eval_episodes, seed=200, max_cycles=args.max_cycles)

    print()
    print("=" * 60)
    print("PETTINGZOO SIMPLE SPREAD BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Random baseline:       {np.mean(random_returns):7.2f} +/- {np.std(random_returns):5.2f}")
    print(f"  Per-agent PPO (eval):   {np.mean(eval_returns):7.2f} +/- {np.std(eval_returns):5.2f}")
    delta = np.mean(eval_returns) - np.mean(random_returns)
    print(f"  Delta:                  {delta:+7.2f}")
    print()
    print("Honest interpretation:")
    print(f"  - PPO is {'better' if delta > 0 else 'worse'} than random by {delta:.2f}")
    print(f"  - {'Short training; PPO may not be converged' if args.n_updates < 50 else 'Longer training'}")
    print(f"  - {'No DMC comparison yet' if True else ''}")
    print(f"  - Real PettingZoo Simple Spread (not hand-coded env)")

    # Save log
    log_path = HERE / "checkpoints" / "pz_baseline" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3",
        "seed": args.seed,
        "mode": "Per-agent PPO on real PettingZoo benchmark",
        "n_episodes_per_update": args.n_episodes,
        "n_updates": args.n_updates,
        "n_eval_episodes": args.n_eval_episodes,
        "max_cycles": args.max_cycles,
        "random_baseline_mean": float(np.mean(random_returns)),
        "random_baseline_std": float(np.std(random_returns)),
        "ppo_eval_mean": float(np.mean(eval_returns)),
        "ppo_eval_std": float(np.std(eval_returns)),
        "delta": float(delta),
        "history": history,
        "honest_note": "Short training (200 episodes), not converged PPO. "
                       "Real convergence needs 500-1000 episodes."
    }, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
