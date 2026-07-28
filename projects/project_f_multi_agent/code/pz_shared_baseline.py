"""pz_shared_baseline.py - PPO with parameter sharing for PettingZoo.

This is a STANDARD baseline in cooperative MARL: all agents share the
same actor and critic networks. This is what MADDPG, COMA, and most
modern MA-RL methods use.

Honest framing:
- Parameter sharing is a well-known trick, not a research claim
- Expected to be MUCH better than per-agent PPO (which underperformed
  random on Simple Spread)
- This is a baseline, not a contribution

What this DOES validate:
- Whether parameter sharing fixes the per-agent PPO failure
- Whether the PettingZoo Simple Spread benchmark is solvable with
  standard techniques at our compute scale
- Sets a "real" PPO baseline for future DMC comparison

What this does NOT validate:
- Best PPO performance (no hyperparameter tuning)
- Comparison to other MA methods (MADDPG, QMIX, COMA)
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

from pettingzoo.mpe import simple_spread_v3


# Shared actor + critic (all agents use the same networks)
class SharedActorCritic(nn.Module):
    """Single actor + critic, shared across all agents.

    Honest framing: This is the standard parameter-sharing trick.
    The actor takes any agent's observation and outputs its action logits;
    same for critic. No agent-specific parameters.
    """
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


def collect_episode_shared(policy, env, seed):
    """Collect episode using shared policy.

    Same as pz_baseline.py but with single shared policy.
    """
    env.reset(seed=seed)
    transitions = []
    ep_returns = 0.0
    agent_order = env.possible_agents

    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_returns += reward
        if term or trunc:
            action = None
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, value, logits = policy.act(obs_t)
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
            transitions.append({
                "obs": obs.copy(),
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "agent": a,
            })
        env.step(action)
    # Joint reward: each transition gets the joint return
    for t in transitions:
        t["reward"] = ep_returns
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


def ppo_update_shared(policy, optimizer, trajectories, n_epochs=4,
                      batch_size=32, clip=0.2):
    """Single PPO update on all collected trajectories (joint batch)."""
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


def train_shared_ppo(n_episodes=20, n_updates=30, seed=0, max_cycles=25):
    """Train single shared-actor PPO on PettingZoo Simple Spread."""
    env = simple_spread_v3.env(N=3, max_cycles=max_cycles, continuous_actions=False)
    obs_dim = env.observation_space("agent_0").shape[0]
    n_actions = env.action_space("agent_0").n
    env.close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # SINGLE shared policy for all agents
    policy = SharedActorCritic(obs_dim, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Shared PPO: 1 policy, {n_params} params, obs_dim={obs_dim}, n_actions={n_actions}")

    history = []
    for u in range(n_updates):
        all_trajectories = []
        all_returns = []
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_shared(
                policy,
                simple_spread_v3.env(N=3, max_cycles=max_cycles, continuous_actions=False),
                seed=seed * 1000 + u * 100 + ep,
            )
            all_trajectories.append(transitions)
            all_returns.append(ep_return)
        loss = ppo_update_shared(policy, optimizer, all_trajectories)
        mean_return = float(np.mean(all_returns))
        history.append({"update": u, "mean_return": mean_return})
        if (u + 1) % 5 == 0 or u == 0:
            print(f"    update {u+1}/{n_updates}: mean_episode_return={mean_return:.2f}")

    return policy, history


def evaluate_shared(policy, n_episodes=20, seed=200, max_cycles=25):
    """Evaluate shared PPO (deterministic)."""
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
                    action, _, _ = policy.act(obs_t, deterministic=True)
            env.step(action)
            if env.agents == []:
                break
        returns.append(ep_return)
    env.close()
    return returns


def run_random_baseline(seed=0, n_episodes=20, max_cycles=25):
    """Random actions baseline."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--n-updates", type=int, default=30)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--max-cycles", type=int, default=25)
    args = p.parse_args()

    print("=" * 60)
    print("PPO with Parameter Sharing — PettingZoo Simple Spread v3")
    print("=" * 60)
    print(f"  seed={args.seed}, n_episodes={args.n_episodes}, n_updates={args.n_updates}")
    print(f"  1 shared actor + 1 shared critic for all 3 agents")
    print(f"  HONEST: short training ({args.n_episodes * args.n_updates} episodes)")
    print()

    # Phase 1: Random baseline
    print("Phase 1: Random baseline...")
    random_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    random_mean = float(np.mean(random_returns))
    random_std = float(np.std(random_returns))
    print(f"  Random mean: {random_mean:.2f} +/- {random_std:.2f}")

    # Phase 2: Shared PPO training
    print()
    print(f"Phase 2: Shared PPO training ({args.n_updates} updates x {args.n_episodes} episodes)...")
    policy, history = train_shared_ppo(
        n_episodes=args.n_episodes,
        n_updates=args.n_updates,
        seed=args.seed,
        max_cycles=args.max_cycles,
    )

    # Phase 3: Evaluate
    print()
    print("Phase 3: Evaluate shared PPO (deterministic)...")
    eval_returns = evaluate_shared(policy, n_episodes=args.n_eval_episodes,
                                    seed=200, max_cycles=args.max_cycles)
    eval_mean = float(np.mean(eval_returns))
    eval_std = float(np.std(eval_returns))
    delta = eval_mean - random_mean

    # Phase 4: Summary
    print()
    print("=" * 60)
    print("PARAMETER-SHARING PPO SUMMARY")
    print("=" * 60)
    print(f"  Random baseline:    {random_mean:7.2f} +/- {random_std:5.2f}")
    print(f"  Per-agent PPO:       -100.51    21.70  (from prior run)")
    print(f"  Shared PPO (eval):   {eval_mean:7.2f} +/- {eval_std:5.2f}")
    print(f"  Shared vs random:    {delta:+7.2f}")
    print(f"  Shared vs per-agent: {eval_mean - (-100.51):+7.2f}")
    print()
    if eval_mean > random_mean:
        verdict = "**Shared PPO BEATS random**"
    else:
        verdict = "Shared PPO worse than random"
    print(f"  Verdict: {verdict}")
    print()
    print("Honest interpretation:")
    print(f"  - Parameter sharing: 1 policy vs 3 policies")
    print(f"  - Implicit credit assignment through shared representations")
    print(f"  - {'Strong improvement over per-agent' if eval_mean > -85 else 'Modest/no improvement'}")
    print(f"  - {'Now we have a real PPO baseline' if eval_mean > random_mean else 'Need to try MADDPG/QMIX'}")

    # Save log
    log_path = HERE / "checkpoints" / "pz_shared_baseline" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3",
        "seed": args.seed,
        "mode": "PPO with parameter sharing (single shared actor + critic)",
        "n_episodes_per_update": args.n_episodes,
        "n_updates": args.n_updates,
        "n_eval_episodes": args.n_eval_episodes,
        "max_cycles": args.max_cycles,
        "n_params": sum(p.numel() for p in policy.parameters()),
        "random_baseline_mean": random_mean,
        "random_baseline_std": random_std,
        "shared_ppo_eval_mean": eval_mean,
        "shared_ppo_eval_std": eval_std,
        "delta_vs_random": float(delta),
        "delta_vs_per_agent": float(eval_mean - (-100.51)),
        "history": history,
        "honest_note": "Short training ({} episodes). Per-agent PPO was -100.51 (worse than random). "
                       "If shared PPO beats random, we have a real PPO baseline for DMC.".format(
            args.n_episodes * args.n_updates),
    }, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
