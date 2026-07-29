#!/usr/bin/env python3
"""aie_lunarlander.py - Active Inference Engine training on LunarLander.

Replaces PPO+Monitor with Friston-style active inference:
- FreeEnergyComputer: variational free energy
- ActiveInferenceEngine: encode + transition + action selection
- Train via free energy minimization + reward prediction
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
import math

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "project_c_causal_world" / "code"))

import envs
from envs import rollout_one_episode, Transition
from active_inference import ActiveInferenceEngine, FreeEnergyComputer


def train_aie(env_name, n_episodes, seed, hidden=64, n_epochs=20, batch_size=32):
    """Train AIE on environment, return final eval reward."""
    env = envs.make_env(env_name, seed=seed + 1)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    aie = ActiveInferenceEngine(obs_dim, obs_dim, action_dim, hidden=hidden)
    opt = torch.optim.Adam(aie.parameters(), lr=3e-4)

    # Collect training trajectories
    train_returns = []
    for ep in range(min(50, n_episodes)):
        e = envs.make_env(env_name, seed=seed * 1000 + ep + 7777)
        obs, _ = e.reset()
        ep_reward = 0.0
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        for t in range(500):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            action = aie.select_action(obs_t, deterministic=False)
            next_obs, reward, term, trunc, _ = e.step(action)
            ep_reward += reward
            ep_obs.append(obs.copy())
            ep_actions.append(action)
            ep_rewards.append(reward)
            obs = next_obs
            if term or trunc:
                break
        e.close()
        train_returns.append(ep_reward)

    # Train for n_epochs on collected data
    X_all = np.stack([np.array(o) for o in ep_obs])
    A_all = np.array(ep_actions, dtype=np.int64)
    R_all = np.array(ep_rewards, dtype=np.float32)

    # Compute predicted reward baseline (avg)
    R_mean = float(R_all.mean())
    R_std = max(float(R_all.std()), 1e-3)

    for epoch in range(n_epochs):
        idx = np.random.permutation(len(X_all))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_all), batch_size):
            bi = idx[start:start + batch_size]
            obs_b = torch.from_numpy(X_all[bi]).float()
            act_b = torch.from_numpy(A_all[bi])
            r_b = torch.from_numpy(R_all[bi])

            opt.zero_grad()
            # Free energy
            mean, log_var = aie.encode(obs_b)
            fe = aie.fe_computer.compute_free_energy(obs_b, mean, log_var)
            # Action prediction
            action_logits = aie.action_sampler(mean)
            action_loss = F.cross_entropy(action_logits, act_b)
            # Reward prediction (normalized)
            r_pred = aie.generation_model(mean).sum(dim=-1)
            r_norm = (r_b - R_mean) / R_std
            reward_loss = F.mse_loss(r_pred, r_norm)
            loss = fe.mean() + action_loss + 0.1 * reward_loss
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}: avg loss = {total_loss / max(1, n_batches):.4f}")

    # Evaluate
    eval_returns = []
    for ep in range(5):
        e = envs.make_env(env_name, seed=seed * 1000 + 9999 + ep)
        obs, _ = e.reset()
        ep_reward = 0.0
        for t in range(500):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action = aie.select_action(obs_t, deterministic=True)
            obs, reward, term, trunc, _ = e.step(action)
            ep_reward += reward
            if term or trunc:
                break
        e.close()
        eval_returns.append(ep_reward)
    env.close()
    return train_returns, eval_returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=15)
    args = p.parse_args()

    print("=" * 60)
    print("AIE training on", args.env)
    print("=" * 60)

    train_returns, eval_returns = train_aie(
        args.env, args.n_episodes, args.seed, n_epochs=args.n_epochs
    )
    print()
    print(f"  Train mean: {np.mean(train_returns):.1f}")
    print(f"  Eval mean:  {np.mean(eval_returns):.1f}")

    log_path = HERE / "checkpoints" / "aie_lunarlander" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed, "mode": "AIE training",
        "train_mean": float(np.mean(train_returns)),
        "eval_mean": float(np.mean(eval_returns)),
    }, indent=2))


if __name__ == "__main__":
    main()