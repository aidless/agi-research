#!/usr/bin/env python3
import argparse, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import envs
from ppo import PPOAgent, PPOConfig
from monitor import train_monitor, MonitorConfig, _quick_auroc, FailureMonitor
from envs import rollout_one_episode


def collect_rollouts(env_factory, agent, n_episodes, history_len, seed_offset, env_name="procgen"):
    out = []
    for i in range(n_episodes):
        env = env_factory(seed_offset + i + 7777)
        ep = rollout_one_episode(env, agent.select_action, max_steps=500)
        ep.env_name = env_name
        if len(ep.transitions) > history_len:
            ep.transitions = ep.transitions[-history_len:]
        out.append(ep)
        env.close()
    return out


def per_episode_monitor_probs(monitor, episodes, obs_dim, n_actions, history_len):
    per_step_dim = obs_dim + n_actions + 1
    means = []
    for ep in episodes:
        probs = []
        for t, _ in enumerate(ep.transitions):
            history = ep.transitions[: t + 1][-history_len:]
            vec = np.zeros(history_len * per_step_dim, dtype=np.float32)
            for k, h in enumerate(history):
                base = k * per_step_dim
                vec[base : base + obs_dim] = h.obs
                vec[base + obs_dim + h.action] = 1.0
                vec[base + obs_dim + n_actions] = h.reward
            probs.append(monitor.predict(vec))
        if probs:
            means.append(float(np.mean(probs)))
        else:
            means.append(0.5)
    return np.array(means)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-ppo-steps', type=int, default=256000)
    p.add_argument('--n-train-episodes', type=int, default=200)
    p.add_argument('--n-eval-episodes', type=int, default=100)
    p.add_argument('--history-len', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--env', default='LunarLander-v3', help='gymnasium env ID')
    p.add_argument('--percentile', type=float, default=10.0, help='failure percentile threshold')
    p.add_argument('--threshold-floor', type=float, default=-1e9, help='lower-bound for threshold (e.g. 0 for LunarLander success envs)')
    p.add_argument('--monitor-epochs', type=int, default=10)
    p.add_argument('--joint', action='store_true', help='joint monitor ablation: Monitor loss backprops into PPOActorNet')
    args = p.parse_args()

    sys.stdout.write('\n[Project A Phase 2] ' + str(args.env).upper() + '  seed=' + str(args.seed) + '\n')
    sys.stdout.write('  PPO=' + str(args.n_ppo_steps) + '  train=' + str(args.n_train_episodes) + '  eval=' + str(args.n_eval_episodes) + '\n')
    sys.stdout.write('  history=' + str(args.history_len) + '  p=' + str(args.percentile) + '\n\n')

    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    sys.stdout.write('  obs_dim=' + str(obs_dim) + '  n_actions=' + str(n_actions) + '\n\n')

    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    n_updates = args.n_ppo_steps // cfg.rollout_len
    all_returns = []
    for u in range(n_updates):
        batch = agent.collect_rollout(env, obs)
        info = agent.update(batch)
        obs = batch['final_obs']
        all_returns.extend(batch['ep_returns'])
        if (u + 1) % 5 == 0:
            mr = np.mean(all_returns[-200:]) if all_returns else 0.0
            sys.stdout.write('  u=' + str(u + 1) + '/' + str(n_updates) + '  mean_r(last200)=' + str(round(mr, 1)) + '\n')
    env.close()