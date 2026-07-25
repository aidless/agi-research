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
from monitor import train_monitor, MonitorConfig, _quick_auroc
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
    p.add_argument('--perturb-eval', type=float, default=0.0, help='gaussian noise std on eval obs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--env', default='LunarLander-v3', help='gymnasium env ID')
    p.add_argument('--percentile', type=float, default=10.0, help='failure percentile threshold')
    p.add_argument('--threshold-floor', type=float, default=-1e9, help='lower-bound for threshold (e.g. 0 for LunarLander success envs)')
    p.add_argument('--monitor-epochs', type=int, default=10)
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

    threshold = max(args.threshold_floor, envs.percentile_failure_threshold(all_returns, args.percentile))
    sys.stdout.write('\n  PPO done. mean=' + str(round(np.mean(all_returns), 2)) + '  p10=' + str(round(threshold, 2)) + '  n_eps=' + str(len(all_returns)) + '\n\n')

    def make_target(seed_offset):
        return envs.make_env(args.env, seed=seed_offset)

    sys.stdout.write('[Stage 2] Collecting ' + str(args.n_train_episodes) + ' train...\n')
    train_eps = collect_rollouts(make_target, agent, args.n_train_episodes, args.history_len, args.seed * 1000, env_name=args.env)
    train_returns = [e.total_reward for e in train_eps]
    fail_n = sum(1 for r in train_returns if r < threshold)
    sys.stdout.write('  collected.  mean=' + str(round(np.mean(train_returns), 2)) + '  fail=' + str(fail_n) + '/' + str(len(train_eps)) + '\n\n')

    sys.stdout.write('[Stage 3] Training Monitor...\n')
    history_dim = args.history_len * (obs_dim + n_actions + 1)
    mcfg = MonitorConfig(history_dim=history_dim, seed=args.seed, epochs=args.monitor_epochs)
    monitor, mmetrics = train_monitor(mcfg, train_eps, history_len=args.history_len, verbose=True, threshold=threshold)

    ckpt_dir = HERE / 'checkpoints' / ('lunarlander_seed' + str(args.seed))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'monitor': monitor.state_dict(), 'history_dim': history_dim, 'config': vars(mcfg)}, str(ckpt_dir / 'monitor.pt'))
    sys.stdout.write('  saved monitor.pt\n\n')

    sys.stdout.write('[Stage 4] Evaluating on ' + str(args.n_eval_episodes) + ' eval...\n')
    eval_eps = collect_rollouts(make_target, agent, args.n_eval_episodes, args.history_len, args.seed * 1000 + 999, env_name=args.env)
    eval_returns = [e.total_reward for e in eval_eps]
    if args.perturb_eval > 0:
        # Apply noise to ALL eval obs before Monitor sees them
        rng = np.random.default_rng(args.seed + 99999)
        for ep in eval_eps:
            for tr in ep.transitions:
                tr.obs = tr.obs + rng.normal(0, args.perturb_eval, size=tr.obs.shape).astype('float32')
    eval_probs = per_episode_monitor_probs(monitor, eval_eps, obs_dim, n_actions, args.history_len)

    fail_labels = np.array([1.0 if r < threshold else 0.0 for r in eval_returns])
    if fail_labels.std() > 1e-9:
        auroc = _quick_auroc(fail_labels, eval_probs)
        pearson = float(np.corrcoef(eval_probs, eval_returns)[0, 1])
    else:
        auroc = float('nan')
        pearson = float('nan')

    sys.stdout.write('\n=== Phase 2 LUNARLANDER (seed=' + str(args.seed) + ') ===\n')
    sys.stdout.write('  Eval episodes: ' + str(len(eval_eps)) + '\n')
    sys.stdout.write('  Failure-rate:  ' + str(round(fail_labels.mean(), 3)) + '\n')
    sys.stdout.write('  Reward mean+/-std: ' + str(round(np.mean(eval_returns), 2)) + '+/-' + str(round(np.std(eval_returns), 2)) + '\n')
    sys.stdout.write('  Monitor mean prob mean+/-std: ' + str(round(eval_probs.mean(), 3)) + '+/-' + str(round(eval_probs.std(), 3)) + '\n')
    sys.stdout.write('  AUROC: ' + str(auroc) + '\n')
    sys.stdout.write('  Pearson(prob,reward): ' + str(pearson) + '\n\n')

    if fail_labels.mean() == 0:
        sys.stdout.write('  RESULT: no failure cases.\n')
    elif not (auroc != auroc) and auroc > 0.55:
        sys.stdout.write('  RESULT: H1 directional support! AUROC=' + str(round(auroc, 3)) + ' > 0.55\n')
    else:
        sys.stdout.write('  RESULT: AUROC=' + str(auroc) + ' not above 0.55\n')

    out = {
        'env': args.env,
        'seed': args.seed,
        'n_ppo_steps': args.n_ppo_steps,
        'threshold_percentile': args.percentile,
        'threshold': float(threshold),
        'n_train_eps': len(train_eps),
        'n_eval_eps': len(eval_eps),
        'fail_rate': float(fail_labels.mean()),
        'eval_reward_mean': float(np.mean(eval_returns)),
        'eval_reward_std': float(np.std(eval_returns)),
        'eval_prob_mean': float(eval_probs.mean()),
        'eval_prob_std': float(eval_probs.std()),
        'auroc': float(auroc) if auroc == auroc else None,
        'pearson_prob_reward': float(pearson) if pearson == pearson else None,
        'monitor_train': mmetrics,
        'p_training_returns_mean': float(np.mean(all_returns)),
        'p_std': float(np.std(all_returns)),
    }
    log_path = ckpt_dir / 'phase2_log.json'
    log_path.write_text(json.dumps(out, indent=2))
    sys.stdout.write('  saved ' + str(log_path) + '\n')


if __name__ == '__main__':
    main()














