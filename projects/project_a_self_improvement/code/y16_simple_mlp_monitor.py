"""y16_simple_mlp_monitor.py - H2.0-B: Simple MLP Monitor (ablation).

Replaces the SlotAttention-based Monitor in y13_monitor_regularizer.py
with a simple 2-layer MLP. The slot attention had:
  slot = SlotAttention(n_slots=4, slot_dim=32, n_iters=3, hidden=64, ...)
  monitor = MLP head on top of slot output

H2.0-B uses:
  monitor = simple 2-layer MLP directly on flattened trajectory window
  No slot attention. Same training procedure (BCE on failure labels).
  Same Y1.3 reward shaping (env_reward - lambda * Monitor_prob).

This is an ABLATION: testing if simpler architecture is better.
"""
import argparse
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode


class SimpleMLPMonitor(nn.Module):
    """Simple 2-layer MLP directly on flattened trajectory window.
    No slot attention, no recurrence. Just a basic MLP.
    """
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


def build_slot_input(transitions, history_len, obs_dim, n_actions):
    """Same as y13/y15: build a fixed-size window vector."""
    per_step = obs_dim + n_actions + 1
    arr = np.zeros((history_len, per_step), dtype=np.float32)
    for i, t in enumerate(transitions[-history_len:]):
        arr[i, :obs_dim] = t.obs
        if 0 <= t.action < n_actions:
            arr[i, obs_dim + t.action] = 1.0
        arr[i, obs_dim + n_actions] = t.reward
    return arr.flatten()  # FLATTEN to 1D for simple MLP


def train_simple_mlp_monitor(monitor, episodes, threshold, history_len,
                              obs_dim, n_actions, n_epochs=15):
    inputs, labels = [], []
    for ep in episodes:
        if len(ep.transitions) < 1: continue
        inputs.append(build_slot_input(ep.transitions, history_len, obs_dim, n_actions))
        labels.append(1.0 if ep.total_reward < threshold else 0.0)
    pos_idx = [i for i, l in enumerate(labels) if l == 1.0]
    neg_idx = [i for i, l in enumerate(labels) if l == 0.0]
    if len(pos_idx) > 0 and len(neg_idx) > len(pos_idx) * 2:
        np.random.seed(42)
        chosen = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 4), replace=False)
        keep = np.concatenate([np.array(pos_idx), chosen])
        np.random.shuffle(keep)
        inputs = [inputs[i] for i in keep]
        labels = [labels[i] for i in keep]
    X = torch.from_numpy(np.stack(inputs)).float()
    y = torch.from_numpy(np.array(labels, dtype=np.float32))
    opt = torch.optim.Adam(monitor.parameters(), lr=3e-4)
    for epoch in range(n_epochs):
        for start in range(0, len(y), 16):
            xb = X[start:start+16]; yb = y[start:start+16]
            if len(xb) == 0: continue
            preds = monitor(xb)
            loss = F.binary_cross_entropy(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return monitor


class MLPSPPO(PPOAgent):
    """PPO with SimpleMLPMonitor (no slot attention)."""
    def __init__(self, cfg, monitor, history_len, obs_dim, n_actions,
                 monitor_lambda=0.5):
        super().__init__(cfg)
        self.monitor = monitor
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.monitor_lambda = monitor_lambda

    def collect_shaped_rollout(self, env, obs0, transitions_for_monitor):
        cfg = self.cfg
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        obs = obs0
        ep_returns = []
        cur_ret = 0.0
        rollout_transitions = []

        for _ in range(cfg.rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = self.policy(obs_t)
                act = dist.sample()
                logp = dist.log_prob(act)
                val = self.value(obs_t)
            a = int(act.item())
            new_obs, r, term, trunc, _ = env.step(a)
            obs_buf.append(obs); act_buf.append(a); logp_buf.append(logp.item())
            val_buf.append(val.item()); rew_buf.append(float(r))
            done_buf.append(term or trunc)
            cur_ret += float(r)
            from envs import Transition
            rollout_transitions.append(Transition(obs=obs.copy(), action=a, reward=float(r)))
            obs = new_obs
            if term or trunc:
                ep_returns.append(cur_ret)
                cur_ret = 0.0
                obs, _ = env.reset()

        shaped_rew_buf = list(rew_buf)
        if self.monitor is not None and len(rollout_transitions) >= self.history_len:
            for i in range(self.history_len - 1, len(rollout_transitions)):
                window_trans = rollout_transitions[i - self.history_len + 1 : i + 1]
                x = build_slot_input(window_trans, self.history_len,
                                      self.obs_dim, self.n_actions)
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                with torch.no_grad():
                    monitor_prob = float(self.monitor(x_t).item())
                shaped_rew_buf[i] = rew_buf[i] - self.monitor_lambda * monitor_prob

        with torch.no_grad():
            last_val = self.value(torch.as_tensor(obs, dtype=torch.float32)).item()
        val_buf.append(last_val)

        adv_buf = np.zeros(cfg.rollout_len, dtype=np.float32)
        gae = 0.0
        next_val = last_val
        for t in reversed(range(cfg.rollout_len)):
            nonterminal = 1.0 - float(done_buf[t])
            delta = shaped_rew_buf[t] + cfg.gamma * next_val * nonterminal - val_buf[t]
            gae = delta + cfg.gamma * 0.95 * nonterminal * gae
            adv_buf[t] = gae
            next_val = val_buf[t]
        ret_buf = adv_buf + np.array(val_buf[:-1], dtype=np.float32)

        return {
            "obs": np.array(obs_buf, dtype=np.float32),
            "act": np.array(act_buf, dtype=np.int64),
            "logp": np.array(logp_buf, dtype=np.float32),
            "adv": adv_buf,
            "ret": ret_buf,
            "val": np.array(val_buf[:-1], dtype=np.float32),
            "ep_returns": ep_returns,
            "final_obs": obs,
            "rew_orig_mean": float(np.mean(rew_buf)),
            "rew_shaped_mean": float(np.mean(shaped_rew_buf)),
        }


class MLPSPPO_Random(PPOAgent):
    """Same as MLPSPPO but with random-weight MLP (no training)."""
    def __init__(self, cfg, history_len, obs_dim, n_actions,
                 monitor_lambda=0.5, seed=0):
        super().__init__(cfg)
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.monitor_lambda = monitor_lambda
        self.rng = np.random.RandomState(seed + 99999)
        per_step = obs_dim + n_actions + 1
        input_dim = history_len * per_step
        # Random weights MLP (no training)
        self.W1 = torch.randn(input_dim, 64) * 0.1
        self.b1 = torch.zeros(64)
        self.W2 = torch.randn(64, 1) * 0.1
        self.b2 = torch.zeros(1)

    def _random_monitor(self, x_t):
        h = torch.relu(x_t @ self.W1 + self.b1)
        out = torch.sigmoid(h @ self.W2 + self.b2).squeeze(-1)
        return float(out.item())

    def collect_shaped_rollout(self, env, obs0, transitions_for_monitor):
        cfg = self.cfg
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        obs = obs0
        ep_returns = []
        cur_ret = 0.0
        rollout_transitions = []

        for _ in range(cfg.rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = self.policy(obs_t)
                act = dist.sample()
                logp = dist.log_prob(act)
                val = self.value(obs_t)
            a = int(act.item())
            new_obs, r, term, trunc, _ = env.step(a)
            obs_buf.append(obs); act_buf.append(a); logp_buf.append(logp.item())
            val_buf.append(val.item()); rew_buf.append(float(r))
            done_buf.append(term or trunc)
            cur_ret += float(r)
            from envs import Transition
            rollout_transitions.append(Transition(obs=obs.copy(), action=a, reward=float(r)))
            obs = new_obs
            if term or trunc:
                ep_returns.append(cur_ret)
                cur_ret = 0.0
                obs, _ = env.reset()

        shaped_rew_buf = list(rew_buf)
        if self.history_len is not None and len(rollout_transitions) >= self.history_len:
            for i in range(self.history_len - 1, len(rollout_transitions)):
                window_trans = rollout_transitions[i - self.history_len + 1 : i + 1]
                x = build_slot_input(window_trans, self.history_len,
                                      self.obs_dim, self.n_actions)
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                monitor_prob = self._random_monitor(x_t)
                shaped_rew_buf[i] = rew_buf[i] - self.monitor_lambda * monitor_prob

        with torch.no_grad():
            last_val = self.value(torch.as_tensor(obs, dtype=torch.float32)).item()
        val_buf.append(last_val)

        adv_buf = np.zeros(cfg.rollout_len, dtype=np.float32)
        gae = 0.0
        next_val = last_val
        for t in reversed(range(cfg.rollout_len)):
            nonterminal = 1.0 - float(done_buf[t])
            delta = shaped_rew_buf[t] + cfg.gamma * next_val * nonterminal - val_buf[t]
            gae = delta + cfg.gamma * 0.95 * nonterminal * gae
            adv_buf[t] = gae
            next_val = val_buf[t]
        ret_buf = adv_buf + np.array(val_buf[:-1], dtype=np.float32)

        return {
            "obs": np.array(obs_buf, dtype=np.float32),
            "act": np.array(act_buf, dtype=np.int64),
            "logp": np.array(logp_buf, dtype=np.float32),
            "adv": adv_buf,
            "ret": ret_buf,
            "val": np.array(val_buf[:-1], dtype=np.float32),
            "ep_returns": ep_returns,
            "final_obs": obs,
            "rew_orig_mean": float(np.mean(rew_buf)),
            "rew_shaped_mean": float(np.mean(shaped_rew_buf)),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps-total", type=int, default=100000)
    p.add_argument("--n-warmup-steps", type=int, default=25000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--monitor-lambda", type=float, default=0.5)
    p.add_argument("--use-random-monitor", action="store_true",
                   help="Use random-weight MLP (no training) - control")
    p.add_argument("--out-tag", default="y16")
    args = p.parse_args()

    out = []
    out.append(f"[Y2.0-B: Simple MLP Monitor] env={args.env} seed={args.seed} random={args.use_random_monitor}")
    out.append("=" * 70)

    # Phase 1: PPO warm-up
    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    per_step = obs_dim + n_actions + 1
    input_dim = args.history_len * per_step
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    n_warmup_updates = args.n_warmup_steps // cfg.rollout_len
    for u in range(n_warmup_updates):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    out.append(f"[Phase 1] PPO warm-up done: {args.n_warmup_steps} steps")

    # Phase 2: collect rollouts for Monitor training
    train_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()
    threshold = max(0.0, float(np.percentile([e.total_reward for e in train_eps], 10.0)))
    out.append(f"[Data] {len(train_eps)} rollouts, threshold={threshold:.1f}")

    # Phase 3: train SimpleMLPMonitor (or use random)
    if not args.use_random_monitor:
        monitor = SimpleMLPMonitor(input_dim, hidden=64)
        train_simple_mlp_monitor(monitor, train_eps, threshold,
                                  args.history_len, obs_dim, n_actions)
        for p_ in monitor.parameters():
            p_.requires_grad = False
        out.append("[Monitor] SimpleMLPMonitor trained and frozen")
    else:
        monitor = None
        out.append("[Monitor] Using RANDOM-weight MLP (negative control)")

    # Phase 4: PPO continued with Monitor reward shaping
    if args.use_random_monitor:
        shaped_agent = MLPSPPO_Random(cfg, args.history_len, obs_dim, n_actions,
                                        args.monitor_lambda, args.seed)
    else:
        shaped_agent = MLPSPPO(cfg, monitor, args.history_len, obs_dim, n_actions,
                                 args.monitor_lambda)
    shaped_agent.policy = agent.policy
    shaped_agent.value = agent.value
    shaped_agent.opt = agent.opt
    n_total_updates = args.n_ppo_steps_total // cfg.rollout_len
    n_shaped_updates = n_total_updates - n_warmup_updates
    for u in range(n_shaped_updates):
        batch = shaped_agent.collect_shaped_rollout(env, obs, None)
        shaped_agent.update(batch)
        obs = batch["final_obs"]
    out.append(f"[Phase 4] PPO MLP-shaped done: {n_shaped_updates * cfg.rollout_len} more steps")

    # Phase 5: Evaluation
    eval_returns = []
    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx)
        ep = rollout_one_episode(e, shaped_agent.select_action, max_steps=500)
        eval_returns.append(ep.total_reward)
        e.close()
    env.close()

    out.append("")
    out.append("=" * 70)
    mode = "Y2.0-B: PPO + SimpleMLPMonitor (RANDOM)" if args.use_random_monitor else "Y2.0-B: PPO + SimpleMLPMonitor (TRAINED)"
    out.append(mode + " EVALUATION")
    out.append("=" * 70)
    out.append(f"  Mean: {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")
    print(chr(10).join(out))

    suffix = "_random" if args.use_random_monitor else "_trained"
    log_dir = HERE / "checkpoints" / ("full_integration_" + args.out_tag + suffix + "_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": mode,
        "monitor_arch": "simple MLP (no slot attention)",
        "monitor_signal": "RANDOM weights" if args.use_random_monitor else "TRAINED on 200 rollouts",
        "monitor_lambda": args.monitor_lambda,
        "n_ppo_steps_total": args.n_ppo_steps_total,
        "n_warmup_steps": args.n_warmup_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_eval_episodes": args.n_eval_episodes,
        "eval_mean": float(np.mean(eval_returns)),
        "eval_std": float(np.std(eval_returns)),
        "per_episode_eval": [float(x) for x in eval_returns],
    }, indent=2))


if __name__ == "__main__":
    main()
