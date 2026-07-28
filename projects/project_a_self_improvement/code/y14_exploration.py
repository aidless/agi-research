"""y14_exploration.py - DEC-0011 H1.4: Monitor as EXPLORATION BONUS.

Different from Y1.3 (reward shaping):
  Y1.3:  shaped_reward = env_reward - lambda * Monitor_prob  (PENALTY)
  Y1.4:  shaped_reward = env_reward + beta  * Monitor_prob  (BONUS)

The sign is reversed. H1.4 hypothesizes that visiting Monitor-flagged
states is USEFUL (for learning how to handle them) and should be
rewarded, not penalized.

This is a different intervention, not a parameter tweak on Y1.3.
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
PC_CODE = Path(r"E:\agi-research\projects\project_c_causal_world\code")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PC_CODE))

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode
from slot_attention import SlotAttention


class SlotMonitor(nn.Module):
    def __init__(self, slot_attention, n_slots, slot_dim, hidden=64):
        super().__init__()
        self.slot_attention = slot_attention
        self.head = nn.Sequential(
            nn.Linear(n_slots * slot_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        slots = self.slot_attention(x)
        flat = slots.reshape(slots.size(0), -1)
        return torch.sigmoid(self.head(flat)).squeeze(-1)


def build_slot_input(transitions, history_len, obs_dim, n_actions):
    per_step = obs_dim + n_actions + 1
    arr = np.zeros((history_len, per_step), dtype=np.float32)
    for i, t in enumerate(transitions[-history_len:]):
        arr[i, :obs_dim] = t.obs
        if 0 <= t.action < n_actions:
            arr[i, obs_dim + t.action] = 1.0
        arr[i, obs_dim + n_actions] = t.reward
    return arr


def train_slot_monitor(slot_monitor, episodes, threshold, history_len,
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
    opt = torch.optim.Adam(slot_monitor.parameters(), lr=3e-4)
    for epoch in range(n_epochs):
        for start in range(0, len(y), 16):
            xb = X[start:start+16]; yb = y[start:start+16]
            if len(xb) == 0: continue
            preds = slot_monitor(xb)
            loss = F.binary_cross_entropy(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return slot_monitor


class ExplorationShapedPPO(PPOAgent):
    """Like y13_monitor_regularizer but uses PLUS (exploration bonus) instead of MINUS.

    shaped_reward = env_reward + beta * Monitor_prob
    """
    def __init__(self, cfg, slot_monitor, history_len, obs_dim, n_actions,
                 monitor_beta=0.5):
        super().__init__(cfg)
        self.slot_monitor = slot_monitor
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.monitor_beta = monitor_beta

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
        if self.slot_monitor is not None and len(rollout_transitions) >= self.history_len:
            windows = []
            valid_idx = []
            for i in range(self.history_len - 1, len(rollout_transitions)):
                window_trans = rollout_transitions[i - self.history_len + 1 : i + 1]
                windows.append(build_slot_input(window_trans, self.history_len,
                                                  self.obs_dim, self.n_actions))
                valid_idx.append(i)
            X = torch.from_numpy(np.stack(windows)).float()
            with torch.no_grad():
                monitor_probs = self.slot_monitor(X).cpu().numpy()
            for k, idx in enumerate(valid_idx):
                # NOTE: PLUS not MINUS. This is the exploration bonus.
                shaped_rew_buf[idx] = rew_buf[idx] + self.monitor_beta * float(monitor_probs[k])

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


class ExplorationShapedPPO_Random(PPOAgent):
    """Same as above but uses RANDOM monitor signal (no trained Monitor)."""
    def __init__(self, cfg, history_len, obs_dim, n_actions, monitor_beta=0.5, seed=0):
        super().__init__(cfg)
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.monitor_beta = monitor_beta
        self.rng = np.random.RandomState(seed + 99999)

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
                random_monitor_prob = float(self.rng.uniform(0.0, 1.0))
                shaped_rew_buf[i] = rew_buf[i] + self.monitor_beta * random_monitor_prob

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
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--monitor-beta", type=float, default=0.5,
                   help="H1.4: monitor_prob BONUS (positive, NOT penalty)")
    p.add_argument("--use-random-monitor", action="store_true",
                   help="Use random uniform monitor instead of trained Monitor (negative control)")
    p.add_argument("--out-tag", default="y14")
    args = p.parse_args()

    out = []
    out.append(f"[Y1.4: Monitor as EXPLORATION BONUS] env={args.env} seed={args.seed} random={args.use_random_monitor}")
    out.append(f"  monitor_beta={args.monitor_beta}  n_warmup={args.n_warmup_steps}")
    out.append("=" * 70)

    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    n_warmup_updates = args.n_warmup_steps // cfg.rollout_len
    for u in range(n_warmup_updates):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    env.close()
    out.append(f"[Phase 1] PPO warm-up done: {args.n_warmup_steps} steps")

    train_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        train_eps.append(ep)
        e.close()
    threshold = max(0.0, float(np.percentile([e.total_reward for e in train_eps], 10.0)))
    out.append(f"[Data] {len(train_eps)} rollouts, threshold={threshold:.1f}")

    slot_monitor = None
    if not args.use_random_monitor:
        per_step = obs_dim + n_actions + 1
        slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                              n_iters=3, hidden_dim=64, input_dim=per_step)
        slot_monitor = SlotMonitor(slot, args.n_slots, args.slot_dim, hidden=64)
        train_slot_monitor(slot_monitor, train_eps, threshold,
                           args.history_len, obs_dim, n_actions)
        for p_ in slot_monitor.parameters():
            p_.requires_grad = False
        out.append("[Monitor] SlotMonitor trained and frozen (used as EXPLORATION BONUS)")
    else:
        out.append("[Monitor] Using RANDOM uniform [0,1] (negative control)")

    if args.use_random_monitor:
        shaped_agent = ExplorationShapedPPO_Random(cfg, args.history_len, obs_dim, n_actions,
                                                     args.monitor_beta, args.seed)
    else:
        shaped_agent = ExplorationShapedPPO(cfg, slot_monitor, args.history_len,
                                              obs_dim, n_actions, args.monitor_beta)
    shaped_agent.policy = agent.policy
    shaped_agent.value = agent.value
    shaped_agent.opt = agent.opt
    n_total_updates = args.n_ppo_steps_total // cfg.rollout_len
    n_shaped_updates = n_total_updates - n_warmup_updates
    for u in range(n_shaped_updates):
        batch = shaped_agent.collect_shaped_rollout(env, obs, None)
        shaped_agent.update(batch)
        obs = batch["final_obs"]
    out.append(f"[Phase 4] PPO EXPLORATION-shaped done: {n_shaped_updates * cfg.rollout_len} more steps")

    eval_returns = []
    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx)
        ep = rollout_one_episode(e, shaped_agent.select_action, max_steps=500)
        eval_returns.append(ep.total_reward)
        e.close()
    env.close()

    out.append("")
    out.append("=" * 70)
    mode = "Y1.4: PPO + EXPLORATION BONUS (random monitor)" if args.use_random_monitor else "Y1.4: PPO + EXPLORATION BONUS (real Monitor)"
    out.append(mode + " EVALUATION")
    out.append("=" * 70)
    out.append(f"  Mean: {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")
    print(chr(10).join(out))

    suffix = "_random" if args.use_random_monitor else "_real"
    log_dir = HERE / "checkpoints" / ("full_integration_" + args.out_tag + suffix + "_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": mode,
        "monitor_beta": args.monitor_beta,
        "monitor_signal": "RANDOM uniform [0, 1]" if args.use_random_monitor else "real (trained SlotMonitor)",
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
