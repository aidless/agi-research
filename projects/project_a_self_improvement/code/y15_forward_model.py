"""y15_forward_model.py - H2.0-A: Forward model + PPO (ICM-style exploration).

This is a NEW intervention, NOT a Monitor use case. Instead of
using the failure-prediction Monitor for reward shaping, we use a
forward model that predicts next_state from (current_state, action).
The prediction ERROR is used as an exploration bonus.

ICM-style: bonus = ||predicted_next_state - actual_next_state||^2
Random control: same architecture but with random weights (no
learning). This provides a baseline of "any prediction error"
without informative signal.
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


class ForwardModel(nn.Module):
    """Predicts next_state given (current_state, action_onehot).
    Used for ICM-style exploration: bonus = MSE(predicted, actual).
    """
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
    def forward(self, state, action):
        a_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        x = torch.cat([state, a_onehot], dim=-1)
        return self.net(x)


class FMSPPO(PPOAgent):
    """PPO with forward model exploration bonus.

    Bonus = MSE(FM(s, a), actual_s') added to reward.
    For random FM: same code, just don't train the FM (random init).
    """
    def __init__(self, cfg, fm, bonus_coeff=0.5):
        super().__init__(cfg)
        self.fm = fm
        self.bonus_coeff = bonus_coeff

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
            rollout_transitions.append((obs.copy(), a, new_obs.copy()))
            obs = new_obs
            if term or trunc:
                ep_returns.append(cur_ret)
                cur_ret = 0.0
                obs, _ = env.reset()

        # Apply forward model exploration bonus
        shaped_rew_buf = list(rew_buf)
        with torch.no_grad():
            for i, (s, a, s_next) in enumerate(rollout_transitions):
                s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                a_t = torch.tensor([a], dtype=torch.long)
                s_next_pred = self.fm(s_t, a_t)
                s_next_actual = torch.as_tensor(s_next, dtype=torch.float32).unsqueeze(0)
                fm_error = float(((s_next_pred - s_next_actual) ** 2).mean().item())
                shaped_rew_buf[i] = rew_buf[i] + self.bonus_coeff * fm_error

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
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bonus-coeff", type=float, default=0.5)
    p.add_argument("--fm-train-steps", type=int, default=25000,
                   help="Steps to train the forward model (using random rollouts)")
    p.add_argument("--use-random-fm", action="store_true",
                   help="Don't train FM (use random weights) - control")
    p.add_argument("--out-tag", default="y15")
    args = p.parse_args()

    out = []
    out.append(f"[Y2.0-A: Forward model + PPO exploration bonus] env={args.env} seed={args.seed} random_fm={args.use_random_fm}")
    out.append(f"  bonus_coeff={args.bonus_coeff}  fm_train_steps={args.fm_train_steps}")
    out.append("=" * 70)

    # Phase 0: PPO warm-up (no FM yet, same as Y1.3)
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
    out.append(f"[Phase 1] PPO warm-up done: {args.n_warmup_steps} steps")

    # Phase 1.5: Collect rollouts for FM training (same as PPO training data)
    train_data = []
    for i in range(args.fm_train_steps // 500):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        for j in range(len(ep.transitions) - 1):
            tr = ep.transitions[j]
            tr_next = ep.transitions[j + 1]
            train_data.append((tr.obs, tr.action, tr_next.obs))
        e.close()
    out.append(f"[Phase 1.5] Collected {len(train_data)} (s, a, s') transitions for FM training")

    # Phase 2: Train Forward Model
    fm = ForwardModel(obs_dim, n_actions, hidden=64)
    if not args.use_random_fm:
        opt = torch.optim.Adam(fm.parameters(), lr=3e-4)
        s_arr = np.array([d[0] for d in train_data])
        a_arr = np.array([d[1] for d in train_data])
        sn_arr = np.array([d[2] for d in train_data])
        S = torch.from_numpy(s_arr).float()
        A = torch.from_numpy(a_arr).long()
        SN = torch.from_numpy(sn_arr).float()
        for epoch in range(15):
            perm = np.random.permutation(len(S))
            for start in range(0, len(S), 64):
                bi = perm[start:start + 64]
                pred = fm(S[bi], A[bi])
                loss = F.mse_loss(pred, SN[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        out.append(f"[Phase 2] Forward model trained on {len(train_data)} transitions")
    else:
        for p_ in fm.parameters():
            p_.requires_grad = False
        out.append(f"[Phase 2] Forward model RANDOM (not trained) - control")

    # Phase 3: PPO continued with FM exploration bonus
    fm_agent = FMSPPO(cfg, fm, args.bonus_coeff)
    fm_agent.policy = agent.policy
    fm_agent.value = agent.value
    fm_agent.opt = agent.opt
    env.close()
    env = envs.make_env(args.env, seed=args.seed + 1)
    obs, _ = env.reset()
    n_total_updates = args.n_ppo_steps_total // cfg.rollout_len
    n_shaped_updates = n_total_updates - n_warmup_updates
    for u in range(n_shaped_updates):
        batch = fm_agent.collect_shaped_rollout(env, obs, None)
        fm_agent.update(batch)
        obs = batch["final_obs"]
    out.append(f"[Phase 3] PPO FM-shaped done: {n_shaped_updates * cfg.rollout_len} more steps")

    # Phase 4: Evaluation
    eval_returns = []
    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx)
        ep = rollout_one_episode(e, fm_agent.select_action, max_steps=500)
        eval_returns.append(ep.total_reward)
        e.close()
    env.close()

    out.append("")
    out.append("=" * 70)
    mode = "Y2.0-A: PPO + FM exploration bonus (RANDOM FM)" if args.use_random_fm else "Y2.0-A: PPO + FM exploration bonus (TRAINED FM)"
    out.append(mode + " EVALUATION")
    out.append("=" * 70)
    out.append(f"  Mean: {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")
    print(chr(10).join(out))

    suffix = "_random" if args.use_random_fm else "_trained"
    log_dir = HERE / "checkpoints" / ("full_integration_" + args.out_tag + suffix + "_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": mode,
        "fm_signal": "RANDOM weights" if args.use_random_fm else "TRAINED on 5000+ transitions",
        "bonus_coeff": args.bonus_coeff,
        "n_ppo_steps_total": args.n_ppo_steps_total,
        "n_warmup_steps": args.n_warmup_steps,
        "n_eval_episodes": args.n_eval_episodes,
        "eval_mean": float(np.mean(eval_returns)),
        "eval_std": float(np.std(eval_returns)),
        "per_episode_eval": [float(x) for x in eval_returns],
    }, indent=2))


if __name__ == "__main__":
    main()
