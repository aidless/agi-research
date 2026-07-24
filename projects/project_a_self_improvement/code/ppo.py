"""
ppo.py — Minimal PPO implementation from scratch.

Why not stable-baselines3?
- We want every line to be readable in the paper's appendix.
- We want CPU-runnable, deterministic, and easy to ablate.
- This file is ~120 lines of actual logic + comments.

REVIEW-ME:
- This is a vanilla PPO. The paper will use this exact code.
- Hyperparameters are conservative defaults (Schulman 2017).
- The Monitor never appears in this file by design (decoupled).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# -------- Policy network (small MLP for CartPole / LunarLander) --------

class Policy(nn.Module):
    """
    Stochastic policy: outputs logits over discrete actions.

    REVIEW-ME: Could be swapped for a CNN or larger MLP in later tasks.
    This is intentionally tiny so v1 experiment finishes in <10 min CPU.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32)
            dist = self.forward(t)
            return int(dist.sample().item())


# -------- Value network (used in PPO loss, NOT used by Monitor later) --------

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# -------- Rollout buffer --------

@dataclass
class PPOConfig:
    obs_dim: int
    n_actions: int
    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 10
    batch_size: int = 64
    rollout_len: int = 2048
    hidden: int = 64
    seed: int = 0


class PPOAgent:
    """
    PPO with clipped objective, GAE-lambda advantage.

    REVIEW-ME: GAE-lambda is set to 0.95 (default). Worth ablating
    in Appendix B if reviewer asks about variance.
    """

    def __init__(self, cfg: PPOConfig, device: str = "cpu"):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = device
        self.policy = Policy(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(device)
        self.value = ValueNet(cfg.obs_dim, cfg.hidden).to(device)
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=cfg.lr,
        )

    def select_action(self, obs: np.ndarray) -> int:
        return self.policy.act(obs)

    def collect_rollout(self, env, obs0: np.ndarray) -> dict:
        """Run `cfg.rollout_len` steps in env, return collected batch."""
        cfg = self.cfg
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        obs = obs0
        ep_returns: List[float] = []
        cur_ret = 0.0

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
            obs = new_obs
            if term or trunc:
                ep_returns.append(cur_ret)
                cur_ret = 0.0
                obs, _ = env.reset()

        # bootstrap value for last state
        with torch.no_grad():
            last_val = self.value(torch.as_tensor(obs, dtype=torch.float32)).item()
        val_buf.append(last_val)

        # GAE-lambda advantage
        adv_buf = np.zeros(cfg.rollout_len, dtype=np.float32)
        gae = 0.0
        next_val = last_val
        for t in reversed(range(cfg.rollout_len)):
            nonterminal = 1.0 - float(done_buf[t])
            delta = rew_buf[t] + cfg.gamma * next_val * nonterminal - val_buf[t]
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
        }

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        act = torch.as_tensor(batch["act"], dtype=torch.long)
        logp_old = torch.as_tensor(batch["logp"], dtype=torch.float32)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32)
        # normalise advantage (helps stability)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_batches": 0}
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                mb = idx[start:start + cfg.batch_size]
                mbi = torch.as_tensor(mb, dtype=torch.long)
                dist = self.policy(obs[mbi])
                logp = dist.log_prob(act[mbi])
                ratio = torch.exp(logp - logp_old[mbi])
                surr1 = ratio * adv[mbi]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[mbi]
                pol_loss = -torch.min(surr1, surr2).mean()

                v_pred = self.value(obs[mbi])
                v_loss = F.mse_loss(v_pred, ret[mbi])

                ent = dist.entropy().mean()
                loss = pol_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.opt.step()

                info["policy_loss"] += float(pol_loss.item())
                info["value_loss"] += float(v_loss.item())
                info["entropy"] += float(ent.item())
                info["n_batches"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            info[k] /= max(1, info["n_batches"])
        return info
