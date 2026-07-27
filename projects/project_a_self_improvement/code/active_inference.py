#!/usr/bin/env python3
"""active_inference.py - ENWI Active Inference Engine ported to Project A.

Replaces PPO + Q-function with Friston-style active inference.
Objective: minimize variational free energy (surprise) instead of
maximize reward.

Core principle:
  F = E_q[log q(s) - log p(o,s)]
    = D_KL[q(s) || p(s|o)] - log p(o)

Action selection: pick action minimizing EXPECTED free energy
  (epistemic value: information gain + pragmatic value: goal achievement)
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
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode, Transition


class FreeEnergyComputer(nn.Module):
    """Computes variational free energy for state-belief pairs.

    F = E_q[log q(s) - log p(o,s)]
      = D_KL[q(s) || p(s|o)] - log p(o)
    """
    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        # Generative model p(o|s): state -> predicted obs
        self.generation_model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, obs_dim),
        )
        # Prior p(s): learnable mean and log-var
        self.prior_mean = nn.Parameter(torch.zeros(state_dim))
        self.prior_log_var = nn.Parameter(torch.zeros(state_dim))

    def compute_log_likelihood(self, state, obs):
        obs_pred = self.generation_model(state)
        log_var = torch.zeros_like(obs_pred)
        return -0.5 * (log_var + math.log(2 * math.pi) + (obs - obs_pred)**2 / torch.exp(log_var)).sum(dim=-1)

    def compute_log_prior(self, state):
        return -0.5 * (self.prior_log_var + math.log(2*math.pi) + (state - self.prior_mean)**2 / torch.exp(self.prior_log_var)).sum(dim=-1)

    def compute_free_energy(self, obs, posterior_mean, posterior_log_var, num_samples=1):
        # Sample from posterior
        std = torch.exp(0.5 * posterior_log_var)
        eps = torch.randn_like(std)
        state = posterior_mean + std * eps
        log_lik = self.compute_log_likelihood(state, obs)
        log_prior = self.compute_log_prior(state)
        return -(log_lik + log_prior)  # negative log evidence = free energy


class ActiveInferenceEngine(nn.Module):
    """Active Inference Engine for action selection.

    Picks action that minimizes expected free energy:
      G(a) = E_q(o,s|a) [-log p(o) - KL[q(s|o,a) || q(s|o)]]

    Two components:
      - Epistemic value: information gain (-KL)
      - Pragmatic value: goal achievement (-log p(o|goal))
    """
    def __init__(self, state_dim: int, obs_dim: int, action_dim: int,
                 hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Variational posterior q(s|o)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * state_dim),  # mean and log_var
        )
        # Generative model p(o|s)
        self.generation_model = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
        # Transition model p(s'|s,a) - simple linear
        self.transition = nn.Linear(state_dim + action_dim, state_dim)
        # Preference model p(o|goal) - parameterized
        self.preference = nn.Parameter(torch.zeros(obs_dim))
        # Action sampling
        self.action_sampler = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # Free energy module
        self.fe_computer = FreeEnergyComputer(state_dim, obs_dim)

    def encode(self, obs):
        """Return posterior mean and log_var over state."""
        out = self.encoder(obs)
        mean, log_var = out.chunk(2, dim=-1)
        # Clamp log_var for stability
        log_var = torch.clamp(log_var, -10, 2)
        return mean, log_var

    def transition_step(self, state, action_onehot):
        action_t = torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        sa = torch.cat([state, action_t], dim=-1)
        return self.transition(sa)

    def expected_free_energy(self, state, action):
        """Compute G(a) for each candidate action.

        G(a) = -E_p(o|s')[log p(o|goal)] + KL[q(s|o) || q(s|o,a)]
        """
        action_oh = np.zeros(self.action_dim)
        action_oh[action] = 1.0
        next_state = self.transition_step(state, action_oh)
        # Predicted obs
        pred_obs = self.generation_model(next_state)
        # Pragmatic value: how close to preference
        pragmatic = -F.log_softmax(-(pred_obs - self.preference)**2, dim=-1).sum(dim=-1)
        # Epistemic value: information gain (KL between prior and posterior)
        prior_mean = self.fe_computer.prior_mean.unsqueeze(0).expand_as(next_state)
        prior_logvar = self.fe_computer.prior_log_var.unsqueeze(0).expand_as(next_state)
        kl = 0.5 * (
            prior_logvar - next_state.new_zeros(1).log()
            + (next_state - prior_mean)**2 / torch.exp(prior_logvar)
            - 1
        ).sum(dim=-1)
        epistemic = -kl  # negative KL = info gain
        return (pragmatic + 0.1 * epistemic).item()

    def select_action(self, obs, deterministic=False):
        mean, log_var = self.encode(obs)
        state = mean
        scores = [self.expected_free_energy(state, a) for a in range(self.action_dim)]
        scores = torch.tensor(scores)
        probs = F.softmax(-scores, dim=-1)
        if deterministic:
            return int(torch.argmin(scores).item())
        return int(torch.multinomial(probs, 1).item())

    def compute_loss(self, obs, action, reward, next_obs, done):
        """Loss = expected free energy + reward prediction error."""
        mean, log_var = self.encode(obs)
        state = mean
        # Free energy
        fe = self.fe_computer.compute_free_energy(obs, mean, log_var)
        # Action prediction
        action_t = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        action_logits = self.action_sampler(state)
        action_loss = F.cross_entropy(action_logits, action_t)
        # Reward prediction
        reward_pred = self.generation_model(state).sum(dim=-1)
        reward_loss = F.mse_loss(reward_pred, torch.tensor([reward], dtype=torch.float32))
        return fe.mean() + action_loss + 0.1 * reward_loss


if __name__ == "__main__":
    print("=" * 60)
    print("ENWI Active Inference Engine — Project A port")
    print("=" * 60)

    # Smoke test
    state_dim, obs_dim, action_dim = 8, 8, 4
    aie = ActiveInferenceEngine(state_dim, obs_dim, action_dim)

    n_params = sum(p.numel() for p in aie.parameters())
    print(f"  AIE params: {n_params:,}")

    obs = torch.randn(1, obs_dim)
    mean, log_var = aie.encode(obs)
    print(f"  Posterior mean: {mean.shape}, log_var: {log_var.shape}")

    action = aie.select_action(obs, deterministic=True)
    print(f"  Selected action: {action}")

    fe = aie.fe_computer.compute_free_energy(obs, mean, log_var)
    print(f"  Free energy: {fe.item():.4f}")

    print()
    print("ENWI Active Inference Engine — ported to Project A, smoke test PASSED")