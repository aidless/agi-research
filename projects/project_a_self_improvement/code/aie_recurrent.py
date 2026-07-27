"""aie_recurrent.py - Active Inference Engine with recurrence + baseline subtraction.

Y1 improvements over aie_train_full.py:
1. Recurrent latent state: hidden state carries across timesteps
2. Value baseline: learned baseline for variance reduction
3. Higher reward weight (0.1 -> 0.5) to address reward signal under-utilization
4. Longer training budget (50K -> 100K env steps)

Hypothesis: these changes will improve convergence on LunarLander.
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
from collections import deque

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import envs
from envs import make_env, Transition


class RecurrentAIE(nn.Module):
    """Active Inference Engine with recurrent latent state.

    Architecture:
      encoder: obs -> posterior (mean, log_var)
      gru: (latent_state, posterior) -> next_latent_state
      transition: latent_state + action -> next_latent_state (model-based prior)
      generation: latent_state -> predicted_obs
      action_sampler: latent_state -> action_logits
      value_baseline: latent_state -> value (for variance reduction)
    """
    def __init__(self, obs_dim=8, latent_dim=32, action_dim=4, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encoder: obs -> posterior
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),  # mean and log_var
        )
        # Recurrent latent state update
        self.gru = nn.GRUCell(latent_dim, latent_dim)
        # Transition model (action-conditioned)
        self.transition = nn.Linear(latent_dim + action_dim, latent_dim)
        # Generation model
        self.generation = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
        # Action sampler
        self.action_sampler = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # Value baseline (for variance reduction)
        self.value_baseline = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Prior (for KL term)
        self.prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.prior_log_var = nn.Parameter(torch.zeros(latent_dim))

    def encode(self, obs, prev_latent):
        """obs + prev_latent -> posterior, new latent."""
        out = self.encoder(obs)
        post_mean, post_log_var = out.chunk(2, dim=-1)
        post_log_var = torch.clamp(post_log_var, -10, 2)
        # Sample
        std = torch.exp(0.5 * post_log_var)
        eps = torch.randn_like(std)
        new_latent = post_mean + std * eps
        # Update via GRU
        new_latent = self.gru(new_latent, prev_latent)
        return new_latent, post_mean, post_log_var

    def transition_step(self, latent, action_onehot):
        action_t = torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        sa = torch.cat([latent, action_t], dim=-1)
        return self.transition(sa)

    def expected_free_energy(self, latent, action):
        """G(a) = -E_p(o|s')[log p(o|goal)] + KL[q(s|o,a) || q(s|o)]"""
        action_oh = np.zeros(self.action_dim)
        action_oh[action] = 1.0
        next_latent = self.transition_step(latent, action_oh)
        pred_obs = self.generation(next_latent)
        # Pragmatic value: how close to preference (assume upright, landed, in_pad)
        # Simple proxy: high obs[1] (height) and low obs[2], obs[3] (velocity)
        preference = torch.tensor([0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
        pragmatic = -((pred_obs - preference) ** 2).sum(dim=-1)
        # Epistemic: KL between prior and transition-based posterior
        prior_logvar = self.prior_log_var.unsqueeze(0)
        post_logvar = torch.zeros_like(prior_logvar)
        kl = 0.5 * (
            prior_logvar
            - post_logvar
            + (next_latent - self.prior_mean.unsqueeze(0))**2 / torch.exp(prior_logvar)
            - 1
        ).sum(dim=-1)
        return float((pragmatic - 0.1 * kl).item())

    def select_action(self, latent, deterministic=False):
        scores = [self.expected_free_energy(latent, a) for a in range(self.action_dim)]
        scores = torch.tensor(scores)
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        probs = F.softmax(-scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() < 1e-6:
            probs = torch.ones_like(probs) / self.action_dim
        if deterministic:
            return int(torch.argmin(scores).item())
        return int(torch.multinomial(probs, 1).item())

    def compute_loss(self, obs_seq, latent_seq, action_seq, reward_seq, next_obs_seq):
        """Compute AIE loss with value baseline.

        loss = free_energy + action_loss + reward_loss
             + value_baseline_loss (for variance reduction)
             + (reward - value_baseline) ** 2 * log_prob
        """
        # Use the final latent state
        latent = latent_seq[-1]

        # Free energy on final obs
        post_mean, post_log_var = self.encoder(obs_seq[-1]).chunk(2, dim=-1)
        post_log_var = torch.clamp(post_log_var, -10, 2)
        kl = 0.5 * (
            post_log_var
            - self.prior_log_var.unsqueeze(0)
            + (latent.unsqueeze(0) - self.prior_mean.unsqueeze(0))**2 / torch.exp(self.prior_log_var).unsqueeze(0)
            - 1
        ).sum(dim=-1)
        recon = ((self.generation(latent) - obs_seq[-1]) ** 2).sum(dim=-1)
        fe_loss = kl.mean() + recon.mean()

        # Action prediction
        action_logits = self.action_sampler(latent)
        action_loss = F.cross_entropy(action_logits, action_seq[-1])

        # Reward prediction
        r_pred = self.value_baseline(latent).squeeze()
        reward_loss = F.mse_loss(r_pred, reward_seq[-1])

        # Variance reduction: use baseline for advantage
        advantage = reward_seq[-1] - r_pred.detach()
        action_logp = F.log_softmax(action_logits, dim=-1)[action_seq[-1]]
        policy_loss = -advantage * action_logp

        # Total loss
        total = (
            0.5 * fe_loss          # free energy (lower weight)
            + 1.0 * action_loss    # action prediction
            + 0.5 * reward_loss    # reward prediction (higher than original 0.1)
            + 1.0 * policy_loss    # variance-reduced policy
        )
        return total, {"fe": float(fe_loss), "action": float(action_loss),
                       "reward": float(reward_loss), "policy": float(policy_loss)}


def collect_episode(env_name, aie, seed, max_steps=500):
    """Collect one episode using AIE policy."""
    env = make_env(env_name, seed=seed)
    obs, _ = env.reset()
    ep_obs = []
    ep_actions = []
    ep_rewards = []
    ep_next_obs = []
    ep_latents = []
    latent = torch.zeros(1, aie.latent_dim)  # (batch=1, latent_dim) for GRU
    ep_reward = 0.0

    for t in range(max_steps):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        latent, _, _ = aie.encode(obs_t, latent)
        action = aie.select_action(latent.squeeze(0), deterministic=False)
        next_obs, reward, term, trunc, _ = env.step(action)
        ep_obs.append(obs_t.squeeze(0))
        ep_actions.append(torch.tensor(action, dtype=torch.long))
        ep_rewards.append(torch.tensor(float(reward)))
        ep_next_obs.append(torch.from_numpy(next_obs).float())
        ep_latents.append(latent.squeeze(0).detach())
        ep_reward += reward
        obs = next_obs
        if term or trunc:
            break
    env.close()
    return ep_obs, ep_actions, ep_rewards, ep_next_obs, ep_latents, ep_reward


def evaluate(env_name, aie, n_episodes, seed, max_steps=500):
    """Evaluate AIE deterministically."""
    eval_returns = []
    for ep in range(n_episodes):
        env = make_env(env_name, seed=seed * 99999 + ep)
        obs, _ = env.reset()
        latent = torch.zeros(1, aie.latent_dim)  # (batch=1, latent_dim) for GRU
        ep_reward = 0.0
        for t in range(max_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            latent, _, _ = aie.encode(obs_t, latent)
            with torch.no_grad():
                action = aie.select_action(latent.squeeze(0), deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                break
        env.close()
        eval_returns.append(ep_reward)
    return float(np.mean(eval_returns)), float(np.std(eval_returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-outer", type=int, default=10)
    p.add_argument("--n-episodes-per-outer", type=int, default=10)
    p.add_argument("--n-epochs-per-outer", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    print("=" * 70)
    print("RECURRENT AIE (with GRU + value baseline)")
    print("=" * 70)
    print(f"  latent_dim={args.latent_dim}, hidden={args.hidden}")
    print(f"  n_outer={args.n_outer}, episodes_per_outer={args.n_episodes_per_outer}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    aie = RecurrentAIE(obs_dim=8, latent_dim=args.latent_dim, action_dim=4,
                        hidden=args.hidden)
    opt = torch.optim.Adam(aie.parameters(), lr=3e-4)

    log = {
        "env": args.env, "seed": args.seed,
        "mode": "Recurrent AIE with GRU + value baseline",
        "outer_history": [],
        "eval_history": [],
    }

    # Initial evaluation
    eval_mean, eval_std = evaluate(args.env, aie, n_episodes=5, seed=args.seed)
    log["eval_history"].append({"step": 0, "mean": eval_mean, "std": eval_std})
    print(f"  step 0: eval={eval_mean:.1f}+/-{eval_std:.1f} (untrained)")

    for outer in range(args.n_outer):
        # Collect episodes
        episodes = []
        train_returns = []
        for ep in range(args.n_episodes_per_outer):
            ep_data = collect_episode(args.env, aie, args.seed * 1000 + outer * 100 + ep)
            episodes.append(ep_data)
            train_returns.append(ep_data[-1])

        train_mean = float(np.mean(train_returns))
        train_std = float(np.std(train_returns))

        # Train
        epoch_losses = []
        for epoch in range(args.n_epochs_per_outer):
            np.random.shuffle(episodes)
            ep_loss_sum = 0.0
            n_batches = 0
            for start in range(0, len(episodes), args.batch_size):
                batch = episodes[start:start + args.batch_size]
                if len(batch) < 2:
                    continue
                opt.zero_grad()
                batch_loss = 0.0
                for ep_data in batch:
                    obs_seq, action_seq, reward_seq, next_obs_seq, latent_seq, _ = ep_data
                    if len(obs_seq) == 0:
                        continue
                    loss, _ = aie.compute_loss(
                        obs_seq, latent_seq, action_seq, reward_seq, next_obs_seq,
                    )
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss / max(1, len(batch))
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(aie.parameters(), 1.0)
                opt.step()
                ep_loss_sum += float(batch_loss.detach())
                n_batches += 1
            if n_batches > 0:
                epoch_losses.append(ep_loss_sum / n_batches)

        # Evaluate
        eval_mean, eval_std = evaluate(args.env, aie, n_episodes=5, seed=args.seed + outer * 100)

        log["outer_history"].append({
            "outer": outer,
            "train_mean": train_mean, "train_std": train_std,
            "eval_mean": eval_mean, "eval_std": eval_std,
            "epoch_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
        })
        log["eval_history"].append({
            "step": (outer + 1) * args.n_episodes_per_outer,
            "mean": eval_mean, "std": eval_std,
        })
        print(f"  outer {outer+1}/{args.n_outer}: train={train_mean:.1f}+/-{train_std:.1f}, "
              f"eval={eval_mean:.1f}+/-{eval_std:.1f}, "
              f"loss={np.mean(epoch_losses) if epoch_losses else 0:.3f}")

    final_mean, final_std = evaluate(args.env, aie, n_episodes=20, seed=args.seed + 99999)
    log["final_eval_mean"] = final_mean
    log["final_eval_std"] = final_std
    print(f"\n  FINAL: eval={final_mean:.1f}+/-{final_std:.1f} (n=20)")

    log_path = HERE / "checkpoints" / "aie_recurrent" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
