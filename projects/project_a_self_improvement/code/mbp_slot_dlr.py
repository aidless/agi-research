"""mbp_slot_dlr.py - Model-based planning with slot world model + DLR verifier.

The hypothesis: instead of *gating* PPO actions (which fails), use the
world model + DLR verifier to *plan* a better action sequence.

Architecture (closed-loop MPC):
  1. Encode current obs to slot features (SlotAttention)
  2. For each candidate action a in {0, 1, 2, 3}:
     a. Predict next slot: slot_dynamics(slot_t, a) -> slot_{t+1}
     b. Decode slot back to obs: linear projection slot_{t+1} -> obs_pred
     c. Evaluate DLR predicates on obs_pred -> safety score
  3. Pick action with max safety score
  4. Execute action; repeat from step 1

Why this is different from v0.1-v0.4C:
  - v0.1-v0.4C: Monitor OVERRIDES PPO at inference.
  - MBP: world model PLANS a better action, replacing PPO with planning.

Why this is different from DEC-0011 HALT:
  - DEC-0011: action gating didn't work (do-nothing is wrong).
  - MBP: planning picks the BEST action, not the "safest" do-nothing.

Training:
  1. Slot world model: predicts next-step slot features (already at 0.000007 err).
  2. Slot-to-obs decoder: linear projection from slot features back to obs.
  3. DLR predicates: 95.5% accurate on 7 predicates.

Combined pipeline evaluates each candidate action and picks the best.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PA_CODE = Path(r"E:\agi-research\projects\project_a_self_improvement\code")
PC_CODE = Path(r"E:\agi-research\projects\project_c_causal_world\code")
PE_CODE = Path(r"E:\agi-research\projects\project_e_verification\code")
sys.path.insert(0, str(PA_CODE))
sys.path.insert(0, str(PC_CODE))
sys.path.insert(0, str(PE_CODE))
sys.path.insert(0, str(PE_CODE / "code"))

import envs
from ppo import PPOConfig, PPOAgent
from envs import rollout_one_episode
from slot_attention import SlotAttention
from differentiable_logic import SoftLogic
from dlr_attention import (
    ObsToSlots, AttnSlotPredicateNet,
    collect_dataset, train_jointly,
)


class SlotWorldModel(nn.Module):
    """Combined slot attention + dynamics + decoder.

    Pipeline:
      obs -> SlotAttention -> current_slot
      current_slot + action -> dynamics MLP -> next_slot
      next_slot -> linear decoder -> predicted_obs
    """
    def __init__(self, obs_dim=8, n_slots=4, slot_dim=32, n_actions=4,
                 hidden=64, n_iters=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_actions = n_actions
        # Slot attention
        self.slot_attention = SlotAttention(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=n_iters,
            hidden_dim=hidden, input_dim=obs_dim + n_actions,
        )
        # Dynamics: per-slot MLP
        self.dynamics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim + n_actions, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, slot_dim),
            ) for _ in range(n_slots)
        ])
        # Decoder: slot -> obs
        self.decoder = nn.Linear(n_slots * slot_dim, obs_dim)

    def encode(self, x):
        """obs sequence -> slot features."""
        return self.slot_attention(x)

    def predict_next(self, slots, action):
        """slots + action -> next_slots, predicted_obs."""
        b = slots.size(0)
        action_oh = torch.zeros(b, self.n_actions, device=slots.device)
        action_oh[:, action] = 1.0
        next_slots = []
        for k in range(self.n_slots):
            inp = torch.cat([slots[:, k], action_oh], dim=-1)
            next_slots.append(self.dynamics[k](inp))
        next_slots = torch.stack(next_slots, dim=1)  # (b, n_slots, slot_dim)
        pred_obs = self.decoder(next_slots.reshape(b, -1))
        return next_slots, pred_obs

    def forward(self, x, action):
        slots = self.encode(x)
        return self.predict_next(slots, action)


def collect_obs_action_dataset(env_name, agent, n_episodes, seed, max_steps=500):
    """Collect (obs_t, action_t, obs_{t+1}) triples from agent."""
    triples = []
    for ep in range(n_episodes):
        e = envs.make_env(env_name, seed=seed * 1000 + ep + 1)
        obs, _ = e.reset()
        prev_obs = obs.copy()
        for t in range(max_steps):
            action = agent.select_action(obs)
            obs, _, term, trunc, _ = e.step(action)
            triples.append((prev_obs.copy(), action, obs.copy()))
            prev_obs = obs.copy()
            if term or trunc:
                break
        e.close()
    obs_t = np.stack([t[0] for t in triples])
    actions = np.array([t[1] for t in triples], dtype=np.int64)
    obs_next = np.stack([t[2] for t in triples])
    return obs_t, actions, obs_next


def train_slot_world_model(slot_wm, obs_t, actions, obs_next,
                            n_epochs=20, lr=1e-3, batch_size=128):
    """Train slot attention + dynamics + decoder."""
    opt = torch.optim.Adam(slot_wm.parameters(), lr=lr)
    obs_dim = 8
    n_actions = 4

    X = torch.from_numpy(obs_t).float()
    A = torch.from_numpy(actions).long()
    Y = torch.from_numpy(obs_next).float()
    N = X.shape[0]
    print(f"  Train set: {N} (obs, action, obs_next) triples")

    for epoch in range(n_epochs):
        idx = np.random.permutation(N)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            bi = idx[start:start + batch_size]
            x_b = X[bi]
            a_b = A[bi]
            y_b = Y[bi]

            # Build input: (obs_t, action_oh)
            action_oh = torch.zeros(len(bi), n_actions)
            action_oh[torch.arange(len(bi)), a_b] = 1.0
            x_with_action = torch.cat([x_b, action_oh], dim=-1).unsqueeze(1)  # (b, 1, obs+n_actions)

            # Forward
            next_slots, pred_obs = slot_wm(x_with_action, a_b)

            loss = F.mse_loss(pred_obs, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{n_epochs}: loss = {total_loss / max(1, n_batches):.6f}")
    return slot_wm


def compute_safety_score(obs_proj, predicate_nets, obs):
    """Compute the weighted safety score from DLR predicates on obs.

    Returns float in [0, 1].
    """
    sl = SoftLogic()
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        slots = obs_proj(obs_t)
        upright = predicate_nets["upright"](slots).squeeze()
        low_velocity = predicate_nets["low_velocity"](slots).squeeze()
        # Weighted safety: 0.5 * upright + 0.5 * low_velocity
        score = 0.5 * upright + 0.5 * low_velocity
    return float(score.item())


def mbp_step(slot_wm, obs_proj, predicate_nets, obs, current_slots,
             safety_weight=1.0, ppo_fallback_prob=0.3):
    """Model-based planning: pick action that maximizes predicted safety.

    For each action a:
      1. Predict next_slot, next_obs from current_slots + a
      2. Compute DLR safety on next_obs
    Returns: chosen action.
    """
    obs_dim = 8
    n_actions = 4
    action_scores = []

    # Encode current obs to slots (1-step)
    action_oh = torch.zeros(1, n_actions)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    x_with_action = torch.cat([obs_t, action_oh], dim=-1).unsqueeze(1)

    with torch.no_grad():
        # Re-encode current obs to slots
        current_slots_encoded = slot_wm.encode(x_with_action)
        for a in range(n_actions):
            # Predict next_slot given current_slots + action
            next_slots, pred_obs = slot_wm.predict_next(current_slots_encoded, a)
            pred_obs_np = pred_obs.numpy()[0]
            # Compute DLR safety on predicted obs
            safety = compute_safety_score(obs_proj, predicate_nets, pred_obs_np)
            action_scores.append(safety)

    best_action = int(np.argmax(action_scores))
    return best_action, action_scores


def evaluate_mbp(env_name, agent, slot_wm, obs_proj, predicate_nets,
                 n_episodes, seed, ppo_fallback_prob=0.3):
    """Run MBP-gated episodes."""
    returns = []
    action_choices = []  # which action was picked each step

    for ep_idx in range(n_episodes):
        e = envs.make_env(env_name, seed=seed * 1000 + ep_idx)
        obs, _ = e.reset()
        ep_reward = 0.0
        ep_actions = []
        for t in range(500):
            # MBP pick
            chosen_action, scores = mbp_step(
                slot_wm, obs_proj, predicate_nets, obs, None,
                ppo_fallback_prob=ppo_fallback_prob,
            )
            # Mix with PPO: explore with PPO with prob ppo_fallback_prob
            if np.random.rand() < ppo_fallback_prob:
                chosen_action = agent.select_action(obs)
            ep_actions.append(chosen_action)
            obs, reward, term, trunc, _ = e.step(chosen_action)
            ep_reward += reward
            if term or trunc:
                break
        e.close()
        returns.append(ep_reward)
        action_choices.append(ep_actions)
    return returns, action_choices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-steps", type=int, default=100_000)
    p.add_argument("--n-train-episodes", type=int, default=30)
    p.add_argument("--n-wm-epochs", type=int, default=20)
    p.add_argument("--n-dlr-epochs", type=int, default=30)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--ppo-fallback-prob", type=float, default=0.5,
                   help="probability of using PPO instead of MBP (exploration)")
    args = p.parse_args()

    print("=" * 70)
    print("MODEL-BASED PLANNING with slot WM + DLR verifier")
    print("=" * 70)
    print(f"  env={args.env} seed={args.seed}")
    print(f"  ppo_fallback_prob={args.ppo_fallback_prob} (mixing MBP with PPO)")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Train PPO
    print("[Phase 1] Train PPO...")
    cfg = PPOConfig(obs_dim=8, n_actions=4, rollout_len=2048, seed=args.seed)
    ppo_agent = PPOAgent(cfg)
    train_env = envs.make_env(args.env, seed=args.seed + 1)
    obs, _ = train_env.reset()
    for u in range(args.n_ppo_steps // cfg.rollout_len):
        batch = ppo_agent.collect_rollout(train_env, obs)
        ppo_agent.update(batch)
        obs = batch["final_obs"]
    train_env.close()
    print("  PPO trained")

    # 2. Collect data and train slot world model
    print("[Phase 2] Train slot world model...")
    obs_t, actions, obs_next = collect_obs_action_dataset(
        args.env, ppo_agent, args.n_train_episodes, args.seed,
    )
    slot_wm = SlotWorldModel(obs_dim=8, n_slots=4, slot_dim=32, n_actions=4)
    train_slot_world_model(slot_wm, obs_t, actions, obs_next,
                            n_epochs=args.n_wm_epochs, lr=1e-3)

    # 3. Train DLR predicates
    print("[Phase 3] Train DLR predicates...")
    X_train, Y_train = collect_dataset(args.env, args.n_train_episodes, args.seed)
    obs_proj, predicate_nets, losses = train_jointly(
        X_train, Y_train, n_epochs=args.n_dlr_epochs, lr=1e-3,
        n_slots=4, slot_dim=32, hidden=64,
    )
    print("  DLR trained")

    # 4. Evaluate MBP
    print("[Phase 4] Evaluate MBP-gated episodes...")
    returns_mbp, actions_mbp = evaluate_mbp(
        args.env, ppo_agent, slot_wm, obs_proj, predicate_nets,
        args.n_eval_episodes, args.seed,
        ppo_fallback_prob=args.ppo_fallback_prob,
    )

    # 5. Evaluate ungated PPO baseline
    print("[Phase 5] Evaluate ungated PPO baseline...")
    returns_ungated = []
    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + ep_idx)
        ep = rollout_one_episode(e, ppo_agent.select_action, max_steps=500)
        e.close()
        returns_ungated.append(ep.total_reward)

    # 6. Summary
    print()
    print("=" * 70)
    print("MODEL-BASED PLANNING SUMMARY")
    print("=" * 70)
    print(f"  Ungated PPO mean:    {np.mean(returns_ungated):.1f} +/- {np.std(returns_ungated):.1f}")
    print(f"  MBP-gated mean:      {np.mean(returns_mbp):.1f} +/- {np.std(returns_mbp):.1f}")
    print(f"  Delta:               {np.mean(returns_mbp) - np.mean(returns_ungated):+.2f}")
    print(f"  MBP/PPO mix:         {100*(1-args.ppo_fallback_prob):.0f}% MBP / {100*args.ppo_fallback_prob:.0f}% PPO")

    log_path = HERE / "checkpoints" / "mbp_slot_dlr" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "Model-based planning with slot WM + DLR verifier",
        "n_ppo_steps": args.n_ppo_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_wm_epochs": args.n_wm_epochs,
        "n_dlr_epochs": args.n_dlr_epochs,
        "n_eval_episodes": args.n_eval_episodes,
        "ppo_fallback_prob": args.ppo_fallback_prob,
        "ungated_mean": float(np.mean(returns_ungated)),
        "ungated_std": float(np.std(returns_ungated)),
        "mbp_mean": float(np.mean(returns_mbp)),
        "mbp_std": float(np.std(returns_mbp)),
        "delta": float(np.mean(returns_mbp) - np.mean(returns_ungated)),
        "per_episode_mbp": [float(x) for x in returns_mbp],
        "per_episode_ungated": [float(x) for x in returns_ungated],
    }, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
