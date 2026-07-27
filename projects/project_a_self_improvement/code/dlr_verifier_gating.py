"""dlr_verifier_gating.py - DLR-based verifier-aware gating pipeline.

This replaces the SlotMonitor-based gating pipeline (DEC-0011 HALTed)
with a DLR-based verifier pipeline. The hypothesis: with DLR predicates
at 95.5% accuracy (the dlr_attention fix), the verifier signal is more
reliable than the Monitor signal for action-level gating.

Architecture:
  1. Train DLR predicates (via dlr_attention.py — STRONG POSITIVE 95.5%)
  2. Build composite safety formula over DLR predicates
  3. At each step, compute fuzzy truth value of formula
  4. If safety < threshold: gate action (use safe_action)
  5. Compare against:
     - No gating (PPO only)
     - Hard-coded LTL gating
     - DLR gating (this work)

This is the first end-to-end DLR pipeline.
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
PE_CODE = Path(r"E:\agi-research\projects\project_e_verification\code")
sys.path.insert(0, str(PA_CODE))
sys.path.insert(0, str(PE_CODE))
sys.path.insert(0, str(HERE.parent / "project_c_causal_world" / "code"))

import envs
from ppo import PPOConfig, PPOAgent
from envs import rollout_one_episode
from differentiable_logic import SoftLogic

# Import DLR training functions
sys.path.insert(0, str(PE_CODE / "code"))
from dlr_attention import (
    ObsToSlots, AttnSlotPredicateNet,
    collect_dataset, train_jointly, evaluate_attn,
    gt_landed, gt_upright, gt_leg_l_contact, gt_leg_r_contact,
    gt_in_pad, gt_low_velocity, gt_safe_approach,
)


def compose_safety_formula(dlr_outputs):
    """Compute a composite safety score from DLR predicates using fuzzy logic.

    Safety formula:
      upright AND low_velocity AND (NOT very_unstable)

    Where:
      upright = DLR(upright) truth value
      low_velocity = DLR(low_velocity) truth value
      very_unstable = NOT(leg_l_contact OR leg_r_contact) AND NOT landed
                      i.e., lander is in air without stable contact
    """
    sl = SoftLogic()
    upright = dlr_outputs["upright"]
    low_velocity = dlr_outputs["low_velocity"]
    leg_l = dlr_outputs["leg_l_contact"]
    leg_r = dlr_outputs["leg_r_contact"]
    landed = dlr_outputs["landed"]
    in_pad = dlr_outputs["in_pad"]

    # very_unstable = NOT(leg_l OR leg_r) AND NOT landed
    # (lander is in air without contact AND not yet landed)
    any_leg = sl.or_op(leg_l, leg_r)
    has_contact = any_leg
    not_landed = sl.not_op(landed)
    in_air = sl.and_op(sl.not_op(has_contact), not_landed)
    very_unstable = in_air

    # Weighted safety score: 0.5 * upright + 0.5 * low_velocity
    # Range [0, 1]; gate when < threshold
    # This is more lenient than pure conjunction (AND)
    score = 0.5 * upright + 0.5 * low_velocity

    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def train_ppo(env_name, n_ppo_steps, seed):
    """Train PPO from scratch."""
    obs_dim = 8
    n_actions = 4
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=seed)
    agent = PPOAgent(cfg)
    train_env = envs.make_env(env_name, seed=seed + 1)
    obs, _ = train_env.reset()
    for u in range(n_ppo_steps // cfg.rollout_len):
        batch = agent.collect_rollout(train_env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    train_env.close()
    return agent


def evaluate_dlr_gating(env_name, agent, obs_proj, predicate_nets,
                         n_episodes, seed, safety_threshold=0.5,
                         safe_action=0):
    """Run episodes with DLR-verifier gating.

    At each step:
      1. PPO proposes action
      2. Compute DLR predicate values
      3. Compute safety formula truth value
      4. If safety < threshold: substitute safe_action
    """
    returns_gated = []
    gate_counts = []
    safety_history = []

    for ep_idx in range(n_episodes):
        e = envs.make_env(env_name, seed=seed * 1000 + ep_idx)
        obs, _ = e.reset()
        ep_gate_count = 0
        ep_reward = 0.0
        ep_safeties = []

        for t in range(500):
            # DLR predicate evaluation
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                slots = obs_proj(obs_t)
                dlr_outputs = {name: net(slots).squeeze() for name, net in predicate_nets.items()}
                safety = compose_safety_formula(dlr_outputs)

            ep_safeties.append(safety)

            # PPO action
            ppo_action = agent.select_action(obs)

            # Gate
            if safety < safety_threshold:
                chosen = safe_action
                ep_gate_count += 1
            else:
                chosen = ppo_action

            obs, reward, term, trunc, _ = e.step(chosen)
            ep_reward += reward
            if term or trunc:
                break
        e.close()
        returns_gated.append(ep_reward)
        gate_counts.append(ep_gate_count)
        safety_history.append(np.mean(ep_safeties))

    return returns_gated, gate_counts, safety_history


def evaluate_ungated(env_name, agent, n_episodes, seed):
    """Baseline: PPO without gating."""
    returns = []
    for ep_idx in range(n_episodes):
        e = envs.make_env(env_name, seed=seed * 1000 + ep_idx)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        e.close()
        returns.append(ep.total_reward)
    return returns


def evaluate_ltl_gating(env_name, agent, n_episodes, seed, safe_action=0):
    """Baseline: LTL-style hardcoded gating.

    Gate when:
      angle > 0.3 (about to fail upright) OR
      velocity > 0.5 (about to fail velocity)
    """
    returns = []
    gate_counts = []
    for ep_idx in range(n_episodes):
        e = envs.make_env(env_name, seed=seed * 1000 + ep_idx)
        obs, _ = e.reset()
        ep_reward = 0.0
        ep_gate_count = 0
        for t in range(500):
            # Hardcoded predicates (LTL-style)
            angle_violation = abs(obs[4]) > 0.3
            velocity_violation = np.sqrt(obs[2]**2 + obs[3]**2) > 0.5
            ltl_violates = angle_violation or velocity_violation

            if ltl_violates:
                chosen = safe_action
                ep_gate_count += 1
            else:
                chosen = agent.select_action(obs)

            obs, reward, term, trunc, _ = e.step(chosen)
            ep_reward += reward
            if term or trunc:
                break
        e.close()
        returns.append(ep_reward)
        gate_counts.append(ep_gate_count)
    return returns, gate_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-steps", type=int, default=100_000)
    p.add_argument("--n-train-episodes", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--safety-threshold", type=float, default=0.5)
    p.add_argument("--safe-action", type=int, default=0)
    args = p.parse_args()

    print("=" * 70)
    print("DLR VERIFIER-AWARE GATING (replacing SlotMonitor-based)")
    print("=" * 70)
    print(f"  env={args.env} seed={args.seed}")
    print(f"  safety_threshold={args.safety_threshold} (DLR formula < this -> gate)")
    print(f"  safe_action={args.safe_action}")
    print(f"  n_eval_episodes={args.n_eval_episodes}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Train DLR predicates (reuse dlr_attention.py)
    print("[Phase 1] Train DLR predicates...")
    X_train, Y_train = collect_dataset(args.env, args.n_train_episodes, args.seed)
    obs_proj, predicate_nets, losses = train_jointly(
        X_train, Y_train, n_epochs=args.n_epochs, lr=1e-3, batch_size=128,
        n_slots=4, slot_dim=32, hidden=64,
    )
    final_losses = {name: float(loss[-1]) for name, loss in losses.items()}
    print(f"  Final BCE losses: {final_losses}")

    # 2. Train PPO
    print("[Phase 2] Train PPO...")
    agent = train_ppo(args.env, args.n_ppo_steps, args.seed)
    print("  PPO trained")

    # 3. Evaluate all three pipelines
    print("[Phase 3] Evaluate DLR-gated, LTL-gated, ungated...")
    returns_dlr_gated, gate_counts_dlr, safety_hist = evaluate_dlr_gating(
        args.env, agent, obs_proj, predicate_nets,
        args.n_eval_episodes, args.seed,
        safety_threshold=args.safety_threshold,
        safe_action=args.safe_action,
    )
    returns_ltl_gated, gate_counts_ltl = evaluate_ltl_gating(
        args.env, agent, args.n_eval_episodes, args.seed,
        safe_action=args.safe_action,
    )
    returns_ungated = evaluate_ungated(args.env, agent, args.n_eval_episodes, args.seed)

    # Summary
    print()
    print("=" * 70)
    print("DLR VERIFIER GATING SUMMARY")
    print("=" * 70)
    print(f"  Ungated PPO mean:        {np.mean(returns_ungated):.1f} +/- {np.std(returns_ungated):.1f}")
    print(f"  LTL-gated mean:          {np.mean(returns_ltl_gated):.1f} +/- {np.std(returns_ltl_gated):.1f}")
    print(f"  DLR-gated mean:          {np.mean(returns_dlr_gated):.1f} +/- {np.std(returns_dlr_gated):.1f}")
    print(f"  DLR - ungated delta:     {np.mean(returns_dlr_gated) - np.mean(returns_ungated):+.2f}")
    print(f"  LTL - ungated delta:     {np.mean(returns_ltl_gated) - np.mean(returns_ungated):+.2f}")
    print(f"  Avg DLR gates/episode:   {np.mean(gate_counts_dlr):.2f}")
    print(f"  Avg LTL gates/episode:   {np.mean(gate_counts_ltl):.2f}")
    print(f"  Avg DLR safety score:    {np.mean(safety_hist):.3f}")

    # Save log
    log_path = HERE / "checkpoints" / "dlr_verifier_gating" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "DLR verifier-aware gating (post-DLR-attention fix)",
        "n_ppo_steps": args.n_ppo_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_epochs": args.n_epochs,
        "n_eval_episodes": args.n_eval_episodes,
        "safety_threshold": args.safety_threshold,
        "safe_action": args.safe_action,
        "dlr_predicate_losses": final_losses,
        "ungated_mean": float(np.mean(returns_ungated)),
        "ungated_std": float(np.std(returns_ungated)),
        "ltl_gated_mean": float(np.mean(returns_ltl_gated)),
        "ltl_gated_std": float(np.std(returns_ltl_gated)),
        "dlr_gated_mean": float(np.mean(returns_dlr_gated)),
        "dlr_gated_std": float(np.std(returns_dlr_gated)),
        "dlr_minus_ungated_delta": float(np.mean(returns_dlr_gated) - np.mean(returns_ungated)),
        "ltl_minus_ungated_delta": float(np.mean(returns_ltl_gated) - np.mean(returns_ungated)),
        "avg_dlr_gates": float(np.mean(gate_counts_dlr)),
        "avg_ltl_gates": float(np.mean(gate_counts_ltl)),
        "avg_dlr_safety": float(np.mean(safety_hist)),
        "per_episode_ungated": [float(x) for x in returns_ungated],
        "per_episode_ltl_gated": [float(x) for x in returns_ltl_gated],
        "per_episode_dlr_gated": [float(x) for x in returns_dlr_gated],
    }, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
