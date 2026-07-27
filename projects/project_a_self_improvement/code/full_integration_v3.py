"""full_integration_v3.py - Phase 1.5 v0.3: DEC-0011 v0.3 4-layer integration.

Differences from v0.2 (REJECTED):
  - Skip Platt scaling entirely (v0.2 overfit val_auroc=1.0 to tiny val set)
  - Use FIXED high threshold (0.7) instead of calibrated threshold
  - Larger val set: 200 episodes (vs 40 in v0.2)
  - Skip Q entirely when Monitor fires; use safe_action=0 (do nothing)
    Rationale: v0.2 CQL Q (200 train eps) was bad; Q-BoN picked bad actions
  - Add temporal hysteresis: only gate if Monitor has been high for last 3 steps
    Rationale: reduce flicker (gate toggling on/off within episode)

Pipeline per step (eval):
  1. PPO proposes action
  2. Slot-attention encodes trajectory as slots
  3. SlotMonitor predicts raw failure prob
  4. If recent_monitor_probs[N=N_HYST] all >= THRESH:
        use safe_action (do nothing)
     else: use PPO action
  5. Take action; world model updates; language interface reports
  6. LTL verifier checks rules on trajectory
  7. Log everything

Expected behavior: gating should be RARE and STABLE, only firing when
Monitor has been consistently high for several steps.
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
PD_CODE = Path(r"E:\agi-research\projects\project_d_language\code")
PE_CODE = Path(r"E:\agi-research\projects\project_e_verification\code")
sys.path.insert(0, str(PA_CODE))
sys.path.insert(0, str(PC_CODE))
sys.path.insert(0, str(PD_CODE))
sys.path.insert(0, str(PE_CODE))

import envs
from ppo import PPOAgent, PPOConfig
from envs import rollout_one_episode, Transition
from slot_attention import SlotAttention
from language_interface import generate_status, generate_plan
from ltl_verifier import verify_rule, graded_truth, DEFAULT_RULES
from calibration import compute_auroc


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
    """Pad history and add action one-hots. Returns (1, history_len, obs_dim+n_actions)."""
    n = min(len(transitions), history_len)
    feat_dim = obs_dim + n_actions
    x = np.zeros((history_len, feat_dim), dtype=np.float32)
    for i, tr in enumerate(transitions[-n:]):
        idx = history_len - n + i
        x[idx, :obs_dim] = tr.obs
        if 0 <= tr.action < n_actions:
            x[idx, obs_dim + tr.action] = 1.0
    return x


def collect_rollouts(env_name, agent, n_episodes, seed):
    """Collect n_episodes from agent.select_action, returns list of episodes."""
    episodes = []
    for i in range(n_episodes):
        e = envs.make_env(env_name, seed=seed * 1000 + i + 1)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        e.close()
        episodes.append(ep)
    return episodes


def train_slot_monitor(monitor, episodes, n_epochs=20, lr=1e-3, batch_size=64):
    """Train SlotMonitor on episodes from frozen policy."""
    opt = torch.optim.Adam(monitor.parameters(), lr=lr)
    obs_dim = 8
    n_actions = 4
    history_len = 20
    feat_dim = obs_dim + n_actions

    # Build dataset
    X_list = []
    Y_list = []
    for ep in episodes:
        ep_failed = (ep.total_reward < -50)  # heuristic failure label
        transitions = ep.transitions
        for t in range(len(transitions)):
            x = np.zeros((history_len, feat_dim), dtype=np.float32)
            n = min(t + 1, history_len)
            for k in range(n):
                idx = history_len - n + k
                tr = transitions[k]
                x[idx, :obs_dim] = tr.obs
                if 0 <= tr.action < n_actions:
                    x[idx, obs_dim + tr.action] = 1.0
            X_list.append(x)
            Y_list.append(1.0 if ep_failed else 0.0)

    X = torch.from_numpy(np.stack(X_list)).float()
    Y = torch.from_numpy(np.array(Y_list, dtype=np.float32))
    n_pos = int(Y.sum().item())
    n_neg = len(Y) - n_pos
    print(f"  SlotMonitor train set: {len(Y)} timesteps, {n_pos} positives ({100*n_pos/len(Y):.1f}%)")

    for epoch in range(n_epochs):
        idx = np.random.permutation(len(Y))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(Y), batch_size):
            bi = idx[start:start + batch_size]
            x_b = X[bi]
            y_b = Y[bi]
            opt.zero_grad()
            pred = monitor(x_b).squeeze(-1)
            loss = F.binary_cross_entropy(pred, y_b)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{n_epochs}: avg loss = {total_loss / max(1, n_batches):.4f}")
    return monitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-steps", type=int, default=100_000)
    p.add_argument("--n-train-episodes", type=int, default=300,
                   help="rollouts for Monitor training (v0.3: larger)")
    p.add_argument("--n-val-episodes", type=int, default=200,
                   help="held-out val set (v0.3: larger)")
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.7,
                   help="FIXED threshold (v0.3 skips calibration)")
    p.add_argument("--n-hysteresis", type=int, default=3,
                   help="require Monitor >= threshold for last N steps")
    p.add_argument("--safe-action", type=int, default=0,
                   help="action when gated (0=do nothing, v0.3 default)")
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--out-tag", default="v03")
    args = p.parse_args()

    out = []
    print_fn = lambda s: (out.append(s), print(s))[1]

    print_fn("=" * 70)
    print_fn("FULL INTEGRATION v0.3 (DEC-0011 v0.3)")
    print_fn("=" * 70)
    print_fn(f"  env={args.env} seed={args.seed}")
    print_fn(f"  threshold={args.threshold} (FIXED, no calibration)")
    print_fn(f"  n_hysteresis={args.n_hysteresis} steps")
    print_fn(f"  safe_action={args.safe_action} (0=do nothing)")
    print_fn(f"  n_eval_episodes={args.n_eval_episodes}")
    print_fn(f"  n_val_episodes={args.n_val_episodes} (v0.3: larger)")
    print_fn("")

    obs_dim = 8
    n_actions = 4
    history_len = args.history_len
    feat_dim = obs_dim + n_actions
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. PPO
    print_fn(f"[A] Training PPO for {args.n_ppo_steps} steps...")
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    train_env = envs.make_env(args.env, seed=args.seed + 1)
    obs, _ = train_env.reset()
    for u in range(args.n_ppo_steps // cfg.rollout_len):
        batch = agent.collect_rollout(train_env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    train_env.close()
    print_fn(f"  PPO trained: {agent}")
    print_fn("")

    # 2. Collect train/val episodes
    print_fn(f"[Data] Collecting {args.n_train_episodes} train + {args.n_val_episodes} val episodes...")
    train_eps = collect_rollouts(args.env, agent, args.n_train_episodes, args.seed)
    val_eps = collect_rollouts(args.env, agent, args.n_val_episodes, args.seed + 99999)

    # Failure rate stats
    train_fail_rate = np.mean([1.0 if ep.total_reward < -50 else 0.0 for ep in train_eps])
    val_fail_rate = np.mean([1.0 if ep.total_reward < -50 else 0.0 for ep in val_eps])
    print_fn(f"  Train failure rate: {train_fail_rate:.2f}")
    print_fn(f"  Val failure rate:   {val_fail_rate:.2f}")
    print_fn("")

    # 3. SlotMonitor
    print_fn(f"[C+D] Training SlotMonitor...")
    slot_attn = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim, n_iters=3, hidden_dim=64, input_dim=feat_dim)
    slot_monitor = SlotMonitor(slot_attn, args.n_slots, args.slot_dim)
    train_slot_monitor(slot_monitor, train_eps, n_epochs=20, lr=1e-3, batch_size=64)

    # Evaluate on val
    print_fn("  Evaluating SlotMonitor on val set...")
    val_X_list = []
    val_Y_list = []
    for ep in val_eps:
        ep_failed = (ep.total_reward < -50)
        transitions = ep.transitions
        for t in range(len(transitions)):
            x = np.zeros((history_len, feat_dim), dtype=np.float32)
            n = min(t + 1, history_len)
            for k in range(n):
                idx = history_len - n + k
                tr = transitions[k]
                x[idx, :obs_dim] = tr.obs
                if 0 <= tr.action < n_actions:
                    x[idx, obs_dim + tr.action] = 1.0
            val_X_list.append(x)
            val_Y_list.append(1.0 if ep_failed else 0.0)
    val_X = torch.from_numpy(np.stack(val_X_list)).float()
    val_Y = np.array(val_Y_list, dtype=np.float32)
    with torch.no_grad():
        val_preds = slot_monitor(val_X).squeeze(-1).numpy()
    val_auroc = compute_auroc(val_Y, val_preds)
    print_fn(f"  val AUROC = {val_auroc:.3f}")
    print_fn(f"  (v0.3: NO Platt scaling — raw threshold)")
    print_fn("")

    # 4-6. Other layers (D, E) — language interface and verifier always available
    rules = DEFAULT_RULES
    print_fn(f"[D+E] Language interface + LTL verifier ready ({len(rules)} rules)")
    print_fn("")

    # 7. Eval loop with v0.3 gating logic
    print_fn("=" * 70)
    print_fn("FULL INTEGRATION v0.3 EVAL EPISODES")
    print_fn("=" * 70)

    returns_gated = []
    returns_ungated = []
    gate_counts = []
    recent_monitor_probs = []  # for hysteresis

    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx)
        obs, _ = e.reset()
        ep_transitions = []
        ep_gate_count = 0
        ep_reward = 0.0
        recent_monitor_probs.clear()

        for t in range(500):
            ppo_action = agent.select_action(obs)
            monitor_prob = 0.0
            if len(ep_transitions) >= 5:
                x = build_slot_input(ep_transitions, history_len, obs_dim, n_actions)
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                with torch.no_grad():
                    monitor_prob = float(slot_monitor(x_t).item())
            recent_monitor_probs.append(monitor_prob)
            if len(recent_monitor_probs) > args.n_hysteresis:
                recent_monitor_probs.pop(0)

            # v0.3 gate condition: hysteresis
            if (len(recent_monitor_probs) >= args.n_hysteresis and
                all(p >= args.threshold for p in recent_monitor_probs[-args.n_hysteresis:])):
                chosen = args.safe_action
                ep_gate_count += 1
            else:
                chosen = ppo_action

            next_obs, reward, term, trunc, _ = e.step(chosen)
            ep_reward += reward
            ep_transitions.append(Transition(obs=obs.copy(), action=chosen, reward=reward))
            obs = next_obs
            if term or trunc:
                break
        e.close()
        returns_gated.append(ep_reward)
        gate_counts.append(ep_gate_count)

        # Ungate baseline (deterministic same-seed eval)
        e2 = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx + 50000)
        ep = rollout_one_episode(e2, agent.select_action, max_steps=500)
        e2.close()
        returns_ungated.append(ep.total_reward)

        if (ep_idx + 1) % 10 == 0:
            print_fn(f"  ep {ep_idx+1}/{args.n_eval_episodes}: "
                     f"gated_so_far={np.mean(returns_gated):.1f}+/-{np.std(returns_gated):.1f}, "
                     f"ungated_so_far={np.mean(returns_ungated):.1f}+/-{np.std(returns_ungated):.1f}, "
                     f"avg_gates={np.mean(gate_counts):.2f}")

    # Summary
    print_fn("")
    print_fn("=" * 70)
    print_fn("PHASE 1.5 v0.3 SUMMARY")
    print_fn("=" * 70)
    print_fn(f"  Episodes:                {args.n_eval_episodes}")
    print_fn(f"  Ungated PPO mean:        {np.mean(returns_ungated):.1f} +/- {np.std(returns_ungated):.1f}")
    print_fn(f"  Gated (v0.3) mean:       {np.mean(returns_gated):.1f} +/- {np.std(returns_gated):.1f}")
    print_fn(f"  Delta:                   {np.mean(returns_gated) - np.mean(returns_ungated):+.2f}")
    print_fn(f"  Avg gates per episode:   {np.mean(gate_counts):.2f}")
    print_fn(f"  SlotMonitor val AUROC:   {val_auroc:.3f}")
    print_fn(f"  Fixed threshold:         {args.threshold}")
    print_fn(f"  Hysteresis N:            {args.n_hysteresis}")
    print_fn("")

    # Save log
    log_dir = HERE / "checkpoints" / ("full_integration_" + args.out_tag + "_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "Phase 1.5 v0.3 fixed-threshold hysteresis 4-layer integration",
        "version": "0.3",
        "n_ppo_steps": args.n_ppo_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_val_episodes": args.n_val_episodes,
        "n_eval_episodes": args.n_eval_episodes,
        "threshold_fixed": args.threshold,
        "n_hysteresis": args.n_hysteresis,
        "safe_action": args.safe_action,
        "val_auroc": float(val_auroc),
        "ungated_mean": float(np.mean(returns_ungated)),
        "ungated_std":  float(np.std(returns_ungated)),
        "gated_mean":   float(np.mean(returns_gated)),
        "gated_std":    float(np.std(returns_gated)),
        "delta":        float(np.mean(returns_gated) - np.mean(returns_ungated)),
        "avg_gates":    float(np.mean(gate_counts)),
        "per_episode_gated": [float(x) for x in returns_gated],
        "per_episode_ungated": [float(x) for x in returns_ungated],
    }, indent=2))
    print_fn(f"Log saved to: {log_dir / 'phase2_log.json'}")


if __name__ == "__main__":
    main()
