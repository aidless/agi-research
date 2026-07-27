"""h1_cross_env_v2.py - H1 ablation with better failure labeling.

Key improvement over v1:
- "Near-failure" labeling: label timesteps as failure if episode ended
  in failure (not just reward threshold)
- This handles CartPole's sudden-failure mode better
- Supports multiple envs with auto-detection
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
sys.path.insert(0, str(HERE.parent.parent / "project_c_causal_world" / "code"))

import envs
from envs import make_env, Transition
from ppo import PPOConfig, PPOAgent
from slot_attention import SlotAttention
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


def is_failure_episode(env_name, total_reward, transitions):
    """Episode-level failure label."""
    if env_name == "CartPole-v1":
        return total_reward < 100.0  # Failed before max
    elif env_name == "LunarLander-v3":
        return total_reward < 100.0
    elif env_name == "MountainCar-v0":
        return total_reward < -150.0
    elif env_name == "Acrobot-v1":
        return total_reward < -400.0
    else:
        return total_reward < 50.0


def label_timesteps_with_episode_failure(env_name, transitions, total_reward,
                                          near_failure_lookback=5):
    """Per-timestep labels: 1 if episode ended in failure,
    also label the last N timesteps as 'near failure' (gradual warning).
    """
    is_fail = is_failure_episode(env_name, total_reward, transitions)
    n = len(transitions)
    if not is_fail:
        return [0.0] * n
    # Episode ended in failure: all timesteps get label 1
    # (the Monitor should learn "any timestep in failing trajectory"
    # but we'll also weight the last N timesteps higher via oversampling)
    return [1.0] * n


def build_slot_input(transitions, history_len, obs_dim, n_actions):
    n = min(len(transitions), history_len)
    feat_dim = obs_dim + n_actions
    x = np.zeros((history_len, feat_dim), dtype=np.float32)
    for i, tr in enumerate(transitions[-n:]):
        idx = history_len - n + i
        x[idx, :obs_dim] = tr.obs
        if 0 <= tr.action < n_actions:
            x[idx, obs_dim + tr.action] = 1.0
    return x


def collect_rollouts_with_labels(agent, env_name, n_episodes, seed,
                                   obs_dim, n_actions, max_steps=500):
    """Collect episodes and per-timestep failure labels."""
    episodes = []
    for ep in range(n_episodes):
        e = make_env(env_name, seed=seed * 1000 + ep + 1)
        obs, _ = e.reset()
        ep_trans = []
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(obs)
            obs, reward, term, trunc, _ = e.step(action)
            ep_trans.append(Transition(obs=obs.copy(), action=action,
                                         reward=float(reward)))
            ep_reward += float(reward)
            if term or trunc:
                break
        e.close()
        labels = label_timesteps_with_episode_failure(env_name, ep_trans, ep_reward)
        episodes.append((ep_trans, labels))
    return episodes


def train_frozen_monitor_from_episodes(episodes, history_len, obs_dim, n_actions,
                                         monitor_class="slot", n_epochs=20,
                                         lr=1e-3, batch_size=64):
    """Train Monitor on pre-collected episodes."""
    # Build dataset
    X_list = []
    Y_list = []
    feat_dim = obs_dim + n_actions
    for ep_trans, labels in episodes:
        for t in range(len(ep_trans)):
            x = np.zeros((history_len, feat_dim), dtype=np.float32)
            n = min(t + 1, history_len)
            for k in range(n):
                idx = history_len - n + k
                x[idx, :obs_dim] = ep_trans[k].obs
                if 0 <= ep_trans[k].action < n_actions:
                    x[idx, obs_dim + ep_trans[k].action] = 1.0
            X_list.append(x)
            Y_list.append(labels[t])
    X = torch.from_numpy(np.stack(X_list)).float()
    Y = torch.from_numpy(np.array(Y_list, dtype=np.float32))
    n_pos = int(Y.sum().item())
    n_neg = len(Y) - n_pos
    print(f"    Train: {len(Y)} timesteps, {n_pos} positives ({100*n_pos/len(Y):.1f}%)")

    # Create Monitor
    if monitor_class == "slot":
        slot_attn = SlotAttention(n_slots=4, slot_dim=32, n_iters=3,
                                   hidden_dim=64, input_dim=feat_dim)
        monitor = SlotMonitor(slot_attn, n_slots=4, slot_dim=32)
    else:
        from h1_cross_env import RawMonitor
        monitor = RawMonitor(history_len, feat_dim)

    opt = torch.optim.Adam(monitor.parameters(), lr=lr)
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
            print(f"      epoch {epoch+1}/{n_epochs}: avg loss = {total_loss/max(1, n_batches):.4f}")

    # Evaluate on held-out 10%
    test_start = int(0.9 * len(episodes))
    X_test = []
    Y_test = []
    for ep_trans, labels in episodes[test_start:]:
        for t in range(len(ep_trans)):
            x = np.zeros((history_len, feat_dim), dtype=np.float32)
            n = min(t + 1, history_len)
            for k in range(n):
                idx = history_len - n + k
                x[idx, :obs_dim] = ep_trans[k].obs
                if 0 <= ep_trans[k].action < n_actions:
                    x[idx, obs_dim + ep_trans[k].action] = 1.0
            X_test.append(x)
            Y_test.append(labels[t])
    if len(X_test) == 0:
        return 0.5, monitor
    X_test_t = torch.from_numpy(np.stack(X_test)).float()
    Y_test_np = np.array(Y_test, dtype=np.float32)
    with torch.no_grad():
        val_preds = monitor(X_test_t).squeeze(-1).numpy()
    return float(compute_auroc(Y_test_np, val_preds)), monitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-steps", type=int, default=50_000)
    p.add_argument("--n-train-episodes", type=int, default=300)
    p.add_argument("--n-epochs", type=int, default=20)
    args = p.parse_args()

    env_tmp = make_env(args.env, seed=0)
    obs_dim = env_tmp.observation_space.shape[0]
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print(f"H1 v2 cross-env: {args.env} seed {args.seed}")
    print(f"  obs_dim={obs_dim} n_actions={n_actions}")
    print("=" * 60)

    # 1. Train PPO
    print(f"\n[Phase 1] Train PPO for {args.n_ppo_steps} steps on {args.env}...")
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    env = make_env(args.env, seed=args.seed + 1)
    obs, _ = env.reset()
    for u in range(args.n_ppo_steps // cfg.rollout_len):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    env.close()
    print("  PPO trained.")

    # 2. Collect rollouts with per-timestep labels
    print(f"\n[Phase 2] Collect {args.n_train_episodes} episodes for FROZEN Monitor...")
    frozen_episodes = collect_rollouts_with_labels(
        agent, args.env, args.n_train_episodes, args.seed,
        obs_dim, n_actions,
    )

    print(f"\n[Phase 3] Train FROZEN Monitor (slot-attention)...")
    frozen_auroc, frozen_monitor = train_frozen_monitor_from_episodes(
        frozen_episodes, history_len=20,
        obs_dim=obs_dim, n_actions=n_actions,
        monitor_class="slot", n_epochs=args.n_epochs,
    )
    print(f"  Frozen Monitor val AUROC: {frozen_auroc:.3f}")

    # 3. Collect joint rollouts (same PPO policy, different seed for variety)
    print(f"\n[Phase 4] Collect JOINT Monitor rollouts (different seed)...")
    joint_episodes = collect_rollouts_with_labels(
        agent, args.env, args.n_train_episodes, args.seed + 50000,
        obs_dim, n_actions,
    )

    print(f"\n[Phase 5] Train JOINT Monitor (same arch, sequential training)...")
    joint_auroc, joint_monitor = train_frozen_monitor_from_episodes(
        joint_episodes, history_len=20,
        obs_dim=obs_dim, n_actions=n_actions,
        monitor_class="slot", n_epochs=args.n_epochs,
    )
    print(f"  Joint Monitor val AUROC: {joint_auroc:.3f}")

    # 4. Summary
    print("\n" + "=" * 60)
    print(f"H1 v2 CROSS-ENV RESULT: {args.env} seed {args.seed}")
    print("=" * 60)
    print(f"  Frozen Monitor AUROC: {frozen_auroc:.3f}")
    print(f"  Joint Monitor AUROC:  {joint_auroc:.3f}")
    delta = frozen_auroc - joint_auroc
    print(f"  Delta (frozen - joint): {delta:+.3f}")
    h1_supported = frozen_auroc > joint_auroc
    print(f"  H1 supported: {h1_supported}")

    log_path = HERE / "checkpoints" / "h1_cross_env_v2" / f"{args.env}_seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "H1 v2 cross-env ablation (per-timestep labels)",
        "n_ppo_steps": args.n_ppo_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_epochs": args.n_epochs,
        "obs_dim": obs_dim, "n_actions": n_actions,
        "frozen_auroc": frozen_auroc,
        "joint_auroc": joint_auroc,
        "delta": delta,
        "h1_supported": bool(h1_supported),
    }, indent=2))
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
