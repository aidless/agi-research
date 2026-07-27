"""full_integration_v2.py - Phase 1.5 v0.2: All 4 layers + calibration guards.

Differences from full_integration.py (v0.1):
  - Train/val split for SlotMonitor (default 80/20)
  - Val-set Platt scaling: 1-param logistic on raw failure probs
  - Adaptive threshold: pick gate_threshold for target FPR (default 10%)
  - Q coverage guard: only gate if Q has seen >= min_q_coverage (s,a) pairs
  - n_eval_episodes default raised from 5 to 50
  - phase2_log.json now includes val_auroc, platt_a, platt_b,
    cal_threshold, q_coverage, etc.

Pipeline per step (eval):
  1. PPO proposes action
  2. Slot-attention encodes trajectory as slots
  3. SlotMonitor predicts raw failure prob
  4. Platt-scale the raw prob
  5. If calibrated_prob >= cal_threshold AND q_coverage_ok:
        use Q-BoN argmax; else use PPO action
  6. Take action; world model updates; language interface reports
  7. LTL verifier checks rules on trajectory
  8. Log everything
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
from envs import rollout_one_episode
from slot_attention import SlotAttention
from language_interface import generate_status, generate_plan
from ltl_verifier import verify_rule, graded_truth, DEFAULT_RULES
from calibration import (
    compute_auroc, platt_fit, platt_apply,
    find_threshold_for_fpr, count_unique_sa_pairs,
)


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


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, state, action):
        a_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        x = torch.cat([state, a_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def build_slot_input(transitions, history_len, obs_dim, n_actions):
    per_step = obs_dim + n_actions + 1
    arr = np.zeros((history_len, per_step), dtype=np.float32)
    for i, t in enumerate(transitions[-history_len:]):
        arr[i, :obs_dim] = t.obs
        if 0 <= t.action < n_actions:
            arr[i, obs_dim + t.action] = 1.0
        arr[i, obs_dim + n_actions] = t.reward
    return arr


def predict_monitor(slot_monitor, episodes, history_len, obs_dim, n_actions):
    """Return raw failure probabilities and labels for each episode."""
    inputs, labels = [], []
    for ep in episodes:
        if len(ep.transitions) < 1: continue
        inputs.append(build_slot_input(ep.transitions, history_len, obs_dim, n_actions))
    if not inputs:
        return np.zeros(0), np.zeros(0)
    X = torch.from_numpy(np.stack(inputs)).float()
    with torch.no_grad():
        p = slot_monitor(X).cpu().numpy()
    return p.astype(np.float64)


def train_slot_monitor(slot_monitor, train_eps, val_eps, threshold,
                       history_len, obs_dim, n_actions, n_epochs=15):
    """Train SlotMonitor on train_eps; return (val_auroc, val_labels, val_raw_probs)."""
    # Build train inputs/labels
    inputs, labels = [], []
    for ep in train_eps:
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
    # Val predictions
    val_raw = predict_monitor(slot_monitor, val_eps, history_len, obs_dim, n_actions)
    val_labels = np.array([1.0 if ep.total_reward < threshold else 0.0
                            for ep in val_eps if len(ep.transitions) >= 1], dtype=np.float64)
    val_auroc = compute_auroc(val_labels, val_raw) if len(val_labels) > 0 else float("nan")
    return val_auroc, val_labels, val_raw


def train_q_cql(q_net, episodes, n_epochs=15, cql_alpha=1.0, gamma=0.99):
    transitions = []
    for ep in episodes:
        ts = ep.transitions
        for i in range(len(ts) - 1):
            transitions.append((ts[i].obs.copy(), ts[i].action, ts[i].reward,
                                  ts[i+1].obs.copy(), False))
        if len(ts) > 0:
            transitions.append((ts[-1].obs.copy(), ts[-1].action, ts[-1].reward,
                                  ts[-1].obs.copy(), True))
    s_all = np.stack([t[0] for t in transitions])
    a_all = np.array([t[1] for t in transitions], dtype=np.int64)
    r_all = np.array([t[2] for t in transitions], dtype=np.float32)
    s_next_all = np.stack([t[3] for t in transitions])
    done_all = np.array([t[4] for t in transitions], dtype=np.float32)
    opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    for epoch in range(n_epochs):
        idx = np.random.permutation(len(transitions))
        for start in range(0, len(transitions), 32):
            bi = idx[start:start+32]
            s = torch.from_numpy(s_all[bi]).float()
            a = torch.from_numpy(a_all[bi])
            r = torch.from_numpy(r_all[bi])
            s_next = torch.from_numpy(s_next_all[bi]).float()
            done = torch.from_numpy(done_all[bi])
            q_pred = q_net(s, a)
            with torch.no_grad():
                q_next_all = torch.stack([
                    q_net(s_next, torch.full((s_next.size(0),), ai, dtype=torch.long))
                    for ai in range(q_net.n_actions)
                ], dim=-1)
                q_next_max = q_next_all.max(dim=-1).values
                td_target = r + gamma * q_next_max * (1.0 - done)
            td_loss = F.mse_loss(q_pred, td_target)
            q_all_a = torch.stack([
                q_net(s, torch.full((s.size(0),), ai, dtype=torch.long))
                for ai in range(q_net.n_actions)
            ], dim=-1)
            cql_loss = (torch.logsumexp(q_all_a, dim=-1) - q_pred).mean()
            loss = td_loss + cql_alpha * cql_loss
            opt.zero_grad(); loss.backward(); opt.step()
    return q_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--n-ppo-steps", type=int, default=100000)
    p.add_argument("--n-train-episodes", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--gate-threshold", type=float, default=0.5,
                   help="v0.2 IGNORED unless --use-fixed-threshold is set")
    p.add_argument("--use-fixed-threshold", action="store_true",
                   help="Skip calibration, use --gate-threshold as-is (v0.1 behavior)")
    p.add_argument("--target-fpr", type=float, default=0.10,
                   help="Target false-positive rate on val set for calibrated threshold")
    p.add_argument("--min-q-coverage", type=int, default=50,
                   help="Min unique (s,a) pairs in Q training data; else never gate")
    p.add_argument("--val-fraction", type=float, default=0.20,
                   help="Fraction of train_eps to hold out for calibration")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-tag", default="v2",
                   help="Subdirectory tag for checkpoints to keep v0.1 vs v0.2 separate")
    args = p.parse_args()

    out = []
    out.append(f"[Phase 1.5 v0.2: Calibrated 4-Layer Integration] env={args.env} seed={args.seed}")
    out.append(f"  target_fpr={args.target_fpr}  min_q_coverage={args.min_q_coverage}"
               f"  val_fraction={args.val_fraction}  n_eval={args.n_eval_episodes}")
    out.append("=" * 70)

    # 1. Train PPO
    env = envs.make_env(args.env, seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    obs, _ = env.reset()
    for u in range(args.n_ppo_steps // cfg.rollout_len):
        batch = agent.collect_rollout(env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    env.close()
    out.append("[A] PPO trained.")

    # 2. Collect rollouts
    all_eps = []
    for i in range(args.n_train_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + i + 7777)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        all_eps.append(ep)
        e.close()
    threshold = max(0.0, float(np.percentile([e.total_reward for e in all_eps], 10.0)))
    # Train / val split
    n_val = max(1, int(round(args.n_train_episodes * args.val_fraction)))
    n_tr  = args.n_train_episodes - n_val
    np.random.seed(args.seed + 1000)
    perm = np.random.permutation(args.n_train_episodes)
    val_idx = perm[:n_val].tolist()
    tr_idx  = perm[n_val:].tolist()
    train_eps = [all_eps[i] for i in tr_idx]
    val_eps   = [all_eps[i] for i in val_idx]
    out.append(f"[Data] {len(all_eps)} rollouts, threshold={threshold:.1f}"
               f"  (train={n_tr}, val={n_val})")

    # 3. Layer A: SlotMonitor
    per_step = obs_dim + n_actions + 1
    slot = SlotAttention(n_slots=args.n_slots, slot_dim=args.slot_dim,
                          n_iters=3, hidden_dim=64, input_dim=per_step)
    slot_monitor = SlotMonitor(slot, args.n_slots, args.slot_dim, hidden=64)
    val_auroc, val_labels, val_raw = train_slot_monitor(
        slot_monitor, train_eps, val_eps, threshold,
        args.history_len, obs_dim, n_actions)
    out.append(f"[A] SlotMonitor trained (val AUROC = {val_auroc:.3f})")

    # 3b. Platt scaling
    if args.use_fixed_threshold or len(val_labels) < 2:
        cal_threshold = args.gate_threshold
        platt_a, platt_b = 1.0, 0.0
        out.append(f"[Cal] Using FIXED threshold = {cal_threshold:.3f}"
                   f"  (Platt skipped)")
    else:
        platt_a, platt_b = platt_fit(val_raw, val_labels)
        val_cal = platt_apply(val_raw, platt_a, platt_b)
        cal_threshold = find_threshold_for_fpr(val_cal, val_labels, args.target_fpr)
        out.append(f"[Cal] Platt fit: a={platt_a:.3f} b={platt_b:.3f}"
                   f"  -> target FPR={args.target_fpr:.2f}, threshold={cal_threshold:.3f}")
        # Sanity report: FPR/TPR at chosen threshold
        gate_mask = val_cal >= cal_threshold
        n_neg = int((val_labels == 0).sum()); n_pos = int((val_labels == 1).sum())
        fpr = float((gate_mask & (val_labels == 0)).sum() / max(1, n_neg))
        tpr = float((gate_mask & (val_labels == 1)).sum() / max(1, n_pos))
        out.append(f"[Cal] val FPR={fpr:.2f}  val TPR={tpr:.2f}  (n_neg={n_neg}, n_pos={n_pos})")

    # 4. Layer Q: Q-function with CQL
    q_net = QNetwork(obs_dim, n_actions, hidden=64)
    train_q_cql(q_net, train_eps, n_epochs=15, cql_alpha=1.0)
    q_coverage = count_unique_sa_pairs(train_eps, obs_dim, n_actions)
    q_coverage_ok = q_coverage >= args.min_q_coverage
    out.append(f"[Q] Q-function trained. coverage={q_coverage} (need>={args.min_q_coverage})"
               f" -> gate_enabled={q_coverage_ok}")

    # 5-6. Layers E and D
    rules = DEFAULT_RULES
    out.append(f"[E] Verifier ready with {len(rules)} rules")
    out.append("[D] Language interface ready (template-based)")

    # 7. Eval loop
    out.append("")
    out.append("=" * 70)
    out.append("FULL INTEGRATION EPISODES (v0.2)")
    out.append("=" * 70)

    returns_gated = []
    returns_ungated = []
    gate_counts = []
    gate_skip_qcov = 0
    all_traces = []

    for ep_idx in range(args.n_eval_episodes):
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx)
        obs, _ = e.reset()
        ep_transitions = []
        ep_gate_count = 0
        ep_reward = 0.0
        ep_skip = 0

        for t in range(500):
            ppo_action = agent.select_action(obs)
            monitor_prob_cal = 0.0
            if len(ep_transitions) >= 5:
                x = build_slot_input(ep_transitions, args.history_len, obs_dim, n_actions)
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                with torch.no_grad():
                    raw_p = float(slot_monitor(x_t).item())
                if args.use_fixed_threshold:
                    monitor_prob_cal = raw_p
                else:
                    monitor_prob_cal = float(platt_apply(np.array([raw_p]), platt_a, platt_b)[0])

            use_q = (monitor_prob_cal >= cal_threshold) and q_coverage_ok
            if use_q:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                    q_vals = torch.stack([
                        q_net(obs_t, torch.tensor([a], dtype=torch.long))
                        for a in range(n_actions)
                    ]).squeeze(-1)
                    chosen = int(torch.argmax(q_vals).item())
                ep_gate_count += 1
            else:
                if monitor_prob_cal >= cal_threshold and not q_coverage_ok:
                    ep_skip += 1
                chosen = ppo_action

            next_obs, reward, term, trunc, _ = e.step(chosen)
            ep_reward += reward
            from envs import Transition
            ep_transitions.append(Transition(obs=obs.copy(), action=chosen, reward=reward))

            if t % args.log_every == 0:
                status = generate_status(obs, monitor_prob_cal,
                                          recent_actions=[tr.action for tr in ep_transitions[-10:]])
                plan = generate_plan(monitor_prob_cal, cal_threshold)
                out.append(f"  ep{ep_idx} t={t:3d} action={chosen} reward_so_far={ep_reward:.0f}"
                           f" | {status[:100]}")
                out.append(f"     {plan}")

            obs = next_obs
            if term or trunc:
                break
        e.close()
        returns_gated.append(ep_reward)
        gate_counts.append(ep_gate_count)
        gate_skip_qcov += ep_skip
        all_traces.append(ep_transitions)

        # Ungate baseline (same env seed offset as v0.1)
        e = envs.make_env(args.env, seed=args.seed * 1000 + 9999 + ep_idx + 50000)
        ep = rollout_one_episode(e, agent.select_action, max_steps=500)
        e.close()
        returns_ungated.append(ep.total_reward)

        # Verifier
        trace_dicts = []
        for tr in ep_transitions:
            trace_dicts.append({
                "x_pos": float(tr.obs[0]), "y_pos": float(tr.obs[1]),
                "x_vel": float(tr.obs[2]), "y_vel": float(tr.obs[3]),
                "angle": float(tr.obs[4]), "ang_vel": float(tr.obs[5]),
                "leg_l": float(tr.obs[6]), "leg_r": float(tr.obs[7]),
            })
        rule_results = []
        for rule in rules:
            try:
                f, c = graded_truth(rule, trace_dicts)
                rule_results.append({"rule": rule, "freq": f, "conf": c})
            except Exception as e:
                rule_results.append({"rule": rule, "error": str(e)})
        out.append(f"  ep{ep_idx} FINAL: gated={ep_reward:.0f} ungated={returns_ungated[-1]:.0f}"
                   f" gates={ep_gate_count} (skipped_qcov={ep_skip})")
        for r in rule_results:
            out.append(f"    Verifier: {r['rule']} -> freq={r.get('freq', 'N/A')} conf={r.get('conf', 'N/A')}")
        out.append("")

    # Summary
    out.append("=" * 70)
    out.append("PHASE 1.5 v0.2 FULL INTEGRATION SUMMARY")
    out.append("=" * 70)
    out.append(f"  Episodes:                {args.n_eval_episodes}")
    out.append(f"  Ungated PPO mean:        {np.mean(returns_ungated):.1f} +/- {np.std(returns_ungated):.1f}")
    out.append(f"  Gated (Monitor+Q) mean:  {np.mean(returns_gated):.1f} +/- {np.std(returns_gated):.1f}")
    out.append(f"  Delta:                   {np.mean(returns_gated) - np.mean(returns_ungated):+.2f}")
    out.append(f"  Avg gates per episode:   {np.mean(gate_counts):.2f}")
    out.append(f"  Gates skipped (q_cov):   {gate_skip_qcov}")
    out.append(f"  SlotMonitor val AUROC:   {val_auroc:.3f}")
    out.append(f"  Platt:                   a={platt_a:.3f}, b={platt_b:.3f}")
    out.append(f"  Calibrated threshold:    {cal_threshold:.3f}  (target FPR={args.target_fpr})")
    out.append(f"  Q coverage:              {q_coverage}  (min={args.min_q_coverage})")
    out.append("")

    print(chr(10).join(out))

    # Save per-seed JSON (separate dir to avoid clobbering v0.1)
    log_dir = HERE / "checkpoints" / ("full_integration_" + args.out_tag + "_" + args.env + "_seed" + str(args.seed))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase2_log.json").write_text(json.dumps({
        "env": args.env, "seed": args.seed,
        "mode": "Phase 1.5 v0.2 calibrated 4-layer integration",
        "version": "0.2",
        "n_ppo_steps": args.n_ppo_steps,
        "n_train_episodes": args.n_train_episodes,
        "n_val_episodes": n_val,
        "n_eval_episodes": args.n_eval_episodes,
        "target_fpr": args.target_fpr,
        "min_q_coverage": args.min_q_coverage,
        "use_fixed_threshold": args.use_fixed_threshold,
        "val_auroc": float(val_auroc) if not (isinstance(val_auroc, float) and val_auroc != val_auroc) else None,
        "platt_a": float(platt_a),
        "platt_b": float(platt_b),
        "cal_threshold": float(cal_threshold),
        "q_coverage": int(q_coverage),
        "q_coverage_ok": bool(q_coverage_ok),
        "ungated_mean": float(np.mean(returns_ungated)),
        "ungated_std":  float(np.std(returns_ungated)),
        "gated_mean":   float(np.mean(returns_gated)),
        "gated_std":    float(np.std(returns_gated)),
        "delta":        float(np.mean(returns_gated) - np.mean(returns_ungated)),
        "avg_gates":    float(np.mean(gate_counts)),
        "gates_skipped_qcov": int(gate_skip_qcov),
        "rules_checked": rules,
        "per_episode_gated": [float(x) for x in returns_gated],
        "per_episode_ungated": [float(x) for x in returns_ungated],
    }, indent=2))


if __name__ == "__main__":
    main()
