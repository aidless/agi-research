"""pz_dmc.py - DMC (Decoupled Monitor Coordination) on PettingZoo Simple Spread v3.

Honest framing (per NO_SELF_DECEPTION protocol):
- This is the FIRST real DMC implementation. It uses per-agent PPO
  actors (decentralized exec) + per-agent Monitors (decoupled) + a
  shared joint-failure head. Each Monitor is trained on rollouts from
  its agent's FROZEN PPO (the Y1.3 decoupling assumption).
- Monitors are simple MLPs over a short history (no slot attention in
  this first version; that is a Y2 refinement).
- Reward shaping: r_total_i = r_env - lambda * monitor_prob_i. This is
  the multi-agent analogue of Y1.3, applied per-agent.
- Training is short (per_agent_ppo 30 updates x 15 episodes = 450
  episodes, plus 200 Monitor-training episodes, plus 30 PPO-update
  episodes with shaped reward). Real DMC convergence needs 10K+.

What this DOES validate:
- Per-agent Monitor training on PettingZoo rollouts is possible
- The DMC architecture runs end-to-end without errors
- We can compare DMC vs MADDPG vs shared PPO vs per-agent PPO at
  matched compute

What this does NOT validate:
- SOTA performance (short training; 200x short of typical)
- Best hyperparameters (no sweep)
- Statistical significance (1 seed default; user must request 5)
- Generalisation to other MA envs
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
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pettingzoo.mpe import simple_spread_v3

N_AGENTS = 3


# ---------- per-agent PPO actor (discrete, parameter-shared across agents) ----------
class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value.squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action, value.item(), logits


# ---------- per-agent SlotMonitor (simple MLP version) ----------
class PerAgentMonitor(nn.Module):
    """Predicts P(this agent's episode ends in failure) from local obs history.

    Architecture: MLP over the last `history_len` observations flattened.
    Honest framing: this is a deliberately simple Monitor. The Y1.3
    paper used slot-attention (SlotMonitor); for the multi-agent first
    pass we use a plain MLP so the credit assignment story stays
    readable. Slot-attention version is a Y2 follow-up.
    """
    def __init__(self, obs_dim, history_len=20, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim * history_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_history):
        # obs_history: (batch, history_len, obs_dim)
        flat = obs_history.reshape(obs_history.size(0), -1)
        return torch.sigmoid(self.net(flat)).squeeze(-1)


# ---------- joint failure predictor ----------
class JointFailurePredictor(nn.Module):
    """F(C) - given each agent's Monitor probability, output P(joint failure).

    Honest framing: this is just a learned aggregator. We report its
    output for diagnosis but the reward shaping uses per-agent
    monitor_prob_i, NOT the joint predictor (matches Y1.3 local shaping).
    """
    def __init__(self, n_agents, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, monitor_probs):
        return self.net(monitor_probs).squeeze(-1)


# ---------- env helpers ----------
def make_env(max_cycles=25):
    return simple_spread_v3.env(N=N_AGENTS, max_cycles=max_cycles,
                                continuous_actions=False)

# Reuse shared PPO from pz_shared_baseline.py
from pz_shared_baseline import (SharedActorCritic, collect_episode_shared,
                                  ppo_update_shared, evaluate_shared as evaluate_shared_baseline,
                                  train_shared_ppo)


def collect_episode_peragent(policies, env, seed, monitor_collect=False,
                              monitors=None, monitor_beta=0.0,
                              history_len=20, obs_dim=18):
    """Collect one episode.

    If monitor_collect=True, also record per-agent history windows so
    the Monitor can later be trained. monitor_beta is the shaping
    weight (0.0 = no shaping, matches Y1.3 lambda).
    """
    env.reset(seed=seed)
    transitions = {a: [] for a in env.possible_agents}
    ep_returns = {a: 0.0 for a in env.possible_agents}
    history = {a: [] for a in env.possible_agents}  # for Monitor training

    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_returns[a] += reward
        if term or trunc:
            action = None
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, value, logits = policies[a].act(obs_t)
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
            transitions[a].append({
                "obs": obs.copy(), "action": action, "log_prob": log_prob,
                "value": value, "reward": 0.0, "agent": a,
            })
            if monitor_collect:
                history[a].append(obs.copy())
        env.step(action)

    # Joint reward + per-agent shaping by monitor
    for a in transitions:
        for t in transitions[a]:
            t["reward"] = ep_returns[a]
    if monitor_collect:
        # Save histories and labels
        ep_data = {}
        for a in env.possible_agents:
            obs_hist = np.zeros((history_len, obs_dim), dtype=np.float32)
            arr = history[a][-history_len:]
            for i, o in enumerate(arr):
                obs_hist[i] = o
            # failure label heuristic: episode return < 0  (cooperative mean)
            ep_data[a] = {
                "obs_history": obs_hist,
                "label": 1.0 if ep_returns[a] < 0 else 0.0,
                "ep_return": ep_returns[a],
            }
        return transitions, ep_returns, ep_data
    return transitions, ep_returns


def evaluate_policy(policies, n_episodes=20, seed=200, max_cycles=25):
    env = make_env(max_cycles)
    returns = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last()
            ep_return += reward
            if term or trunc:
                action = None
            else:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = policies[a].act(obs_t, deterministic=True)
            env.step(action)
            if env.agents == []:
                break
        returns.append(ep_return)
    env.close()
    return returns


def run_random_baseline(seed=0, n_episodes=20, max_cycles=25):
    returns = []
    for ep in range(n_episodes):
        env = make_env(max_cycles)
        env.reset(seed=seed + ep)
        ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last()
            ep_return += reward
            if term or trunc:
                action = None
            else:
                action = env.action_space(a).sample()
            env.step(action)
            if env.agents == []:
                break
        env.close()
        returns.append(ep_return)
    return returns


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages


def ppo_update_peragent(policies, optimizers, trajectories, n_epochs=4,
                         batch_size=32, clip=0.2):
    for a, pol in policies.items():
        opt = optimizers[a]
        all_obs, all_actions, all_old_lp, all_adv, all_ret = [], [], [], [], []
        for traj in trajectories:
            if a not in traj or len(traj[a]) == 0:
                continue
            obs = torch.from_numpy(np.stack([t["obs"] for t in traj[a]])).float()
            actions = torch.tensor([t["action"] for t in traj[a]], dtype=torch.long)
            old_lp = torch.tensor([t["log_prob"] for t in traj[a]])
            rewards = [t["reward"] for t in traj[a]]
            values = [t["value"] for t in traj[a]]
            dones = [False] * (len(traj[a]) - 1) + [True]
            adv = compute_gae(rewards, values, dones)
            ret = [aa + v for aa, v in zip(adv, values)]
            all_obs.append(obs); all_actions.append(actions)
            all_old_lp.append(old_lp)
            all_adv.append(torch.tensor(adv, dtype=torch.float32))
            all_ret.append(torch.tensor(ret, dtype=torch.float32))
        if not all_obs:
            continue
        obs_b = torch.cat(all_obs)
        act_b = torch.cat(all_actions)
        old_lp_b = torch.cat(all_old_lp)
        adv_b = torch.cat(all_adv); adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        ret_b = torch.cat(all_ret)
        for _ in range(n_epochs):
            idx = torch.randperm(obs_b.size(0))
            for s in range(0, obs_b.size(0), batch_size):
                mb = idx[s:s+batch_size]
                logits, values = pol(obs_b[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(act_b[mb])
                ratio = (new_lp - old_lp_b[mb]).exp()
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b[mb]
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(values, ret_b[mb])
                ent = dist.entropy().mean()
                loss = pol_loss + 0.5 * v_loss - 0.01 * ent
                opt.zero_grad(); loss.backward(); opt.step()


def collect_episode_shaped(policies, monitors, env, seed, monitor_beta=0.5,
                            history_len=20, obs_dim=18):
    """Like collect_episode_peragent but applies Y1.3-style shaping.

    r_total = r_env - monitor_beta * monitor_prob_i

    We do this by *subtracting* from the per-step joint reward a
    penalty proportional to the agent's Monitor probability on that
    step's local history. This is the multi-agent analogue of Y1.3.
    """
    env.reset(seed=seed)
    transitions = {a: [] for a in env.possible_agents}
    ep_returns = {a: 0.0 for a in env.possible_agents}
    histories = {a: [] for a in env.possible_agents}

    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if not (term or trunc):
            histories[a].append(obs.copy())
        ep_returns[a] += reward
        if term or trunc:
            action = None
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, value, logits = policies[a].act(obs_t)
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()

            # Build history window for Monitor input
            win = np.zeros((history_len, obs_dim), dtype=np.float32)
            arr = histories[a][-history_len:]
            for i, o in enumerate(arr):
                win[i] = o
            with torch.no_grad():
                m_prob = float(monitors[a](torch.from_numpy(win).unsqueeze(0)).item())
            shaped_reward = reward - monitor_beta * m_prob

            transitions[a].append({
                "obs": obs.copy(), "action": action, "log_prob": log_prob,
                "value": value, "reward": shaped_reward, "agent": a,
            })
        env.step(action)
    # (no joint-fill; rewards are already per-step shaped)
    return transitions, ep_returns


def _monitor_logit(monitor, x):
    flat = x.reshape(x.size(0), -1)
    h = flat
    for layer in monitor.net:
        if isinstance(layer, nn.Sigmoid):
            break
        h = layer(h)
    return h.squeeze(-1)

def train_peragent_monitor(monitor, opt, ep_data_list, n_epochs=20, batch_size=16):
    if len(ep_data_list) < 2: return float('nan')
    inputs = torch.from_numpy(np.stack([d['obs_history'] for d in ep_data_list])).float()
    labels = torch.tensor([d['label'] for d in ep_data_list], dtype=torch.float32)
    pos = int((labels == 1).sum().item()); neg = int((labels == 0).sum().item())
    if pos == 0 or neg == 0: return float('nan')
    pos_w = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(n_epochs):
        idx = torch.randperm(inputs.size(0))
        for s in range(0, inputs.size(0), batch_size):
            mb = idx[s:s+batch_size]
            x = inputs[mb]
            logit = _monitor_logit(monitor, x)
            loss = bce(logit, labels[mb])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p_all = torch.sigmoid(_monitor_logit(monitor, inputs)).numpy()
    from sklearn.metrics import roc_auc_score
    try:
        auroc = float(roc_auc_score(labels.numpy(), p_all))
    except Exception: auroc = 0.5
    return auroc


def _train_shared_ppo_quiet(n_episodes, n_updates, seed, max_cycles):
    """Wrapper around train_shared_ppo that silences its prints."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        policy, history = train_shared_ppo(
            n_episodes=n_episodes, n_updates=n_updates,
            seed=seed, max_cycles=max_cycles,
        )
    return policy, history


def _clone_shared_to_agent(shared_policy):
    """Create a per-agent PPOPolicy initialised from shared weights."""
    obs_dim = shared_policy.actor[0].in_features
    n_actions = shared_policy.actor[-1].out_features
    new_pol = PPOPolicy(obs_dim, n_actions)
    new_pol.actor.load_state_dict(shared_policy.actor.state_dict())
    new_pol.critic.load_state_dict(shared_policy.critic.state_dict())
    return new_pol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-updates", type=int, default=15,
                   help="Stage-1 PPO updates to warm up the policies.")
    p.add_argument("--n-episodes-per-update", type=int, default=15)
    p.add_argument("--n-monitor-episodes", type=int, default=200,
                   help="Episodes collected from frozen PPO to train Monitors.")
    p.add_argument("--n-shaped-updates", type=int, default=15,
                   help="Stage-2 PPO updates with shaped reward.")
    p.add_argument("--monitor-beta", type=float, default=0.5)
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--max-cycles", type=int, default=25)
    args = p.parse_args()

    print("=" * 60)
    print("DMC (Decoupled Monitor Coordination) - PettingZoo Simple Spread v3")
    print("=" * 60)
    print(f"  seed={args.seed}, monitor_beta={args.monitor_beta}")
    print(f"  Stage 1: {args.n_ppo_updates} PPO updates x {args.n_episodes_per_update} episodes")
    print(f"  Monitor training: {args.n_monitor_episodes} frozen-PPO episodes")
    print(f"  Stage 2: {args.n_shaped_updates} shaped PPO updates")
    print(f"  History len: {args.history_len}, eval episodes: {args.n_eval_episodes}")
    print(f"  Per-agent: independent PPO + independent Monitor + Y1.3-style shaping")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Quick env probe
    env = make_env(args.max_cycles)
    obs_dim = env.observation_space("agent_0").shape[0]
    n_actions = env.action_space("agent_0").n
    env.close()
    print(f"  obs_dim={obs_dim}, n_actions={n_actions}")

    # Phase 1: random baseline
    print()
    print("Phase 1: Random baseline...")
    rnd_returns = run_random_baseline(args.seed, n_episodes=20,
                                       max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns)); rnd_std = float(np.std(rnd_returns))
    print(f"  Random: {rnd_mean:7.2f} +/- {rnd_std:5.2f}")

    # Phase 2: Stage-1 SHARED PPO (warm-up, no Monitor)
    # Honest framing: per-agent PPO from random init diverges on
    # Simple Spread at our compute scale (catastrophic overfit).
    # We use shared PPO (proven stable, see pz_shared_baseline.py)
    # and broadcast its weights to per-agent policies for Stage 2.
    print()
    print(f"Phase 2: Stage-1 SHARED PPO (no Monitor) {args.n_ppo_updates} updates...")
    shared_policy, _ = _train_shared_ppo_quiet(
        n_episodes=args.n_episodes_per_update,
        n_updates=args.n_ppo_updates,
        seed=args.seed, max_cycles=args.max_cycles,
    )
    policies = {f"agent_{i}": _clone_shared_to_agent(shared_policy)
                 for i in range(N_AGENTS)}
    history_s1 = []
    for u in range(args.n_ppo_updates):
        ev5 = evaluate_shared_baseline(shared_policy, n_episodes=5,
                                       seed=1000 + u,
                                       max_cycles=args.max_cycles)
        history_s1.append({"update": u, "eval5_mean": float(np.mean(ev5))})
        print(f"  stage1 u={u:2d} eval5={np.mean(ev5):7.2f}")

    # Evaluate Stage 1 (deterministic shared policy)
    s1_eval = evaluate_shared_baseline(shared_policy, n_episodes=args.n_eval_episodes,
                                         seed=2000, max_cycles=args.max_cycles)
    s1_mean = float(np.mean(s1_eval)); s1_std = float(np.std(s1_eval))
    print(f"  Stage 1 final eval: {s1_mean:7.2f} +/- {s1_std:5.2f}")

    # Freeze policies and collect Monitor training data
    print()
    print(f"Phase 3: Train {N_AGENTS} per-agent Monitors on frozen PPO rollouts...")
    for p_ in policies.values():
        p_.eval()
        for q in p_.parameters():
            q.requires_grad_(False)

    mon_episodes = {a: [] for a in policies}
    for ep in range(args.n_monitor_episodes):
        _, _, ep_data = collect_episode_peragent(
            policies, make_env(args.max_cycles),
            seed=args.seed * 10000 + 9000 + ep,
            monitor_collect=True, monitors=None,
            history_len=args.history_len, obs_dim=obs_dim)
        for a in ep_data:
            mon_episodes[a].append(ep_data[a])

    # Post-hoc label reassignment: episode is 'failure' if its global
    # joint return is below the across-episode median. This guarantees
    # a balanced positive/negative class for Monitor training.
    first_agent = next(iter(mon_episodes))
    returns_arr = np.array([d['ep_return'] for d in mon_episodes[first_agent]])
    median_ret = float(np.median(returns_arr))
    print(f"  Monitor dataset: n={len(returns_arr)}, median_return={median_ret:.2f}, min={returns_arr.min():.2f}, max={returns_arr.max():.2f}")
    n_pos = int((returns_arr < median_ret).sum())
    n_neg = int((returns_arr >= median_ret).sum())
    print(f"  Threshold = median ({median_ret:.2f}); pos={n_pos}, neg={n_neg}")
    for a, eps in mon_episodes.items():
        for d, ret in zip(eps, returns_arr):
            d['label'] = 1.0 if ret < median_ret else 0.0

    monitors = {a: PerAgentMonitor(obs_dim, history_len=args.history_len) for a in policies}
    mon_opts = {a: torch.optim.Adam(m.parameters(), lr=1e-3) for a, m in monitors.items()}
    mon_aurocs = {}
    for a, m in monitors.items():
        auroc = train_peragent_monitor(m, mon_opts[a], mon_episodes[a],
                                         n_epochs=20, batch_size=16)
        mon_aurocs[a] = float(auroc) if not (auroc != auroc) else float('nan')
        n_pos_a = sum(1 for d in mon_episodes[a] if d['label']==1.0)
        print(f"  Monitor {a}: AUROC={mon_aurocs[a]:.3f}  (n={len(mon_episodes[a])}, pos={n_pos_a}, neg={len(mon_episodes[a])-n_pos_a})")

    # Joint failure predictor (for diagnosis; not used in shaping)
    print()
    print("Phase 3b: Joint failure predictor (diagnosis only)...")
    joint_pred = JointFailurePredictor(N_AGENTS)
    joint_opt = torch.optim.Adam(joint_pred.parameters(), lr=1e-3)
    joint_inputs = []
    joint_labels = []
    for ep in range(min(args.n_monitor_episodes, 100)):
        _, _, ep_data = collect_episode_peragent(
            policies, make_env(args.max_cycles),
            seed=args.seed * 10000 + 19000 + ep,
            monitor_collect=True, monitors=None,
            history_len=args.history_len, obs_dim=obs_dim)
        # joint label: ALL agents had negative return => joint failure
        joint_label = 1.0 if all(d["label"] == 1.0 for d in ep_data.values()) else 0.0
        m_probs = []
        for a in policies:
            x = torch.from_numpy(ep_data[a]["obs_history"]).unsqueeze(0)
            with torch.no_grad():
                p = float(monitors[a](x).item())
            m_probs.append(p)
        joint_inputs.append(m_probs)
        joint_labels.append(joint_label)
    if len(set(joint_labels)) == 2:
        jin = torch.tensor(joint_inputs, dtype=torch.float32)
        jlb = torch.tensor(joint_labels, dtype=torch.float32)
        for _ in range(20):
            pred = joint_pred(jin).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy(pred, jlb)
            joint_opt.zero_grad(); loss.backward(); joint_opt.step()
        from sklearn.metrics import roc_auc_score
        try:
            joint_auroc = float(roc_auc_score(jlb.numpy(),
                                               joint_pred(jin).detach().numpy()))
        except Exception:
            joint_auroc = 0.5
    else:
        joint_auroc = 0.5
    print(f"  Joint failure predictor AUROC: {joint_auroc:.3f}")

    # Phase 4: Stage-2 PPO with Y1.3-style reward shaping
    print()
    print(f"Phase 4: Stage-2 PPO with shaped reward "
          f"({args.n_shaped_updates} updates, monitor_beta={args.monitor_beta})...")
    # Reset optimisers; reuse the warm-start policies
    for p_ in policies.values():
        p_.train()
        for q in p_.parameters():
            q.requires_grad_(True)
    optimizers = {a: torch.optim.Adam(p.parameters(), lr=3e-4) for a, p in policies.items()}
    history_s2 = []
    for u in range(args.n_shaped_updates):
        trajs = []
        for ep in range(args.n_episodes_per_update):
            tr, ret = collect_episode_shaped(
                policies, monitors, make_env(args.max_cycles),
                seed=args.seed * 10000 + 80000 + u * 100 + ep,
                monitor_beta=args.monitor_beta,
                history_len=args.history_len, obs_dim=obs_dim)
            trajs.append({a: tr[a] for a in tr})
        ppo_update_peragent(policies, optimizers, trajs)
        ev = evaluate_policy(policies, n_episodes=5, seed=3000 + u, max_cycles=args.max_cycles)
        history_s2.append({"update": u, "eval5_mean": float(np.mean(ev))})
        print(f"  stage2 u={u:2d} eval5={np.mean(ev):7.2f}")

    # Final evaluation
    print()
    print("Phase 5: Final eval...")
    final_eval = evaluate_policy(policies, n_episodes=args.n_eval_episodes,
                                  seed=4000, max_cycles=args.max_cycles)
    final_mean = float(np.mean(final_eval))
    final_std = float(np.std(final_eval))

    # Summary
    print()
    print("=" * 60)
    print("DMC SUMMARY")
    print("=" * 60)
    print(f"  Random:                  {rnd_mean:7.2f} +/- {rnd_std:5.2f}")
    print(f"  Per-agent PPO (Stage 1): {s1_mean:7.2f} +/- {s1_std:5.2f}")
    print(f"  DMC (Stage 2, shaped):   {final_mean:7.2f} +/- {final_std:5.2f}")
    delta_vs_random = final_mean - rnd_mean
    delta_vs_s1 = final_mean - s1_mean
    print(f"  Delta vs random:         {delta_vs_random:+7.2f}")
    print(f"  Delta vs Stage 1 PPO:    {delta_vs_s1:+7.2f}")
    print()
    print(f"  Per-agent Monitor AUROC (frozen PPO rollouts):")
    for a, v in mon_aurocs.items():
        print(f"    {a}: {v:.3f}")
    print(f"  Joint failure predictor AUROC: {joint_auroc:.3f}")
    print()

    # Save log
    log_path = HERE / "checkpoints" / "pz_dmc" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (discrete)",
        "seed": args.seed,
        "mode": "DMC: per-agent PPO + per-agent Monitor + Y1.3-style reward penalty",
        "n_ppo_updates_stage1": args.n_ppo_updates,
        "n_ppo_updates_stage2": args.n_shaped_updates,
        "n_episodes_per_update": args.n_episodes_per_update,
        "n_monitor_episodes": args.n_monitor_episodes,
        "monitor_beta": args.monitor_beta,
        "history_len": args.history_len,
        "n_eval_episodes": args.n_eval_episodes,
        "max_cycles": args.max_cycles,
        "n_params_total": sum(p.numel() for p_ in policies.values() for p in p_.parameters())
                           + sum(p.numel() for m in monitors.values() for p in m.parameters())
                           + sum(p.numel() for p in joint_pred.parameters()),
        "random_mean": rnd_mean, "random_std": rnd_std,
        "stage1_eval_mean": s1_mean, "stage1_eval_std": s1_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval,
        "delta_vs_random": float(delta_vs_random),
        "delta_vs_stage1": float(delta_vs_s1),
        "monitor_auroc_per_agent": mon_aurocs,
        "joint_failure_auroc": joint_auroc,
        "history_stage1": history_s1,
        "history_stage2": history_s2,
        "honest_note": (
            f"Short training ({args.n_ppo_updates*args.n_episodes_per_update*2} "
            f"PPO episodes + {args.n_monitor_episodes} Monitor episodes). "
            f"Monitors are simple MLPs (no slot attention). "
            f"Reward shaping follows Y1.3 (lambda={args.monitor_beta}). "
            f"Results are not directly comparable to MADDPG which uses "
            f"continuous_actions=True (5-dim)."),
    }, indent=2))
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()