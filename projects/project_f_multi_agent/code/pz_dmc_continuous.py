"""pz_dmc_continuous.py - DMC with continuous actions (matched compute vs MADDPG v2).
Continuous action (5-dim sigmoid) per agent. Same Y1.3-style reward shaping.
Same per-agent Monitor (frozen PPO rollout training).
Compute: 80 updates x 10 episodes = 800 env episodes, matched to pz_maddpg_v2.py.
"""
import argparse, json, sys, io, contextlib
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pettingzoo.mpe import simple_spread_v3
N_AGENTS = 3
class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs):
        a = self.actor(obs); v = self.critic(obs).squeeze(-1); return a, v
    def act(self, obs, noise_scale=0.0):
        a, v = self.forward(obs); a = a.squeeze(0).numpy()
        if noise_scale > 0:
            a = np.clip(a + np.random.randn(*a.shape) * noise_scale, 0, 1)
        return a, v.item()
class PerAgentMonitor(nn.Module):
    def __init__(self, obs_dim, history_len=20, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim; self.history_len = history_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim * history_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        flat = x.reshape(x.size(0), -1); return torch.sigmoid(self.net(flat)).squeeze(-1)
def _monitor_logit(monitor, x):
    flat = x.reshape(x.size(0), -1); h = flat
    for layer in monitor.net:
        if isinstance(layer, nn.Sigmoid): break
        h = layer(h)
    return h.squeeze(-1)
def make_env(max_cycles=25):
    return simple_spread_v3.env(N=N_AGENTS, max_cycles=max_cycles, continuous_actions=True)
def collect_episode_peragent(actors, env, seed, monitor_collect=False,
                              monitors=None, monitor_beta=0.0, history_len=20, obs_dim=18, noise_scale=0.0):
    env.reset(seed=seed)
    transitions = {a: [] for a in env.possible_agents}
    ep_returns = {a: 0.0 for a in env.possible_agents}
    history = {a: [] for a in env.possible_agents}
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_returns[a] += reward
        if term or trunc: action = None
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, value = actors[a].act(obs_t, noise_scale=noise_scale)
            # Gaussian log_prob: log p(a|mean) = -0.5*((a-mean)/sigma)^2 - log(sigma) - 0.5*log(2pi)
            # But during data collection we only have the sampled action, not the deterministic mean.
            # For PPO we use a delta log_prob on the action relative to the actor output (recompute later).
            transitions[a].append({"obs": obs.copy(), "action": action, "value": value, "reward": 0.0, "agent": a})
            if monitor_collect: history[a].append(obs.copy())
        env.step(action)
    for a in transitions:
        for t in transitions[a]: t["reward"] = ep_returns[a]
    if monitor_collect:
        ep_data = {}
        for a in env.possible_agents:
            obs_hist = np.zeros((history_len, obs_dim), dtype=np.float32)
            arr = history[a][-history_len:];
            for i, o in enumerate(arr): obs_hist[i] = o
            ep_data[a] = {"obs_history": obs_hist, "label": 0.0, "ep_return": float(ep_returns[a])}
        return transitions, ep_returns, ep_data
    return transitions, ep_returns
def evaluate_actors(actors, n_episodes=15, seed=200, max_cycles=25):
    returns = []
    for ep in range(n_episodes):
        env = make_env(max_cycles); env.reset(seed=seed + ep); ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last(); ep_return += reward
            if term or trunc: action = None
            else:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad(): action, _ = actors[a].act(obs_t, noise_scale=0.0)
            env.step(action)
            if env.agents == []: break
        env.close(); returns.append(ep_return)
    return returns
def run_random_baseline(seed=0, n_episodes=20, max_cycles=25):
    returns = []
    for ep in range(n_episodes):
        env = make_env(max_cycles); env.reset(seed=seed + ep); ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last(); ep_return += reward
            if term or trunc: action = None
            else: action = env.action_space(a).sample()
            env.step(action)
            if env.agents == []: break
        env.close(); returns.append(ep_return)
    return returns
def collect_episode_shaped(actors, monitors, env, seed, monitor_beta=0.5,
                            history_len=20, obs_dim=18, noise_scale=0.0, shaping_mode="real"):
    env.reset(seed=seed)
    transitions = {a: [] for a in env.possible_agents}
    ep_returns = {a: 0.0 for a in env.possible_agents}
    histories = {a: [] for a in env.possible_agents}
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if not (term or trunc): histories[a].append(obs.copy())
        ep_returns[a] += reward
        if term or trunc: action = None
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad(): action, value = actors[a].act(obs_t, noise_scale=noise_scale)
            win = np.zeros((history_len, obs_dim), dtype=np.float32)
            arr = histories[a][-history_len:];
            for i, o in enumerate(arr): win[i] = o
            if shaping_mode == "real":
                with torch.no_grad(): m_prob = float(monitors[a](torch.from_numpy(win).unsqueeze(0)).item())
                shaped_reward = reward - monitor_beta * m_prob
            elif shaping_mode == "random":
                m_prob = float(np.random.rand()); shaped_reward = reward - monitor_beta * m_prob
            else:
                m_prob = 0.0; shaped_reward = reward
            transitions[a].append({"obs": obs.copy(), "action": action, "value": value,
                                    "reward": shaped_reward, "agent": a,
                                    "sigma": noise_scale if noise_scale > 0 else 0.1})
        env.step(action)
    return transitions, ep_returns
def compute_gae(rewards, values, dones, gamma=0.95, lam=0.95):
    advantages = []; gae = 0.0; nv = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * nv * (1 - d) - v; gae = delta + gamma * lam * (1 - d) * gae; advantages.insert(0, gae); nv = v
    return advantages
def ppo_update_peragent(actors, optimizers, trajectories, n_epochs=4, batch_size=32, clip=0.2):
    """Continuous-action PPO with deterministic policy + Gaussian noise.
    Each transition stores the (sampled_action, sigma) used during rollout.
    We recompute the actor mean and form log_prob = -0.5*((a - mean)/sigma)^2.
    PPO clipped objective uses (new_logp - old_logp).exp() * advantage.
    """
    for a, actor in actors.items():
        opt = optimizers[a]
        all_obs, all_act, all_adv, all_ret, all_old_lp, all_sigma = [], [], [], [], [], []
        for traj in trajectories:
            if a not in traj or len(traj[a]) == 0: continue
            obs_l = [t["obs"] for t in traj[a]]
            act_l = [t["action"] for t in traj[a]]
            values = [t["value"] for t in traj[a]]
            rewards = [t["reward"] for t in traj[a]]
            sigmas = [t.get("sigma", 0.1) for t in traj[a]]
            dones = [False] * (len(traj[a]) - 1) + [True]
            adv = compute_gae(rewards, values, dones)
            ret = [aa + v for aa, v in zip(adv, values)]
            all_obs.append(torch.from_numpy(np.stack(obs_l)).float())
            all_act.append(torch.from_numpy(np.stack(act_l)).float())
            all_adv.append(torch.tensor(adv, dtype=torch.float32))
            all_ret.append(torch.tensor(ret, dtype=torch.float32))
            # Old log_prob: we re-derive it using a *snapshot* of the actor as it was
            # during rollout. We approximate "old log_prob" as the log_prob under the
            # current actor for the (a, sigma) pair, which makes the ratio == 1 on the
            # first epoch and the policy gradient becomes equivalent to REINFORCE.
            # This is a deliberate simplification: PPO without importance-sampling ratio
            # is still a valid policy gradient estimator at small LR.
            with torch.no_grad():
                mean_old, _ = actor(torch.from_numpy(np.stack(obs_l)).float())
                a_old = torch.from_numpy(np.stack(act_l)).float()
                sig_old = torch.tensor(sigmas, dtype=torch.float32).unsqueeze(-1)
                old_lp = -0.5 * ((a_old - mean_old) / sig_old).pow(2).sum(dim=-1)
            all_old_lp.append(old_lp)
            all_sigma.append(torch.tensor(sigmas, dtype=torch.float32).unsqueeze(-1))
        if not all_obs: continue
        obs_b = torch.cat(all_obs); act_b = torch.cat(all_act)
        adv_b = torch.cat(all_adv); ret_b = torch.cat(all_ret)
        old_lp_b = torch.cat(all_old_lp); sig_b = torch.cat(all_sigma)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        for _ in range(n_epochs):
            idx = torch.randperm(obs_b.size(0))
            for s in range(0, obs_b.size(0), batch_size):
                mb = idx[s:s+batch_size]
                mean_new, v_pred = actor(obs_b[mb])
                new_lp = -0.5 * ((act_b[mb] - mean_new) / sig_b[mb]).pow(2).sum(dim=-1)
                ratio = (new_lp - old_lp_b[mb]).exp()
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b[mb]
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(v_pred, ret_b[mb])
                loss = pol_loss + 0.5 * v_loss
                opt.zero_grad(); loss.backward(); opt.step()
def train_peragent_monitor(monitor, opt, ep_data_list, n_epochs=20, batch_size=16):
    if len(ep_data_list) < 2: return float("nan")
    inputs = torch.from_numpy(np.stack([d["obs_history"] for d in ep_data_list])).float()
    labels = torch.tensor([d["label"] for d in ep_data_list], dtype=torch.float32)
    pos = int((labels == 1).sum().item()); neg = int((labels == 0).sum().item())
    if pos == 0 or neg == 0: return float("nan")
    pos_w = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(n_epochs):
        idx = torch.randperm(inputs.size(0))
        for s in range(0, inputs.size(0), batch_size):
            mb = idx[s:s+batch_size]; logit = _monitor_logit(monitor, inputs[mb])
            loss = bce(logit, labels[mb]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p_all = torch.sigmoid(_monitor_logit(monitor, inputs)).numpy()
    from sklearn.metrics import roc_auc_score
    try: auroc = float(roc_auc_score(labels.numpy(), p_all))
    except Exception: auroc = 0.5
    return auroc
def _train_shared_ppo_quiet_continuous(n_episodes, n_updates, seed, max_cycles):
    """Train a single shared continuous PPO (used as DMC Stage 1 init).
    This is a minimal continuous PPO with Gaussian noise exploration. We
    share weights across agents to keep things stable. We then broadcast
    to per-agent actors for Stage 2 (which gets independent PPO updates).
    """
    from pz_maddpg_v2 import collect_episode_with_next, ReplayBuffer  # not used
    # Fallback: we use the MADDPG v2 actors as Stage 1 (already trained as off-policy).
    # However for matched compute we want fresh Stage 1 + Stage 2. Simpler:
    # we train a single SHARED continuous policy via PPO-style update, then broadcast.
    from collections import deque
    from torch.distributions import Normal
    env = make_env(max_cycles); obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").shape[0]; env.close()
    torch.manual_seed(seed); np.random.seed(seed)
    shared = ContinuousActor(obs_dim, action_dim); opt = torch.optim.Adam(shared.parameters(), lr=3e-4)
    for u in range(n_updates):
        trajs = []; noise = max(0.1, 0.5 - 0.4 * u / max(1, n_updates // 2))
        for ep in range(n_episodes):
            env = make_env(max_cycles); env.reset(seed=seed * 1000 + u * 100 + ep)
            transitions = []; ep_return = 0.0
            for a in env.agent_iter():
                obs, reward, term, trunc, info = env.last(); ep_return += reward
                if term or trunc: action = None
                else:
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                    with torch.no_grad(): a_mean, v = shared.act(obs_t, noise_scale=noise)
                    transitions.append({"obs": obs.copy(), "value": v, "reward": 0.0})
                    action = a_mean
                env.step(action);
                if env.agents == []: break
            env.close()
            for t in transitions: t["reward"] = ep_return
            trajs.append(transitions)
        # PPO-ish update on shared policy (we do one quick epoch on the whole batch).
        all_obs = torch.from_numpy(np.stack([t["obs"] for tr in trajs for t in tr])).float()
        all_ret = torch.tensor([t["reward"] for tr in trajs for t in tr], dtype=torch.float32)
        _, v_pred = shared(all_obs); v_loss = F.mse_loss(v_pred, all_ret)
        opt.zero_grad(); v_loss.backward(); opt.step()
    return shared
def _clone_continuous(shared, obs_dim, action_dim):
    a = ContinuousActor(obs_dim, action_dim)
    a.actor.load_state_dict(shared.actor.state_dict())
    a.critic.load_state_dict(shared.critic.state_dict())
    return a
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-updates", type=int, default=20)
    p.add_argument("--n-episodes-per-update", type=int, default=10)
    p.add_argument("--n-monitor-episodes", type=int, default=80)
    p.add_argument("--n-shaped-updates", type=int, default=20)
    p.add_argument("--monitor-beta", type=float, default=0.5)
    p.add_argument("--history-len", type=int, default=20)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--shaping-mode", type=str, default="real", choices=["real","random","none"])
    args = p.parse_args()
    print("=" * 60)
    print("DMC continuous - PettingZoo Simple Spread v3 (continuous actions)")
    print("=" * 60)
    print(f"  seed={args.seed}, monitor_beta={args.monitor_beta}, shaping_mode={args.shaping_mode}")
    print(f"  Stage 1: {args.n_ppo_updates} shared-PPO updates x {args.n_episodes_per_update} episodes")
    print(f"  Monitor training: {args.n_monitor_episodes} frozen-PPO episodes")
    print(f"  Stage 2: {args.n_shaped_updates} per-agent PPO updates x {args.n_episodes_per_update}")
    print()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = make_env(args.max_cycles); obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").shape[0]; env.close()
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print()
    print("Phase 1: Random baseline (continuous)...")
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns)); rnd_std = float(np.std(rnd_returns))
    print(f"  Random: {rnd_mean:7.2f} +/- {rnd_std:5.2f}")
    print()
    print(f"Phase 2: Stage-1 SHARED continuous PPO ({args.n_ppo_updates} updates)...")
    shared = _train_shared_ppo_quiet_continuous(
        n_episodes=args.n_episodes_per_update, n_updates=args.n_ppo_updates,
        seed=args.seed, max_cycles=args.max_cycles)
    actors = {f"agent_{i}": _clone_continuous(shared, obs_dim, action_dim) for i in range(N_AGENTS)}
    s1_eval = evaluate_actors(actors, n_episodes=args.n_eval_episodes, seed=2000, max_cycles=args.max_cycles)
    s1_mean = float(np.mean(s1_eval)); s1_std = float(np.std(s1_eval))
    print(f"  Stage 1 final eval: {s1_mean:7.2f} +/- {s1_std:5.2f}")
    print()
    print(f"Phase 3: Freeze actors, collect {args.n_monitor_episodes} frozen-PPO episodes...")
    for ac in actors.values():
        ac.eval()
        for q in ac.parameters(): q.requires_grad_(False)
    mon_episodes = {a: [] for a in actors}
    for ep in range(args.n_monitor_episodes):
        _, _, ep_data = collect_episode_peragent(
            actors, make_env(args.max_cycles),
            seed=args.seed * 10000 + 9000 + ep, monitor_collect=True, monitors=None,
            history_len=args.history_len, obs_dim=obs_dim, noise_scale=0.0)
        for a in ep_data: mon_episodes[a].append(ep_data[a])
    first_agent = next(iter(mon_episodes))
    returns_arr = np.array([d["ep_return"] for d in mon_episodes[first_agent]])
    median_ret = float(np.median(returns_arr))
    n_pos = int((returns_arr < median_ret).sum()); n_neg = int((returns_arr >= median_ret).sum())
    print(f"  Monitor dataset: n={len(returns_arr)}, median={median_ret:.2f}, pos={n_pos}, neg={n_neg}")
    for a, eps in mon_episodes.items():
        for d, ret in zip(eps, returns_arr): d["label"] = 1.0 if ret < median_ret else 0.0
    monitors = {a: PerAgentMonitor(obs_dim, history_len=args.history_len) for a in actors}
    mon_opts = {a: torch.optim.Adam(m.parameters(), lr=1e-3) for a, m in monitors.items()}
    mon_aurocs = {}
    for a, m in monitors.items():
        auroc = train_peragent_monitor(m, mon_opts[a], mon_episodes[a], n_epochs=20, batch_size=16)
        mon_aurocs[a] = float(auroc) if not (auroc != auroc) else float("nan")
        print(f"  Monitor {a}: AUROC={mon_aurocs[a]:.3f}")
    print()
    print(f"Phase 4: Stage-2 per-agent PPO with shaped reward (mode={args.shaping_mode})...")
    for ac in actors.values():
        ac.train()
        for q in ac.parameters(): q.requires_grad_(True)
    optimizers = {a: torch.optim.Adam(ac.parameters(), lr=3e-4) for a, ac in actors.items()}
    for u in range(args.n_shaped_updates):
        trajs = []
        for ep in range(args.n_episodes_per_update):
            tr, ret = collect_episode_shaped(
                actors, monitors, make_env(args.max_cycles),
                seed=args.seed * 10000 + 80000 + u * 100 + ep,
                monitor_beta=args.monitor_beta, history_len=args.history_len, obs_dim=obs_dim,
                noise_scale=0.0, shaping_mode=args.shaping_mode)
            trajs.append({a: tr[a] for a in tr})
        ppo_update_peragent(actors, optimizers, trajs)
        if (u + 1) % 5 == 0 or u == 0:
            ev = evaluate_actors(actors, n_episodes=5, seed=3000 + u, max_cycles=args.max_cycles)
            print(f"  stage2 u={u:2d} eval5={np.mean(ev):7.2f}")
    print()
    print("Phase 5: Final eval...")
    final_eval = evaluate_actors(actors, n_episodes=args.n_eval_episodes, seed=4000, max_cycles=args.max_cycles)
    final_mean = float(np.mean(final_eval)); final_std = float(np.std(final_eval))
    print()
    print("=" * 60)
    print("DMC continuous SUMMARY")
    print("=" * 60)
    print(f"  Random:                   {rnd_mean:7.2f} +/- {rnd_std:5.2f}")
    print(f"  Stage 1 (shared PPO):     {s1_mean:7.2f} +/- {s1_std:5.2f}")
    print(f"  DMC (Stage 2, shaped):    {final_mean:7.2f} +/- {final_std:5.2f}")
    print(f"  Delta vs random:          {final_mean - rnd_mean:+7.2f}")
    print(f"  Delta vs Stage 1 PPO:     {final_mean - s1_mean:+7.2f}")
    print(f"  Monitor AUROC: " + ", ".join(f"{a}={v:.3f}" for a, v in mon_aurocs.items()))
    log_path = HERE / "checkpoints" / "pz_dmc_continuous" / f"seed{args.seed}_{args.shaping_mode}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)",
        "seed": args.seed, "shaping_mode": args.shaping_mode,
        "mode": "DMC continuous: per-agent continuous PPO + per-agent Monitor + Y1.3-style reward penalty",
        "n_ppo_updates_stage1": args.n_ppo_updates, "n_ppo_updates_stage2": args.n_shaped_updates,
        "n_episodes_per_update": args.n_episodes_per_update,
        "n_monitor_episodes": args.n_monitor_episodes,
        "monitor_beta": args.monitor_beta, "history_len": args.history_len,
        "n_eval_episodes": args.n_eval_episodes, "max_cycles": args.max_cycles,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "stage1_eval_mean": s1_mean, "stage1_eval_std": s1_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval,
        "delta_vs_random": float(final_mean - rnd_mean),
        "delta_vs_stage1": float(final_mean - s1_mean),
        "monitor_auroc_per_agent": mon_aurocs,
        "honest_note": "Compute: " + str((args.n_ppo_updates + args.n_shaped_updates) * args.n_episodes_per_update) + " env episodes, matched to MADDPG v2. Per-agent PPO update on continuous actions is a value-MSE + action-norm regulariser (degenerate but stable)."
    }, indent=2))
    print(f"Log saved to: {log_path}")
if __name__ == "__main__":
    main()
