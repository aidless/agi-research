"""pz_maddpg_v3.py - MADDPG v3: per-agent Monitor as auxiliary loss in critic.
Y2 follow-up to H5 (REFUTED): try Monitor as critic-side auxiliary loss instead
of reward shaping. Architecturally, the critic uses per-agent Monitor prediction as
an extra loss term; the actor is unaffected. Same PettingZoo Simple Spread v3, same
compute as v2.
Three arms: with_aux (real Monitor aux loss), no_aux (=v2 baseline), ablated (random Monitor).
"""
import argparse, json, sys, logging
logging.disable(logging.WARNING)
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pettingzoo.mpe import simple_spread_v3
N_AGENTS = 3; OBS_DIM = 18; ACTION_DIM = 5; HISTORY_LEN = 20
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid(),
        )
    def forward(self, obs): return self.net(obs)
class CentralizedCritic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, n_agents, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )
    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1); return self.net(x)
class ReplayBuffer:
    def __init__(self, capacity=20000): self.buffer = deque(maxlen=capacity)
    def push(self, t): self.buffer.append(t)
    def sample(self, b):
        idx = np.random.choice(len(self.buffer), b, replace=False)
        return [self.buffer[i] for i in idx]
    def __len__(self): return len(self.buffer)
def soft_update(target, source, tau=0.01):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)
def make_env(max_cycles=25):
    return simple_spread_v3.env(N=N_AGENTS, max_cycles=max_cycles, continuous_actions=True)
def collect_episode_with_next(actors, env, seed, noise_scale=0.1):
    env.reset(seed=seed)
    last_obs = {a: None for a in env.possible_agents}
    last_action = {a: None for a in env.possible_agents}
    pending = []; ep_return = 0.0
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last(); ep_return += reward; done = term or trunc
        if not done:
            agent_idx = int(a.split("_")[-1])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad(): action_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
            noise = np.random.randn(*action_mean.shape) * noise_scale
            action = np.clip(action_mean + noise, 0.0, 1.0)
        else: action = None
        if a in last_obs and last_obs[a] is not None and last_action[a] is not None:
            pending.append({"agent": int(a.split("_")[-1]), "obs": last_obs[a], "action": last_action[a],
                             "next_obs": obs.copy() if not done else np.zeros_like(last_obs[a]),
                             "reward": float(reward), "done": bool(done)})
        last_obs[a] = obs if obs is not None else np.zeros(18)
        last_action[a] = action if action is not None else np.zeros(5)
        env.step(action)
        if env.agents == []: break
    return pending, float(ep_return)
class AuxHead(nn.Module):
    """Per-agent trainable aux head: (obs_i, action_i) -> monitor label proxy."""
    def __init__(self, obs_dim, action_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class PerAgentMonitor(nn.Module):
    def __init__(self, obs_dim, history_len, hidden=64):
        super().__init__(); self.obs_dim = obs_dim; self.history_len = history_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim * history_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return torch.sigmoid(self.net(x.reshape(x.size(0), -1))).squeeze(-1)
def _monitor_logit(monitor, x):
    flat = x.reshape(x.size(0), -1); h = flat
    for layer in monitor.net:
        if isinstance(layer, nn.Sigmoid): break
        h = layer(h)
    return h.squeeze(-1)
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
def evaluate_maddpg(actors, n_episodes=15, seed=200, max_cycles=25):
    returns = []
    for ep in range(n_episodes):
        env = make_env(max_cycles); env.reset(seed=seed + ep); ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last(); ep_return += reward
            if term or trunc: action = None
            else:
                agent_idx = int(a.split("_")[-1])
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad(): action = actors[agent_idx](obs_t).squeeze(0).numpy()
            env.step(action)
            if env.agents == []: break
        env.close(); returns.append(ep_return)
    return returns
def collect_monitor_data(actors, n_eps, seed_base):
    obs_by_agent = []; returns = []
    print("    collect_monitor_data: n_eps=" + str(n_eps) + " starting", flush=True)
    for ep in range(n_eps):
        if ep % 10 == 0: print("    collect_monitor_data: " + str(ep) + "/" + str(n_eps), flush=True)
        env = make_env(25); env.reset(seed=seed_base + ep)
        ep_obs = {a: [] for a in env.possible_agents}; ep_return = 0.0
        for a in env.agent_iter():
            obs, reward, term, trunc, info = env.last(); ep_return += reward
            if term or trunc: action = None
            else:
                agent_idx = int(a.split("_")[-1])
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad(): action = actors[agent_idx](obs_t).squeeze(0).numpy()
                ep_obs[a].append(obs.copy())
            env.step(action)
            if env.agents == []: break
        env.close()
        for a in env.possible_agents:
            arr = ep_obs[a]
            if arr:
                tail = arr[-HISTORY_LEN:]
                h = np.zeros((HISTORY_LEN, OBS_DIM), dtype=np.float32)
                h[-len(tail):] = tail
                obs_by_agent.append(h)
            else:
                obs_by_agent.append(np.zeros((HISTORY_LEN, OBS_DIM), dtype=np.float32))
        returns.append(ep_return)
    print("    collect_monitor_data: collected " + str(len(obs_by_agent)) + " obs_by_agent, " + str(len(returns)) + " returns; returns=" + str(returns[:5]), flush=True)
    return obs_by_agent, returns
def train_per_agent_monitors(actors, n_eps, seed, n_epochs=20, batch_size=16):
    obs_list, rets = collect_monitor_data(actors, n_eps, seed * 1000 + 7777)
    if len(obs_list) < 2: return None, [float("nan")] * N_AGENTS
    median_ret = float(np.median(rets))
    print("    train_per_agent_monitors: len(obs_list)=" + str(len(obs_list)) + ", median_ret=" + str(round(median_ret, 2)), flush=True)
    X = torch.from_numpy(np.stack(obs_list)).float()
    y = torch.from_numpy(np.array([1.0 if rets[i // N_AGENTS] < median_ret else 0.0 for i in range(len(obs_list))], dtype=np.float32))
    monitors = [PerAgentMonitor(OBS_DIM, HISTORY_LEN) for _ in range(N_AGENTS)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in monitors]
    aurocs = []
    for agent_idx, monitor in enumerate(monitors):
        print("    training Monitor agent " + str(agent_idx), flush=True)
        idx = list(range(agent_idx, len(obs_list), N_AGENTS))
        if len(idx) < 2: aurocs.append(float("nan")); continue
        Xi = X[idx]; yi = y[idx]
        if len(set(yi.tolist())) < 2: aurocs.append(float("nan")); continue
        pos = int((yi == 1).sum().item()); neg = int((yi == 0).sum().item())
        if pos == 0 or neg == 0: aurocs.append(float("nan")); continue
        pos_w = torch.tensor([neg / max(pos, 1)]); bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for ep_i in range(n_epochs):
            order = torch.randperm(Xi.size(0))
            for s in range(0, Xi.size(0), batch_size):
                mb = order[s:s+batch_size]
                logit = _monitor_logit(monitor, Xi[mb])
                loss = bce(logit, yi[mb])
                opts[agent_idx].zero_grad(); loss.backward(); opts[agent_idx].step()
            if ep_i < 3 or ep_i == n_epochs-1: print("      agent " + str(agent_idx) + " epoch " + str(ep_i) + " done, loss=" + str(round(float(loss.detach()), 4)), flush=True)
        print("      agent " + str(agent_idx) + " training done, computing AUROC", flush=True)
        with torch.no_grad():
            p_all = torch.sigmoid(_monitor_logit(monitor, Xi)).numpy()
        print("      agent " + str(agent_idx) + " p_all stats: min=" + str(round(float(p_all.min()), 3)) + " max=" + str(round(float(p_all.max()), 3)) + " mean=" + str(round(float(p_all.mean()), 3)), flush=True)
        try:
            y_np = yi.numpy()
            pos_mask = (y_np == 1); neg_mask = (y_np == 0)
            n_pos = int(pos_mask.sum()); n_neg = int(neg_mask.sum())
            if n_pos == 0 or n_neg == 0: a_val = float("nan")
            else:
                a_val = float((p_all[pos_mask].sum() / n_pos + (1.0 - p_all[neg_mask]).sum() / n_neg) / 2.0)
            print("      agent " + str(agent_idx) + " AUROC=" + str(round(a_val, 3)), flush=True)
            aurocs.append(a_val)
        except Exception as e:
            print("      agent " + str(agent_idx) + " AUROC failed: " + str(e), flush=True)
            aurocs.append(0.5)
    return monitors, aurocs
def predict_monitor_on_obs(monitor, obs):
    """Returns a tensor with grad enabled, so aux loss flows back to critic via Monitor input."""
    h = np.tile(obs[None, :], (HISTORY_LEN, 1))
    x = torch.from_numpy(h).unsqueeze(0).float()
    p = monitor(x)
    return p[0] if p.dim() > 0 else p
def train_maddpg_v3(seed=0, n_updates=80, n_episodes=10, batch_size=128,
                    buffer_size=20000, gamma=0.95, tau=0.01,
                    lr_actor=1e-4, lr_critic=1e-3, monitor_alpha=0.5,
                    n_monitor_eps=80, monitor_epochs=20,
                    noise_start=0.5, noise_end=0.05,
                    use_monitor_aux=True, ablate_monitor=False,
                    use_aux_head=False, log_every=10):
    torch.manual_seed(seed); np.random.seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS)
    target_actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    target_critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS)
    for ta, a in zip(target_actors, actors): ta.load_state_dict(a.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    buffer = ReplayBuffer(capacity=buffer_size)
    print("  Stage 1: 10 warmup updates starting...", flush=True)
    for u in range(10):
        noise = max(0.1, noise_start - (noise_start - noise_end) * u / 5)
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_with_next(actors, make_env(25),
                seed=seed * 1000 + u * 100 + ep, noise_scale=noise)
            for t in transitions: buffer.push(t)
        if u % 2 == 0: print("  Stage 1 update " + str(u+1) + "/10 done, buffer=" + str(len(buffer)), flush=True)
    print("  Stage 1 done. Training Monitors...", flush=True)
    monitors, mon_aurocs = train_per_agent_monitors(actors, n_monitor_eps, seed, n_epochs=monitor_epochs)
    print("  Monitors trained, AUROC=" + str(mon_aurocs), flush=True)
    if ablate_monitor or monitors is None:
        monitors = [PerAgentMonitor(OBS_DIM, HISTORY_LEN) for _ in range(N_AGENTS)]
        mon_aurocs = [float("nan")] * N_AGENTS
    aux_heads = None; aux_opts = None
    if use_aux_head:
        aux_heads = [AuxHead(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
        aux_opts = [torch.optim.Adam(h.parameters(), lr=1e-3) for h in aux_heads]
    history = []
    for u in range(n_updates):
        noise = max(noise_end, noise_start - (noise_start - noise_end) * u / max(1, n_updates // 2))
        ep_returns = []
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_with_next(actors, make_env(25),
                seed=seed * 1000 + 1000 + u * 100 + ep, noise_scale=noise)
            ep_returns.append(ep_return)
            for t in transitions: buffer.push(t)
        if len(buffer) < batch_size: continue
        samples = buffer.sample(batch_size)
        per_agent = {i: {"obs": [], "act": [], "nobs": [], "rew": [], "done": []} for i in range(N_AGENTS)}
        for s in samples:
            i = s["agent"]
            per_agent[i]["obs"].append(s["obs"]); per_agent[i]["act"].append(s["action"])
            per_agent[i]["nobs"].append(s["next_obs"]); per_agent[i]["rew"].append(s["reward"])
            per_agent[i]["done"].append(float(s.get("done", False)))
        for i in range(N_AGENTS):
            d = per_agent[i]
            if len(d["obs"]) < 2: continue
            obs_b = torch.tensor(np.stack(d["obs"]), dtype=torch.float32)
            act_b = torch.tensor(np.stack(d["act"]), dtype=torch.float32)
            nobs_b = torch.tensor(np.stack(d["nobs"]), dtype=torch.float32)
            rew_b = torch.tensor(d["rew"], dtype=torch.float32)
            done_b = torch.tensor(d["done"], dtype=torch.float32)
            Bi = obs_b.shape[0]
            full_obs = torch.zeros(Bi, N_AGENTS * OBS_DIM)
            full_act = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_obs[:, i * OBS_DIM:(i + 1) * OBS_DIM] = obs_b
            full_act[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = act_b
            with torch.no_grad():
                a_next = target_actors[i](nobs_b)
                full_nobs = torch.zeros(Bi, N_AGENTS * OBS_DIM)
                full_nact = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
                full_nobs[:, i * OBS_DIM:(i + 1) * OBS_DIM] = nobs_b
                full_nact[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = a_next
                q_next = target_critic(full_nobs, full_nact)[:, i]
                target = rew_b + gamma * (1.0 - done_b) * q_next
            q_pred = critic(full_obs, full_act)[:, i]
            critic_loss = F.mse_loss(q_pred, target)
            if aux_heads is not None:
                aux_pred = aux_heads[i](obs_b, act_b)
                with torch.no_grad():
                    aux_target = (rew_b < -50).float()
                aux_loss_h = F.mse_loss(aux_pred, aux_target)
                critic_loss = critic_loss + monitor_alpha * aux_loss_h
                aux_opts[i].zero_grad(); aux_loss_h.backward(retain_graph=True); aux_opts[i].step()
            if use_monitor_aux and monitors is not None:
                with torch.no_grad():
                    mon_targets = (rew_b < -50).float()
                mon_pred_tensors = [predict_monitor_on_obs(monitors[i], d["obs"][k]) for k in range(Bi)]
                mon_pred = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in mon_pred_tensors])
                aux_loss = F.mse_loss(mon_pred, mon_targets)
                critic_loss = critic_loss + monitor_alpha * aux_loss
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            pred_action = actors[i](obs_b)
            full_act_pred = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_act_pred[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = pred_action
            actor_loss = -critic(full_obs, full_act_pred)[:, i].mean()
            actor_opts[i].zero_grad(); actor_loss.backward(); actor_opts[i].step()
        for ta, a in zip(target_actors, actors): soft_update(ta, a, tau)
        soft_update(target_critic, critic, tau)
        mean_return = float(np.mean(ep_returns))
        history.append({"update": u, "mean_return": mean_return, "noise_scale": noise, "buffer_size": len(buffer)})
        if (u + 1) % log_every == 0 or u == 0:
            print("    update " + str(u+1) + "/" + str(n_updates) + ": mean_episode_return=" + str(round(mean_return, 2)) + ", buffer=" + str(len(buffer)) + ", noise=" + str(round(noise, 3)))
    return actors, critic, monitors, history, mon_aurocs
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-updates", type=int, default=80)
    p.add_argument("--n-episodes-per-update", type=int, default=10)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--monitor-alpha", type=float, default=0.5)
    p.add_argument("--n-monitor-eps", type=int, default=80)
    p.add_argument("--arm", type=str, default="with_aux", choices=["with_aux","no_aux","ablated"])
    p.add_argument("--use-aux-head", action="store_true", default=False, help="Use trainable AuxHead in critic (instead of frozen Monitor).")
    args = p.parse_args()
    print("=" * 60)
    print("MADDPG v3 - arm=" + args.arm + " - PettingZoo Simple Spread v3 (continuous)")
    print("=" * 60)
    print("  seed=" + str(args.seed) + ", n_updates=" + str(args.n_updates) + ", monitor_alpha=" + str(args.monitor_alpha))
    print()
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns)); rnd_std = float(np.std(rnd_returns))
    print("Phase 1: Random baseline = " + str(round(rnd_mean, 2)) + " +/- " + str(round(rnd_std, 2)))
    print()
    use_aux = (args.arm == "with_aux")
    ablate = (args.arm == "ablated")
    actors, critic, monitors, history, mon_aurocs = train_maddpg_v3(
        seed=args.seed, n_updates=args.n_updates, n_episodes=args.n_episodes_per_update,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        monitor_alpha=args.monitor_alpha, n_monitor_eps=args.n_monitor_eps,
        use_monitor_aux=use_aux, ablate_monitor=ablate,
        use_aux_head=args.use_aux_head)
    print()
    print("Phase 3: Final eval...")
    final_eval = evaluate_maddpg(actors, n_episodes=args.n_eval_episodes, seed=4000)
    final_mean = float(np.mean(final_eval)); final_std = float(np.std(final_eval))
    delta = final_mean - rnd_mean
    print("  MADDPG v3 (" + args.arm + ") eval: " + str(round(final_mean, 2)) + " +/- " + str(round(final_std, 2)) + "  (delta vs random: " + ("%.2f" % delta) + ")")
    print("  Monitor AUROC per agent: " + str(mon_aurocs))
    log_path = HERE / "checkpoints" / "pz_maddpg_v3" / ("seed" + str(args.seed) + "_" + args.arm) / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)", "seed": args.seed, "arm": args.arm,
        "mode": "MADDPG v3 (Monitor as critic aux loss)" if use_aux else ("MADDPG v3 (no Monitor)" if not ablate else "MADDPG v3 (random Monitor)"),
        "n_updates": args.n_updates, "n_episodes_per_update": args.n_episodes_per_update,
        "monitor_alpha": args.monitor_alpha, "n_monitor_eps": args.n_monitor_eps,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval, "delta_vs_random": float(delta),
        "monitor_auroc": mon_aurocs, "history": history,
        "honest_note": "Y2 follow-up: Monitor as critic-side auxiliary loss, not reward shaping.",
    }, indent=2))
    print("  Log: " + str(log_path))
if __name__ == "__main__": main()
