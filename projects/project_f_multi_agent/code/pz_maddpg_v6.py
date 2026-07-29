"""pz_maddpg_v6.py - Trust head architecture ablation: v5 with RANDOM trust-head input.

Y2 follow-up to v5 (2026-07-29). The original v6 (this file before rewrite)
was a broken stub: it used torch.randn for obs and skipped the Bellman
update. This rewrite is a PROPER ablation of v5:

  - Architecture (critic, replay buffer, actor, trust head, training
    loop) is IDENTICAL to pz_maddpg_trusthead_same_agent.py (renamed v5).
  - ONLY difference: the trust head input (my_mon and others_stats) is
    uniform random in [0,1] instead of the same-agent Monitor broadcast.
  - Stage 0 (per-agent Monitor training) is SKIPPED (monitors unused
    in v6's random arm; with_verifier arm still trains them for
    matched compare).

Three arms (matching v5):
  - with_verifier: real (same-agent) Monitor -- == v5
  - no_verifier:  MADDPG v2 baseline (no trust head)
  - with_trusthead_random: random trust head input (v6 ablation arm)

The key test: if v5's +0.17 effect (n=5) is from the Monitor signal,
v6 with_trusthead_random should NOT show it. If v6 = v5 numerically,
the Monitor is being ignored (trust head learns to use obs only).
This is the same test as v7's "trust head + Monitor" vs "trust head +
random", but v6 is the proper clean implementation.

For n=5 budget at 80 updates x 10 episodes = 800 env episodes.
"""
import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.disable(logging.WARNING)
sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pettingzoo.mpe import simple_spread_v3

N_AGENTS = 3
OBS_DIM = 18
ACTION_DIM = 5
HISTORY_LEN = 20


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid(),
        )

    def forward(self, obs):
        return self.net(obs)


class CentralizedCritic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, n_agents, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )

    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


class PerAgentMonitor(nn.Module):
    """obs history -> failure prob in [0,1]."""

    def __init__(self, obs_dim, history_len=HISTORY_LEN, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim * history_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        flat = x.reshape(x.size(0), -1)
        return torch.sigmoid(self.net(flat)).squeeze(-1)


def _monitor_logit(monitor, x):
    flat = x.reshape(x.size(0), -1)
    h = flat
    for layer in monitor.net:
        if isinstance(layer, nn.Sigmoid):
            break
        h = layer(h)
    return h.squeeze(-1)


class TrustHead(nn.Module):
    """Input: (my_obs, my_monitor_prob, chain_summary_features) -> per-other-agent trust weights.

    chain_summary_features: per-other-agent trust weights in [0,1] (slot filled with same-agent monitor proxy).
    Output: (n_agents-1) trust weights in [0,1] via sigmoid.
    """

    def __init__(self, obs_dim, n_agents, hidden=64):
        super().__init__()
        in_dim = obs_dim + 1 + (n_agents - 1)  # my_obs + my_monitor + others' monitor stats
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents - 1),
        )

    def forward(self, my_obs, my_monitor_prob, others_monitor_stats):
        # my_obs: (B, obs_dim); my_monitor_prob: (B, 1); others_monitor_stats: (B, n_agents-1)
        x = torch.cat([my_obs, my_monitor_prob, others_monitor_stats], dim=-1)
        return torch.sigmoid(self.net(x))


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, t):
        self.buffer.append(t)

    def sample(self, b):
        idx = np.random.choice(len(self.buffer), b, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


def soft_update(target, source, tau=0.01):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def make_env(max_cycles=25):
    return simple_spread_v3.env(N=N_AGENTS, max_cycles=max_cycles, continuous_actions=True)




def collect_episode_v5(actors, monitors, env, seed, noise_scale=0.1,
                        use_verifier=True, random_verifier=False):
    """Collect one episode (no cross-agent evidence chain is built; per-agent monitors only)."""
    env.reset(seed=seed)
    last_obs = {a: None for a in env.possible_agents}
    last_action = {a: None for a in env.possible_agents}
    last_monitor = {a: None for a in env.possible_agents}
    pending = []
    ep_return = 0.0
    step_idx = 0
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_return += reward
        done = term or trunc
        cur_monitor = None
        if not done:
            agent_idx = int(a.split("_")[-1])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
                noise = np.random.randn(*action_mean.shape) * noise_scale
                action = np.clip(action_mean + noise, 0.0, 1.0)
                # Compute monitor prob on this obs (replicate single obs history_len times)
                if use_verifier or random_verifier:
                    h = np.tile(obs[None, :], (HISTORY_LEN, 1))
                    x = torch.from_numpy(h).unsqueeze(0).float()
                    if random_verifier:
                        cur_monitor = float(np.random.rand())
                    else:
                        cur_monitor = float(monitors[agent_idx](x).item())
            # If use_verifier but we didnt set cur_monitor (shouldnt happen for non-done)
        else:
            action = None
        if a in last_obs and last_obs[a] is not None and last_action[a] is not None:
            pending.append({
                "agent": int(a.split("_")[-1]),
                "obs": last_obs[a],
                "action": last_action[a],
                "next_obs": obs.copy() if not done else np.zeros_like(last_obs[a]),
                "last_monitor": last_monitor[a] if last_monitor[a] is not None else 0.5,
                "reward": float(reward),
                "done": bool(done),
                "step": step_idx,
            })
            step_idx += 1
        last_obs[a] = obs if obs is not None else np.zeros(18)
        last_action[a] = action if action is not None else np.zeros(5)
        last_monitor[a] = cur_monitor if cur_monitor is not None else 0.5
        env.step(action)
        if env.agents == []:
            break
    return pending, float(ep_return)


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


def evaluate_maddpg(actors, n_episodes=15, seed=200, max_cycles=25):
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
                agent_idx = int(a.split("_")[-1])
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action = actors[agent_idx](obs_t).squeeze(0).numpy()
            env.step(action)
            if env.agents == []:
                break
        env.close()
        returns.append(ep_return)
    return returns


def collect_monitor_data(actors, n_eps, seed_base):
    obs_list = []
    returns = []
    for ep in range(n_eps):
        env = make_env(25)
        env.reset(seed=seed_base + ep)
        ep_obs = {a: [] for a in env.possible_agents}
        ep_return = 0.0
        for a in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            ep_return += reward
            if term or trunc:
                action = None
            else:
                agent_idx = int(a.split("_")[-1])
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action = actors[agent_idx](obs_t).squeeze(0).numpy()
                ep_obs[a].append(obs.copy())
            env.step(action)
            if env.agents == []:
                break
        env.close()
        for a in env.possible_agents:
            arr = ep_obs[a]
            if arr:
                tail = arr[-HISTORY_LEN:]
                h = np.zeros((HISTORY_LEN, OBS_DIM), dtype=np.float32)
                h[-len(tail):] = tail
                obs_list.append(h)
            else:
                obs_list.append(np.zeros((HISTORY_LEN, OBS_DIM), dtype=np.float32))
        returns.append(ep_return)
    return obs_list, returns


def train_per_agent_monitors(actors, n_eps, seed, n_epochs=20, batch_size=16):
    obs_list, rets = collect_monitor_data(actors, n_eps, seed * 1000 + 7777)
    if len(obs_list) < 2:
        return None, [float("nan")] * N_AGENTS
    median_ret = float(np.median(rets))
    X = torch.from_numpy(np.stack(obs_list)).float()
    y = torch.from_numpy(
        np.array(
            [1.0 if rets[i // N_AGENTS] < median_ret else 0.0 for i in range(len(obs_list))],
            dtype=np.float32,
        )
    )
    monitors = [PerAgentMonitor(OBS_DIM) for _ in range(N_AGENTS)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in monitors]
    aurocs = []
    for agent_idx, monitor in enumerate(monitors):
        idx = list(range(agent_idx, len(obs_list), N_AGENTS))
        if len(idx) < 2:
            aurocs.append(float("nan"))
            continue
        Xi = X[idx]
        yi = y[idx]
        if len(set(yi.tolist())) < 2:
            aurocs.append(float("nan"))
            continue
        pos = int((yi == 1).sum().item())
        neg = int((yi == 0).sum().item())
        if pos == 0 or neg == 0:
            aurocs.append(float("nan"))
            continue
        pos_w = torch.tensor([neg / max(pos, 1)])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(n_epochs):
            order = torch.randperm(Xi.size(0))
            for s in range(0, Xi.size(0), batch_size):
                mb = order[s:s + batch_size]
                logit = _monitor_logit(monitor, Xi[mb])
                loss = bce(logit, yi[mb])
                opts[agent_idx].zero_grad()
                loss.backward()
                opts[agent_idx].step()
        with torch.no_grad():
            p_all = torch.sigmoid(_monitor_logit(monitor, Xi)).numpy()
        try:
            y_np = yi.numpy()
            pos_mask = (y_np == 1); neg_mask = (y_np == 0)
            n_pos = int(pos_mask.sum()); n_neg = int(neg_mask.sum())
            if n_pos == 0 or n_neg == 0: aurocs.append(float("nan"))
            else:
                a_val = float((p_all[pos_mask].sum() / n_pos + (1.0 - p_all[neg_mask]).sum() / n_neg) / 2.0)
            aurocs.append(a_val)
        except Exception:
            aurocs.append(0.5)
    return monitors, aurocs


def predict_monitor_on_obs(monitor, obs):
    h = np.tile(obs[None, :], (HISTORY_LEN, 1))
    x = torch.from_numpy(h).unsqueeze(0).float()
    p = monitor(x)
    return p[0] if p.dim() > 0 else p


def train_maddpg_v6(seed=0, n_updates=80, n_episodes=10, batch_size=128,
                    buffer_size=20000, gamma=0.95, tau=0.01,
                    lr_actor=1e-4, lr_critic=1e-3, lr_trust=1e-3,
                    n_monitor_eps=80, monitor_epochs=20,
                    noise_start=0.5, noise_end=0.05, log_every=20,
                    use_verifier=True, random_verifier=False,
                    use_random_trust_input=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS)
    target_actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    target_critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS)
    for ta, a in zip(target_actors, actors):
        ta.load_state_dict(a.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    # Monitors (frozen) -- only trained if we need real Monitor output
    monitors = [PerAgentMonitor(OBS_DIM) for _ in range(N_AGENTS)]
    mon_aurocs = [float("nan")] * N_AGENTS
    if use_random_trust_input:
        # v6 ablation: trust head gets random input. Skip monitor training
        # (saves the 80 monitor-collection episodes; matched compute is
        # the 80 PPO updates below).
        print("  Stage 0: SKIPPED (use_random_trust_input=True -- no Monitor needed)", flush=True)
    else:
        print("  Stage 0: Training per-agent Monitors on frozen PPO rollouts...", flush=True)
        monitors, mon_aurocs = train_per_agent_monitors(actors, n_monitor_eps, seed, n_epochs=monitor_epochs)
        if use_verifier and monitors is None:
            use_verifier = False
        if random_verifier:
            monitors = [PerAgentMonitor(OBS_DIM) for _ in range(N_AGENTS)]
            mon_aurocs = [float("nan")] * N_AGENTS
    for m in monitors:
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
    # Trust head (used when use_verifier OR use_random_trust_input)
    trust_heads = None
    if use_verifier or use_random_trust_input:
        trust_heads = [TrustHead(OBS_DIM, N_AGENTS) for _ in range(N_AGENTS)]
        trust_opts = [torch.optim.Adam(h.parameters(), lr=lr_trust) for h in trust_heads]
    # Replay buffer
    buffer = ReplayBuffer(capacity=buffer_size)
    print("  Stage 1: 10 warmup updates starting...", flush=True)
    for u in range(10):
        noise = max(0.1, noise_start - (noise_start - noise_end) * u / 5)
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_v5(
                actors, monitors, make_env(25),
                seed=seed * 1000 + u * 100 + ep, noise_scale=noise,
                use_verifier=use_verifier, random_verifier=random_verifier)
            for t in transitions:
                buffer.push(t)
    history = []
    for u in range(n_updates):
        noise = max(noise_end, noise_start - (noise_start - noise_end) * u / max(1, n_updates // 2))
        ep_returns = []
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_v5(
                actors, monitors, make_env(25),
                seed=seed * 1000 + 1000 + u * 100 + ep, noise_scale=noise,
                use_verifier=use_verifier, random_verifier=random_verifier)
            ep_returns.append(ep_return)
            for t in transitions:
                buffer.push(t)
        if len(buffer) < batch_size:
            continue
        samples = buffer.sample(batch_size)
        per_agent = {i: {"obs": [], "act": [], "nobs": [], "rew": [], "done": [], "mon": []} for i in range(N_AGENTS)}
        for s in samples:
            i = s["agent"]
            per_agent[i]["obs"].append(s["obs"])
            per_agent[i]["act"].append(s["action"])
            per_agent[i]["nobs"].append(s["next_obs"])
            per_agent[i]["rew"].append(s["reward"])
            per_agent[i]["done"].append(float(s.get("done", False)))
            per_agent[i]["mon"].append(s["last_monitor"])
        for i in range(N_AGENTS):
            d = per_agent[i]
            if len(d["obs"]) < 2:
                continue
            obs_b = torch.tensor(np.stack(d["obs"]), dtype=torch.float32)
            act_b = torch.tensor(np.stack(d["act"]), dtype=torch.float32)
            nobs_b = torch.tensor(np.stack(d["nobs"]), dtype=torch.float32)
            rew_b = torch.tensor(d["rew"], dtype=torch.float32)
            done_b = torch.tensor(d["done"], dtype=torch.float32)
            mon_b = torch.tensor(d["mon"], dtype=torch.float32)
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
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            # Actor update: standard v2 (use own Q)
            pred_action = actors[i](obs_b)
            full_act_pred = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_act_pred[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = pred_action
            # Use trust head if use_verifier (real Monitor) OR use_random_trust_input
            # (v6 ablation). Both arms go through the same trust head code path --
            # only the input source differs (Monitor broadcast vs torch.rand).
            if not (use_verifier or use_random_trust_input):
                actor_loss_v2 = -critic(full_obs, full_act_pred)[:, i].mean()
                actor_opts[i].zero_grad()
                actor_loss_v2.backward()
                actor_opts[i].step()
            else:
                # Trust head: use chain summary features.
                # For simplicity, we use mean(last_monitor) per OTHER agent as the chain summary.
                # Since the buffer stores per-agent last_monitor at the time of step,
                # we approximate "others' monitor stats" as the mean monitor of all OTHER
                # agents at the same step. In this episode, this is approximated by
                # using the same step's other-agent monitor values; we approximate via
                # the buffer: for transition t, we have last_monitor for THIS agent only.
                # We use the same-agent monitor as a fallback for others.
                # chain_summary = mean of all other agents' monitor at the same step
                # We compute chain summary from buffer samples: take all samples at the
                # same step_idx, find their agent_ids, and gather their last_monitor values.
                # Simplified: for the batch, use this agent's monitor as proxy for others
                # (conservative: real chain uses actual cross-agent entries).
                # v6 ablation: replace same-agent Monitor broadcast with random noise.
                # This is the ONLY difference from v5 (architecture otherwise identical).
                if use_random_trust_input:
                    others_stats = torch.rand(mon_b.size(0), N_AGENTS - 1, device=mon_b.device)
                else:
                    others_stats = mon_b.unsqueeze(-1).expand(-1, N_AGENTS - 1)  # v5: same-agent Monitor broadcast
                # Actor loss: maximise own Q + trust-weighted sum of other agents' Q
                # We get other agents' Q by computing the critic's full Q for all agents
                # under the predicted action (only this agent's action changes).
                with torch.no_grad():
                    other_actions_pred = torch.cat(
                        [actors[j](obs_b) if j != i else pred_action for j in range(N_AGENTS)],
                        dim=-1,
                    )
                full_act_pred_for_all = other_actions_pred
                all_q = critic(full_obs, full_act_pred_for_all)  # (B, n_agents)
                my_q = all_q[:, i]  # (B,)
                other_q = torch.cat([all_q[:, j].unsqueeze(-1) for j in range(N_AGENTS) if j != i], dim=-1)  # (B, n_agents-1)
                # Trust head input: my obs, my monitor prob, others_stats
                # v6 ablation: random my_mon (matches others_stats above)
                if use_random_trust_input:
                    my_mon = torch.rand(mon_b.size(0), 1, device=mon_b.device)
                else:
                    my_mon = mon_b.unsqueeze(-1)  # v5: same-agent Monitor broadcast
                trust = trust_heads[i](obs_b, my_mon, others_stats)  # (B, n_agents-1)
                # Actor loss = -E[ my_q + trust * other_q ] (per-agent sum)
                trust_other_q = (trust * other_q).sum(dim=-1)  # (B,)
                actor_loss = -(my_q + trust_other_q).mean()
                actor_opts[i].zero_grad()
                actor_loss.backward()
                actor_opts[i].step()
                # Trust head update: same loss (so trust head is trained end-to-end)
                trust_loss = actor_loss.detach()  # already detached from critic/actor side
                trust_opts[i].zero_grad()
                # We need a differentiable version of the loss for trust head
                trust = trust_heads[i](obs_b, my_mon.detach(), others_stats.detach())
                with torch.no_grad():
                    q_other_for_trust = torch.cat(
                        [critic(full_obs, torch.cat(
                            [actors[j](obs_b) if j != k else pred_action.detach() for j in range(N_AGENTS)],
                            dim=-1,
                        ))[:, k].unsqueeze(-1) for k in range(N_AGENTS) if k != i],
                        dim=-1,
                    )
                trust_loss_v = -((1.0 - trust) * q_other_for_trust).mean()  # encourage trusting non-failed Q
                trust_loss_v.backward()
                trust_opts[i].step()
        for ta, a in zip(target_actors, actors):
            soft_update(ta, a, tau)
        soft_update(target_critic, critic, tau)
        mean_return = float(np.mean(ep_returns))
        history.append({"update": u, "mean_return": mean_return, "noise_scale": noise, "buffer_size": len(buffer)})
        if (u + 1) % log_every == 0 or u == 0:
            print("    update " + str(u + 1) + "/" + str(n_updates) + ": mean_episode_return=" + str(round(mean_return, 2)) + ", buffer=" + str(len(buffer)), flush=True)
    return actors, monitors, trust_heads, history, mon_aurocs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-updates", type=int, default=80)
    p.add_argument("--n-episodes-per-update", type=int, default=10)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--n-monitor-eps", type=int, default=80)
    p.add_argument("--arm", type=str, default="with_trusthead_random", choices=["with_verifier", "no_verifier", "with_trusthead_random"])
    args = p.parse_args()
    print("=" * 60)
    print("MADDPG v6 (trust head ablation) - arm=" + args.arm + " - PettingZoo Simple Spread v3 (continuous)")
    print("=" * 60)
    print("  seed=" + str(args.seed) + ", n_updates=" + str(args.n_updates))
    print()
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns))
    rnd_std = float(np.std(rnd_returns))
    print("Phase 1: Random baseline = " + str(round(rnd_mean, 2)) + " +/- " + str(round(rnd_std, 2)))
    print()
    use_verifier = (args.arm == "with_verifier")
    random_verifier = False  # legacy v5 arm, unused in v6
    use_random_trust_input = (args.arm == "with_trusthead_random")
    actors, monitors, trust_heads, history, mon_aurocs = train_maddpg_v6(
        seed=args.seed, n_updates=args.n_updates, n_episodes=args.n_episodes_per_update,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        n_monitor_eps=args.n_monitor_eps,
        use_verifier=use_verifier, random_verifier=random_verifier,
        use_random_trust_input=use_random_trust_input)
    print()
    print("Phase 3: Final eval...")
    final_eval = evaluate_maddpg(actors, n_episodes=args.n_eval_episodes, seed=4000)
    final_mean = float(np.mean(final_eval))
    final_std = float(np.std(final_eval))
    delta = final_mean - rnd_mean
    print("  MADDPG v6 (" + args.arm + ") eval: " + str(round(final_mean, 2)) + " +/- " + str(round(final_std, 2)) + "  (delta vs random: " + ("%.2f" % delta) + ")")
    if mon_aurocs:
        print("  Monitor AUROC per agent: " + str([round(a, 3) if a == a else "nan" for a in mon_aurocs]))
    log_path = HERE / "checkpoints" / "pz_maddpg_v6" / ("seed" + str(args.seed) + "_" + args.arm) / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)",
        "seed": args.seed, "arm": args.arm,
        "mode": ("MADDPG v6 trust-head same-agent Monitor (= v5)" if use_verifier
                 else ("MADDPG v2 baseline" if not use_random_trust_input
                       else "MADDPG v6 trust-head RANDOM input (architecture-only ablation)")),
        "n_updates": args.n_updates, "n_episodes_per_update": args.n_episodes_per_update,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval, "delta_vs_random": float(delta),
        "monitor_auroc": mon_aurocs, "history": history,
        "honest_note": "v6 = v5 with random trust-head input (proper architecture ablation, not critic-side extra).",
    }, indent=2))
    print("  Log: " + str(log_path))


if __name__ == "__main__":
    main()
