"""pz_maddpg_v8.py - MADDPG v8: DLR cross-agent predicates as trust signal.

3-arm ablation (no v5 replication; v5 results exist separately):
- no_verifier: v2 baseline (no DLR, no trust head)
- dlr_only: DLR predicates in critic (no trust head)
- v8: DLR predicates + trust head at actor
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
DLR_PRED_DIM = 24
HISTORY_LEN = 20


def extract_dlr_preds(obs):
    """Extract 24 DLR cross-agent predicates from a single agent's obs."""
    preds = np.zeros(DLR_PRED_DIM, dtype=np.float32)
    own_pos = obs[2:4]
    landmark_abs = obs[4:10].reshape(3, 2) + own_pos
    others_abs = obs[10:14].reshape(2, 2) + own_pos
    dists = np.linalg.norm(landmark_abs - own_pos, axis=1)
    other_dists_0 = np.linalg.norm(others_abs[0:1] - landmark_abs, axis=1)
    other_dists_1 = np.linalg.norm(others_abs[1:2] - landmark_abs, axis=1)
    all_dists = np.stack([dists, other_dists_0, other_dists_1])
    closest_agent = np.argmin(all_dists, axis=0)
    for j in range(3):
        if closest_agent[j] == 0:
            preds[j] = 1.0
    for j in range(3):
        if dists[j] < 0.5:
            preds[9 + j] = 1.0
    for j in range(3):
        if np.min(all_dists[:, j]) < 0.5:
            preds[18 + j] = 1.0
    for k in range(2):
        if np.linalg.norm(others_abs[k]) < 0.3:
            preds[21 + k] = 1.0
    return preds


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
    """Critic input: obs + actions + (optional) DLR predicates."""
    def __init__(self, total_obs_dim, total_action_dim, n_agents, total_dlr_dim=0, hidden=128):
        super().__init__()
        in_dim = total_obs_dim + total_action_dim + total_dlr_dim
        self.has_dlr = total_dlr_dim > 0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )
    def forward(self, all_obs, all_actions, all_dlr=None):
        if self.has_dlr and all_dlr is not None:
            x = torch.cat([all_obs, all_actions, all_dlr], dim=-1)
        else:
            x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


class TrustHead(nn.Module):
    """Input: (my_obs, my_dlr) -> per-other-agent trust weights.

    Note: unlike v5 which stored 3 separate Monitor signals, v8 stores
    only this agent's DLR predicates (24-dim). The trust head input
    is 18 + 24 = 42 features, producing n_agents-1 = 2 trust weights.
    """
    def __init__(self, obs_dim, dlr_dim, n_agents, hidden=64):
        super().__init__()
        in_dim = obs_dim + dlr_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents - 1),
        )
    def forward(self, my_obs, my_dlr):
        x = torch.cat([my_obs, my_dlr], dim=-1)
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


def collect_episode_v8(actors, trust_heads, env, seed, noise_scale=0.1,
                        use_dlr_trust=True, use_dlr_critic=False):
    """Collect one episode.

    Transition format:
    - agent: int (which agent did the transition)
    - obs: (18,) this agent's obs
    - action: (5,) this agent's action
    - dlr_all: (3*24,) = (72,) ALL agents' dlr at this timestep
    - reward, done
    """
    env.reset(seed=seed)
    last_obs = {a: None for a in env.possible_agents}
    last_action = {a: None for a in env.possible_agents}
    last_dlr_all = {a: None for a in env.possible_agents}
    pending = []
    ep_return = 0.0
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_return += reward
        done = term or trunc
        cur_dlr_all = None
        if not done:
            agent_idx = int(a.split("_")[-1])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
                noise = np.random.randn(*action_mean.shape) * noise_scale
                action = np.clip(action_mean + noise, 0.0, 1.0)
            # Compute DLR for all 3 agents at this timestep from their obs
            # We have obs only for the current agent. For others, we approximate.
            # Better: at training time, the trust head sees (my_obs, my_dlr).
            # The cross-agent info is in the GLOBAL DLR structure but we don't have it
            # from a single agent's obs alone.
            # Simplest: store this agent's dlr (24-dim) and use that.
            cur_dlr = extract_dlr_preds(obs)
            cur_dlr_all = cur_dlr  # (24,) only this agent's
        else:
            action = None
        if a in last_obs and last_obs[a] is not None and last_action[a] is not None:
            pending.append({
                "agent": int(a.split("_")[-1]),
                "obs": last_obs[a],
                "action": last_action[a],
                "next_obs": obs.copy() if not done else np.zeros_like(last_obs[a]),
                "dlr": cur_dlr_all if cur_dlr_all is not None else np.zeros(DLR_PRED_DIM, dtype=np.float32),
                "reward": float(reward),
                "done": bool(done),
            })
        last_obs[a] = obs if obs is not None else np.zeros(18)
        last_action[a] = action if action is not None else np.zeros(5)
        last_dlr_all[a] = cur_dlr_all if cur_dlr_all is not None else np.zeros(DLR_PRED_DIM, dtype=np.float32)
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


def train_maddpg_v8(seed=0, n_updates=80, n_episodes=10, batch_size=128,
                    buffer_size=20000, gamma=0.95, tau=0.01,
                    lr_actor=1e-4, lr_critic=1e-3, lr_trust=1e-3,
                    noise_start=0.5, noise_end=0.05, log_every=20,
                    use_dlr_trust=True, use_dlr_critic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    total_dlr_dim = N_AGENTS * DLR_PRED_DIM if use_dlr_critic else 0
    critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS, total_dlr_dim)
    target_actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    target_critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS, total_dlr_dim)
    for ta, a in zip(target_actors, actors):
        ta.load_state_dict(a.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    trust_heads = None
    trust_opts = None
    if use_dlr_trust:
        trust_heads = [TrustHead(OBS_DIM, DLR_PRED_DIM, N_AGENTS) for _ in range(N_AGENTS)]
        trust_opts = [torch.optim.Adam(h.parameters(), lr=lr_trust) for h in trust_heads]
    buffer = ReplayBuffer(capacity=buffer_size)
    history = []
    for u in range(n_updates):
        noise = max(noise_end, noise_start - (noise_start - noise_end) * u / max(1, n_updates // 2))
        ep_returns = []
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_v8(
                actors, trust_heads, make_env(25),
                seed=seed * 1000 + u * 100 + ep, noise_scale=noise,
                use_dlr_trust=use_dlr_trust, use_dlr_critic=use_dlr_critic)
            ep_returns.append(ep_return)
            for t in transitions:
                buffer.push(t)
        if len(buffer) < batch_size:
            continue
        samples = buffer.sample(batch_size)
        per_agent = {i: {"obs": [], "act": [], "rew": [], "done": [], "dlr": []} for i in range(N_AGENTS)}
        for s in samples:
            i = s["agent"]
            per_agent[i]["obs"].append(s["obs"])
            per_agent[i]["act"].append(s["action"])
            per_agent[i]["rew"].append(s["reward"])
            per_agent[i]["done"].append(float(s.get("done", False)))
            per_agent[i]["dlr"].append(s["dlr"])
        for i in range(N_AGENTS):
            d = per_agent[i]
            if len(d["obs"]) < 2:
                continue
            obs_b = torch.tensor(np.stack(d["obs"]), dtype=torch.float32)
            act_b = torch.tensor(np.stack(d["act"]), dtype=torch.float32)
            dlr_b = torch.tensor(np.stack(d["dlr"]), dtype=torch.float32)
            rew_b = torch.tensor(d["rew"], dtype=torch.float32)
            done_b = torch.tensor(d["done"], dtype=torch.float32)
            Bi = obs_b.shape[0]
            full_obs = torch.zeros(Bi, N_AGENTS * OBS_DIM)
            full_act = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_dlr = torch.zeros(Bi, N_AGENTS * DLR_PRED_DIM)
            full_obs[:, i * OBS_DIM:(i + 1) * OBS_DIM] = obs_b
            full_act[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = act_b
            if use_dlr_critic:
                full_dlr[:, i * DLR_PRED_DIM:(i + 1) * DLR_PRED_DIM] = dlr_b
            with torch.no_grad():
                target = rew_b
            q_pred = critic(full_obs, full_act, full_dlr if use_dlr_critic else None)[:, i]
            critic_loss = F.mse_loss(q_pred, target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            pred_action = actors[i](obs_b)
            full_act_pred = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_act_pred[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = pred_action
            if use_dlr_trust:
                # dlr_b is (Bi, DLR_PRED_DIM) - this agent's DLR preds only.
                # Trust head input: (my_obs 18, my_dlr 24) = 42 features.
                # This is a STRONGER test than v5 (Monitor + trust):
                # we use DLR cross-agent predicates (which encode relationships
                # between agents) instead of per-agent failure probability.
                my_dlr = dlr_b
                # No others_dlr available (we only have this agent's obs).
                # The trust head input is just (my_obs, my_dlr).
                with torch.no_grad():
                    other_q = []
                    for k in range(N_AGENTS):
                        if k == i:
                            other_q.append(critic(full_obs, full_act_pred, full_dlr if use_dlr_critic else None)[:, k])
                        else:
                            other_q.append(critic(full_obs, full_act, full_dlr if use_dlr_critic else None)[:, k])
                    other_q_t = torch.stack(other_q, dim=-1)
                    other_q_no_i = torch.cat([other_q_t[:, :i], other_q_t[:, i+1:]], dim=-1)
                my_q = critic(full_obs, full_act_pred, full_dlr if use_dlr_critic else None)[:, i]
                trust = trust_heads[i](obs_b, my_dlr)
                actor_loss = -(my_q + (trust * other_q_no_i).sum(dim=-1)).mean()
                actor_opts[i].zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_opts[i].step()
                trust_loss = -((1.0 - trust) * other_q_no_i).sum(dim=-1).mean()
                trust_opts[i].zero_grad()
                trust_loss.backward()
                trust_opts[i].step()
            else:
                actor_loss = -critic(full_obs, full_act_pred, full_dlr if use_dlr_critic else None)[:, i].mean()
                actor_opts[i].zero_grad()
                actor_loss.backward()
                actor_opts[i].step()
        for ta, a in zip(target_actors, actors):
            soft_update(ta, a, tau)
        soft_update(target_critic, critic, tau)
        mean_return = float(np.mean(ep_returns))
        history.append({"update": u, "mean_return": mean_return, "noise_scale": noise})
        if (u + 1) % log_every == 0 or u == 0:
            print("    update " + str(u + 1) + "/" + str(n_updates) + ": mean_episode_return=" + str(round(mean_return, 2)) + ", buffer=" + str(len(buffer)), flush=True)
    return actors, trust_heads, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-updates", type=int, default=80)
    p.add_argument("--n-episodes-per-update", type=int, default=10)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--arm", type=str, default="v8",
                   choices=["v8", "no_verifier", "dlr_only"])
    args = p.parse_args()
    print("=" * 60)
    print("MADDPG v8 - arm=" + args.arm)
    print("=" * 60)
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns))
    rnd_std = float(np.std(rnd_returns))
    print("Phase 1: Random baseline = " + str(round(rnd_mean, 2)) + " +/- " + str(round(rnd_std, 2)))
    use_dlr_trust = (args.arm == "v8")
    use_dlr_critic = (args.arm in ("v8", "dlr_only"))
    actors, trust_heads, history = train_maddpg_v8(
        seed=args.seed, n_updates=args.n_updates, n_episodes=args.n_episodes_per_update,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        use_dlr_trust=use_dlr_trust, use_dlr_critic=use_dlr_critic)
    print()
    print("Phase 3: Final eval...")
    final_eval = evaluate_maddpg(actors, n_episodes=args.n_eval_episodes, seed=4000)
    final_mean = float(np.mean(final_eval))
    final_std = float(np.std(final_eval))
    delta = final_mean - rnd_mean
    print("  MADDPG v8 (" + args.arm + ") eval: " + str(round(final_mean, 2)) + " +/- " + str(round(final_std, 2)) + "  (delta vs random: " + ("%.2f" % delta) + ")")
    log_path = HERE / "checkpoints" / "pz_maddpg_v8" / ("seed" + str(args.seed) + "_" + args.arm) / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)",
        "seed": args.seed, "arm": args.arm,
        "mode": "MADDPG v8 (DLR cross-agent predicates + trust head)" if args.arm == "v8" else
                ("DLR-only (DLR in critic, no trust)" if args.arm == "dlr_only" else
                "MADDPG v2 baseline"),
        "n_updates": args.n_updates,
        "n_episodes_per_update": args.n_episodes_per_update,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval, "delta_vs_random": float(delta),
        "history": history,
        "honest_note": "v8: DLR cross-agent predicates as alternative to Monitor signal.",
    }, indent=2))
    print("  Log: " + str(log_path))


if __name__ == "__main__":
    main()
