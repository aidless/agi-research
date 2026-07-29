"""pz_maddpg_v4.py - MADDPG v4: simplified TarMAC inter-agent comms in critic.

Y2 follow-up to v3 (aux loss = NEGATIVE). Architecture:
- per-agent MLP message encoder: obs -> 32-dim message
- critic input: (all_obs, all_actions, all_messages) -> Q
- actor unchanged: still obs -> action
- three arms: with_comms, no_comms, random_comms
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
MSG_DIM = 32


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid(),
        )

    def forward(self, obs):
        return self.net(obs)


class MessageEncoder(nn.Module):
    def __init__(self, obs_dim, msg_dim=MSG_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, msg_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class CommsCritic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, n_agents, msg_dim, hidden=128):
        super().__init__()
        in_dim = total_obs_dim + total_action_dim + n_agents * msg_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )

    def forward(self, all_obs, all_actions, all_msgs):
        x = torch.cat([all_obs, all_actions, all_msgs], dim=-1)
        return self.net(x)


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


def collect_episode_with_msgs(actors, msg_encoders, env, seed, noise_scale=0.1, use_comms=True, random_msgs=False):
    env.reset(seed=seed)
    last_obs = {a: None for a in env.possible_agents}
    last_action = {a: None for a in env.possible_agents}
    last_msg = {a: None for a in env.possible_agents}
    pending = []
    ep_return = 0.0
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_return += reward
        done = term or trunc
        cur_msg = None
        if not done:
            agent_idx = int(a.split("_")[-1])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
                noise = np.random.randn(*action_mean.shape) * noise_scale
                action = np.clip(action_mean + noise, 0.0, 1.0)
                if use_comms:
                    cur_msg = msg_encoders[agent_idx](obs_t).squeeze(0).numpy()
                elif random_msgs:
                    cur_msg = (np.random.randn(MSG_DIM).astype(np.float32) * 0.1)
                else:
                    cur_msg = np.zeros(MSG_DIM, dtype=np.float32)
        else:
            action = None
        if a in last_obs and last_obs[a] is not None and last_action[a] is not None:
            pending.append({
                "agent": int(a.split("_")[-1]),
                "obs": last_obs[a],
                "action": last_action[a],
                "next_obs": obs.copy() if not done else np.zeros_like(last_obs[a]),
                "msg": last_msg[a],
                "reward": float(reward),
                "done": bool(done),
            })
        last_obs[a] = obs if obs is not None else np.zeros(18)
        last_action[a] = action if action is not None else np.zeros(5)
        last_msg[a] = cur_msg if cur_msg is not None else np.zeros(MSG_DIM, dtype=np.float32)
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


def train_maddpg_v4(seed=0, n_updates=80, n_episodes=10, batch_size=128,
                    buffer_size=20000, gamma=0.95, tau=0.01,
                    lr_actor=1e-4, lr_critic=1e-3, lr_msg=1e-3,
                    noise_start=0.5, noise_end=0.05, log_every=20,
                    use_comms=True, random_msgs=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    msg_encoders = [MessageEncoder(OBS_DIM) for _ in range(N_AGENTS)]
    critic = CommsCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS, MSG_DIM)
    target_actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    target_msg_encoders = [MessageEncoder(OBS_DIM) for _ in range(N_AGENTS)]
    target_critic = CommsCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS, MSG_DIM)
    for ta, a in zip(target_actors, actors):
        ta.load_state_dict(a.state_dict())
    for tm, m in zip(target_msg_encoders, msg_encoders):
        tm.load_state_dict(m.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in actors]
    msg_opts = [torch.optim.Adam(m.parameters(), lr=lr_msg) for m in msg_encoders]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    buffer = ReplayBuffer(capacity=buffer_size)
    print("  Stage 1: 10 warmup updates starting...", flush=True)
    for u in range(10):
        noise = max(0.1, noise_start - (noise_start - noise_end) * u / 5)
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_with_msgs(
                actors, msg_encoders, make_env(25),
                seed=seed * 1000 + u * 100 + ep, noise_scale=noise,
                use_comms=use_comms, random_msgs=random_msgs)
            for t in transitions:
                buffer.push(t)
    history = []
    for u in range(n_updates):
        noise = max(noise_end, noise_start - (noise_start - noise_end) * u / max(1, n_updates // 2))
        ep_returns = []
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_with_msgs(
                actors, msg_encoders, make_env(25),
                seed=seed * 1000 + 1000 + u * 100 + ep, noise_scale=noise,
                use_comms=use_comms, random_msgs=random_msgs)
            ep_returns.append(ep_return)
            for t in transitions:
                buffer.push(t)
        if len(buffer) < batch_size:
            continue
        samples = buffer.sample(batch_size)
        per_agent = {i: {"obs": [], "act": [], "nobs": [], "msg": [], "rew": [], "done": []} for i in range(N_AGENTS)}
        for s in samples:
            i = s["agent"]
            per_agent[i]["obs"].append(s["obs"])
            per_agent[i]["act"].append(s["action"])
            per_agent[i]["nobs"].append(s["next_obs"])
            per_agent[i]["msg"].append(s["msg"])
            per_agent[i]["rew"].append(s["reward"])
            per_agent[i]["done"].append(float(s.get("done", False)))
        for i in range(N_AGENTS):
            d = per_agent[i]
            if len(d["obs"]) < 2:
                continue
            obs_b = torch.tensor(np.stack(d["obs"]), dtype=torch.float32)
            act_b = torch.tensor(np.stack(d["act"]), dtype=torch.float32)
            nobs_b = torch.tensor(np.stack(d["nobs"]), dtype=torch.float32)
            msg_b = torch.tensor(np.stack(d["msg"]), dtype=torch.float32)
            rew_b = torch.tensor(d["rew"], dtype=torch.float32)
            done_b = torch.tensor(d["done"], dtype=torch.float32)
            Bi = obs_b.shape[0]
            full_obs = torch.zeros(Bi, N_AGENTS * OBS_DIM)
            full_act = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_msg = torch.zeros(Bi, N_AGENTS * MSG_DIM)
            full_obs[:, i * OBS_DIM:(i + 1) * OBS_DIM] = obs_b
            full_act[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = act_b
            full_msg[:, i * MSG_DIM:(i + 1) * MSG_DIM] = msg_b
            with torch.no_grad():
                a_next = target_actors[i](nobs_b)
                m_next = target_msg_encoders[i](nobs_b)
                full_nobs = torch.zeros(Bi, N_AGENTS * OBS_DIM)
                full_nact = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
                full_nmsg = torch.zeros(Bi, N_AGENTS * MSG_DIM)
                full_nobs[:, i * OBS_DIM:(i + 1) * OBS_DIM] = nobs_b
                full_nact[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = a_next
                full_nmsg[:, i * MSG_DIM:(i + 1) * MSG_DIM] = m_next
                q_next = target_critic(full_nobs, full_nact, full_nmsg)[:, i]
                target = rew_b + gamma * (1.0 - done_b) * q_next
            q_pred = critic(full_obs, full_act, full_msg)[:, i]
            critic_loss = F.mse_loss(q_pred, target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            pred_action = actors[i](obs_b)
            full_act_pred = torch.zeros(Bi, N_AGENTS * ACTION_DIM)
            full_act_pred[:, i * ACTION_DIM:(i + 1) * ACTION_DIM] = pred_action
            actor_loss = -critic(full_obs, full_act_pred, full_msg)[:, i].mean()
            actor_opts[i].zero_grad()
            actor_loss.backward()
            actor_opts[i].step()
            if use_comms and not random_msgs:
                msg_pred = msg_encoders[i](obs_b)
                full_msg_pred = torch.zeros(Bi, N_AGENTS * MSG_DIM)
                full_msg_pred[:, i * MSG_DIM:(i + 1) * MSG_DIM] = msg_pred
                msg_loss = -critic(full_obs, full_act, full_msg_pred)[:, i].mean()
                msg_opts[i].zero_grad()
                msg_loss.backward()
                msg_opts[i].step()
        for ta, a in zip(target_actors, actors):
            soft_update(ta, a, tau)
        for tm, m in zip(target_msg_encoders, msg_encoders):
            soft_update(tm, m, tau)
        soft_update(target_critic, critic, tau)
        mean_return = float(np.mean(ep_returns))
        history.append({"update": u, "mean_return": mean_return, "noise_scale": noise, "buffer_size": len(buffer)})
        if (u + 1) % log_every == 0 or u == 0:
            print("    update " + str(u + 1) + "/" + str(n_updates) + ": mean_episode_return=" + str(round(mean_return, 2)) + ", buffer=" + str(len(buffer)), flush=True)
    return actors, msg_encoders, critic, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-updates", type=int, default=80)
    p.add_argument("--n-episodes-per-update", type=int, default=10)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--arm", type=str, default="with_comms", choices=["with_comms", "no_comms", "random_comms"])
    args = p.parse_args()
    print("=" * 60)
    print("MADDPG v4 - arm=" + args.arm + " - PettingZoo Simple Spread v3 (continuous)")
    print("=" * 60)
    print("  seed=" + str(args.seed) + ", n_updates=" + str(args.n_updates))
    print()
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns))
    rnd_std = float(np.std(rnd_returns))
    print("Phase 1: Random baseline = " + str(round(rnd_mean, 2)) + " +/- " + str(round(rnd_std, 2)))
    print()
    use_comms = (args.arm == "with_comms")
    random_msgs = (args.arm == "random_comms")
    actors, msg_encoders, critic, history = train_maddpg_v4(
        seed=args.seed, n_updates=args.n_updates, n_episodes=args.n_episodes_per_update,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        use_comms=use_comms, random_msgs=random_msgs)
    print()
    print("Phase 3: Final eval...")
    final_eval = evaluate_maddpg(actors, n_episodes=args.n_eval_episodes, seed=4000)
    final_mean = float(np.mean(final_eval))
    final_std = float(np.std(final_eval))
    delta = final_mean - rnd_mean
    print("  MADDPG v4 (" + args.arm + ") eval: " + str(round(final_mean, 2)) + " +/- " + str(round(final_std, 2)) + "  (delta vs random: " + ("%.2f" % delta) + ")")
    log_path = HERE / "checkpoints" / "pz_maddpg_v4" / ("seed" + str(args.seed) + "_" + args.arm) / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)",
        "seed": args.seed,
        "arm": args.arm,
        "mode": "MADDPG v4 (comms in critic)" if use_comms else ("MADDPG v2 baseline" if not random_msgs else "MADDPG v4 (random comms)"),
        "n_updates": args.n_updates,
        "n_episodes_per_update": args.n_episodes_per_update,
        "random_mean": rnd_mean,
        "random_std": rnd_std,
        "final_eval_mean": final_mean,
        "final_eval_std": final_std,
        "per_episode_final_eval": final_eval,
        "delta_vs_random": float(delta),
        "history": history,
        "honest_note": "Y2 inter-agent comms follow-up to v3.",
    }, indent=2))
    print("  Log: " + str(log_path))


if __name__ == "__main__":
    main()
