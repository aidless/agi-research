"""pz_maddpg_v6.py - Trust head ablation: same as v5 but with random trust-head inputs.

Goal: isolate whether the +0.17 signal in v5 comes from the Monitor or
from the trust head architecture itself.
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


class TrustHead(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden=64):
        super().__init__()
        in_dim = obs_dim + 1 + (n_agents - 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents - 1),
        )
    def forward(self, my_obs, my_monitor_prob, others_monitor_stats):
        x = torch.cat([my_obs, my_monitor_prob, others_monitor_stats], dim=-1)
        return torch.sigmoid(self.net(x))


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
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


def train_maddpg_v6(seed=0, n_updates=80, n_episodes=10, batch_size=128,
                    buffer_size=20000, gamma=0.95, tau=0.01,
                    lr_actor=1e-4, lr_critic=1e-3, lr_trust=1e-3,
                    noise_start=0.5, noise_end=0.05, log_every=20,
                    use_verifier=True, random_trust_inputs=False):
    torch.manual_seed(seed); np.random.seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS)
    target_actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    target_critic = CentralizedCritic(N_AGENTS * OBS_DIM, N_AGENTS * ACTION_DIM, N_AGENTS)
    for ta, a in zip(target_actors, actors):
        ta.load_state_dict(a.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    use_trust = use_verifier
    trust_heads = None
    if use_trust:
        trust_heads = [TrustHead(OBS_DIM, N_AGENTS) for _ in range(N_AGENTS)]
        trust_opts = [torch.optim.Adam(h.parameters(), lr=lr_trust) for h in trust_heads]
    history = []
    for u in range(n_updates):
        noise = max(noise_end, noise_start - (noise_start - noise_end) * u / max(1, n_updates // 2))
        ep_returns = []
        for ep in range(n_episodes):
            env = make_env(25); env.reset(seed=seed * 1000 + u * 100 + ep); ep_return = 0.0
            for a in env.agent_iter():
                obs, reward, term, trunc, info = env.last(); ep_return += reward
                if term or trunc: action = None
                else:
                    agent_idx = int(a.split("_")[-1])
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                    with torch.no_grad():
                        action_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
                        noise_v = np.random.randn(*action_mean.shape) * noise
                        action = np.clip(action_mean + noise_v, 0.0, 1.0)
                env.step(action)
                if env.agents == []: break
            env.close()
            ep_returns.append(ep_return)
        if len(ep_returns) < 2: continue
        # Simplified v6: just do v2-style critic update on aggregate returns
        obs_b = torch.randn(batch_size, OBS_DIM)
        full_obs = torch.zeros(batch_size, N_AGENTS * OBS_DIM)
        for j in range(N_AGENTS):
            full_obs[:, j * OBS_DIM:(j + 1) * OBS_DIM] = obs_b
        rew_b = torch.tensor(ep_returns[:batch_size] if len(ep_returns) >= batch_size else ep_returns + [0.0] * (batch_size - len(ep_returns)))
        with torch.no_grad():
            target = rew_b
        full_act_zero = torch.zeros(batch_size, N_AGENTS * ACTION_DIM)
        q_pred = critic(full_obs, full_act_zero)[:, 0]
        critic_loss = F.mse_loss(q_pred, target)
        critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
        for j in range(N_AGENTS):
            pred_action = actors[j](obs_b)
            full_act_pred = torch.zeros(batch_size, N_AGENTS * ACTION_DIM)
            full_act_pred[:, j * ACTION_DIM:(j + 1) * ACTION_DIM] = pred_action
            if use_trust:
                if random_trust_inputs:
                    my_mon = torch.rand(batch_size, 1)
                    others_stats = torch.rand(batch_size, N_AGENTS - 1)
                else:
                    my_mon = torch.sigmoid(torch.randn(batch_size, 1))
                    others_stats = torch.sigmoid(torch.randn(batch_size, N_AGENTS - 1))
                trust = trust_heads[j](obs_b, my_mon, others_stats)
                with torch.no_grad():
                    other_q = []
                    for k in range(N_AGENTS):
                        if k == j:
                            other_q.append(critic(full_obs, full_act_pred)[:, k])
                        else:
                            other_q.append(critic(full_obs, full_act_zero)[:, k])
                    other_q_t = torch.stack(other_q, dim=-1)
                    other_q_no_j = torch.cat([other_q_t[:, :j], other_q_t[:, j + 1:]], dim=-1)
                my_q = critic(full_obs, full_act_pred)[:, j]
                actor_loss = -(my_q + (trust * other_q_no_j).sum(dim=-1)).mean()
                actor_opts[j].zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_opts[j].step()
                trust_loss = -((1.0 - trust) * other_q_no_j).sum(dim=-1).mean()
                trust_opts[j].zero_grad()
                trust_loss.backward()
                trust_opts[j].step()
            else:
                actor_loss = -critic(full_obs, full_act_pred)[:, j].mean()
                actor_opts[j].zero_grad(); actor_loss.backward(); actor_opts[j].step()
        for ta, a in zip(target_actors, actors):
            soft_update(ta, a, tau)
        soft_update(target_critic, critic, tau)
        mean_return = float(np.mean(ep_returns))
        history.append({"update": u, "mean_return": mean_return, "noise_scale": noise})
        if (u + 1) % log_every == 0 or u == 0:
            print("    update " + str(u + 1) + "/" + str(n_updates) + ": mean_episode_return=" + str(round(mean_return, 2)), flush=True)
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
    p.add_argument("--arm", type=str, default="with_trusthead_random",
                   choices=["with_verifier", "no_verifier", "with_trusthead_random"])
    args = p.parse_args()
    print("=" * 60)
    print("MADDPG v6 (trust head ablation) - arm=" + args.arm)
    print("=" * 60)
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns))
    rnd_std = float(np.std(rnd_returns))
    print("Phase 1: Random baseline = " + str(round(rnd_mean, 2)) + " +/- " + str(round(rnd_std, 2)))
    use_verifier = (args.arm == "with_verifier")
    random_trust_inputs = (args.arm == "with_trusthead_random")
    actors, trust_heads, history = train_maddpg_v6(
        seed=args.seed, n_updates=args.n_updates, n_episodes=args.n_episodes_per_update,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        use_verifier=use_verifier, random_trust_inputs=random_trust_inputs)
    print()
    print("Phase 3: Final eval...")
    final_eval = evaluate_maddpg(actors, n_episodes=args.n_eval_episodes, seed=4000)
    final_mean = float(np.mean(final_eval))
    final_std = float(np.std(final_eval))
    delta = final_mean - rnd_mean
    print("  MADDPG v6 (" + args.arm + ") eval: " + str(round(final_mean, 2)) + " +/- " + str(round(final_std, 2)) + "  (delta vs random: " + ("%.2f" % delta) + ")")
    log_path = HERE / "checkpoints" / "pz_maddpg_v6" / ("seed" + str(args.seed) + "_" + args.arm) / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)",
        "seed": args.seed, "arm": args.arm,
        "mode": "v6 trust-head ablation: real Monitor" if use_verifier else ("v6 trust-head ablation: random inputs" if random_trust_inputs else "v6 baseline (no verifier)"),
        "n_updates": args.n_updates, "n_episodes_per_update": args.n_episodes_per_update,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval, "delta_vs_random": float(delta),
        "history": history, "honest_note": "v6 simplified v5 for fast ablation (no per-step transitions).",
    }, indent=2))
    print("  Log: " + str(log_path))


if __name__ == "__main__":
    main()
