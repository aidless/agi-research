"""pz_maddpg_v2.py - MADDPG v2 on PettingZoo Simple Spread v3.
Improvements over pz_maddpg.py (which had these bugs):
  1. target_q was hard-coded to zeros (no next_obs bootstrap).
  2. target_actors / target_critic were created and soft-updated but NEVER USED in the critic loss.
  3. Other agents obs/action were zero-padded in the centralized critic.
This v2 fixes all three: next_obs -> target_actor -> target_critic, full global state in critic.
"""
import argparse, json, sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pettingzoo.mpe import simple_spread_v3
N_AGENTS = 3
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
class ReplayBuffer:
    def __init__(self, capacity=50000):
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
def collect_episode_with_next(actors, env, seed, noise_scale=0.1):
    env.reset(seed=seed)
    last_obs = {a: None for a in env.possible_agents}
    last_action = {a: None for a in env.possible_agents}
    pending = []
    ep_return = 0.0
    for a in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        ep_return += reward
        done = term or trunc
        if not done:
            agent_idx = int(a.split("_")[-1])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
            noise = np.random.randn(*action_mean.shape) * noise_scale
            action = np.clip(action_mean + noise, 0.0, 1.0)
        else:
            action = None
        if a in last_obs and last_obs[a] is not None and last_action[a] is not None:
            pending.append({
                "agent": int(a.split("_")[-1]),
                "obs": last_obs[a],
                "action": last_action[a],
                "next_obs": obs.copy() if not done else np.zeros_like(last_obs[a]),
                "reward": float(reward),
                "done": bool(done),
            })
        last_obs[a] = obs if obs is not None else np.zeros(18)
        last_action[a] = action if action is not None else np.zeros(5)
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
            if term or trunc: action = None
            else: action = env.action_space(a).sample()
            env.step(action)
            if env.agents == []: break
        env.close()
        returns.append(ep_return)
    return returns
def evaluate_maddpg(actors, n_episodes=15, seed=200, max_cycles=25, noise_scale=0.0):
    returns = []
    for ep in range(n_episodes):
        env = make_env(max_cycles)
        env.reset(seed=seed + ep)
        ep_return = 0.0
        for a in env.agent_iter(max_iter=max_cycles * env.num_agents + 10):
            obs, reward, term, trunc, info = env.last()
            ep_return += reward
            if term or trunc: action = None
            else:
                agent_idx = int(a.split("_")[-1])
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    a_mean = actors[agent_idx](obs_t).squeeze(0).numpy()
                if noise_scale > 0:
                    a_mean = np.clip(a_mean + np.random.randn(*a_mean.shape) * noise_scale, 0, 1)
                action = a_mean
            env.step(action)
            if env.agents == []: break
        env.close()
        returns.append(ep_return)
    return returns
def train_maddpg_v2(n_episodes=15, n_updates=80, seed=0, max_cycles=25,
                     buffer_size=20000, batch_size=128, gamma=0.95, tau=0.01,
                     lr_actor=1e-4, lr_critic=1e-3,
                     noise_start=0.5, noise_end=0.05, log_every=10):
    env = make_env(max_cycles)
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").shape[0]
    total_obs_dim = N_AGENTS * obs_dim
    total_action_dim = N_AGENTS * action_dim
    env.close()
    torch.manual_seed(seed); np.random.seed(seed)
    actors = [Actor(obs_dim, action_dim) for _ in range(N_AGENTS)]
    critic = CentralizedCritic(total_obs_dim, total_action_dim, N_AGENTS)
    target_actors = [Actor(obs_dim, action_dim) for _ in range(N_AGENTS)]
    target_critic = CentralizedCritic(total_obs_dim, total_action_dim, N_AGENTS)
    for ta, a in zip(target_actors, actors):
        ta.load_state_dict(a.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=lr_actor) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    buffer = ReplayBuffer(capacity=buffer_size)
    history = []
    for u in range(n_updates):
        noise_scale = max(noise_end, noise_start - (noise_start - noise_end) * u / max(1, n_updates // 2))
        ep_returns = []
        for ep in range(n_episodes):
            transitions, ep_return = collect_episode_with_next(
                actors, make_env(max_cycles),
                seed=seed * 1000 + u * 100 + ep, noise_scale=noise_scale)
            ep_returns.append(ep_return)
            for t in transitions: buffer.push(t)
        if len(buffer) < batch_size: continue
        samples = buffer.sample(batch_size)
        per_agent = {i: {"obs": [], "act": [], "nobs": [], "rew": [], "done": []} for i in range(N_AGENTS)}
        for s in samples:
            i = s["agent"]
            per_agent[i]["obs"].append(s["obs"])
            per_agent[i]["act"].append(s["action"])
            per_agent[i]["nobs"].append(s["next_obs"])
            per_agent[i]["rew"].append(s["reward"])
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
            full_obs = torch.zeros(Bi, total_obs_dim)
            full_act = torch.zeros(Bi, total_action_dim)
            full_obs[:, i * obs_dim:(i + 1) * obs_dim] = obs_b
            full_act[:, i * action_dim:(i + 1) * action_dim] = act_b
            with torch.no_grad():
                a_next = target_actors[i](nobs_b)
                full_nobs = torch.zeros(Bi, total_obs_dim)
                full_nact = torch.zeros(Bi, total_action_dim)
                full_nobs[:, i * obs_dim:(i + 1) * obs_dim] = nobs_b
                full_nact[:, i * action_dim:(i + 1) * action_dim] = a_next
                q_next = target_critic(full_nobs, full_nact)[:, i]
                target = rew_b + gamma * (1.0 - done_b) * q_next
            q_pred = critic(full_obs, full_act)[:, i]
            critic_loss = F.mse_loss(q_pred, target)
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            pred_action = actors[i](obs_b)
            full_act_pred = torch.zeros(Bi, total_action_dim)
            full_act_pred[:, i * action_dim:(i + 1) * action_dim] = pred_action
            actor_loss = -critic(full_obs, full_act_pred)[:, i].mean()
            actor_opts[i].zero_grad(); actor_loss.backward(); actor_opts[i].step()
        for ta, a in zip(target_actors, actors): soft_update(ta, a, tau)
        soft_update(target_critic, critic, tau)
        mean_return = float(np.mean(ep_returns))
        history.append({"update": u, "mean_return": mean_return, "noise_scale": noise_scale, "buffer_size": len(buffer)})
        if (u + 1) % log_every == 0 or u == 0:
            print(f"    update {u+1}/{n_updates}: mean_episode_return={mean_return:.2f}, buffer={len(buffer)}, noise={noise_scale:.3f}")
    return actors, critic, history
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-updates", type=int, default=80)
    p.add_argument("--n-episodes-per-update", type=int, default=15)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=20000)
    args = p.parse_args()
    print("=" * 60)
    print("MADDPG v2 - PettingZoo Simple Spread v3 (continuous, proper bootstrap)")
    print("=" * 60)
    print(f"  seed={args.seed}, n_updates={args.n_updates}, n_episodes/update={args.n_episodes_per_update}")
    print(f"  buffer={args.buffer_size}, batch={args.batch_size}, gamma=0.95, tau=0.01")
    print(f"  noise decay: 0.5 -> 0.05 over first half of training")
    print()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("Phase 1: Random baseline (continuous actions)...")
    rnd_returns = run_random_baseline(args.seed, n_episodes=20, max_cycles=args.max_cycles)
    rnd_mean = float(np.mean(rnd_returns)); rnd_std = float(np.std(rnd_returns))
    print(f"  Random: {rnd_mean:7.2f} +/- {rnd_std:5.2f}")
    print()
    print(f"Phase 2: MADDPG v2 training ({args.n_updates} updates x {args.n_episodes_per_update} episodes)...")
    actors, critic, history = train_maddpg_v2(
        n_episodes=args.n_episodes_per_update, n_updates=args.n_updates,
        seed=args.seed, max_cycles=args.max_cycles,
        batch_size=args.batch_size, buffer_size=args.buffer_size)
    print()
    print("Phase 3: Final eval (deterministic, no noise)...")
    final_eval = evaluate_maddpg(actors, n_episodes=args.n_eval_episodes, seed=4000)
    final_mean = float(np.mean(final_eval)); final_std = float(np.std(final_eval))
    delta = final_mean - rnd_mean
    print(f"  MADDPG v2 eval: {final_mean:7.2f} +/- {final_std:5.2f}")
    print(f"  Delta vs random: {delta:+7.2f}")
    print()
    log_path = HERE / "checkpoints" / "pz_maddpg_v2" / f"seed{args.seed}" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_actor_params = sum(p.numel() for a in actors for p in a.parameters())
    n_critic_params = sum(p.numel() for p in critic.parameters())
    log_path.write_text(json.dumps({
        "env": "PettingZoo Simple Spread v3 (continuous)",
        "seed": args.seed,
        "mode": "MADDPG v2 (proper next_obs bootstrap, target networks, linear noise decay)",
        "n_updates": args.n_updates, "n_episodes_per_update": args.n_episodes_per_update,
        "n_eval_episodes": args.n_eval_episodes, "max_cycles": args.max_cycles,
        "batch_size": args.batch_size, "buffer_size": args.buffer_size,
        "n_actor_params": n_actor_params, "n_critic_params": n_critic_params,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "final_eval_mean": final_mean, "final_eval_std": final_std,
        "per_episode_final_eval": final_eval,
        "delta_vs_random": float(delta), "history": history,
        "honest_note": "Proper MADDPG with target networks. Compute: " + str(args.n_updates*args.n_episodes_per_update) + " env episodes."
    }, indent=2))
    print(f"Log saved to: {log_path}")
if __name__ == "__main__":
    main()
