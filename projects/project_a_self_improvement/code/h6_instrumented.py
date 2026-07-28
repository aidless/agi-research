"""h6_instrumented.py - H6: Joint Monitor AUROC vs PPO training step (instrumented).
H6 pre-registered claim: joint Monitor discrimination power decreases monotonically as PPO updates accumulate, due to the policy gradient dragging the Monitor signal.
Test: train PPO on LunarLander-v3 for 5K steps, every 500 steps (i) collect 30 fresh rollouts, (ii) train joint Monitor for 5 epochs, (iii) evaluate on a HELD-OUT set collected ONCE at the start (so we measure Monitor quality on a fixed distribution while train distribution drifts).
H6 verdict: VALIDATED if AUROC strictly decreases (Spearman rho < 0, p<0.10); REFUTED if AUROC does not decrease; PARTIAL if noisy.
"""
import argparse, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from collections import deque
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import envs
from ppo import PPOAgent, PPOConfig
from monitor import FailureMonitor, MonitorConfig, _quick_auroc
from envs import rollout_one_episode
def make_env_factory(env_name):
    def factory(seed): return envs.make_env(env_name, seed=seed)
    return factory
def collect_rollouts(env_factory, agent, n_episodes, seed_offset):
    out = []
    for i in range(n_episodes):
        env = env_factory(seed_offset + i + 7777)
        ep = rollout_one_episode(env, agent.select_action, max_steps=500)
        ep.env_name = "lunarlander"
        out.append(ep); env.close()
    return out
def monitor_auroc_on(monitor, episodes, history_len, n_actions, threshold):
    X, y = [], []
    for ep in episodes:
        if len(ep.transitions) < 1: continue
        v = ep.history_vector(history_len=history_len, n_actions=n_actions)
        X.append(v); y.append(1.0 if ep.total_reward < threshold else 0.0)
    if not X or len(set(y)) < 2: return float("nan")
    X = torch.from_numpy(np.stack(X))
    y = np.array(y, dtype=np.float32)
    with torch.no_grad(): p = monitor(X).numpy()
    try: return float(_quick_auroc(y, p))
    except Exception: return float("nan")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-ppo-steps", type=int, default=5000)
    p.add_argument("--history-len", type=int, default=32)
    p.add_argument("--monitor-interval", type=int, default=500)
    p.add_argument("--n-monitor-rollouts", type=int, default=30)
    p.add_argument("--n-eval-rollouts", type=int, default=20)
    p.add_argument("--monitor-epochs-per-step", type=int, default=5)
    p.add_argument("--percentile", type=float, default=10.0)
    args = p.parse_args()
    print("=== H6 instrumented: seed=" + str(args.seed) + ", n_ppo_steps=" + str(args.n_ppo_steps) + ", interval=" + str(args.monitor_interval) + " ===")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = envs.make_env("LunarLander-v3", seed=args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    history_dim = args.history_len * (obs_dim + n_actions + 1)
    mcfg = MonitorConfig(history_dim=history_dim, seed=args.seed, epochs=args.monitor_epochs_per_step)
    monitor = FailureMonitor(mcfg)
    opt = torch.optim.Adam(monitor.parameters(), lr=mcfg.lr)
    obs, _ = env.reset()
    n_updates = args.n_ppo_steps // cfg.rollout_len
    factory = make_env_factory("LunarLander-v3")
    running_threshold = -50.0
    print("  Collecting held-out set (" + str(args.n_eval_rollouts) + " rollouts) from initial PPO...")
    heldout_rollouts = collect_rollouts(factory, agent, args.n_eval_rollouts, args.seed * 100000 + 9999)
    heldout_returns = [ep.total_reward for ep in heldout_rollouts]
    heldout_threshold = float(np.percentile(heldout_returns, args.percentile))
    print("  Heldout threshold (p" + str(args.percentile) + " of heldout returns): " + str(round(heldout_threshold, 2)))
    init_auroc = monitor_auroc_on(monitor, heldout_rollouts, args.history_len, n_actions, heldout_threshold)
    print("  Step 0 (Monitor random): heldout AUROC = " + str(round(init_auroc, 3)))
    traj = []
    all_returns = deque(maxlen=200)
    next_eval_step = args.monitor_interval
    for u in range(n_updates):
        batch = agent.collect_rollout(env, obs)
        info = agent.update(batch)
        obs = batch["final_obs"]
        all_returns.extend(batch["ep_returns"])
        if len(all_returns) >= 50:
            running_threshold = float(np.percentile(list(all_returns), args.percentile))
        cur_step = (u + 1) * cfg.rollout_len
        if cur_step >= next_eval_step:
            fresh = collect_rollouts(factory, agent, args.n_monitor_rollouts, args.seed * 100000 + cur_step)
            X, y = [], []
            for ep in fresh:
                if len(ep.transitions) < 1: continue
                v = ep.history_vector(history_len=args.history_len, n_actions=n_actions)
                X.append(v); y.append(1.0 if ep.total_reward < running_threshold else 0.0)
            if X and len(set(y)) == 2:
                X = torch.from_numpy(np.stack(X))
                y = torch.from_numpy(np.array(y, dtype=np.float32))
                for _ in range(args.monitor_epochs_per_step):
                    opt.zero_grad()
                    p_pred = monitor(X)
                    loss = F.binary_cross_entropy(p_pred, y)
                    loss.backward(); opt.step()
            auroc = monitor_auroc_on(monitor, heldout_rollouts, args.history_len, n_actions, heldout_threshold)
            traj.append({"step": cur_step, "heldout_auroc": auroc, "running_threshold": running_threshold, "mean_train_return": float(np.mean(list(all_returns)))})
            print("  Step " + str(cur_step) + ": heldout_AUROC=" + str(round(auroc, 3)) + " train_threshold=" + str(round(running_threshold, 2)) + " mean_train_return=" + str(round(float(np.mean(list(all_returns))), 2)))
            next_eval_step += args.monitor_interval
    valid = [t for t in traj if t["heldout_auroc"] == t["heldout_auroc"]]
    if len(valid) >= 3:
        from scipy.stats import spearmanr
        steps = [t["step"] for t in valid]
        aurocs = [t["heldout_auroc"] for t in valid]
        rho, pval = spearmanr(steps, aurocs)
        verdict = "VALIDATED" if (rho < 0 and pval < 0.10) else ("REFUTED" if rho > 0 else "PARTIAL")
        print()
        print("  Spearman rho(step, AUROC) = " + str(round(rho, 3)) + ", p=" + str(round(pval, 3)))
        print("  H6 verdict: " + verdict)
    else:
        verdict = "INSUFFICIENT_DATA"; rho, pval = float("nan"), float("nan")
        print("  Not enough valid AUROC points.")
    log_path = HERE / "checkpoints" / "h6_instrumented" / ("seed" + str(args.seed)) / "h6_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"seed": args.seed, "n_ppo_steps": args.n_ppo_steps, "monitor_interval": args.monitor_interval, "n_eval_rollouts": args.n_eval_rollouts, "n_monitor_rollouts": args.n_monitor_rollouts, "heldout_threshold": heldout_threshold, "init_auroc": init_auroc, "traj": traj, "spearman_rho": float(rho) if rho == rho else None, "spearman_p": float(pval) if pval == pval else None, "h6_verdict": verdict}, indent=2))
    print("  Log: " + str(log_path))
if __name__ == "__main__": main()
