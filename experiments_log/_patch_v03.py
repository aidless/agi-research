p = r"projects\project_a_self_improvement\code\full_integration_v3.py"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()
old = """    cfg = PPOConfig(total_steps=args.n_ppo_steps)
    agent = PPOAgent(obs_dim, n_actions, cfg)
    agent.train(args.env, total_steps=args.n_ppo_steps)"""
new = """    cfg = PPOConfig(obs_dim=obs_dim, n_actions=n_actions, rollout_len=2048, seed=args.seed)
    agent = PPOAgent(cfg)
    train_env = envs.make_env(args.env, seed=args.seed + 1)
    obs, _ = train_env.reset()
    for u in range(args.n_ppo_steps // cfg.rollout_len):
        batch = agent.collect_rollout(train_env, obs)
        agent.update(batch)
        obs = batch["final_obs"]
    train_env.close()"""
assert old in s, "pattern not found"
s = s.replace(old, new)
with open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched")
