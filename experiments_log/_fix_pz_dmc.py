import pathlib
p = pathlib.Path(r'E:\agi-research\projects\project_f_multi_agent\code\pz_dmc.py')
t = p.read_text(encoding='utf-8')

old1 = (
    '    for a, pol in policies.items():\n'
    '        opt = optimizers[a]\n'
    '        all_obs, all_actions, all_old_lp, all_adv, all_ret = [], [], [], [], []\n'
    '        for traj in trajectories:\n'
    '            obs = torch.from_numpy(np.stack([t["obs"] for t in traj])).float()\n'
    '            actions = torch.tensor([t["action"] for t in traj], dtype=torch.long)\n'
    '            old_lp = torch.tensor([t["log_prob"] for t in traj])\n'
    '            rewards = [t["reward"] for t in traj]\n'
    '            values = [t["value"] for t in traj]\n'
    '            dones = [False] * (len(traj) - 1) + [True]\n'
)
new1 = (
    '    for a, pol in policies.items():\n'
    '        opt = optimizers[a]\n'
    '        all_obs, all_actions, all_old_lp, all_adv, all_ret = [], [], [], [], []\n'
    '        for traj in trajectories:\n'
    '            if a not in traj or len(traj[a]) == 0:\n'
    '                continue\n'
    '            obs = torch.from_numpy(np.stack([t["obs"] for t in traj[a]])).float()\n'
    '            actions = torch.tensor([t["action"] for t in traj[a]], dtype=torch.long)\n'
    '            old_lp = torch.tensor([t["log_prob"] for t in traj[a]])\n'
    '            rewards = [t["reward"] for t in traj[a]]\n'
    '            values = [t["value"] for t in traj[a]]\n'
    '            dones = [False] * (len(traj[a]) - 1) + [True]\n'
)
assert old1 in t, 'old1 not found'
t = t.replace(old1, new1)

# Fix 2+3: replace "[tr[a] for a in tr]" with dict in both stages
t = t.replace('            trajs.append([tr[a] for a in tr])\n', '            trajs.append({a: tr[a] for a in tr})\n')

p.write_text(t, encoding='utf-8')
import ast
ast.parse(t)
print('AST OK, len', len(t))