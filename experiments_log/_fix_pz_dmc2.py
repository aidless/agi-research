import pathlib
p = pathlib.Path(r'E:\agi-research\projects\project_f_multi_agent\code\pz_dmc.py')
t = p.read_text(encoding='utf-8')

# Replace the two "if monitor_collect and monitors is not None" guards.
old1 = "            if monitor_collect and monitors is not None:\n                history[a].append(obs.copy())\n"
new1 = "            if monitor_collect:\n                history[a].append(obs.copy())\n"
assert old1 in t, 'old1 missing'
t = t.replace(old1, new1)

old2 = "    if monitor_collect and monitors is not None:\n        # Save histories and labels\n        ep_data = {}\n"
new2 = "    if monitor_collect:\n        # Save histories and labels\n        ep_data = {}\n"
assert old2 in t, 'old2 missing'
t = t.replace(old2, new2)

p.write_text(t, encoding='utf-8')
import ast
ast.parse(t)
print('AST OK, len', len(t))