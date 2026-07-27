p = r"projects\project_a_self_improvement\code\h1_cross_env.py"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()
old = 'HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nsys.path.insert(0, str(HERE.parent / "project_c_causal_world" / "code"))'
new = 'HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nsys.path.insert(0, str(HERE.parent.parent / "project_c_causal_world" / "code"))'
assert old in s, "pattern not found"
s = s.replace(old, new)
with open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched parent.parent")
