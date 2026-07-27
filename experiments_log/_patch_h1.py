p = r"projects\project_a_self_improvement\code\h1_cross_env.py"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()
# Add project_c to path BEFORE the imports
old = "HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))"
new = """HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "project_c_causal_world" / "code"))"""
assert old in s
s = s.replace(old, new)
# Remove the duplicate sys.path for slot_attention later
old2 = "from slot_attention import SlotAttention"
new2 = "from slot_attention import SlotAttention  # noqa: E402"
assert old2 in s
s = s.replace(old2, new2)
with open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched")
