#!/usr/bin/env python3
"""policy_check.py - Cedar-like policy enforcement for the agent.

Decision priority (cascade):
  1. DENY     if path in deny_paths
  2. DEFER    if op matches require_human_approval prefix
  3. ALLOW    if path is in allow_paths (and path contains a writable)
  4. DEFER    if op is unknown (not in allow_commands or require list)
  5. DENY     if op not in allow_commands and op is one we recognise
"""
import argparse
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / ".policy" / "agent.yaml"
LOG = ROOT / ".policy" / ".audit.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


def norm_path(p):
    if not p:
        return ""
    return p.replace(chr(92), "/").strip().lower()


def load_policy():
    if yaml is not None:
        with open(POLICY, encoding="utf-8") as f:
            return yaml.safe_load(f)
    pol = {"allow_paths": [], "deny_paths": [], "allow_commands": [], "require_human_approval": []}
    section = None
    for raw in POLICY.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith(":"):
            section = s[:-1]
            continue
        if s.startswith("- "):
            v = s[2:].strip()
            if section in pol and isinstance(pol[section], list):
                pol[section].append(v)
    return pol


def append_log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + chr(10))


def pattern_to_regex(pat):
    p = norm_path(pat)
    parts = re.split(r"(\*\*|\*)", p)
    out = ""
    for chunk in parts:
        if chunk == "**":
            out += ".*"
        elif chunk == "*":
            out += "[^/]*"
        else:
            out += re.escape(chunk)
    return "^" + out + "$"


def match_glob(path, patterns):
    p = norm_path(path)
    if not p:
        return False
    for pat in patterns:
        if re.match(pattern_to_regex(pat), p):
            return True
    return False


def starts_with_any(op, lst):
    if not op:
        return False
    return any(op.startswith(x.strip()) for x in lst)


def check(args):
    pol = load_policy()
    op = args.op or ""
    path = args.path or ""
    allow_paths = pol.get("allow_paths", []) or []
    deny_paths = pol.get("deny_paths", []) or []
    allow_cmds = pol.get("allow_commands", []) or []
    require_human = pol.get("require_human_approval", []) or []
    verdict = "ALLOW"
    reason = "default"
    # 1. Path deny check (highest priority)
    if path and deny_paths and match_glob(path, deny_paths):
        verdict = "DENIED"
        reason = "path in deny_paths"
    # 2. Op require-human check
    elif op and starts_with_any(op, require_human):
        verdict = "DEFER-TO-HUMAN"
        reason = "op in require_human_approval"
    # 3. Path allow check (covers file ops)
    elif path and (not allow_paths or match_glob(path, allow_paths)):
        verdict = "ALLOW"
        reason = "path in allow_paths"
    # 4. Op allow check (covers command ops when no path given)
    elif op and starts_with_any(op, allow_cmds):
        verdict = "ALLOW"
        reason = "op in allow_commands"
    # 5. Anything else with a suspicious op (write, edit, rm, etc.) is DEFER
    elif op:
        verdict = "DEFER-TO-HUMAN"
        reason = "unrecognised op; defer"
    else:
        verdict = "ALLOW"
        reason = "no op or path"
    log_line = str(datetime.now().isoformat(timespec="seconds")) + "  " + verdict + "  op=" + op + "  path=" + (path or "-") + "  reason=" + reason
    append_log(log_line)
    print(verdict)
    print("  reason: " + reason)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="", help="operation name")
    p.add_argument("--path", default="", help="target path")
    args = p.parse_args()
    check(args)


if __name__ == "__main__":
    main()
