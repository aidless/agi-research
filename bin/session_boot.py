"""
session_boot.py - print a starter prompt for a fresh Codex session.
Reads PROGRESS.md, decisions, and recent commits.

Usage:
    python E:\agi-research\bin\session_boot.py

Trend #1: long-horizon autonomy. This is the read-side.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


def gitlog(n=5):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-" + str(n), "--pretty=format:%h %ad %s", "--date=short"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception as e:
        return f"(fail: {e})"


def rtxt(p, n=50):
    pth = Path(p)
    if not pth.exists():
        return "(missing)"
    t = pth.read_text(encoding="utf-8")
    L = t.split("\n")
    if len(L) > n:
        t = "\n".join(L[:n]) + f"\n...({len(L)-n} more)"
    return t


def pending_decisions():
    out = []
    d = ROOT / "decisions"
    if d.exists():
        for f in sorted(d.glob("*.md")):
            try:
                c = f.read_text(encoding="utf-8")
                if "Status: OPEN" in c or "Status: PENDING" in c:
                    out.append(f.stem)
            except Exception:
                pass
    return ", ".join(out) if out else "(none)"


def main():
    print("=" * 60)
    print("AGI Research Workspace - Session Boot")
    print("Time:", datetime.now().isoformat(timespec="seconds"))
    print("=" * 60)
    print()
    print("## Recent 5 commits")
    print(gitlog(5))
    print()
    print("## Pending decisions")
    print("  ", pending_decisions())
    print()
    print("## PROGRESS.md (top 30 lines)")
    print(rtxt(ROOT / "PROGRESS.md", 30))


if __name__ == "__main__":
    main()
