#!/usr/bin/env python3
"""session_debrief.py - archive session work and produce a status report.

Trend #1 (long-horizon autonomy) + #3 (agentic learning).
Implement the write-side of session management.
"""
import subprocess, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


def git_diff_stats():
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--stat", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip() or "(no unstaged diff)"
    except Exception as e:
        return f"(fail: {e})"


def git_recent_commits(n=10):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-" + str(n), "--pretty=format:%h %ad %s", "--date=short"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception as e:
        return f"(fail: {e})"


def main():
    print("=" * 60)
    print("AGI Research Workspace - Session Debrief")
    print("Time:", datetime.now().isoformat(timespec="seconds"))
    print("=" * 60)
    print()
    print("## Recent 10 commits")
    print(git_recent_commits(10))
    print()
    print("## Uncommitted files")
    uf = []
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        )
        uf = [ln.strip() for ln in out.split(chr(10)) if ln.strip()]
    except Exception:
        pass
    if uf:
        for f in uf:
            print(" ", f)
    else:
        print("  (clean working tree)")
    print()
    print("## Uncommitted changes (git diff HEAD --stat)")
    print(git_diff_stats())


if __name__ == "__main__":
    main()
