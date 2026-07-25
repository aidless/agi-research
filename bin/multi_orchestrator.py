#!/usr/bin/env python3
"""multi_orchestrator.py - pipeline multi-agent roles against a task.

Reads E:\agi-research\prompts\{planner,executor,reviewer,safety}.md
and walks the human + Codex through a 4-stage pipeline:
  Planner -> Executor -> Reviewer -> Safety.

Output goes to a single context file the user (or Codex) can paste into
a fresh Codex session and execute.

Usage:
  python multi_orchestrator.py "Run Phase 1 Step 4"
"""
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
TASKS_DIR = ROOT / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


def read(name):
    return (PROMPTS / (name + ".md")).read_text(encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        print("usage: multi_orchestrator.py <task-description>")
        sys.exit(1)
    task = " ".join(sys.argv[1:])
    planner = read("planner")
    executor = read("executor")
    reviewer = read("reviewer")
    safety = read("safety")
    output = []
    output.append("# Orchestration context for task: " + task)
    output.append("")
    output.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    output.append("")
    output.append("## Role 1: Planner")
    output.append(planner)
    output.append("")
    output.append("## Role 2: Executor (only after Planner outputs)")
    output.append(executor)
    output.append("")
    output.append("## Role 3: Reviewer (only after Executor outputs)")
    output.append(reviewer)
    output.append("")
    output.append("## Role 4: Safety (gates destructive ops)")
    output.append(safety)
    output.append("")
    output.append("## The task:")
    output.append(task)
    print(chr(10).join(output))
    # save
    out_path = TASKS_DIR / ("task-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".md")
    out_path.write_text(chr(10).join(output), encoding="utf-8")
    print()
    print("[saved] " + str(out_path))


if __name__ == "__main__":
    main()
