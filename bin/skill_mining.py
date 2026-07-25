#!/usr/bin/env python3
"""skill_mining.py - extract top lessons from .experience_log/ retro entries.

Trend #3 (Agentic Learning). Reads all retro entries from
E:\agi-research\.experience_log\, finds "What bit us" / "What broke" sections,
summarises.
"""
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / ".experience_log"


def lessons_from_log(text):
    out = []
    in_broke = False
    cur_block = []
    for line in text.split(chr(10)):
        if "What bit us" in line or "What broke" in line:
            in_broke = True
            continue
        if in_broke and (line.startswith("# ") or line.startswith("## ")):
            in_broke = False
        if in_broke and re.match(r"\s*[\-\d]", line):
            cur_block.append(line.strip())
        else:
            if cur_block:
                out.append(" | ".join(cur_block))
                cur_block = []
    if cur_block:
        out.append(" | ".join(cur_block))
    return out


def main():
    print("=" * 60)
    print("AGI Workspace - Skill Mining")
    print("Source:", LOG_DIR)
    print("Time:", datetime.now().isoformat(timespec="seconds"))
    print("=" * 60)
    print()
    if not LOG_DIR.exists():
        print("(no experience log yet; this CLI is bound for the future)")
        return
    files = sorted(LOG_DIR.glob("*.md"))
    if not files:
        print("(no retrospective entries yet)")
        return
    all_lessons = []
    for f in files:
        print("Reading " + f.name + "...")
        all_lessons.extend([(f.stem, l) for l in lessons_from_log(f.read_text(encoding="utf-8"))])
    print()
    print("Total lessons extracted: " + str(len(all_lessons)))
    print()
    print("Top 10 (most-recent first):")
    print("-" * 60)
    for f, l in all_lessons[-10:]:
        print("  [" + f + "] " + l)
    print()
    print("-" * 60)
    print("To use these lessons:")
    print("  1. Manually review the above.")
    print("  2. Promote the most-actionable ones into prompts/*.md")
    print("  3. Re-run future sessions with the new prompt content")


if __name__ == "__main__":
    main()
