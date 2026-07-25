#!/usr/bin/env python3
"""paper_draft.py - assemble a paper draft from outline + notes."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
PAPERS = ROOT / "literature" / "papers"


def main():
    if len(sys.argv) < 2:
        print("usage: paper_draft.py <project-folder-name>")
        print("e.g. paper_draft.py project_a_self_improvement")
        sys.exit(1)
    name = sys.argv[1]
    proj = PROJECTS / name
    if not proj.exists():
        print("not a project: " + str(proj))
        sys.exit(1)
    outlines = list(proj.glob("paper_outline*.md"))
    if not outlines:
        print("no outlines in " + str(proj))
        sys.exit(1)
    print("# DRAFT: " + name)
    print()
    print("> Auto-generated from paper outline + literature notes.")
    print("> Edit and submit; not a final manuscript.")
    print()
    for ol in outlines:
        print(ol.read_text(encoding="utf-8"))
        print()
    notes = list(PAPERS.glob("*.md"))
    if notes:
        print("# Bibliography (workspace paper notes)")
        print()
        import re
        for n in sorted(notes):
            if n.name.startswith("_"):
                continue
            t = n.read_text(encoding="utf-8")
            tm = re.search(r"^# (.+)$", t, re.MULTILINE)
            title = tm.group(1) if tm else n.stem
            print("- " + title + " (" + n.name + ")")


if __name__ == "__main__":
    main()
