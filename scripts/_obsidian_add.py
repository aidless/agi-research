#!/usr/bin/env python3
"""
_obsidian_add.py - Helper to add new outputs to the Obsidian knowledge base.

Usage:
    python _obsidian_add.py --title "..." --tags "..." --file path/to/content.md
    python _obsidian_add.py --title "..." --tags "..." --content "..." --folder Results
"""
import argparse, os, sys
from datetime import datetime
from pathlib import Path

VAULT = Path(r"E:\ObsidianKnowledgeBase")
TEMPLATE = """---
tags: [{tags}]
date: {date}
title: "{title}"
{frontmatter_block}
---

# {title}

{content}

## Source

- Generated: {date}
- Added to Obsidian: {date}

## See also

{see_also}
"""

def slugify(s):
    s = s.lower()
    for c in [" ", "\t", "\n", ":", "/", "\\", "|", "?", "*", "<", ">", '"', ","]:
        s = s.replace(c, "-")
    return s[:80]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--file")
    p.add_argument("--content")
    p.add_argument("--folder", default="Papers")
    p.add_argument("--links", default="")
    p.add_argument("--frontmatter", default="")
    p.add_argument("--vault", default=str(VAULT))
    args = p.parse_args()
    if not args.file and not args.content:
        p.error("Need --file or --content")
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        content = args.content
    fm_block = ""
    if args.frontmatter:
        for line in args.frontmatter.split(","):
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                fm_block += f"{k.strip()}: {v.strip()}\n"
    see_also = "\n".join(f"- [[{t.strip()}]]" for t in args.links.split("|") if t.strip()) or "- (none)"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    slug = slugify(args.title)
    note = TEMPLATE.format(tags=args.tags, date=date, title=args.title,
                           frontmatter_block=fm_block, content=content, see_also=see_also)
    out_dir = Path(args.vault) / "00-outputs" / args.folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timestamp}_{slug}.md"
    out_path.write_text(note, encoding="utf-8")
    print(f"Wrote {out_path.relative_to(args.vault)}")

if __name__ == "__main__":
    main()