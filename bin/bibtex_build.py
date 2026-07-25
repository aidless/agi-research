#!/usr/bin/env python3
"""bibtex_build.py - assemble paper notes/*.md into a BibTeX file.

Each .md has a paper title, year, and (if user can find) arxiv id.
We extract: first # heading (title), first author-year filename hint,
and any arxiv id mentioned in the prose. Output a .bib.

Usage:
  python bibtex_build.py > E:\agi-research\references.bib
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "literature" / "papers"


def slug_to_key(stem):
    """e.g. '1991_sutton_dyna_q' -> 'sutton1991dyna' (rough)."""
    parts = stem.split(chr(95))
    if len(parts) < 2:
        return stem
    year = parts[0]
    author = parts[1] if len(parts) > 1 else "anon"
    rest = chr(10).join(parts[2:]) if len(parts) > 2 else ""
    return (author + year + rest.split("_")[0])[:30]


def main():
    if not PAPERS.exists():
        print("% no papers dir")
        return
    bib = []
    print("@% preamble:")
    for f in sorted(PAPERS.glob("*.md")):
        if f.name.startswith("_"):
            continue
        text = f.read_text(encoding="utf-8")
        title_m = re.search(r"^# (.+)$", text, re.MULTILINE)
        title = title_m.group(1) if title_m else f.stem
        arxiv_m = re.search(r"arXiv[: ]+([\d.]+)", text)
        arxiv = arxiv_m.group(1) if arxiv_m else ""
        year_m = re.search(r"\b(19|20)\d{2}\b", text)
        year = year_m.group(0) if year_m else "TODO"
        key = slug_to_key(f.stem)
        bib.append("@misc{" + key + ",")
        bib.append("  title  = {" + title + "},")
        if arxiv:
            bib.append("  note   = {arXiv:" + arxiv + "},")
        bib.append("  year   = {" + year + "},")
        bib.append("  file   = {" + str(f.relative_to(ROOT)).replace(chr(92), "/") + "},")
        bib.append("}")
        bib.append("")
    bib_text = chr(10).join(bib)
    print(bib_text)


if __name__ == "__main__":
    main()
