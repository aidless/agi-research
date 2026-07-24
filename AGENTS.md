# AGENTS.md

This folder is your AGI research workspace managed by Codex (an AI agent).

## Working conventions
- Codex executes commands, writes code, drafts papers, and proposes plans here.
- You (the human) make decisions: which direction to commit, which result counts,
  which draft to ship. Your time is spent on review + judgment.
- Codex has no memory across sessions. Always re-state the current goal at the
  start of a new session by opening this file or pointing Codex at
  `E:\agi-research\PROGRESS.md`.

## Directory layout (authoritative)
- `README.md`        : entry point, "what to do today"
- `ROADMAP.md`       : 5-year vision + quarterly milestones
- `PROGRESS.md`      : living log (status, blockers, recent wins)
- `00_daily/`        : daily review notes template
- `literature/`      : my paper reviews + your self-written summaries
- `projects/`        : per-project folders (paper, code, experiments, notes)
- `grant_applications/` : GPU credit application drafts
- `community/`       : Twitter / Reddit / Discord / email drafts
- `experiments_log/` : per-experiment log files (one .md per run)
- `decisions/`       : Decision Records (one .md per major decision)

## How to interact with Codex
Always open a new session by typing something like:
  "Pick up where PROGRESS.md left off. Goal is X. Next milestone is Y."

Codex will read PROGRESS.md and the relevant project README, then resume work.
You review the outputs, leave comments in CODE files via # REVIEW: markers, and
update PROGRESS.md at end of each session.
