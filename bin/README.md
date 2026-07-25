# bin/ - Workspace automation tools (Project F Phase 1)

Tools here implement Trend #1 (long-horizon autonomy) and Trend #5 (safety).
Each is a thin Python 3.10+ script that runs without external dependencies.

## Current tools

### session_boot.py
- Reads PROGRESS.md, decisions/, recent commits.
- Emits a starter prompt for a fresh Codex session.
- Use at the start of every Codex session.

### session_debrief.py
- Reports recent commits, uncommitted files, git diff summary.
- Use at the end of every Codex session; save output as
  experiments_log/<date>-session-N.md.

### skill_mining.py
- Reads .experience_log/ retro entries.
- Extracts "What bit us" lessons via regex.
- Prints top 10 most-recent lessons.
- Use weekly to update prompts/*.md.

## Pending tools (Project F backlog)

- paper_draft.py - generate paper drafts from notes/
- bibtex_build.py - compile all *.md paper notes into a BibTeX file
- role_orchestrator.py - spawn planner/executor/reviewer sub-agents
- policy_check.py - enforce .policy/agent.yaml before destructive ops

## Style

- Python 3.10+ (3.11 hermes-agent venv also works)
- ASCII only (avoid PowerShell quoting hell)
- ~150 lines maximum per tool
- Single file each, no external deps beyond standard library + git

## Testing

Run session_boot.py and check it prints reasonable output.
Run session_debrief.py and verify it shows git data.
Run skill_mining.py and verify it returns lessons.
