# Project F - Comprehensive Design Doc (Workspace Automation + Multi-Agent)

> 2026-07-25. Synthesizes F.1-F.4 deliverables shipped today, plus future
> F.5-F.7 (Trend #1, #2, #3, #5 implementation roadmap).

## 1. What Project F is

Project F is the engineering track for the AGI research programme: it builds
the workspace that Project A-E papers ride on.

Specifically:
- E:\agi-research\ is the workspace root.
- bin/, prompts/, decisions/, literature/, projects/, .experience_log/,
  .policy/ are the workspace sub-systems.
- All deliverables use these tools and conform to the conventions in AGENTS.md.

## 2. Mapping to Agent Futures Trends (the 6 trends)

| Trend | F.1-F.7 deliverable | Status |
|-------|--------------------|---------|
| #1 long-horizon autonomy | session_boot.py, session_debrief.py, multi_orchestrator.py | shipped |
| #2 multi-agent | prompts/{planner,executor,reviewer,safety}.md, multi_orchestrator.py | shipped |
| #3 self-improving / Agentic Learning | skill_mining.py, .experience_log/ | shipped |
| #4 MCP tool ecosystem | not yet implemented (2026 H2) | pending |
| #5 safety | policy_check.py, .policy/agent.yaml, audit.log | shipped |
| #6 multimodal | vision-based archive indexing | pending |

## 3. Shipped components (cumulative across sessions)

```
  E:\agi-research\
+ bin/
  + session_boot.py       (1785b) read-side of long-horizon
  + session_debrief.py    (1845b) write-side + commit summary
  + skill_mining.py       (2233b) regex-extract lessons from .experience_log/
  + multi_orchestrator.py (1986b) 4-stage pipeline concatenator
  + bibtex_build.py       (1915b) paper notes -> BibTeX
  + paper_draft.py        (1665b) outline + notes -> draft
  + policy_check.py       (   2.5KB) Cedar-like gate for destructive ops
  + README.md             (1474b) workspace tools documentation
+ prompts/
  + planner.md  (1753b) goal decomposition
  + executor.md (1613b) file/code execution
  + reviewer.md (1489b) diff/review + # REVIEW-ME
  + safety.md   (1417b) enforces .policy/agent.yaml
+ .policy/
  + agent.yaml    (1097b) Cedar-like declarative policy
  + audit.log     (append-only, written by policy_check.py)
+ .experience_log/
  + 2026-07-25-session-retro.md (2966b)
+ .tasks/
  + task-20260725-120747.md (multi_orchestrator first run)
```

## 4. Pending (F.5+ roadmap)

### F.5: MCP wrapper layer (Trend #4) - 1-2 months
- Wrap bin/ tools with Model Context Protocol servers.
- Allow external agents to invoke them via standard MCP interface.
- Prototype: McpServer wrapping policy_check.py + skill_mining.py.

### F.6: Multi-orchestrator end-to-end test (Trend #2) - 1 month
- Spawn 4 sub-agent personas per their prompts/planner.md.
- Each uses its own context window.
- Coordinate via the orchestration context file.
- Validate end-to-end on a real sub-task (e.g. paper outline review).

### F.7: Auto-reflection (Trend #3) - 1 month
- Auto-run skill_mining.py weekly.
- Distill top 3 lessons into prompts/*.md updates.
- Version-control each prompt iteration.
- Measure: do weeks with auto-updated prompts outperform baseline weeks?

## 5. Architecture: how the 4 layers interact

User session:
1. Opens Codex desktop.
2. Says: "Run E:\\agi-research\\bin\\session_boot.py".
3. Reads starter prompt, continues from latest DEC.

Workspace state:
- AGENTS.md (this file structure)
- PROGRESS.md (cross-session state, last)
- decisions/ (DEC-XXXX.md named decisions)
- .experience_log/ (retros)
- bin/ (CLIs, includes session_boot + session_debrief)
- prompts/ (multi-role system messages for sub-agents)
- .policy/ (Cedar-like rules)
- literature/ (paper notes + plan docs)
- projects/ (5 projects each with v0 + v1 outlines)
- experiments_log/ (per-experiment JSON + report)
- .tasks/ (orchestrated agent task contexts)

## 6. How Project F maps to the 4-layer architecture

Our 4-layer AGI architecture (sensor -> WM -> planner -> executor +
self-model) is itself an AGENT. Project F is the implementation of that
agent for the specific task of "build AGI research programme".

- "Sensors" = bin/session_boot.py reads workspace.
- "World Model" = CHANGELOG + decisions/ + git history.
- "Planner" = AGENTS.md + TASKBOOK + ROADMAP.
- "Executor" = Codex + bin/ tools.
- "Self-Model" = ADRs (decision records capture our reasoning).

Trend #1, #2, #3 are Project F being a self-improving agent.
Trend #5 (safety) is Project F gate-keeping destructive ops.

## 6. KPIs for Project F

- **Uptime**: 99% (workspace usable every session).
- **Tool reuse**: 80% of new tasks can reuse existing tools.
- **Decision latency**: time from goal to first DEC-XXXX file - target < 24h.
- **Tool coverage**: number of bin/ tools - currently 7.
- **Policy violations**: count of audit.log DENIED entries per quarter.
