# Agent Futures Plan (2026-07-25)

> Mapping the user-supplied 6 trends about the future of AI agents onto
> our 5-year AGI research program. Saved for future sessions to build on.

## The 6 trends (input)

1. Long-horizon autonomy (persistent agents, days/weeks)
2. Multi-agent collaboration (planner / executor / reviewer / safety)
3. Self-improvement (record success/failure, distill skills, accumulate)
4. Tool ecosystem explosion (MCP, OS-like attach of tools)
5. Safety and alignment (Cedar-like policy engines + audit + budget + approval)
6. Multimodal perception (screen, voice, GUI)

## How each trend touches our program

| Trend | Existing Projects Touched | New Project Suggestion |
|-------|---------------------------|------------------------|
| 1 | A, B, C, D, E (long compute jobs) | Project F (Workspace Automation) |
| 2 | A (Monitor), E (verifier) | Project G (Multi-Agent Verification Pipeline) |
| 3 | A (decoupled self-critic is the RL version) | Project H (Agentic Learning Layer) |
| 4 | B (cross-domain data) | MCP wrapper for our tools |
| 5 | C, E (causal + verification) | Project I (AgentOps Safety Layer) |
| 6 | B + V-JEPA 2-AC (VLA data) | Project J (Multimodal Assistant) |

## Priority ordering (for next 18 months)

P1 (existing, already in motion): A, C, D, E, B
P2 (new, proposed now): F (Workspace Automation)
P2 (new): G (Multi-Agent Verification, branched from E)
P2 (new): I (AgentOps Safety)
P3 (new): H (Agentic Learning Layer)
P3 (new): J (Multimodal Assistant)

## Concrete deliverables (60 days)

### F.1 Session Boot CLI (5 days)
File: E:\agi-research\bin\session_boot.py
Purpose: print a starter prompt for a fresh Codex session by reading
PROGRESS.md, decisions, recent commits.
Status: SHIPPED (this session, 1785 bytes).

### F.2 Session Debrief CLI (5 days)
File: E:\agi-research\bin\session_debrief.py
Purpose: summarise session changes via git, propose next-session prompt.
Status: SHIPPED (this session, 1845 bytes).

### F.3 Experience Log (7 days)
File: E:\agi-research\.experience_log\<date>-<topic>.md
Purpose: capture what worked / what broke, indexed for replay.
Status: SHIPPED initial retro (2966 bytes).

### I.1 Cedar-like Policy YAML (10 days)
File: E:\agi-research\.policy\agent.yaml
Purpose: declarative allow/deny/approval rules for the agent.
Status: SHIPPED seed (this session).

### F.4 Multi-role prompt templates (5 days)
File: E:\agi-research\prompts\{planner, executor, reviewer, safety}.md
Purpose: prompts to spawn sub-agents per role for sub-tasks.

### G.1 Multi-agent verification experiment (14 days)
Real implementation of Project E with three sub-agents:
proposer (LLM), verifier (Z3/Lean), coordinator (decide revise / accept).

## 18-month timeline

- Month 1-2: F.1 + F.2 + F.3 + I.1 (workspace tools stack)
- Month 3-4: F.4 (multi-role prompts), G.1 (multi-agent verification)
- Month 5-6: MCP wrapper for E tools
- Month 7-9: H + J design (Phase 1)
- Month 10-12: H + J implementation
- Month 13-15: multi-agent run on papers
- Month 16-18: evaluate, write "how we work" paper (academic value)

## Notes

- These 6 trends describe the FUTURE of agents. We are simultaneously
an AGI research program (studying the topic) AND a research workflow that
uses agents (Codex). The two roles converge: our workspace IS one instance
of trends #1, #2, #3 in production.
- Trends #4 and #6 are mostly external tooling; we use what is available
rather than build our own. MCP wrapper (~ month 5) is the bridging layer.
- Trend #5 (safety) is the most under-served by current tools; this is
where our `.policy/agent.yaml` becomes a precursor.

## Open questions

1. Should Project F ship as a paper (F.1-F.2 are practical but unglamorous) or a tool (just ship, no academic credit)? Recommendation: tool.
2. Is multi-agent verification actually better than single-LM with Tree-of-Thought? Empirically testable; do not assume.
3. Should we ship a "self-improving agent loop" first, or a "well-engineered multi-agent loop" first? Order unknown ¡ª propose to test both in parallel after F.1/F.2/F.3/F.4 are stable.
