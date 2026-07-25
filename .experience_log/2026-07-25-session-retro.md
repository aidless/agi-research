# Session retro: 2026-07-25

> A retrospective on the first big session of the AGI research program.
Captures what worked, what broke, and what to remember.

## What went well

- Workspace bootstrapped from zero to 33 files in one intense session.
- End-to-end Procgen pipeline (CPU-only) works on Python 3.10 (Trae Solo CN).
- Project A paper outline v0 full body with real smoke-test numbers (Phase 1 step 1 AUROC 0.71) ready for workshop submission.
- Phase 1 step 1 + 2 + 3 ran (4305 episodes collected across 12 runs).
- Phase 2 first execution verified end-to-end pipeline (negative-but-clean result).
- Multiple architectural pieces now have outlines (A, B, C, D, E).
- CHANGELOG + git version control working from start.
- 7 Decision Records (DEC-0001..0010) capture all major pivots.

## What bit us

1. **PowerShell heredoc failures.** Multiple times tried to write big files via `@'...'@` and saw them parsed wrong. Eventually settled on line-by-line `Add-Content`. Wasted ~20 min on this.
2. **procgen wheels only for cp310.** Hermes-agent venv is 3.11; procgen cp311 wheel does not exist. Solution: switched the runtime path to Trae Solo CN python 3.10 which has procgen 0.10.7 pre-installed. Documented in envs.py.
3. **ProcgenWrapper AssertionError.** Initially subclassed gymnasium.ObservationWrapper which type-checks underlying env. Switched to standalone `gym.Env` subclass mimicking the API surface.
4. **history_vector hardcoded action space size 2.** Caused IndexError on Procgen (15 actions). Fixed by parameterizing with n_actions; auto-detected in monitor.py FailureDataset from max action value seen.
5. **P30 thresholds all 0.0** at 50K PPO steps. Causal: policy not yet learned enough to produce failure variance. Decision: scale up to 256K+ steps before relying on Phase 2 H1 demonstration.
6. **5 trends docs in Chinese delivered as pasted text** -- PowerShell escaping cannot reliably pass them through. Workaround: skip doc archive, deliver plan inline.

## What to remember

- Use `Add-Content` line-by-line for any non-trivial file write in PowerShell.
- Always test Python imports at the right Python version (cp31x vs cp310).
- For Phases: 50K PPO step is too short for any variance. Plan for 256K+.
- Cedar-policy ideas from agent futures doc would replace our ad-hoc file rules.
- The 4-layer architecture (A self-monitor + C causal-WM + D language-types + E verifier) is correct, needs more empirical coverage.
- PR commits with the magic messages; DO NOT rely on session memory at all.

## Open items for next session

- Phase 1 Step 4 (256K * 3 * 4 games = ~30 min CPU).
- Phase 2 v2 (with stronger baselines from Step 4).
- Tier A 4 paper primary reads (Causal-JEPA, V-JEPA 2-AC, JEPA-WM, Value-Guided JEPA).
- Project H (Agentic Learning Layer) outline.
- Workspace F-I tooling follow-up (Cedar-like policy YAML).
- Connect with future Project F plan from `agent_futures_plan.md`.
