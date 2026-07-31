# Archimedes Project: H1-H10 Hypothesis Status (2026-07-31)

This file aggregates the verification status of every hypothesis
in the Archimedes Project framework as of 2026-07-31. It
supersedes the H1-H9 list in `papers/y1_9hypothesis_framework.md`
by adding H10 (the LLM self-monitoring pilot, Y4) and updating
H5 (multi-agent decoupling, Y3) and H2 (training-time Monitor
validity) with their most recent follow-up data.

| H | Statement | Status | Key result | n | source |
|---|---|---|---|---|---|
| **H1** | Decoupled Monitor > Joint Monitor (single-agent) | VALIDATED | 5/5 seeds, +39.5 mean, t=6.76, p<0.001 (LunarLander-v3) | 15 | Y1 paper |
| **H1.4** | Monitor as exploration bonus | REFUTED | H1.4 REAL mean 52.7, RANDOM mean 78.3 (LunarLander-v3) | 5 | Y1 H1.4 |
| **H2** | Training-time Monitor > Inference-time intervention | VALIDATED | n=15 seeds, p<0.001 (LunarLander-v3) | 15 | Y1 paper |
| **H3** | DLR predicate transfer across environments | VALIDATED | 4 envs, 3 seeds each, 19 predicates, accuracy >70% | 12 | Y1 paper |
| **H4** | Slot-attention Monitor > Raw-history Monitor | VALIDATED (1 env, 1 seed) | 0.989 vs 0.796 AUROC (LunarLander-v3) | 1 | Y1 paper |
| **H5** | Decoupled Monitor coordination in multi-agent | REFUTED (5/6 pathways) | v8 dlr_only +0.06 at n=100 (p<0.05 Bonf) is the only publishable sub-result; all 5 Monitor-using pathways REFUTED | 100+ | Y3 paper |
| **H6** | Joint Monitor failure is monotonic with PPO updates | REFUTED | non-monotonic; 5-seed instrumented, 10K PPO | 5 | Y1 H6 |
| **H7** | Reference Monitor + Evidence Chain (V1 governance) | VALIDATED | GovBench H1+H2, 7 seeds | 7 | Y1 H7 |
| **H8** | A2A cross-agent trust gate intercepts impersonation | VALIDATED | GovBench H3, 7 seeds | 7 | Y1 H8 |
| **H9** | Self-improvement loop with Monitor feedback | OPEN | Y3 work in progress (multi-step self-modification) | - | Y3 follow-up |
| **H10** | Decoupled Monitor transfers to LLM self-monitoring | REFUTED at chance level (n=100) | F-J Cohen d=+0.030, 95% CI [-0.087, +0.117], all 3 arms within 0.02 of 0.5 | 100 | Y4 paper |

## Status legend

- **VALIDATED** -- the hypothesis is supported by the data
  at the pre-registered significance level.
- **REFUTED** -- the hypothesis is contradicted by the data
  (either direction, significance, or both).
- **OPEN** -- the hypothesis is still under investigation
  (planned or in-progress work).
- **VALIDATED (caveat)** -- supported but with limited scope
  (e.g., H4 was only tested in 1 env, 1 seed; H5 partial
  -- only 1 of 6 pathways gave a positive result).

## What changed since the original H1-H9 list

- **H5** updated from REFUTED (continuous-action DMC) to
  REFUTED (5/6 multi-agent pathways), with the new
  qualification that v8 dlr_only (DLR in critic, NOT Monitor)
  is the only publishable positive result, +0.06 at n=100.
- **H10** is new. It is the LLM self-monitoring analog of H5
  and was REFUTED at all three sample sizes tested (n=5, 20, 100).

## Pre-registration discipline

Each of H5 (multi-agent) and H10 (LLM self-monitoring) had a
pre-registration document written BEFORE the data was collected.
The pre-registrations are linked in
`papers/supplementary_materials.md` Section S11. The analyses
in the Y3 and Y4 papers follow the pre-registered pipeline
without post-hoc modification.
