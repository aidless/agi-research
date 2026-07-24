# WM-PL (World Model + Planning Loop) Analysis - 2026-07-25

> Author: the user (you). Saved to workspace because it cites 2025-2026
> evidence that the previous critique + smoke test did not have.

---

## TL;DR from Codex

- WM-PL has been re-rated upwards by recent evidence.
  Previously B+ as standalone AGI path; now A as "AGI cognitive-layer core"
  with B+ as standalone path.
- The 7 bottlenecks the doc lists agree with my critique of yesterday
  but are sharper.
- The 4-layer architecture (LLM + WM-PL + VLA + Neuro-symbolic) is the
  new working frame. Our 4 projects partially map onto this.

---

## Critical evidence Codex did NOT have yesterday (added now)

| Paper / System | Date | What it changes |
|---|---|---|
| DreamerV3 (Nature 2025) | 2025 | Single config, 150+ tasks, **zero-shot Minecraft diamond** |
| V-JEPA 2-AC | 2025-06 | 1.2B params + 1M-hour video + 62-hour robot, **zero-shot Franka** |
| Genie 3 | 2025-08 | 24fps real-time interactive 3D worlds |
| Cosmos 2.5 | 2025-10 | Open-source WFM, 2M+ downloads |
| General Intuition | 2026-06 | Bezos-backed, **.3B valuation** for game-trained physical WM |
| Causal-JEPA | 2026-02 | **Object-level causal intervention injection** |

Causal-JEPA is the single most relevant new evidence. It directly attacks
yesterday's critique #2 ("causal ladder is assumed, not implemented").

---

## Mapping the 4-layer architecture to our 4 projects

| 4-layer component | Maps to | Status |
|---|---|---|
| LLM (semantic) | Project D - language-as-type-system | P0 (just upgraded) |
| WM-PL (imagination) | Project C - causal world model | P0 (just upgraded) |
| VLA (grounded action) | Project B - cross-domain transfer to physical | P1 |
| Neuro-symbolic (causal verification) | **No project yet** -- this is the gap |

We are missing Neuro-symbolic. The honest acknowledgement: our Project A
is *adjacent* to but not the same as neuro-symbolic verification. Project
A is meta-cognition on the agent; neuro-symbolic is causal formal verification
on the world model.

---

## Bottleneck cross-check vs our prior critique

| Yesterday's critique | Today's bottleneck table (user doc) | Concordance |
|---|---|---|
| Sim-to-Real is ontological | #3 表征接地 / Sim2Real 鸿沟 | Same concern, sharper |
| Causal ladder is assumed | #4 因果 vs 相关 | Same + a pointer (Causal-JEPA) |
| Long horizon needs causal structure | #1 + #2 + #7 交叉 | Same, more specific |
| Language is ontological | (implicit in #5 开放目标) | Same concern, different framing |
| LLM is straw man 2022 view | Addressed by 4-layer model | Updated to hybrid framing |

---

## Three scenarios -- their probability assignments

- A (WM-PL as cognitive layer core): ~55% — Codex agrees, this is highest
- B (absorbed into LLM as implicit WM): ~25% — Codex agrees, plausible
- C (independent WM-PL to AGI): ~20% — Codex disagrees: 5-10%, requires
  causal + open-goal + planning-economics breakthroughs simultaneously

---

## New evidence to read immediately (next 30 days)

1. **Causal-JEPA (arXiv 2602.11389)** - object-level causal intervention
   injection. **CRITICAL for Project C.**
2. **V-JEPA 2-AC (arXiv 2506.09985)** - video pretrain -> zero-shot robot
   deployment. **CRITICAL for Project B (cross-domain transfer).**
3. **JEPA-WM (arXiv 2512.24497)** - empirical study of what drives
   success in physical planning.
4. **Value-Guided Action Planning with JEPA (arXiv 2601.00844)** -
   uses value function to gate action selection in JEPA.
5. **UniZero (arXiv 2406.10667)** - scalable latent WM for games.

Together these define a "2026 Wave" we were missing yesterday.

---

## Action items this session (Codex will)

1. Save this doc to workspace - DONE in this file
2. Write a deep dive on causal WM focused on Causal-JEPA
   (in companion file crl_and_causal_jepa.md)
3. Update ROADMAP with the 4-layer arch note
4. Flag the missing Neuro-symbolic component
