# Reading Plan (2026-07-25)

> **Honest framing**: Codex can read and report on **pre-2025 papers
> from training-data memory** with high reliability. For **2025-2026
> papers** cited in this program, Codex should be treated as a
> **secondary source**: user must verify the actual paper to ground
> the citation. This is non-negotiable for paper-grade work.

---

## Tier A (user must read personally)

These are the papers Codex cannot fully verify from training data.
The user must read them and write the notes (Codex can co-author
after the user has the source).

| # | paper | why important | when read |
|---|---|---|---|
| A1 | **Causal-JEPA (arXiv 2602.11389)** | Project C core method | week 1 |
| A2 | **V-JEPA 2-AC (arXiv 2506.09985)** | Project B core evidence | week 1 |
| A3 | JEPA-WM (arXiv 2512.24497) | Project C empirical design | week 2 |
| A4 | Value-Guided JEPA (arXiv 2601.00844) | Project A vs integrated ablation | week 2 |
| A5 | UniZero (arXiv 2406.10667) | latent-MCTS, Project A plug-in test | week 2 |
| A6 | DreamerV3 (Nature 2025) | main engineering reference | week 3 |

For each: 4-hour read + 1-page note. Each note must include:
- one-sentence problem statement
- mechanism (1 paragraph)
- key result + figure reference
- 3 criticisms you actually have
- where in our program this connects
- confidence level in our citation

---

## Tier B (Codex reads from training-data memory + draft note; user reviews)

Codex drafts a paper note from memory with explicit confidence flags.
User reviews + corrects. **Time cost: ~2 hours per paper for user, vs
~0 if user accepts Codex's first draft with caveats**.

| topic | paper | what we cite it for |
|---|---|---|
| PPO | Schulman 2017 | Project A baseline RL algorithm |
| GAE | Schulman 2015 | Project A advantage estimation |
| MuZero | Schrittwieser 2020 | latent-MCTS family |
| World Models | Ha & Schmidhuber 2018 | existing review in `world_models_review.md` |
| Dreamer V1 | Hafner 2020 | RSSM family origin |
| Dreamer V2 | Hafner 2021 | discrete latent innovation |
| Gato | Reed 2022 | Project D inspiration (multi-task) |
| PaLM-E | Driess 2023 | Project D empirical anchor |
| Scholkopf 2021 CRL | Scholkopf 2021 | Project C theory foundation |
| von Kugelgen 2021 | 2021 | Project C theory |
| Causal-JEPA survey | Bareinboim 2016/2024 | Project C theory |
| Pearl Book of Why | Pearl 2018 | L1-L3 ladder reference |
| Self-Critical seq | Rennie 2017 | decoupled training precedent in NLP |
| Curiosity/ICM | Pathak 2017 | intrinsic motivation baseline |
| Procgen | Cobbe 2019 | Project A paper-env benchmark |
| MiniGrid | Chevalier-Boisvert 2018 | alternative paper env |
| ACT-R | Anderson 2007 | cognitive arch reference |
| LIDA | Franklin 2006 | global workspace |
| SOAR | Laird 2012 | production systems |
| Chollet "On the Measure of Intelligence" | 2019 | KPI framework basis |

**Total Tier B**: 20 papers. 4 weeks if 1/day.

---

## Tier C (already covered)

Already written or in literature/:
- World Models 2018 review (`world_models_review.md`)
- AGI routes overview (`agi_routes_5x_overview.md`)
- Agent taxonomy (`agent_taxonomy_reference.md`)
- WM-PL 2025-2026 analysis (`wmpl_analysis_2026-07-25.md`)
- Self-critique (`self_critique/...`)
- CRL+Causal-JEPA deep dive (`crl_and_causal_jepa.md`) -- but this
  is **honest about uncertainty**; user should re-read with
  actual papers

---

## Tier D (do not read; cite as context only)

These are foundational but we already know the gist from textbooks.
Reading them time-for-value is low:
- Sutton & Barto (just buy and skim relevant chapters)
- Goodfellow Deep Learning (chapter 6, 8 for RL context)
- Murphy PML (skim chapters 1-12)

---

## How reading gets done

### Path 1: codex reads, user reviews (default)
- Codex drafts paper note from memory
- flags confidence at top
- user reads note + skims original paper
- user flags errors, Codex updates
- **Time**: 1 hour per paper for Codex, 30 min review

### Path 2: user reads, Codex co-writes
- user reads paper (3-4 hours)
- user writes a 1-page bullet summary
- Codex co-writes the formal note from user's bullets
- **Time**: 4 hours per paper for user, 1 hour for Codex

### Path 3: collaborate on Twitter-style thread
- user pastes key sentence from paper
- Codex asks 5 clarification questions
- user answers; we have a note
- **Time**: 1 hour per paper

**Recommended**: Path 1 for Tier B (15-20 papers), Path 2 for Tier A
(6 critical papers).

---

## Cadence target

- 1 paper/day (weekday)
- = 20-25 papers/month
- by end of Y0 Q3: ~45 papers read
- by end of Y0: ~60 papers read + noted

This is the **minimum** to be literature-credible for a Y1 main-conference
submission.

---

## Folder structure

```
literature/
  reading_plan.md        <- this file
  papers/                 <- one .md per paper, named YYYY_author_short.md
    _template.md
    2017_schulman_ppo.md
    2020_schrittwieser_muzero.md
    2025_hafner_dreamer_v3_nature.md
    ...
```

---

## Sub-critique: my earlier literature reviews were overconfident

`world_models_review.md` reads as authoritative. **It is not.** It is
Codex synthesising memory, not a primary reading. Same for
`crl_and_causal_jepa.md`. **User should treat all `literature/`
reviews as draft, not as authoritative summaries**, until the
corresponding paper note in `papers/` is marked "PRIMARY" (i.e., the
paper was actually read by user or Codex during this research program).

This is one of the most important methodological notes I owe the user.
