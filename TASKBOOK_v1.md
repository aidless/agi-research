# Archimedes Project Charter v1.0

| field | value |
|---|---|
| project-id | AGI-2026-001 |
| project-name | Archimedes: Self-Improving Hybrid AGI Substrate |
| version | v1.0 |
| start-date | 2026-07-25 |
| principal-investigator | user |
| research-assistant | Codex (AI) |
| duration | 60 months (5 years) |
| current-phase | Year 0 Q1 |

---

## 0. Charter (core belief)

The kernel of this project:

> "The essence of AGI is not the possession of many skills, but the
> ability to acquire new skills. Real general intelligence is not
> measured by how much you already know, but by how effectively you
> can learn in the face of the unknown."
> -- Francois Chollet (DeepMind / ARC-AGI)

We do NOT measure absolute task performance. We measure
**cross-domain learning efficiency**. Every KPI in this document is
rooted there.

---

## 0.5. Operating mode: AIKR (v1.10 framing, 2026-07-25)

> The 5-year program operates in **AIKR mode** (Assumption of
> Insufficient Knowledge and Resources, after Pei Wang`s NARS).
> We accept upfront that:
> 1. We do not know the full AGI architecture today.
> 2. Compute, data, and team are bounded.
> 3. Tasks and resources are open-ended (new problems arrive,
>    new compute/data can be acquired).
> 4. Truth is not static (architectures will be revised).
>
> This is NOT a weakness. It is the operating mode. Our 5-year
> plan is a *plan under* AIKR, not in spite of it. Each
> quarterly review re-rates the plan against the latest
> frontier evidence.

**v1.10 amendments** (2026-07-25, see CHANGELOG.md):

| ID | change | status |
|----|--------|--------|
| 13 | H1 joint ablation shipped (5-seed delta=0.724, 5/5 supported) | done |
| 14 | TMLR H+I corpus synthesis shipped (14 files, 5 headline updates) | done |
| 15 | ROADMAP reading list +6 must-read papers (Lightman/Snell/STaR/DIAMOND/Shumailov/Burns) | done |
| 16 | TTC as Monitor extension proposed (ADR 0011, P3) | queued |
| 17 | Paper A related work: ReAct/Reflexion/Self-Refine/CRITIC | queued |
| 18 | Paper C: DIAMOND baseline + collapse-detection methodology | queued |
| 19 | Paper E: Outer/Inner/Corrigibility three-layer framing | queued |

## 1. Background

### 1.1 Academic + engineering coordinate (2025-2026)

The candidate paths to AGI have converged from single-technology bets
to a **multi-technology fusion**. Representative evidence:

- Scaling LLM: GPT-5 / Claude 4 / Gemini 3 / DeepSeek R2 (engineering mature)
- Neuro-Symbolic: DreamCoder, Scallop, AlphaProof (hybrid works)
- World Models: DreamerV3 (Nature 2025), V-JEPA 2-AC, Causal-JEPA, Genie 3
- Embodied AI: Gato, pi-0, RT-X, Cosmos 2.5
- Cognitive Architecture: LIDA (Global Workspace), ACT-R, SOAR

### 1.2 Key open gaps

- No single route solves Pearl L3 (counterfactual reasoning)
- The interface between LLM and World Model is non-formalised
- Self-improvement has no formal verification layer
- Cross-environment transfer has no agreed benchmark

### 1.3 Why this project is justified

Our four projects (A/B/C/D + pending E) become a publishable AGI-path
contribution. The pieces:

```
LLM (semantic + types)         -> Project D  language-as-type-system
World Model (causal + object)  -> Project C  causal representation
Self-Model (decoupled critic)  -> Project A  frozen-policy monitor
Embodied (VLA grounding)       -> Project B  cross-domain transfer
Neuro-symbolic verification    -> Project E  formal verification (P2)
```

---

## 2. Objectives

### 2.1 Ultimate goal (Year 5)

> At least one reference implementation of our work is cited or
> integrated by 3+ frontier labs.

### 2.2 Phased goals

| year | milestone | deliverable |
|---|---|---|
| Y0 (M1-12) | smoke test + first arXiv | 1 paper, 50 GitHub stars, USD100/mo compute |
| Y1 (M13-24) | main-conference submission | NeurIPS / ICML paper submitted |
| Y2 (M25-36) | integrate A + C | one subsystem prototype |
| Y3 (M37-48) | industry adoption | frontier-lab citation evidence |
| Y4-5 (M49-60) | reference publication | open-source release + invited talk |

### 2.3 Definition of Success (DoS)

- main path: Project A or C paper accepted (main conference, main track)
- fail path: zero main-conference papers in 5 years -> Kill Switch (sect 8)

---

## 3. Scope

### 3.1 In scope

- research code / experiments / papers (arXiv / submissions)
- public GitHub repositories (MIT license)
- public footprint (Twitter / Discord / email)
- resource applications (GPU credits / grants)
- lightweight infrastructure (workspace tooling)

### 3.2 Out of scope

- commercial product / startup entity
- claim of full AGI (we ship sub-modules, not a finished AGI)
- human mentorships / team expansion (unless PhD path is taken)
- production services (no API hosting)

---

## 4. Technical approach

### 4.1 4-layer architecture v2

```
+---------------------------------------+
|        SELF-MODEL (Project A)         |
|     meta-cognition + monitor          |
+---------------------------------------+
            |                ^
            v                |
+--------+ +---------------+ +---------+
| SENSORS| | WORLD MODEL   | | LLM     |
|        | | (Project C)   | | (D: type|
|        | | + object-centric | system) |
|        | |   (slot)      | +---------+
+--------+ +-------+-------+
                |
                v
       Planner (hierarchical + value-guided)
                v
       Executor (VLA-grounded)
                v
       Feedback -> Sensors
                v
       Neuro-Symbolic verify (Project E: P2)
                v
       Cross-domain check (Project B)
```

### 4.2 Mapping to 5 routes

| layer | documents route | reference |
|---|---|---|
| LLM | Scaling + Neuro-Symbolic | PaLM-E / Scallop |
| WM-PL | World Models | DreamerV3 / Causal-JEPA |
| VLA | Embodied | V-JEPA 2-AC / Cosmos |
| Self-Model | Cognitive Arch | SOAR / ACT-R / LIDA |
| Verify | Neuro-Symbolic | Lean / Z3 / CEGIS |

### 4.3 Key design decisions

| decision | resolution | status |
|---|---|---|
| decoupled training monitor | frozen-policy critic | DECIDED (DEC-003) |
| first environment | CartPole-v1 | DECIDED (DEC-002) |
| second environment | LunarLander or Pong | PENDING (DEC-004) |
| compute path | CPU start + GPU credits | in-progress |
| KPI framework | Chollet learning curves | ESTABLISHED this charter |
| route strategy | single bet vs fusion | FUSION (this charter) |
| Project E | P2 launch | PENDING (DEC-007) |

---

## 5. Organisation

### 5.1 Team today

| role | count | contribution |
|---|---|---|
| PI (user) | 1 | decision / review / public identity |
| Codex (AI RA) | 1 | code / lit review / paper draft / experiments |
| critique partners | 0 | to be established (target >= 2 by M6) |

### 5.2 Collaboration protocol

- Codex has no cross-session memory; see AGENTS.md
- every new session opens with PROGRESS.md
- every Codex output marks REVIEW-ME points for the user
- decisions live in decisions/DEC-XXXX.md

### 5.3 Decision protocol

- P1 (deadline <= 30 days): explicit in chat, user must reply
- P2 (30-90 days): written ADR, user comments
- P3 (long-term): discussed in quarterly review

---

## 6. Milestones and deliverables

### 6.1 Quarterly milestones (Year 0)

```
Q1 (M1-3)   - knowledge load + Project A PoC outline + 1 lit review
Q2 (M4-6)   - 1 arXiv paper out + >= USD50/mo compute channel
Q3 (M7-9)   - 1 workshop submission + Project C exp start
Q4 (M10-12) - Year 0 review + paper accepted (workshop) + >=30 GitHub stars
```

### 6.2 Deliverable inventory

| category | item | status |
|---|---|---|
| code | Project A v1 (CartPole) | PASS smoke test |
| code | Project C baseline (CRL) | PENDING |
| lit | WM / Dreamer / MuZero / JEPA / Causality reviews | 1 / 5 done |
| paper | Project A v1 (arXiv) | outline done |
| paper | Project C v0 (arXiv) | PENDING |
| community | Twitter / Reddit / Discord / Email | drafts done |
| compute | HF / Google / Lambda credit applications | drafts done |
| tools | workspace CLI (agi-status) | PAUSED |

### 6.3 Acceptance states

- DRAFT    -> written but user has not reviewed yet
- REVIEWED -> user gave >= 1 feedback
- SHIPPED  -> public / submitted

---

## 7. Resources

### 7.1 Compute evolution

| phase | monthly budget | source |
|---|---|---|
| Y0 Q1-Q2 | USD 0 (CPU only) | local |
| Y0 Q3-Q4 | USD 50-100 | Vast.ai / credits |
| Y1 | USD 200-500 | lab / grants / PhD |
| Y2+ | USD 1000+ | frontier-lab access / industry |

### 7.2 Time investment

- user: full-time (assumed)
- Codex: actual session time
- critique / collaboration: 4-6 hr/day AI + 2 hr/week real humans

### 7.3 Money

- Y0: USD 0 spend (free tiers only)
- Y0.5 onward: apply credits (HF / Google / Lambda)
- Y1+: grants USD 1k-10k

### 7.4 Headcount expansion

| trigger | action |
|---|---|
| Y0 Q4 | add 1 freelance collaborator |
| Y1 Q2 | PhD or join lab (default) |
| Y2 Q1 | 1-2 master interns |

---

## 8. Risks and mitigations

### 8.1 Technical risks

| risk | severity | mitigation |
|---|---|---|
| Pearl L3 unsolvable in 3 years | high | narrow claim to L2+ |
| GPU access never materialises | high | pivot to theory-only |
| decoupled monitor only on narrow tasks | medium | publish negative result |
| WM <-> LLM interface not formalisable | medium | hybrid loose-coupling first |

### 8.2 Execution risks

| risk | mitigation |
|---|---|
| user burnout at Y0.5 / Y2 valley | quarterly review detects signals |
| Codex output drift | PROGRESS.md forced update protocol |
| 6 months no public output | Kill Switch discussion triggered |

### 8.3 External risks

| risk | mitigation |
|---|---|
| frontier lab locks the milestone | pick narrow sub-modules |
| GPU supply crisis | keep theory-only fallback |
| AI safety crackdown | safety section in every paper |

### 8.4 Kill Switch

Triggers (any of):
- 6 months with no public output
- 0 compute persistent >= 12 months
- 0 real-researcher interaction >= 12 months
- user not energised opening this folder

Allow "this project ends". Not failure, just data.

---

## 9. Evaluation / KPIs

### 9.1 Chollet framework KPIs

| metric | formula | Y0-end target |
|---|---|---|
| N-shot transfer efficiency | episodes to reach 95% baseline on OOD task | <= 200 ep |
| cross-domain transfer ratio | data(T') / data(T) after learning T | <= 0.4 |
| novel causal extraction | # causal mechanisms from N labelled interventions | >= 5 (N=50) |
| self-monitoring accuracy | monitor AUROC predicting failure (held-out) | >= 0.7 |
| public footprint | GitHub stars + arXiv cites + Twitter followers | >= 50 + 1 + 50 |

### 9.2 Quarterly review protocol

At end of every quarter:
1. milestone check: hit / missed
2. compute delta: start -> end
3. network delta
4. Kill Switch evaluation
5. decide next 3 P1 items

Template: 00_daily/YYYY-Qn-review.md

---

## 10. Signatures

| role | signature | date |
|---|---|---|
| PI (user) | _pending_ | 2026-07-25 |
| RA (Codex) | _auto-signed on generation_ | 2026-07-25 |

> Once signed, all subsequent decisions anchor on this document.
> Revisions go through the ADR process (decisions/).

---

## Appendix A: Reading list

- Ha & Schmidhuber 2018 - World Models
- Hafner 2020/2021/2024 - Dreamer V1/V2/V3 (Nature 2025)
- Schrittwieser 2020 - MuZero
- LeCun 2022/2024 - JEPA / V-JEPA 2-AC
- Causal-JEPA (arXiv 2602.11389) - Feb 2026 - key new evidence
- V-JEPA 2-AC (arXiv 2506.09985) - Jun 2025 - cross-domain proof
- Bareinboim 2016 / 2024 - Causal Transportability
- Scholkopf 2021 - Causal Representation Learning
- Chollet 2019 - On the Measure of Intelligence (ARC-AGI)
- Sutton & Barto 2017/2024 - RL textbook
- Pearl 2018 - Book of Why

## Appendix B: Tooling

- bin/agi-status.py - workspace CLI (paused)
- AGENTS.md - collaboration protocol
- PROGRESS.md - cross-session state
- ROADMAP.md - 5-year roadmap v2
- decisions/DEC-XXXX.md - decision records

## Appendix C: Decision record index

| ID | topic | status | deadline |
|---|---|---|---|
| DEC-001 | PhD vs independent | OPEN | 2026-09-30 |
| DEC-002 | Project A first env | DECIDED | - |
| DEC-003 | Project A main claim | DECIDED | - |
| DEC-004 | second env | PENDING | M3 |
| DEC-005 | Y1 lab join (default yes) | OPEN | M12 |
| DEC-006 | grant pitch refresh | OPEN | M2 |
| DEC-007 | add Project E | OPEN | M2 |

## PI Signature

> **PI**: 刘泽文 (Liu Zewen)
> **Date**: 2026-07-25
> **Signature**: ＿＿＿＿＿ (手签 / handwritten)
>
> This charter is signed in commitment to the 5-year AGI research
> program AGI-2026-001. All subsequent decisions reference this
> document. Revisions go through the ADR process (decisions/).

> **Copyright (c) 2026 刘泽文 (Liu Zewen)**. Released under MIT License
> with attribution requirement. See LICENSE.