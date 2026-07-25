# ROADMAP.md --- 5 Year Research Programme (v2, 2026-07-25)

> **Versioning note**: v1 of this doc was written 2026-07-24 with a
> world-model-as-sole-cognitive-layer architecture. After the user's
> self-critique and the user's WM-PL analysis (2026-07-25), we're at v2.

---

## 0. Anchor question (5 years, unchanged)

Can we build a system with explicit causal world models whose latent
predicates are typed via language, evaluated by neuro-symbolic verification,
whose self-model computes its own failure modes, all while transferring
zero-shot across physical environments --- and ship at least one
reference piece in 5 years?

## 1. Revised architecture (v2, 4-layer)

`
                   Self-Model  (Project A: meta-cognition)
                         |
                         v
+----------+   +----------------+   +----------+
| Sensors  |-> | World Model    |<- | LLM       |
|          |   | (Project C)    |   | (Project D |
|          |   | + object-      |   | as type    |
|          |   |   centric)     |   | system)    |
+----------+   +--------+-------+   +-----+-----+
                       |                  |
                       +-----+------------+
                             v
                  Planner (hierarchical, value-guided)
                             |
                             v
                  Executor (VLA-grounded)
                             |
                             v
                  Feedback -> Sensors
                             |
                             v
                  Neuro-symbolic verification  (NEW: Project E)
                             |
                             v
                  Cross-domain transfer check (Project B)
`

**4 layers**:
1. **LLM** --- semantic + language-as-type-system (Project D)
2. **WM-PL** --- imagination + planning (Project C + A)
3. **VLA** --- grounded action in physical world (Project B)
4. **Neuro-symbolic** --- causal formal verification (Project E - NEW)

The missing layer is **Project E: Neuro-symbolic verification**.
This was implicit in the user's doc but not in our prior portfolio.

## 2. Updated project priorities (v2)

| Project | Priority | Coupling |
|---|---|---|
| A: Self-Improvement (decoupled Monitor) | P0 | coupled with C |
| C: Causal World Model | P0 | coupled with A, E |
| D: Language-as-type-system | P0 | coupled with C |
| B: Cross-domain (VLA-grounded) | P1 | emerges from A+C+D |
| E: Neuro-symbolic verification | P2 (new) | enables true A+C |

**Action**: add Project E as a P2 research-track project. We do not
start it until C has at least one open-source CRL baseline.

## 3. Reading-list update (2025-2026 wave) --- **MUST READ IN 30 DAYS**

| Paper | Why | Path |
|---|---|---|
| Causal-JEPA (arXiv 2602.11389) | Object-level causal intervention in latent | C, E |
| V-JEPA 2-AC (arXiv 2506.09985) | Video pretrain -> zero-shot Franka | B, C |
| JEPA-WM (arXiv 2512.24497) | What drives physical planning success | C |
| Value-Guided JEPA (arXiv 2601.00844) | Value gates action in JEPA | C, A |
| UniZero (arXiv 2406.10667) | Scalable latent WM | C |
| Dreamer V3 (Nature 2025) | First zero-shot Minecraft diamond | B (SOTA ref) |
| Scholkopf 2021 "Causal Representation Learning" | Theory for C | C |
| von Kugelgen 2021 "Self-Supervised CRL" | Theory for C | C |
| Bareinboim 2016/2024 SCM-ID / Causal Transportability | Theory for C, E | C, E |
| Lightman 2023 "Let`s Verify Step by Step" (arXiv 2305.20050) | PRM canonical, our Monitor is PRM-style | A |
| Snell 2024 "Scaling LLM Test-Time Compute Optimally" (arXiv 2408.03314) | TTC scaling laws, BoN+Monitor free gain | A |
| Zelikman 2022 STaR (arXiv 2203.07859) | LLM cousin of decoupled Monitor | A |
| Alonso 2024 DIAMOND (arXiv 2402.03522) | Diffusion WM alternative to DreamerV3 | C |
| Shumailov 2023 "Self-bias propagation" | Synthetic data collapse warning | C, B |
| Burns 2023 "Weak-to-Strong Generalization" (OpenAI) | Monitor policy scaling is OK | A, E |

**Old reading list** (still valid, lower priority):
World Models 2018, Dreamer V1/V2/V3, MuZero, JEPA papers 2022-2024,
Pearl's ladder.

## 4. Scenario re-rating

- A: WM-PL as AGI cognitive-layer core (the 4-layer approach) --- **55%**
- B: Absorbed as implicit WM in next-gen LLM --- **25%**
- C: Independent WM-PL to AGI --- **5-10%** (user doc says 20%; Codex
     disagrees because assumes 3 simultaneous breakthroughs)

## 5. Kill-switch signals (unchanged from v1)

Same six-month stalemate rules. Add one new signal:
- If after 12 months we have not read any of the 2026 wave papers in full,
  the programme is not running seriously.
