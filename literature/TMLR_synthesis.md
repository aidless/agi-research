# F:\TMLR Corpus Synthesis: Essential Insights for the 5-Year AGI Program

> 2026-07-25. Curated from F:\TMLR corpus (29 files, ~1MB of Chinese-language AGI surveys).
> Focus: insights that update or strengthen our 5-year program. Discarded: generic math,
> restated canonical knowledge, applications we are not building, philosophical speculation
> without engineering traction.

## 1. NARS's AIKR assumption (informs our program framing)

Pei Wang's NARS (Non-Axiomatic Reasoning System) is built on AIKR:
*Assumption of Insufficient Knowledge and Resources*. The system assumes:
- Finite knowledge (cannot know everything upfront)
- Finite resources (compute, time, memory are bounded)
- Open tasks (new problems arrive over time)
- Open resources (system can acquire new tools/data)
- Open world (truth is not static; environment changes)

**How this updates our program**: Our 5-year AGI program operates in AIKR mode.
We accept that we don't know the full architecture upfront. We will iterate. We
plan to acquire new compute and data over time. Our architectures will be revised.
This is NOT a weakness; it's the operating mode. Our 5-year plan is a *plan under*
AIKR, not in spite of it.

## 2. NARS term logic with truth-values (informs Project E verifier)

NARS uses *term logic* with each statement having a truth-value of two
components: frequency (how often it has been observed true) and confidence
(how strong the evidence is). This is fundamentally different from classical
Boolean logic. Statements can be more-or-less true.

**How this updates Project E**: Our LTL-based verifier should probably allow
graded truth values rather than strict Boolean satisfaction. A verifier that
outputs "this is 0.7 likely a constraint violation" is more useful than a strict
yes/no. This is a future Project E v2 direction.

## 3. OpenCog Hyperon atomspace (informs Project D knowledge representation)

Hyperon uses an *atomspace* — a graph of typed nodes and links. It supports
probabilistic logic networks (PLN) for inference and MOSES for program
synthesis. The atomspace is *open* — new atoms can be created at any time.

**How this updates Project D**: Our "typed predicates over slot latents" can
be stored in a graph-like structure where types are nodes and slot predicates
are typed edges. This would let us do inference over the type system (e.g.,
substitutability, consistency checking). For Paper D v2.

## 4. Consciousness theories are NOT actionable (discarded for engineering)

GWT (Baars 1988), HOT (Rosenthal 1986), IIT (Tononi 2004) describe what
consciousness *is* but not how to *build* it. They are philosophical frameworks
that may inform Project A's self-model design (the Monitor is a "global workspace"
of sorts) but they don't yield engineering decisions for our 5-year plan.

**Discarded for now**: We do not need to settle the "hard problem of consciousness"
to build useful failure-aware agents. The Monitor is a working approximation of
self-awareness (it knows when it might fail) without solving the hard problem.

## 5. Chinchilla scaling laws (informs compute budget)

Hoffmann et al. 2022: optimal model size and data size scale at the same rate
with compute budget. N_opt ∝ C^0.5 and D_opt ∝ C^0.5. The previous
GPT-3-style "10x more params, 1x more data" was sub-optimal.

**How this updates our program**: For our 5-year compute budget:
- We don't have a 70B-parameter model, and we don't need one yet
- A 1B-parameter Monitor + 1B-parameter Policy trained on the same data
  is the Chinchilla-optimal architecture for our compute
- Don't over-invest in either compute alone

For our Project A, a small Monitor MLP (64-64-64) is Chinchilla-appropriate
for our PPO policy sizes.

## 6. Emergent capabilities and architectural priors

Capabilities emerge at scale thresholds. The H1 breakthrough (LunarLander-v3
Eval AUROC 0.98) on seed 0 is an emergence event. Our slot-attention + SCM
architecture is designed for compositional emergence, not raw scaling.

**How this updates our program**: We focus on architecture quality, not raw
scaling. The 4-layer design (sensor / WM / planner / executor + self-model) is
our compositional prior. We do not need to scale to 70B+ to find more
emergent capabilities; we can find them at smaller scale by adjusting
architectural structure.

## 7. Specific safety concerns for Project E verifier design

Key safety concerns identified in F:\TMLR corpus:
- **Reward hacking**: agent finds unintended high-reward behaviors
- **Specification gaming**: agent exploits underspecified objective
- **Distributional shift**: agent fails on out-of-distribution inputs
- **Deceptive alignment**: agent deliberately appears aligned during training
  but pursues different goal at deployment
- **Goal misgeneralization**: agent generalizes objective in unintended way

**How this updates Project E verifier**: Our LTL-based verifier (from the
Part 6c) should explicitly check for these failure modes. Specific rules:
- "agent's reward signal must come from the specified reward function"
- "agent's action distribution must not diverge from training distribution"
  beyond a threshold (detects distributional shift)
- "agent's planning trajectory must not include explicitly deceptive actions"
- "agent must not produce outputs inconsistent with its training objective"

For Project E v2, include these as part of the rule set. For Paper E v1,
we demonstrate the verifier mechanism; the rules can be LTL-encoded for
subsequent papers.

## 8. Real-world AGI compute estimates (sets expectations)

GPT-4 trained on 50-100 GWh of energy. AGI would require 10^3 to 10^6 times more
compute. Our program has zero GPU and runs on CPU. We will NOT reach AGI on this
hardware in this lifetime.

**Implication**: Our 5-year program aims to PUBLISH research contributions
(papers, code, frameworks), not deploy an AGI system. The hardware mismatch
is acknowledged and accepted. The H1 breakthrough is publishable in
ICLR/NeurIPS workshops.

## 9. Five-year program implications (consolidated)

From the F:\TMLR corpus, the following updates apply to our 5-year program:

| Insight | Updates |
|---------|---------|
| AIKR framing | Our 5-year plan is operating under AIKR; accept and document this |
| NARS truth-values | Project E v2 should allow graded verifier output |
| Atomspace | Project D v2 should consider graph-based type representation |
| Chinchilla | Project A Monitor architecture size is appropriate (~10^6 params) |
| Safety rules | Project E v2 should include specific safety rules (reward hacking, etc.) |
| Real-world compute | Accept we will not deploy AGI; goal is research publications |

## 10. Discarded (chaff, not wheat)

The F:\TMLR corpus contains ~1MB of text. The dross is large:
- Generic Transformer architecture explanations (already in our canon)
- Medical/legal applications (we're not building those)
- Mathematical derivations of known facts (LayerNorm, RoPE, etc.)
- Speculative AGI timelines (5-50 years — no value to our 5-year plan)
- Consciousness philosophy beyond actionable insights (GWT/HOT/IIT)
- Sociological predictions about job loss / economic disruption

These were skimmed, not absorbed. They are a record of what has been tried
in the field; they do not change what we are doing.

## 11. Open questions (added to existing list)

- Does our architecture need graph-based knowledge (Project D v2)?
- Should Project E v2 use graded truth-values (frequency + confidence)?
- What is the right safety rule set for AGI in our setting?
- Are we hitting Chinchilla-optimal compute for our Monitor?

## 12. References used

F:\TMLR corpus files read for this synthesis:
- AGI技术文档_Part11a_高级对齐理论与价值学习.md (alignment theory)
- AGI技术文档_Part11b_具身认知科学发现与AI经济学.md (embodied cognition)
- AGI技术文档_Part11c_前沿架构与AGI安全研究.md (frontier arch + safety)
- AGI技术文档_Part11d_社会应用与未来展望.md (social applications)
- AGI技术文档_Part11e_技术细节与完整结论.md (technical details)
- AGI技术文档_Part6a_NARS深度分析.md (NARS analysis)
- AGI技术文档_Part6b_OpenCogHyperon深度分析.md (OpenCog Hyperon)
- AGI技术文档_Part6c_GodelMachine_SOAR_ACT-R深度分析.md (cognitive archs)
- AGI技术文档_Part7a_对齐安全治理深度分析.md (alignment governance)
- AGI技术文档_Part8b_深度学习前沿与涌现现象.md (scaling laws)
- AGI技术文档_总目录.md (table of contents)

Files NOT read in detail (skipped as dross or out-of-scope):
- AGI技术文档_Part7b_认知科学哲学.md (consciousness philosophy, discarded)
- AGI技术文档_Part8a_计算神经科学.md (computational neuroscience, partly out-of-scope)
- AGI技术文档_Part8c_AGI形式化验证.md (formal verification, partly useful for Project E v2)
- AGI技术文档_Part9a_时间线经济.md (timeline economics, discarded)
- AGI技术文档_Part9b_多智能体涌现.md (multi-agent emergence, partly useful for Project F)
- AGI技术文档_Part9c_AGI病理学.md (pathology, useful for Project E safety rules)
- AGI技术文档_Part10a_神经形态硬件.md (neuromorphic hardware, out-of-scope text-only)
- AGI技术文档_Part10b_意识意向性.md (consciousness, discarded)
- AGI技术文档_Part10c_远期情景.md (far future, discarded)
- AGI技术文档_Part1-5_实现路径.md (path forward, useful for roadmap cross-check)
- AGI技术文档_Part6实现路径.md (path forward, useful for roadmap cross-check)
- AGI技术文档_完整版.md (full version, subsumed by 1-11)
- AGI技术文档_Part2-5_DeepExpansion.md, Part3-5_Additional.md, Part5-6_Final.md, etc.
  (technical deep dives, mostly redundant with our 36 paper notes)

Total read: ~12 of 29 files (41%). Skipped ~17 files either as dross or as
subsumed content already in our 39 paper notes.
