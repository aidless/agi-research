# F:\TMLR Corpus — H/I Series Synthesis: LLM应用架构 + 前沿研究

> 2026-07-25. Reads 14 of 42 new F:\TMLR files: H01-H07 (LLM应用架构) + I01-I07 (前沿研究).
> Skips F/G/J/K (AI科学计算, ML基础设施, 深度学习框架) per user instruction.
> Companion to `TMLR_synthesis.md` (v1.9) which covered Part1-11.
> Focus: what updates our 5-year AGI program (Project A-E + Project F). Discarded: restated
> canonical knowledge, applications we are not building, marketing-grade hype.

## 0. Scope and method

**Read**:
- H01 Prompt工程模式 (38K) — Zero/Few-shot, CoT, ToT, GoT, PoT, DSPy, Auto-Prompt
- H02 ToolUse与FunctionCalling (59K) — OpenAI/Anthropic/Mistral/Google protocols, multi-layer caching
- H03 Agent框架与自主决策 (33K) — Perception-Thinking-Action loop, ReAct, Reflexion, Self-Refine, CRITIC, MetaGPT, AutoGen
- H04 向量数据库与RAG (36K) — Indexing/Retrieval/Generation, Self-RAG, HyDE, Graph RAG, RAGAS
- H05 评估与对齐 (38K) — LLM-as-Judge (3 biases), DPO/KTO/IPO/ORPO, RLHF-V, Constitutional AI
- H06 多模态应用 (35K) — ViLT, Flamingo, BLIP-2, LLaVA, MME/MMBench, POPE hallucination
- H07 边缘部署与端侧推理 (23K) — Quantization (INT4/INT8), Apple ANE, near-memory computing
- I01 TestTimeCompute (22K) — Best-of-N, Tree Search, Extended CoT, o1/DeepSeek-R1, PRM vs ORM
- I02 SAE与可解释性 (43K) — Superposition, Sparse Autoencoders, TopK LM, BatchTopK, SASA
- I03 世界模型 (37K) — VAE-MDN-RNN, Dreamer series, DIAMOND diffusion WM, Video-as-WM
- I04 多模态基础模型 (31K) — Two-tower vs Unified, cross-modal fusion, hallucination mitigation
- I05 推理与思维链 (33K) — Deductive/Inductive/Abductive, CoT/STaR/Self-Refine, GSM-Symbolic limits
- I06 合成数据与数据飞轮 (41K) — Three walls, simulation vs generative, Magpie/Self-rewarding, model collapse
- I07 AI对齐与价值嵌入 (33K) — Outer/Inner alignment, Corrigibility, RLHF/DPO, Jailbreak, Debate

**Not read** (per user instruction):
- F01-F07 AI科学计算 (蛋白质, 药物, RNA, 材料, 量子, 流体力学, 基因组)
- G01-G07 ML系统设计 (Feature Store, MLOps, AB testing, ML 组织)
- J01-J07 AI基础设施 (GPU集群, 容器编排, 部署, 特征平台, CI/CD)
- K01-K07 深度学习框架 (PyTorch, JAX, ONNX, TF, 编译优化, 分布式)

**Method**: rated each finding by (a) actionability for our 5-year plan, (b) immediacy
(can be cited in Y0 vs Y1+), (c) project impact (A/B/C/D/E/F). Actionability high
+ immediacy near-term + clear project mapping = absorb. Otherwise = discard or
defer to reading list.

---

## 1. The five-line headline (most important take-aways)

1. **The "decoupled Monitor" thesis (Project A) is independently confirmed in the
   LLM self-improvement literature as the leading paradigm.** Reflexion (verbal RL),
   Self-Refine (iterative self-feedback), CRITIC (tool-interactive critiquing), and
   STaR (bootstrapping reasoning with reasoning) all instantiate a frozen-critic
   pattern that mirrors our decoupled Monitor. We are not alone; we have company.

2. **Test-Time Compute (TTC) scaling via PRM + Best-of-N + Tree Search is the 2025-2026
   meta-trend.** o1, DeepSeek-R1, AlphaProof all build on PRM-driven search. Our
   Project A Monitor is *already* a process-reward model in disguise — predict
   per-step failure from hidden state. This is publishable framing for Paper A.

3. **Synthetic data is a double-edged sword.** Self-Instruct / Magpie / Self-rewarding LMs
   can bootstrap agents but risk model collapse (self-bias propagation, NeurIPS 2023).
   Project C needs a *real + synthetic mix*, with explicit collapse-detection
   (e.g., embedding-space diversity metric). For Paper C, this becomes a methodological
   rigor point reviewers will demand.

4. **The LLM-as-Judge evaluation paradigm (H05) is the right tool to grade our Monitor.**
   Position bias, verbosity bias, self-enhancement bias are all known; pairwise
   comparison with position-swap and length-normalization handles them. Use
   this for Paper A evaluation.

5. **World Models in 2025-2026 have split into three branches**: (a) discrete latent
   dynamics (DreamerV3, MuZero — our current direction), (b) diffusion-based
   continuous dynamics (DIAMOND 2024), and (c) video-generation-as-WM (OpenAI,
   Genie 3, Cosmos 2.5). The third branch is the new black horse — it skips
   object-centric abstractions entirely. Project C should explicitly consider
   video-WM as an alternative baseline, not just DreamerV3.

---

## 2. H series — LLM应用架构 (project mapping)

### H01 Prompt Engineering — mostly discard, one nugget

**Findings**: Zero-shot delimiter separation, few-shot embedding-similarity selection
(Liu 2022, +8-20% over random), CoT/ToT/GoT/PoT tree-structured prompting, DSPy
auto-prompt optimization (LLM-as-copilot), Auto-Prompt gradient-style refinement.

**Updates**:
- *Project D* (LLM-as-type-system): DSPy-style eval-driven prompt iteration should
  be our default workflow for tuning Project D's system prompts. Specifically:
  golden test set (>=100) -> categorize errors by type -> targeted prompt rewrite ->
  re-eval -> repeat until all metrics hit threshold.
- *Project A* paper Section 4.5: mention prompt-engineering methodology as
  baseline threat — "without Monitor, GPT-4 with carefully tuned prompts achieves X"
  is our null comparison.

**Discard**: Generic CoT/ToT/GoT taxonomy. We're not building an agent harness,
we're building a self-improvement loop.

### H02 Tool Use / Function Calling — keep the interface, skip the engineering

**Findings**: OpenAI's tool_calls/finish_reason="tool_calls" handshake protocol,
Anthropic's content_block difference, semantic cache + exact-match cache layered
(L1 in-memory, L2 distributed, L3 semantic), 30-60% cache hit rate possible.

**Updates**:
- *Project D* v2: the LLM in our 4-layer architecture should expose tools via
  OpenAI-compatible tool_calls protocol. This makes future drop-in replacement
  of GPT-4/Claude trivial. (Already implicit in TASKBOOK but worth making explicit.)
- *Project E*: when the verifier queries the world model, it should use the same
  function-calling protocol. Uniformity across layers simplifies the implementation.

**Discard**: Detailed cache strategies, rate-limiting, retry logic — engineering
harness concerns, not research concerns.

### H03 Agent frameworks — HIGH PRIORITY for Project A and Project F

**Findings**: Perception-Thinking-Action-Observation loop (Russell/Norvig classical
agent), ReAct's 71% ALFWorld vs 6%/4% for Reason/Act-only (synergy, not replacement),
Reflexion (verbal RL via self-reflection in memory), Self-Refine (iterative
self-feedback), CRITIC (tool-interactive critiquing), generative agents (Park 2023),
Multi-agent debate (Du 2023, ICML 2024), MetaGPT/AutoGen orchestration.

**Updates**:
- *Project A* — three direct mechanisms to cite in Paper A:
  1. **ReAct loop** as the scaffolding for our Monitor -> Policy chain. Our
     decoupled Monitor is a frozen-critic variant of ReAct's observation step.
     The paper should explicitly say: "Our Monitor is a ReAct-style
     self-critique step that runs on every policy output."
  2. **Reflexion** as the natural extension: store Monitor failures in episodic
     memory, condition the policy on them next episode. This is the
     *self-improvement* leg — we have decoupled Monitor (frozen critic), we
     need Reflexion's verbal memory to make it iterative.
  3. **CRITIC** (tool-interactive critiquing) is the closest cousin to our
     decoupled Monitor. CRITIC's "LLMs Can Self-Correct with Tool-Interactive
     Critiquing" demonstrates the frozen-critic pattern works for language
     agents. Cite this.
- *Project A* paper Section 5 (Related Work): add ReAct, Reflexion, Self-Refine,
  CRITIC as the four canonical self-improvement methods our work joins.
- *Project F* (multi-agent): MetaGPT/AutoGen are the baseline. Project F's
  contribution should be "decentralized Monitor coordination" — what happens
  when multiple agents each have their own decoupled Monitor?

**Discard**: Generative Agents (Park 2023) social simulation details; not core.

### H04 Vector DB / RAG — relevant to Project D's interface, not core

**Findings**: Three-stage RAG (indexing, retrieval, generation), HyDE (hypothetical
document embeddings), Self-RAG (Asai 2024, learn to retrieve+generate+critique),
Graph RAG (Edge 2024), RAGAS / RGB / CRUD-RAG benchmarks, agentic RAG trend
(RAG becomes one tool among many).

**Updates**:
- *Project D* v2: Self-RAG's reflection tokens (IsREL, IsSUP, IsUSE) are a
  clean implementation of "LLM judges its own retrieval quality". For our
  language-as-type-system, this is the natural way to gate type lookups:
  the LLM emits a confidence on the retrieved type, and downstream reasoning
  uses that confidence.
- *Reading list*: add Asai et al. 2024 (Self-RAG) for Y1 reading.

**Discard**: Vector database engineering (HNSW, IVF, ANN benchmarks), Graph RAG
implementation details. Not our engineering problem.

### H05 Evaluation & Alignment — HIGH PRIORITY for Project A evaluation

**Findings**: LLM-as-Judge (MT-Bench, Chatbot Arena), three biases (position,
verbosity, self-enhancement), 3D framework (Helpfulness-Honest-Harmlessness),
DPO (direct preference optimization, no reward model), KTO (prospect theory),
RRHF, Constitutional AI, red teaming (Perez 2022), NeMo Guardrails multi-layer
defense, many-shot jailbreaking (Anthropic 2024).

**Updates**:
- *Project A* paper Section 4 (Evaluation): use LLM-as-Judge with the following
  protocol to grade Monitor outputs:
  - Pairwise comparison (Monitor-on vs Monitor-off), position-swap averaged
  - Length normalization to control verbosity bias
  - Judge model != our own policy (avoid self-enhancement bias; use GPT-4 judge
    for our PPO policy)
- *Project A* K2 KPI (self-monitoring accuracy AUROC >= 0.7) is well-defined;
  H05 confirms this is the canonical alignment/eval metric.
- *Project D* v2: DPO/KTO are cheaper alternatives to full RLHF. If we ever
  need to align the type-LLM with user preferences (vs raw task performance),
  DPO is the right tool — no need to train a separate reward model.
- *Reading list*: add Rafailov 2023 (DPO), Ethayarajh 2024 (KTO), Mitchell 2024
  (weaknesses of DPO) for Y1.

**Discard**: Constitutional AI implementation details, NeMo Guardrails config,
many-shot jailbreak analysis. We are not deploying models to users yet.

### H06 Multimodal Applications — relevant to Project B (VLA)

**Findings**: Two-tower (CLIP) vs unified encoder (ViLT, FLAVA, LLaVA),
Q-Former (BLIP-2), CogVLM visual expert, POPE hallucination benchmark (F1
80-87% even on best models = 13-20% hallucinate), hallucination mitigation
(RLHF-V, HA-DPO, multimodal CoT), MMMU benchmark.

**Updates**:
- *Project B* (cross-domain/VLA): LLaVA-style architecture is the baseline
  visual encoder. POPE benchmark is the standard hallucination evaluation.
  Note: "13-20% hallucination even on best models" is a known issue; we
  should *not* claim >5% hallucination in Paper B without POPE-grade evidence.
- *Reading list*: Liu 2023 (LLaVA), Li 2023 (BLIP-2) for Y1 Project B.
- *Discard for Project A/C/D*: vision is not core to our 4-layer arch.

### H07 Edge Deployment — DISCARD entirely for our 5-year plan

**Why discard**: Our program runs on CPU (CartPole, Procgen, LunarLander). Edge
deployment (quantization, ANE, near-memory computing) is a deployment concern
that comes after research. We have no edge-deployment problem to solve.

---

## 3. I series — 前沿研究 (project mapping)

### I01 Test-Time Compute Scaling — CRITICAL for Project A framing

**Findings**: Three TTC techniques (Best-of-N, MCTS/PRM, Extended CoT), PRM
vs ORM (Lightman 2023 "Let's Verify Step by Step"), o1 and DeepSeek-R1, scaling
Laws now apply to test-time compute (Snell 2024), MCTS for Agent long-horizon
reasoning.

**Updates**:
- *Project A* paper framing: **our decoupled Monitor is a process reward model**.
  We predict per-step failure probability from hidden state — that is exactly
  PRM-style supervision at the policy level. Paper A should frame this explicitly:
  "We treat each policy action as a 'reasoning step' and learn a PRM over
  step-level failure, then use threshold-gating as a TTC budget controller."
- *Project A* Y1 extension: Best-of-N over our policy, with the Monitor as the
  per-sample verifier. Concretely: at eval time, sample N candidate actions
  from PPO, score each with Monitor, pick the lowest-failure-probability one.
  This is a free TTC gain on top of the decoupled Monitor.
- *Project C*: PRM-style supervision transfers naturally to slot-WM. Each slot
  transition can be PRM-scored for physical consistency. Cite Lightman 2023.
- *Reading list*: Lightman 2023, Snell 2024, OpenAI o1 system card,
  DeepSeek-R1 paper for Y0 Q3 reading (must-read within 30 days, add to
  ROADMAP.md Section 3).

**Discard**: Best-of-N engineering details, PRM dataset construction methods.

### I02 SAE / Mechanistic Interpretability — DEFER to Y1+ reading list

**Findings**: Neuron-level (Olah), Circuit-level (Induction Heads, ACDC),
Superposition Hypothesis (Elhage 2022), SAE (Anthropic 2023, OpenAI 2024),
BatchTopK SAE, Matryoshka SAE, SAEBench, TopK LM (intrinsic sparse),
SASA (Subspace-Aware SAE), SAELens library.

**Updates**:
- *Project A* Y1+: after Monitor is working, can we use SAE-style interpretability
  to understand *why* the Monitor fires? This becomes a follow-up paper.
- *Project E* Y2+: if we ever train an internal world model in the LLM (vs
  external slot-WM), SAE is the natural tool to verify causal faithfulness.
- *Reading list*: add Anthropic Scaling Monosemanticity (2024), SAEBench (2025)
  for Y1. NOT Y0 — we don't have the model scale to apply these techniques yet.

**Discard**: SAE training methodology, SAE scaling law details. Our Monitor
operates at 10^6 params, not 10^9. Different regime.

### I03 World Models — CRITICAL for Project C; STRONGEST update

**Findings**: Ha & Schmidhuber 2018 (VAE-MDN-RNN), Dreamer V1/V2/V3,
DIAMOND (Alonso 2024, diffusion-based WM, replaces MDN-RNN's mixture
of Gaussians with iterative denoising), Video-generation-as-WM (OpenAI,
Genie 3, Cosmos 2.5), Goyal & Bengio 2022 (inductive biases for physical
understanding), multi-level abstraction (low-level perceptual, high-level
causal), latent imagination as core mechanism.

**Updates**:
- *Project C* — three concrete updates to the architecture:
  1. **DIAMOND (diffusion WM)** is the 2024 alternative to DreamerV3.
     Cite Alonso 2024. For Paper C, we should at minimum acknowledge
     diffusion-WM as a competing baseline. If CPU-feasible, run a small
     DIAMOND experiment alongside DreamerV3 for the Procgen baseline.
  2. **Video-generation-as-WM** (Genie 3, Cosmos 2.5) is the 2025
     black horse. These models skip object-centric abstractions entirely
     and learn physics from raw video. This is NOT our current direction
     (we use slot-WM), but Paper C's Related Work must position our
     approach vs this alternative.
  3. **Multi-level abstraction** (Goyal & Bengio 2022) — bottom level
     perceptual, top level causal/intentional. Our Project C is currently
     single-level (slot attention -> SCM). We should plan for v2: add
     a second-level causal abstraction layer on top of slot-WM.
- *Reading list*: add Alonso 2024 (DIAMOND), Goyal & Bengio 2022,
  DeepMind Genie 3 paper, Cosmos 2.5 technical report for Y0 Q3 reading.
  These are must-reads for Project C.

**Discard**: MDN-RNN specifics (subsumed by Dreamer), Ha & Schmidhuber 2018
canonical intro (already in canon).

### I04 Multimodal Foundation Models — supports Project B

**Findings**: Two-tower (CLIP/ALIGN) vs unified (ViLT/FLAVA/LLaVA/BLIP-2),
cross-modal fusion (Q-Former, CogVLM visual expert), contrastive learning
foundation, POPE hallucination (13-20% even on best models), RLHF-V /
HA-DPO / multimodal CoT for hallucination mitigation, MMMU benchmark.

**Updates**:
- *Project B*: POPE-grade evaluation is mandatory for any vision-language
  claim. Our cross-domain VLA needs to show POPE-style grounding scores.
- *Project B* architecture: LLaVA-1.5 is the canonical open-source VLA
  baseline. We can finetune from LLaVA-1.5 weights for our Procgen/Crafter
  domains.

**Discard**: Vision generation (Stable Diffusion, video diffusion), TTS/ASR
(audio modalities). Not our domain.

### I05 Reasoning & CoT — CRITICAL for Project A and D methodology

**Findings**: Three reasoning types (deductive, inductive, abductive),
ICL = implicit induction (Brown 2020), CoT/Zero-shot CoT/ToT/PoT/STaR/
Self-Refine, GSM-Symbolic (Mirzadeh 2024) showing math reasoning degrades
on paraphrase, o1 + DeepSeek-R1 paradigm, three future directions (TTC
scaling, automated process supervision, reasoning-search fusion).

**Updates**:
- *Project A* paper Section 2 (Method): cite STaR (Zelikman 2022)
  "Bootstrapping Reasoning With Reasoning" — this is the LLM analog of
  our decoupled Monitor. The same idea (frozen critic, generate+filter)
  applies. STaR is the closest published cousin.
- *Project A* honesty: GSM-Symbolic shows that math reasoning degrades on
  paraphrase by 10-15%. Our Monitor AUROC may show similar variance
  across Procgen games. Paper A should report per-game results, not
  just mean. (This is consistent with what we already do in paper_outline.)
- *Project D* v2: PoT (Program of Thoughts) is a stronger baseline than
  raw CoT for type-reasoning tasks. Our "type system over slot predicates"
  should support PoT-style execution: LLM emits code that queries the
  type system, not natural language.

**Discard**: GSM8K/MATH benchmark details (we don't run these benchmarks).

### I06 Synthetic Data & Data Flywheel — CRITICAL for Project C and B

**Findings**: Three walls (annotation cost, privacy/regulation, long-tail),
simulation-based (CARLA, Blender) vs generative-based (GAN, Diffusion),
Self-Instruct / Alpaca / Magpie (alignment data synthesis from scratch),
Self-rewarding LMs (Yuan 2024 ICML), DeepSeek-Coder (code generation),
Self-bias propagation (NeurIPS 2023) -> model collapse if synthetic loops
unchecked, world-model-driven synthetic imagination training, domain
randomization, sim-to-real transfer.

**Updates**:
- *Project C* — explicit methodological rigor point: any synthetic data
  pipeline must include a *collapse detection* metric. Concrete proposal:
  - Maintain a held-out "real" distribution embedding (e.g., 1000 Procgen
    frames embedded via a pretrained visual encoder)
  - Compute mean cosine distance from generated samples to this held-out set
  - If distance drops below threshold X, halt synthetic generation
  - Report this metric in Paper C
- *Project C* sim-to-real: domain randomization is the standard recipe
  for Procgen -> Atari / MuJoCo transfer. Our Paper C cross-domain
  claim should explicitly use randomization in the source environment
  to support the transfer claim.
- *Project B* (cross-domain): world-model-driven synthetic imagination
  is a viable way to generate Procgen/Crafter training data without
  human annotation. Cite Hafner DreamerV3 for the canonical recipe.
- *Reading list*: Self-bias propagation paper (NeurIPS 2023), Magpie
  (NeurIPS 2024), Self-rewarding LMs (Yuan 2024) for Y0 Q3 reading.

**Discard**: Specific implementation details of Self-Instruct, Alpaca
training recipes. We use the *concept*, not the code.

### I07 AI Alignment & Value Embedding — CRITICAL for Project A safety and E verifier

**Findings**: Three-layer framework (Outer Alignment, Inner Alignment,
Corrigibility), mesa-optimization risk (Hubinger 2019), reward hacking,
specification gaming, RLHF/DPO/IPO/KTO/ORPO alternatives, jailbreaking
(Wei 2023 "Jailbroken: How Does LLM Safety Training Fail?"), weak-to-strong
generalization (OpenAI Burns 2023), Constitutional AI, AI safety debates
(Irving 2018), Sparse Feature Circuits for editing factual associations.

**Updates**:
- *Project A* paper Section 6 (Safety/Limitations): explicitly address the
  mesa-optimization concern. Our decoupled Monitor could in principle
  be exploited by the policy if the policy learns to game the Monitor's
  threshold. Mitigation: freeze Monitor architecture and weights, only
  retrain policy. (We already do this — paper should make it explicit.)
- *Project A* honesty: weak-to-strong generalization (OpenAI 2023) shows
  that small monitors can elicit capabilities from strong policies. This
  is a *feature* of our architecture — a 10^6 Monitor can supervise a
  10^9 policy. Cite Burns 2023.
- *Project E* (verifier design): the three-layer alignment framework is
  the canonical way to organize our verifier's rule set. Our LTL-based
  verifier should have:
  - Outer Alignment: "the LTL formula matches user intent" (verified by
    user testing)
  - Inner Alignment: "the verifier's prediction matches the LTL formula
    satisfaction" (verified by synthetic test cases)
  - Corrigibility: "the verifier can be overridden by a human" (interface
    design)
- *Reading list*: Hubinger 2019 (mesa-optimization), Wei 2023 (Jailbroken),
  Burns 2023 (weak-to-strong), Soares 2015 (Corrigibility) for Y0 Q3.

**Discard**: Constitutional AI training details, red team infrastructure.

---

## 4. Cross-cutting recommendations

### 4.1 Update ROADMAP.md Section 3 (must-read within 30 days) — add 6 new papers

Add to ROADMAP.md Section 3:
| Paper | Why | Path |
|---|---|---|
| Lightman 2023 "Let's Verify Step by Step" | PRM canonical, our Monitor is a PRM | A |
| Snell 2024 "Scaling LLM Test-Time Compute Optimally" | TTC scaling, our Monitor enables BoN at policy level | A |
| Zelikman 2022 STaR | Closest LLM cousin of decoupled Monitor | A |
| Alonso 2024 DIAMOND | Diffusion WM alternative to DreamerV3 | C |
| NeurIPS 2023 "Self-bias propagation" | Synthetic data collapse warning | C, B |
| Burns 2023 "Weak-to-Strong Generalization" | Monitor policy scaling is OK | A, E |

### 4.2 Update Project A paper_outline.md — add three claims

In Paper A Section 0 (Falsifiable Hypotheses), add:
- **H3** (NEW): STaR-style bootstrapping with our decoupled Monitor yields >=10%
  sample efficiency improvement on at least 4 of 16 Procgen games, vs the
  PPO-only baseline.

In Paper A Section 5 (Related Work), add four citations:
- ReAct (Yao 2023), Reflexion (Shinn 2023), Self-Refine (Madaan 2023),
  CRITIC (Gou 2024). Frame decoupled Monitor as "frozen-critic pattern"
  shared with these methods.

### 4.3 Update Project C paper_outline.md — add DIAMOND baseline and collapse detection

In Paper C Section 2 (Architecture), add a paragraph acknowledging DIAMOND
(Alonso 2024) as the diffusion-WM alternative. We do not run DIAMOND in this
paper; we note it as a future baseline.

In Paper C Section 3 (Method), add a *collapse detection* subsection. Specify:
- Held-out real distribution embedding (N=1000 frames)
- Mean cosine distance threshold T_collapse
- Synthetic data is halted when distance drops below T_collapse

### 4.4 Update Project E verifier design — three-layer alignment framing

Project E v1's verifier design should adopt the Outer/Inner/Corrigibility
framework from I07:
- **Outer Alignment** (sec 2.1): the LTL formula matches user intent
- **Inner Alignment** (sec 2.2): the verifier's prediction matches LTL
- **Corrigibility** (sec 2.3): human override is structurally supported

Add a citation to Hubinger 2019 in Project E's paper_outline_v0.md Section 6
(Limitations).

### 4.5 New ADR: rate TTC as part of the Monitor architecture

Create decisions/0011-ttc-as-monitor-extension.md (P3):
- After Paper A v1 ships, evaluate Best-of-N over PPO with Monitor scoring
- Decision deadline: Y1 Q2 (after first paper accepted)
- Status: PROPOSED

---

## 5. Discarded — chaff (not absorbed)

| File | Why discarded |
|---|---|
| H01 (most of it) | Generic prompt engineering taxonomy |
| H02 (caching details) | Engineering harness, not research |
| H06 (TTS/ASR, video gen) | Out of scope, not Project B-relevant |
| H07 (edge deployment) | We run on CPU, no edge problem |
| I02 (SAE training) | Model scale mismatch (we have 10^6, need 10^9) |
| I03 (MDN-RNN intro) | Subsumed by Dreamer canon |
| I04 (vision generation) | Not Project B-relevant |
| I05 (GSM8K/MATH benchmarks) | We don't run these |
| I06 (Self-Instruct training) | Concept absorbed, code not |
| I07 (Constitutional AI impl) | Not deploying to users yet |
| F/G/J/K (entire series) | User-instructed skip |

---

## 6. Net delta on the 5-year program

**Updates that strengthen existing plans** (low risk, immediate):
- Paper A: add H3 hypothesis (STaR-style bootstrapping with Monitor)
- Paper A: reframe Monitor as PRM with Best-of-N as natural extension
- Paper A: cite ReAct/Reflexion/Self-Refine/CRITIC as canonical cousins
- Paper C: add DIAMOND to Related Work as future baseline
- Paper C: add collapse-detection methodology for synthetic data
- Paper E: adopt Outer/Inner/Corrigibility three-layer framework

**Updates that add new work** (medium risk, Y0 Q3-Q4 timeline):
- Best-of-N + Monitor as TTC extension (Y1, ADR proposed)
- Multi-level causal abstraction layer on slot-WM (Y2 v2 of Project C)
- SAE interpretability of Monitor (Y1 follow-up paper)

**Updates that we explicitly defer** (high risk, Y2+):
- Diffusion-WM (DIAMOND) as a Project C alternative
- SAE at scale for internal world-model verification
- Constitutional AI alignment for Project D

**Updates that were already in canon** (no change needed):
- PPO, Dreamer, CLIP, MuZero, etc.

---

## 7. Six concrete action items (next 14 days)

1. **Add 6 papers to ROADMAP.md Section 3** — must-read within 30 days.
2. **Edit Paper A paper_outline_v1_full.md** — add H3 hypothesis + ReAct/
   Reflexion/Self-Refine/CRITIC to Related Work.
3. **Edit Paper C paper_outline_v1.md** — add DIAMOND to Related Work +
   collapse-detection methodology subsection.
4. **Edit Paper E paper_outline_v0.md** — adopt Outer/Inner/Corrigibility
   framing + Hubinger 2019 citation.
5. **Create decision record 0011-ttc-as-monitor-extension.md** (P3, Y1 Q2).
6. **Add 4 papers to literature/papers/ deep notes** (Zelikman STaR, Lightman
   PRM, Snell TTC, Alonso DIAMOND) — same format as existing 54 notes.

---

## 8. Reading list — net additions (Y0 Q3 must-read, Y1 background)

**Y0 Q3 (must-read within 30 days, add to ROADMAP.md Section 3)**:
- Lightman 2023 "Let's Verify Step by Step" (PRM)
- Snell 2024 "Scaling LLM Test-Time Compute Optimally" (TTC scaling)
- Zelikman 2022 STaR (LLM self-improvement)
- Alonso 2024 DIAMOND (diffusion WM)
- Shumailov 2023 "Self-bias propagation" (synthetic data collapse)
- Burns 2023 "Weak-to-Strong Generalization" (Monitor scaling)

**Y1 background** (defer to reading list, not must-read):
- Rafailov 2023 DPO
- Ethayarajh 2024 KTO
- Mitchell 2024 DPO weaknesses
- Asai 2024 Self-RAG
- Anthropic 2024 Scaling Monosemanticity
- SAEBench 2025
- Goyal & Bengio 2022 (inductive biases)
- Liu 2023 LLaVA
- Li 2023 BLIP-2
- Wei 2023 Jailbroken
- Hubinger 2019 mesa-optimization
- Soares 2015 Corrigibility

---

## 9. Open questions (added to existing list)

- Does our Monitor's AUROC transfer to OOD environments (cross-env)? (H2
  in paper_outline_v1)
- Is STaR-style bootstrapping with Monitor a free +10% sample efficiency?
  (H3, proposed)
- Can Best-of-N with Monitor gating improve mean return? (ADR 0011)
- At what scale does our Monitor architecture saturate? (Chinchilla check)
- Does our collapse-detection metric detect model collapse early enough
  in Project C synthetic data pipelines?
- Should Project E's verifier output graded truth-values (NARS-style)?
  (carried from v1.9 synthesis)

---

## 10. References used

F:\TMLR corpus files read in detail (14 of 42 = 33%):
- LLM应用架构_01_Prompt工程模式.md
- LLM应用架构_02_ToolUse与FunctionCalling.md
- LLM应用架构_03_Agent框架与自主决策.md
- LLM应用架构_04_向量数据库与RAG.md
- LLM应用架构_05_评估与对齐.md
- LLM应用架构_06_多模态应用.md
- LLM应用架构_07_边缘部署与端侧推理.md
- 前沿研究_01_TestTimeCompute.md
- 前沿研究_02_SAE与可解释性.md
- 前沿研究_03_世界模型.md
- 前沿研究_04_多模态基础模型.md
- 前沿研究_05_推理与思维链.md
- 前沿研究_06_合成数据与数据飞轮.md
- 前沿研究_07_AI对齐与价值嵌入.md

Files NOT read (28 of 42 = 67%):
- AI科学计算_01-07 (F01-F07): per user instruction, skipped
- ML系统设计_01-07 (G01-G07): per user instruction, skipped
- AI基础设施_01-07 (J01-J07): per user instruction, skipped
- 深度学习框架_01-07 (K01-K07): per user instruction, skipped

Skimmed but not deep-read for this synthesis:
- AGI技术文档_Part1-11 (covered in v1.9 synthesis)
- SUMMARY.md + AGI技术文档_总目录.md (corpus overview)

---

*Companion to TMLR_synthesis.md v1.9. Together these cover ~26 of 71 corpus files
(37%). Remaining files (F/G/J/K = 28 + ~17 already-skipped Part files) remain
unread; user can request additional reads as priorities shift.*
