# Project D - Paper v1 outline (Language as Type System over Latent Predicates)

> 2026-07-25. v1 builds on `paper_outline_v0.md`. Most distinct: frames the
> contribution as "typed types over latent predicates" rather than "language
> as command interpreter". Connects directly to Chollet`s ARC and to
> Franka-style robot arms via V-JEPA 2-AC.

## 0. Falsifiable Hypotheses

### H1 (typed predicates > text):
- **Claim**: an LM, given typed predicates over a slot-WM`s latent state, plans
  better rollouts than the same LM given free-form text descriptions of the same
  state on Procgen tasks. Metric: success rate of LM-generated plan executed in
  real env (zero-shot success).
- **Falsifier**: success rate within 5% of text-only baseline on 4/4 games.

### H2 (type vocabulary transfers):
- **Claim**: a type vocabulary learned on Procgen games A-D transfers to
  held-out games E-H; the LM-only baseline degrades more.
- **Falsifier**: transfer degrade > 0.4 ratio (we fail below 60% of in-domain).

## Abstract (~180 words)

We argue language can serve as a type system over latent world-model predicates,
not just an action-output layer. We construct predicate types from slot-WM
latents: each slot has a learned continuous-vector predicate; types are constraints
like "predicates of class (transitive verb) compose", "predicates of class
(container noun) must be cause-effect with (intransitive verb)". An LM receives
as input not raw action sequences but typed predicate operations. Across Procgen
16-game tasks, our typed-language LM achieves higher success on structured planning
tasks than LM-only baselines; the typed interface also transfers across games
more robustly. We argue that this is the architecture the field has been
missing to bridge LLM semantic understanding and world-model dynamics.

## 1. Introduction (~1.5 pages)

### 1.1 The grounding gap
- LLMs plan in language but lack grounded perception
- World models perceive but lack compositional language
- PaLM-E / Gato stitch the two but only as text-conditioned action output
- We argue the gap is the type layer: language that *names* latent predicates

### 1.2 The type-lifting hypothesis
- A latent slot predicate p_i represents an object-property
- The type of p_i says how p_i combines with other predicates
- The LM reasons in type space, not in raw action space
- This is a richer interface than "describe state as text"

### 1.3 Contributions
1. Architecture: a type vocabulary + lifting rules + LM-on-types
2. Empirical demonstration on Procgen 16-game transfer
3. Open-source implementation
4. Clear composition with Projects A, C, E

## 2. Related Work (~1 page)

### 2.1 LLM + world model hybrids
- PaLM-E (Driess 2023): continuous sensor tokens to LLM
- Gato (Reed 2022): multi-task generalist
- V-JEPA 2-AC (Carreira 2025): robot action from video

### 2.2 Type systems in ML
- Neural module networks (Andreas 2016)
- Typed neural programs (Chiang 2020, Neuro-Symbolic)
- Concept learning (Lake 2015)

### 2.3 Cognitive architectures
- ACT-R (Anderson 2007): declarative + procedural split
- LIDA (Franklin 2006): global workspace
- SOAR (Laird 2012): production systems

### 2.4 ARC and skill-acquisition
- Chollet 2019: ARC-AGI as a measure of generalisation
- Our type-lifting is one route to better ARC performance

## 3. Method (~2.5 pages)

### 3.1 Setting
We have Project Cs slot-WM: K slots, each with a continuous predicate vector.
We want to lift these predicates into a type vocabulary, and have an LM
operate on (type, predicate) pairs.

### 3.2 Predicate types
Types are clusters of predicates that compose similarly. Concrete examples:
- (transitive verb): push, pull, kick - pairs with object-typed noun
- (intransitive verb): jump, fall, stop - requires subject-typed noun
- (container noun): box, jar, basket - contains object nouns
- (object noun): coin, key, ball - target of containers

Types are learned via:
- clustering slot predicates with similar role signatures (k-means on
  learned predicate-feature embeddings)
- rules are induced via constrained optimisation over typed-predicate
  composition frequencies

### 3.3 Type-lifted language interface
The LM receives as input:
(slot_i has type(t)) AND (slot_j has type(container)) AND (action transition
expresses verb(push, slot_i, slot_j)) -> predictions over next slot states

This is *not* free-form text; it is structured predicate logic.

### 3.4 Training
Two losses:
1. Type classification loss (slot -> type)
2. LM next-token loss (given typed predicate context)
End-to-end gradient through both.

### 3.5 Inference: Planning with types
To plan a 5-step rollout:
1. Encode current state via slot-WM -> typed predicates
2. Feed to LM, get typed action sequence
3. Decode back to primitive actions via the type-action mapping
4. Execute

The LM never sees raw pixels; the WM never sees text directly.

## 4. Experiments (~1 page)

### 4.1 Tasks
- 16 Procgen games (paper env)
- ARC-AGI as Chollet-style abstract reasoning probe

### 4.2 Baselines
1. LM-only: text descriptions of state, LM predicts actions
2. WM-only: full planning in latent space, no language
3. PaLM-E-style: text + continuous embeddings, joint training
4. Ours: type-lifted interface

### 4.3 Metrics
- Task success rate
- Plan coherence (matches WMs predicted rollout)
- Cross-game transfer (train on A-D, test on E-H)
- Compute per task
- ARC-AGI pass-rate per task

### 4.4 Results (placeholder)
- Type vocabulary successfully covers slot predicates in 4/4 train games
- Our interface beats LM-only by X% on difficult planning tasks
- Transfer degrade: ours -8%, LM-only -25%
- ARC pass-rate: 30% (vs 0% for LM-only baseline, vs 70% for handcrafted)

## 5. Discussion

### 5.1 When type-lifting works
- When slot predicates have consistent role structure across domains
- When the LM can be trained on a manageable type vocabulary (~10-20 types)

### 5.2 Limitations
- Type induction is empirically driven; not principled
- K (slot count) bottleneck
- LLM context length limits for many-step plans

### 5.3 Connection to AGI
This is the integration layer that brings the architectural pieces together.
Without type-lifting, projects A, C, E are isolated. With it, they form the
basis for multi-step reasoning across abstraction levels.

## 6. Conclusion
A type vocabulary over slot predicates is the missing piece for connecting
LLM reasoning and world-model dynamics. The architecture composes naturally
with Projects A (Monitor), C (slot-WM), E (verifier) into a coherent AGI substrate.

## Appendix
A. Type vocabulary induction algorithm (k-means vs spectral)
B. LM training data construction from typed rollouts
C. Compute budget
D. Source code

## References (key)
- Chollet 2019 - On the Measure of Intelligence (ARC-AGI)
- Driess 2023 - PaLM-E
- Reed 2022 - Gato
- Anderson 2007 - ACT-R
- Franklin 2006 - LIDA
- Lake 2015 - concept learning
- Andreas 2016 - neural module networks
- Project C slot-WM (this work)
- Project B cross-domain transfer (this work)
- V-JEPA 2-AC (Carreira 2025)
- Pearl 2018 - Ladder of Causation
