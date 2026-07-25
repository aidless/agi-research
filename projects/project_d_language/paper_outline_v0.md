# Project D — Paper Outline v0 (Language as Type System)

> 2026-07-25. Outline for Project D, championed by the 4-layer architecture.
> Goal venue: ICLR 2027 Workshop on Language Models OR NeurIPS 2026 Workshop
> on Grounded Language Learning. Length: 8 pages + appendix.

---

## Title candidates

1. **Type-Lifted World Models: A Language Interface for Latent Predicates**
2. **Predicate Typing: Composing Language and World-Model Latents**
3. **Symbolic Abstractions over Latent Dynamics: When Language Meets Slot-WM**

## 0. Falsifiable Hypotheses

### H1 (primary): type-lifting improves LM planning
**Claim**: a small LM, when given typed predicates over a slot-WM's latent state,
plans better rollouts than the same LM given free-form text descriptions of the same state
on Procgen tasks. Metric: success rate of LLM-generated plan executed in real env.

### H2 (transfer): types generalise across games
**Claim**: a type vocabulary learned on Procgen games A-D transfers to held-out games
E-H with finite LM context; the LM-only baseline degrades more.

## Abstract (~180 words)

We argue language can serve as a **type system over latent world-model predicates**,
not just an action-output layer. We construct *predicate types* from slot-WM latents:
each slot has a learned continuous-vector predicate; types are constraints like
"predicates of class (transitive verb) compose", "predicates of class (container noun)
must be cause-effect with (intransitive verb)". An LM receives as input not raw
action sequences but typed predicate operations. Across Procgen 16-game tasks,
our typed-language LM achieves X% higher success on structured planning tasks
than LM-only baselines; the typed interface also transfers across games more
robustly. We argue that this is the architecture the field has been missing
to bridge LLM semantic understanding and world-model dynamics.

## 1. Introduction

### 1.1 The grounding gap (1 page)
- LLMs plan in language but lack grounded perception
- WMs perceive but lack compositional language
- PaLM-E / Gato stitch the two but only as text-conditioned action output
- We argue the gap is the *type* layer: language that *names* latent predicates

### 1.2 The type-lifting hypothesis (3 paragraphs)
- A latent slot predicate p_i represents an object-property
- The type of p_i says how p_i combines with other predicates
- The LM reasons in type space, not in raw action space
- This is a richer interface than "describe state as text"

### 1.3 Contributions
1. Architecture: a type vocabulary + lifting rules + LM-on-types
2. Empirical demonstration on Procgen 16-game transfer
3. Open-source implementation
4. Clear composition with Projects A, C, E

## 2. Related Work

### 2.1 LLM + world model hybrids
- PaLM-E (Driess 2023): continuous sensor tokens to LLM
- Gato (Reed 2022): multi-task generalist
- V-JEPA 2-AC (Carreira 2025): robot action from video

### 2.2 Type systems in ML
- Neural module networks (Andreas 2016)
- Typed neural programs (Chiang 2020, Neuro-Symbolic)
- Concept learning (Lake 2015)

### 2.3 Cognitive architecture
- ACT-R declarative + procedural split
- LIDA global workspace
- SOAR production systems

### 2.4 The project context
- Project C slot-WM provides the latent predicates
- Project E verifier validates type-based plans
- Project A Monitor grounds type-prediction quality

## 3. Method

### 3.1 Setting

We have Project C's slot-WM: K slots, each with a continuous predicate vector.
We want to lift these predicates into a *type vocabulary*, and have an LM
operate on (type, predicate) pairs.

### 3.2 Predicate types

Types are clusters of predicates that compose similarly. Concrete examples:

| type                | example predicates | composition rule |
|---------------------|--------------------|------------------|
| transitive verb     | push, pull, kick   | pairs with object-typed noun |
| intransitive verb   | jump, fall, stop   | requires subject-typed noun |
| container noun      | box, jar, basket   | contains object nouns |
| object noun         | coin, key, ball    | target of containers |

Types are learned via:
- clustering slot predicates with similar role signatures (small MLP classifier)
- rules are induced via constrained optimization

### 3.3 Type-lifted language interface

The LM receives as input:
```
(slot_i has type(t)) AND
(slot_j has type(container)) AND
(action transition expresses verb(push, slot_i, slot_j))
-> predictions over next slot states
```

This is *not* free-form text; it's structured predicate logic.

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

## 4. Experiments

### 4.1 Tasks
- 16 Procgen games (paper env)
- Plus ARC-AGI as Chollet-style abstract reasoning probe

### 4.2 Baselines
1. **LM-only**: text descriptions of state, LM predicts actions
2. **WM-only**: full planning in latent space, no language
3. **PaLM-E-style**: text + continuous embeddings, joint training
4. **Ours**: type-lifted interface

### 4.3 Metrics
- Task success rate
- Plan coherence (matches WM's predicted rollout)
- Cross-game transfer (train on A-D, test on E-H)
- Compute per task

### 4.4 Results (placeholder)
- Type vocabulary successfully covers slot predicates in 4/4 train games
- Our interface beats LM-only by X% on difficult planning tasks
- Transfer degrade: ours -8%, LM-only -25%

## 5. Discussion

### 5.1 When type-lifting works
- When slot predicates have consistent role structure
- When the LM can be trained on a manageable type vocabulary

### 5.2 Limitations
- Type induction is empirically driven; not principled
- K (slot count) bottleneck
- LLM context length limits for many-step plans

### 5.3 Connection to AGI

This is the **integration layer** that brings the architectural pieces together.
Without type-lifting, projects A, C, E are isolated. With it, they form the
basis for multi-step reasoning across abstraction levels.

## 6. Conclusion

A type vocabulary over slot predicates is the missing piece for connecting LLM
reasoning and world-model dynamics. The architecture composes naturally with
Projects A (Monitor), C (slot-WM), E (verifier) into a coherent AGI substrate.

## Appendix

A. Type vocabulary induction algorithm
B. LM training data construction
C. Compute budget
D. Source code

## What needs to happen next

1. Run Project C slot-WM on Procgen to obtain slot predicates
2. Implement type induction (simple version: k-means on slot predicates)
3. Construct LM training data from typed-predicate rollouts
4. Compare type-lifted LM with baselines
5. Submit to ICLR 2027 workshop

## Status: outline only. Implementation deferred to Year 1.
