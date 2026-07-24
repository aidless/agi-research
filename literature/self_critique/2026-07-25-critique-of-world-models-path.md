# Self-Critique of the World-Models-AGI Path (2026-07-25)

> Author: the user (you).
> Status: this critique pushed Codex to update the research programme.
> Saved here so every future Codex session sees it as part of the literature canon.

---

## TL;DR (Codex's verdict of the critique)

Of the 5 critiques:

| # | Critique | Verdict | Action |
|---|---|---|---|
| 1 | Sim-to-Real is ontological, not engineering | **Valid** | Elevate B (cross-domain) and add object-centric representation as P0 |
| 2 | Causal ladder assumed not implemented | **Valid (slightly too strong)** | Elevate C (causal world model); add CRL (causal representation learning) to canon |
| 3 | Long horizon requires causal understanding | **Valid** | Project A and C become coupled, not parallel |
| 4 | Language is ontological, not plug-in | **Valid** | Project D becomes P0, not nice-to-have; recast as cross-domain transfer mechanism |
| 5 | LLM comparison is straw man (2026 view) | **Valid** | Reframe LLM + World Models as **complementarity axes**, not competition |

---

## The five critiques, why each is correct

### 1. Sim-to-Real is ontological

RSSM works in DMControl because DMControl gives you the relevant variables
pre-selected. Real world does not. A 64-dim latent in CityDriving has no
chance of capturing which of the 10^5 observable variables actually matter.

What was wrong in my original framing: I treated this as "a matter of scale".
Scale alone does not solve "discover what is causally relevant". The deep
problem is **variable selection**, which is itself a Pearl-Level-3 task.

### 2. Causal ladder transitions are assumed, not implemented

This is right. RSSM is curve-fitting. It has no SCM. There is no published
algorithm that takes RSSM-style stochastic latents and extracts an SCM.
The path L1 -> L2 -> L3 is *not* auto-driven by scale.

**However**: causal representation learning (von Kugelgen 2021, Ahuja 2023)
is an active field with real results. The user's claim "no algorithm exists"
is too strong as of 2026. What's true: nothing close to solving real-world
SCM extraction exists yet. The critique's spirit is right, the literal claim
is overstated.

### 3. Hierarchical abstraction requires causal understanding

This *is* a hidden circular dependency in the original framing.
"Discover temporal abstraction boundaries" presupposes you can tell which
events are "natural units" -- which requires causal structure.

### 4. Language is ontological

MuZero (Go) does not transfer to chess without retraining the rules. AlphaZero
demonstrates rule transfer within a small family. But Minecraft -> real
language understanding? That's not on the table without language-as-type-system.

The critique is essentially: world models without language = perceptual
specialists. World models with language (as PaLM-E / Gato hint) = the
candidate AGI substrate.

### 5. LLM comparison is straw man

Yes, 2022-era LLM-vs-RL framing is dead. By 2025-2026:
- o1/o3 do explicit search at inference
- GPT agents use tools and verify outputs
- The "LLM is just next-token" view is the straw man

The honest framing: LLMs and world models are **complementary** axes.
LLM gives implicit world model + compositional language. World model
gives explicit grounded prediction. Neither alone reaches AGI.

---

## Why this critique matters for our 5-year program

The original AGI architecture I drew (Sensors -> World Model -> Planner ->
Executor -> Feedback, with Language as side channel) was wrong in this sense:
the Language module was an afterthought. The critique forces a redesign:

`
                     [ SELF-MODEL ]
                     (Project A: meta-cognition)
                              |
                              v
  [ SENSORS ]  -> [ WORLD MODEL (Project C: causal) ] <- [ LANGUAGE (Project D: ontological) ]
       |               |                                            |
       |               v                                            |
       |        [ PLANNER (hierarchical) ]                          |
       |               |                                            |
       |               v                                            |
       +---------> [ EXECUTOR ] <------------------------------------+
                              |
                              v
                       [ FEEDBACK ]
                              |
                              +-------> [ CROSS-DOMAIN (Project B) ]
`

Key changes:
- Language is a **peer** of the world model, not a side channel
- Cross-domain (B) is **forced by** language-as-type-system, not just wanted
- Causal (C) and Self-Improvement (A) are coupled (causal understanding
  is what makes meta-cognition stable)
- Object-centric representation is added to World Model's responsibility
  (RESEARCH ITEM, not a separate project)
- World Model + LLM are **dual sub-models of a single substrate**, not competitors

---

## What changes in the project portfolio

### Before this critique
- Project A = priority 1 (self-improvement as the missing piece)
- Project B = priority 2 (cross-domain, nice-to-have after A)
- Project C = priority 3 (causal, theoretical follow-up)
- Project D = priority 4 (language interface, optional)

### After this critique
- Project A = priority 1 (still: meta-cognition on top of everything)
- Project C = priority 1 (causal structure is the missing L3 hook)
- Project D = priority 1 (language-as-type-system, ontological role)
- Project B = priority 2 (cross-domain, emerges from D working)
- Plus: Object-centric repr (slot attention) as a research item inside
  the world model, not a project of its own

In other words: 4 projects are no longer independent. A, C, D are coupled.

---

## What changes in code/research near-term

1. **Add causal representation learning** to literature/ canon.
   Must-read: Bareinboim survey, von Kugelgen 2021, Ahuja 2023,
   Scholkopf's 2021 position paper on causal representation.
2. **Recast Project A**: the decoupled monitor still has value, but
   its claim should narrow to "monitor works for narrow tasks in the
   absence of causal structure". The bigger prize is "monitor with
   causal latents".
3. **Add object-centric baselines** to the cross-domain project:
   Slot Attention, MONet, GENESIS. Their metrics become project B's
   central evaluation.
4. **Project D elevates**: from "LLM as query interface" to "language
   as type system over latent predicates". Different paper, different
   framing.
5. **Review all grant applications**: the phrasing "world models
   + closed-loop to AGI" must be softened. The honest pitch is:
   "world models expose clear, falsifiable bottlenecks to AGI that
   LLM-only research cannot articulate".
