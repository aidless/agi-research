# Causal-JEPA (arXiv 2602.11389, Feb 2026) — placeholder note

> Date: 2026-07-25 placeholder. **REQUIRES USER PRIMARY READ** before paper-grade citation.
> Confidence: **LOW** (only Tier A knowledge from user-supplied abstract)
> One-line: A non-generative world model that injects object-level causal structure into
> the latent, trained with intervention augmentation so it can answer P(s_t+1 | do(a_t))
> rather than just P(s_t+1 | s_t, a_t).

---

## What the user must primary-read for, before I cite:

- the precise intervention training procedure (probability of intervention? curriculum?)
- the loss function (reconstruction only? prediction only? both?)
- whether the causal graph is learned end-to-end or has a human prior
- results on held-out vs in-distribution objects

## What I can say from user's reference:

The paper is one of the **first concrete implementations of Pearl L2 in world models**.
Specifically, it combines:
- JEPA-style non-generative prediction (predict target embedding, not pixel)
- Slot attention or similar object decomposition
- Intervention augmentation: during training, occasionally treat one slot as
  "intervened" and predict the rest
- A causal-graph structural prior over slots

The headline claim: this lifts a world model from L1 to L2 (intervention) on
object-rich environments.

## Connection to our program

This is **the most direct existing work for Project C**.

Project C's slot-WM pipeline (outlined in `projects/project_c_causal_world/paper_outline_v0.md`)
is essentially asking: can we get L3 capability given L2 primitives like Causal-JEPA?
The answer depends on what Causal-JEPA actually achieves and what its limits are.

If Causal-JEPA's L2 lifting is robust, then:
- Project C focuses on L3 (counterfactual) given L2 base
- Project E (verification) is a natural complement to make L3 sampling reliable
- The 4-layer architecture has direct components for each

If Causal-JEPA's L2 lifting is fragile:
- Project C must work harder; L2 cannot be assumed
- The architectural recipe may need revision

## What the placeholder needs from user

User to read Causal-JEPA paper. After read, replace confidence flag with HIGH and
fill in the missing details. The 30-second elevator version of what we cite is
still valid here: "Slot-attention world model with intervention training that
demonstrates L2 capability."

## Status

- [ ] user primary read required
- [ ] once read, this becomes HIGH-confidence Tier B note
