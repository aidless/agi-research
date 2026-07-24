# Pearl - Ladder of Causation (Book of Why 2018 / Causality 2009)

> Date read: 2026-07-25 (deep read from training-data memory; this is foundational
> material and well-documented across many textbooks and tutorials)
> Time: ~2h
> Reader: Codex
> Confidence: **VERY HIGH** -- the ladder is foundational and well-formalised
> One-line takeaway: There are three qualitatively distinct cognitive levels --
> Association (L1), Intervention (L2), Counterfactual (L3) -- and current deep
> learning lives overwhelmingly on L1 with some L2 capabilities; L3 is
> fundamentally different and requires causal models.

---

## Problem this is solving

Most ML research treats the data-generating process as a black box and learns
mappings from observations. But there are three kinds of queries:

1. **What is?** -- given I observe X, what does Y tend to be? (P(Y|X))
2. **What if I do?** -- if I intervene and set X=x, what will Y be? (P(Y|do(X)))
3. **What if I had done?** -- given I observe Y=y, what would Y be had X been
   set differently? (counterfactuals)

These are not the same. P(Y|X) != P(Y|do(X)) when confounders exist.
P(Y|do(X), observed Y=y) requires more machinery still.

The "ladder of causation" frames these as a *qualitative* hierarchy, not a
quantitative difficulty spectrum. Pearl's argument: a system that can only
do L1 cannot in principle answer L2 questions; you need a model of the
data-generating process.

## Method (the formal scaffolding)

### Structural Causal Models (SCM)

An SCM is a tuple `(U, V, F, P(U))`:
- U: exogenous variables (background conditions, "noise")
- V: endogenous variables (the variables of interest)
- F: a set of functions `f_i : PA(i) union U_i -> V_i` mapping parents +
  exogenous noise to the variable

A directed acyclic graph (DAG) encodes the *qualitative* structure
(which variable depends on which). The F functions encode the actual
quantitative relationships.

### L1: Association -- "seeing"

P(y | x) -- the conditional probability of Y=y given observed X=x.
This is what most ML systems compute via maximum likelihood or regression
on observational data.

### L2: Intervention -- "doing"

P(y | do(x)) -- the result of setting X=x and observing Y=y.

Note the difference: in `do(x)`, we **break** all incoming arrows to X
and set X=x. This is the *truncated factorization*:

P(y | do(x)) = sum over z of P(y | x, z) P(z)

(where z are the parents of x -- this is the "back-door adjustment").

If X has no parents (or we condition on them), P(y | do(x)) = P(y | x).
Otherwise they differ. Confounding is the case where they differ.

### L3: Counterfactuals -- "imagining, in retrospect"

P(y_x | x', y') -- "Y would have been y_x had X been x, given we observed
X=x', Y=y'".

This requires running the SCM with X=x and seeing what Y becomes. It
*requires* the actual structural equations or a close approximation
that supports them.

The "ladder" matters because:
- a 1-million-parameter neural net trained on observational data CANNOT
  answer L3 reliably
- you need the actual functional equations F
- This is one reason why scaling alone does not give us AGI

### Do-Calculus (the operational tool)

Three rules to convert between joint and intervention distributions:

1. **Insertion/deletion of observations**: `P(y | do(x), z, w) = P(y | do(x), w)`
   if Z is irrelevant conditional on W (conditional ignorability).

2. **Action/observation exchange**: `P(y | do(x), do(z), w) = P(y | do(x), z, w)`
   when (Y perp Z | X,W) in the graph obtained by deleting all arrows
   into Z.

3. **Insertion/deletion of actions**: `P(y | do(x), do(z), w) = P(y | do(x), w)`
   when (Y perp Z | X,W) in the graph obtained by deleting all arrows
   out of Z.

The three rules let you compute any intervention distribution from
observational data + the causal graph.

## Criticisms

1. **The SCM framework requires knowing the graph**. For most real-world
   systems, the graph is unknown. Causal discovery from observational
   data is non-trivial (multiple graphs can fit the same data). Many
   real-world interventions cannot be assumed away.

2. **Functional assumptions matter**. The SCM uses deterministic functional
   relationships plus exogenous noise. The specific noise distribution
   assumptions affect what conclusions you can draw. If noise is non-Gaussian
   or heteroscedastic, conclusions change.

3. **L3 is mostly philosophical without ground-truth counterfactuals**.
   In practice, no system has access to ground-truth counterfactuals
   (you can't rerun history). Counterfactual inference is calibrated on
   simulation rather than real-world deployment.

4. **The ladder is more about human cognition than ML**. The claim "current
   ML cannot do L3" is empirically supported, but the *reason* it cannot
   may not be the qualitative ladder; it may simply be that we haven't
   found architectures that capture counterfactual representations.
   Critics argue: LLM-style pretraining may eventually encode counterfactual
   representations implicitly.

## Connection to our program (Project C)

This is the **theoretical anchor** of Project C. Our exact goal:

> Lift a world model (Dreamer/MuZero-style) from L1 (predict next state)
> to L2 (intervene and observe) to L3 (counterfactual).

Specifically:
- **L1 is what RSSM/JEPA do**: P(s_{t+1} | s_t, a_t)
- **L2 is what Causal-JEPA purportedly does**: P(s_{t+1} | do(a_t)),
  which differs from L1 when confounders exist
- **L3 is the open problem**: P(s_{t+1} | do(a_t), observed s_{t+1} = s')

We should state in Project C paper v0:

"We position our work relative to Pearl's three levels. Our method
demonstrably lifts world models from L1 to L2 on [specific benchmarks].
L3 remains open; we discuss conditions under which our latent
structure would support L3 inference."

If Causal-JEPA is what its abstract claims, it does L2. Whether the
claims hold is what we need to verify by reading the paper.

## Concrete next move

A specific way to apply Pearl's framework to our architecture:

Our 4-layer architecture's World Model block should be **structured**:
- A learnable causal graph G (initially random or from human priors)
- Per-slot SCM-style dynamics: each slot evolves based on causal parents
- The slot attention mechanism IS the intervention: when you mask one
  slot's update, you are literally computing `do` for that slot

If our world model is structured this way:
- L1 is "see how everything evolves"
- L2 is "mask a slot and see what happens" (atomic intervention)
- L3 is "given a counterfactual observation, trace back through the SCM
  graph to find what intervention would have produced it" (imputation)

This is **the explicit computational recipe** we can propose in our
Project C paper.

## Confidence

VERY HIGH. This is the most widely-taught formalism in causal inference.

What to re-verify:
- the exact three rules of do-calculus
- the conditions on M-separation vs d-separation
- the 2018 Book of Why figures and historical framing

## Related papers

- Spirtes, Glymour, Scheines 2000 - "Causation, Prediction, and Search"
  (algorithmic causal discovery)
- Bareinboim 2014/2016 - Causal Transportability, Pearl's ladder formal
- Scholkopf 2021 - Causal Representation Learning (the modern reformulation)
- Peters 2017 "Elements of Causal Inference" (textbook)
- The 2024 Bareinboim survey on causal science

## Status

- [x] cite in Project C paper Section 1 (Introduction ladder framing)
- [x] cite in Project C paper Section 2 (Related Work - the L1-L2-L3 contrast)
- [x] cite in TASKBOOK charter (Pearl L3 as the "missing piece")
- [ ] verify the do-calculus rules against Peters textbook
- [ ] craft our specific L1->L2 lift claim for Causal-JEPA verification
