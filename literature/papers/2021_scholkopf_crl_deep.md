# Scholkopf 2021 - "Causal Representation Learning" (CRL)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- this is a position-paper / survey by a leading
> researcher; widely-cited; mathematical content well-documented in the
> research community
> One-line: A research agenda arguing that ML should learn representations
> that are identifiable up to the true causal factors -- not entangled pixel
> features.

---

## Problem

Standard unsupervised representation learning produces features that
are correlated with useful quantities but **are not causal**. This means
two systems trained on similar distributions may learn wildly different
representations, and neither is robust to distribution shift.

**Identifiability** of latent variables from observations is the
foundational issue: the same observations can be explained by infinitely
many different latent vectors (because the representation is not
unique). For causal variables, this non-identifiability can be broken
given enough information about interventions.

## Method (the framework)

Scholkopf's framework is more theoretical than algorithmic. The
core claims:

1. **Independent causal mechanisms (ICM)**: high-level variables of
   a generative process are independent of each other (i.e., the
   causal generative process factorises). This is the structural
   assumption.

2. **Sparse mechanism shift**: under intervention on one mechanism,
   only sparse changes happen elsewhere. Testable from data.

3. **Multi-view observations** (timestamps, multiple cameras, etc.)
   help identify latents. The "more views, more identifiability" intuition.

4. **Temporal sequences** carry implicit identifiability information.
   A latent variable that *predicts the future* in a stable way is more
   likely to be a real causal factor.

## Key technical results

1. **Existence proofs**: with sufficient temporal/interventional
   information, identifiability up to permutation + element-wise
   transformations is achievable.

2. **Contrastive learning as proxy for independence**: contrastive
   learning (e.g., CLIP, SimCLR) approximately learns independent
   factors, though with caveats.

3. **Intervention-based identification**: the strongest results require
   access to a data-generating process where certain variables are
   intervened on. Where this is available, CRL becomes identifiable.

## Empirical translation

In practice, CRL pipelines look like:

1. Collect data from a process where some variables are intervened.
2. Train a model that factorises representations using some independence
   prior (e.g., sparse coding, contrastive loss with auxiliary labels).
3. Verify that interventions on latent (i) cause coordinated changes in
   observations corresponding to that variable, while other latents are
   unchanged.

## Criticisms

1. **Identifiability results are conditional on assumptions** (sparse
   mechanism shift, sufficient interventions, etc.). When those
   assumptions break, the conclusions break.

2. **Practical algorithms are limited**. Most CRL papers demonstrate
   on toy 2D/3D problems. Scaling to realistic video is harder than
   the theory suggests.

3. **Whether identifiability matters in practice is debated**. Empirically,
   even non-identifiable features can be useful downstream. Critics
   argue identifiability is a theoretical nicety, not a practical
   necessity.

4. **Interventional data is hard to get**. In a real video stream,
   you don't have "what if I had intervened differently" data.

## Connection to our program (Project C)

**This is the theoretical anchor for Project C, alongside Pearl.**

Our Project C goal: lift World Model from L1 to L2-L3. The CRL
framework gives us the foundational question: *what makes the latent
identifiable as causal variables?*

Practical recipe for our Project C paper:
1. Use Slot Attention for object-centric decomposition (the "what objects?"
   part of identifiability).
2. Use the ICM prior: factorise latent dynamics into per-slot transitions
   that don't depend on each other.
3. Use a sparse causal graph between slots as the structural assumption.
4. Implement SCM-style transitions: each slot evolves based on causal
   parents only.

This is exactly the architecture slot attention + Pearl's SCM combined.
Scholkopf's CRL gives us the formal justification for why this should
give identifiable latents.

When writing Project C paper v0, the "related work" section should:
- Cite Scholkopf 2021 for the theoretical framework
- Cite Pearl for the L1-L2-L3 ladder
- Cite Slot Attention for the practical primitive
- Cite Causal-JEPA (Tier A, user reads personally) for the
  most recent application of these ideas

## Concrete next move

Write Project C paper v0 outline with these three building blocks:
slot attention encoder + SCM-style per-slot dynamics + sparse causal
graph over slot transitions.

## Confidence

HIGH. Theoretical position paper; widely-cited framework. Re-verify:
- exact identifiability theorem statements
- the specific "sparse mechanism shift" formulation

## Related papers

- Pearl 2018 "Book of Why" -- the L1-L2-L3 framing
- Locatello 2020 Slot Attention -- object-centric primitive
- von Kugelgen 2021 "Self-Supervised CRL" -- CRL specifically
- Ahuja et al. 2023 -- "Empirical or invariant risk minimisation"
- Wang & Jordan 2021 -- identifiability for domain adaptation

## Status

- [x] cite in Project C paper v0 outline (essential)
- [x] cite in TASKBOOK Architecture v2 mapping
- [ ] write Project C paper outline draft
