# Pre-Registration: H11 (Cross-environment transfer of H10)

> Pre-registered: 2026-07-28 (BEFORE H10 result is known)
> Project: Project G -- LLM Self-Monitoring
> Author: Liu Zewen (with Codex agent)
> Status: SPEC only, no implementation run yet. Pre-registration
>         is filed BEFORE H10 verdict is known, so the H11 decision
>         rule cannot be biased by the H10 outcome.

---

## H11 hypothesis

**Statement**: If H10 (frozen Monitor > joint Monitor on LLM traces)
is VALIDATED on the original LLM (e.g., Qwen-1.5B) and dataset
(e.g., GSM8K-style math), then H11 tests whether the **decoupling
advantage transfers** to a different (a) reasoning dataset (e.g.,
MATH or another formal-reasoning dataset) and (b) a different frozen
LM (e.g., Phi-3-mini or another small LM).

The H11 hypothesis is **only meaningful after H10 is run**. If H10
is REFUTED, H11 is moot and should be replaced by a different
follow-up.

## Decision rule (pre-registered)

**H11 is VALIDATED** if H10 was VALIDATED AND ALL of:
1. On the new dataset: Frozen Monitor AUROC > Joint Monitor AUROC
   by delta > 0.05
2. On the new dataset: Welch t > 2.0 on n=5 seeds
3. On the new frozen LM: Frozen Monitor AUROC > Joint Monitor AUROC
   by delta > 0.05
4. On the new frozen LM: Welch t > 2.0 on n=5 seeds
5. Negative control: Random Monitor signal AUROC < Frozen Monitor
   AUROC on both new setups

**H11 is REFUTED** if H10 was VALIDATED AND ANY of:
- Frozen < Joint on the new dataset (decoupling does NOT transfer)
- Frozen < Joint on the new LM (decoupling is Qwen-specific)
- Frozen == Random on either new setup

**H11 is INCONCLUSIVE** if:
- H10 was VALIDATED, AND
- Frozen > Joint on both new setups by delta > 0.05 BUT t < 2.0
  at n=5

Inconclusive case: extend to n=15 (pre-registered extension).

## If H10 is REFUTED

If H10 is REFUTED (frozen ~ joint on Qwen+GSM8K), then H11 is
**moot** and the agent should pre-register a different follow-up,
e.g.:

- **H11b**: Slot-Monitor vs raw-history Monitor on LLM traces
  (does slot attention help in the LLM domain, independent of
  the decoupling question?)
- **H11c**: 2-step self-improvement loop on LLM traces (Monitor
  re-trained on updated LLM, does it improve?)
- **H11d**: Cross-domain generalization of DLR predicates to LLM
  reasoning (Project E meets Project G)

The choice depends on the H10 outcome:
- If H10 is REFUTED on Qwen+GSM8K: try H11b (slot attention
  ablation) -- it's the cleanest null-follow-up.
- If H10 is REFUTED on Phi-3-mini+MATH: try H11d (DLR transfer).
- If H10 is REFUTED on both: pivot to a different direction
  entirely (e.g., Project F multi-agent LLM rollouts).

## Pre-registered experimental design (assuming H10 VALIDATED)

**Environment**: 2 new setups, each n=5 seeds:
- Setup A: New dataset (e.g., MATH or CommonsenseQA), same LM
  (Qwen-1.5B or whatever was used in H10)
- Setup B: New LM (e.g., Phi-3-mini), same dataset (GSM8K-style
  or whatever was used in H10)

**Architectures (3 arms, same as H10)**:
1. Frozen Monitor
2. Joint Monitor
3. Random Monitor (negative control)

**Per-seed sample size**: 200 rollouts train, 50 rollouts eval
(matches H10).

**Decision rule aggregation**:
- Run H11 on Setup A: 5 seeds, compute delta and Welch t
- Run H11 on Setup B: 5 seeds, compute delta and Welch t
- H11 is VALIDATED only if BOTH setups pass the decision rule

**Total compute**:
- 2 setups x 3 arms x 5 seeds = 30 Monitor training runs
- Plus 10 random-Monitor evaluations (no training, just evaluation)
- Estimated compute: ~30 minutes on GPU, ~3 hours on CPU

## Pre-registered sample size

n=5 per arm per setup. Extension to n=15 if inconclusive.

## What this pre-registration commits to

1. **No silent extensions**: any change to n, to the new
   dataset/LM choice, or to the failure-label definition will be
   documented in a follow-up pre-registration file before data
   collection resumes.
2. **H11 is contingent on H10**: if H10 is REFUTED, H11 is moot
   and the agent pre-registers a different follow-up.
3. **Both setups must pass** for H11 to be VALIDATED. A single
   negative result on one setup is enough for REFUTATION.
4. **Negative results are first-class**: if H11 is REFUTED, the
   paper will report it with the same precision as a positive
   result.

## What this pre-registration does NOT commit to

- The specific new dataset (MATH vs CommonsenseQA vs other). The
  user may pick based on availability.
- The specific new LM (Phi-3-mini vs Llama-3.2-1B vs other). The
  user may pick based on availability.
- The failure-label definition for the new dataset. The label
  generator file has placeholders; user may customize.

These choices will be documented when the experiment is run, but
they do not change the decision rule.

## Why H11 is interesting (the framing)

H10 tests whether decoupled Monitors help in the LLM domain at all.
H11 tests whether the **transfer properties** of the decoupling
principle are similar to those in classical RL.

In classical RL, the decoupling principle (H1) is **LunarLander-
specific in part**: cross-env on CartPole is inconclusive (PPO too
saturated) and on MountainCar is untestable (PPO fails to converge).
The Y1.3 finding (+50 on LunarLander) is also LunarLander-specific.

If H11 is VALIDATED, the decoupling principle is **more portable**
in the LLM domain than in classical RL. This would be a positive
result on the principle itself.

If H11 is REFUTED, the decoupling principle is **just as brittle
in LLMs as in classical RL**. This is also a publishable finding.

Either outcome advances the understanding of the decoupling
principle.

## Relation to existing hypotheses

- **H10** (Project G): prerequisite for H11. If H10 is REFUTED,
  H11 is moot.
- **H1** (Project A): the source principle. H11 tests whether H1''s
  cross-env limitation applies in the LLM domain.
- **H9** (Project A, OPEN): self-improvement loop. H11c is the LLM
  analogue of H9.
- **Y1.3** (Project A): the +50 LunarLander finding. H11 tests
  whether Y1.3''s env-specificity applies in the LLM domain.

## NO_SELF_DECEPTION.md compliance

This pre-registration follows the project's anti-self-deception
protocol:
1. **Decision rule is pre-committed** (above).
2. **Negative control is included** (Random Monitor).
3. **Mechanism hypothesis is stated** (joint Monitor gradient
   dragged by LLM update, leading to brittle failure concept).
4. **Replication plan is stated** (n=5 -> n=15 if needed).
5. **Limitations are acknowledged** (H11 is contingent on H10).
6. **Boundary on what is NOT claimed**: explicitly stated.
7. **Contingency plan is stated**: if H10 is REFUTED, H11 is moot
   and replaced by H11b/c/d.

---

*Pre-registration filed 2026-07-28 BEFORE H10 result is known.
File: experiments_log/2026-07-28-PRE-REGISTERED-H11.md*