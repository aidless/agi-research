# Pre-Registration: H12 (Small LM as DLR Type Checker)

> Pre-registered: 2026-07-29 (BEFORE any data collection)
> Project: Project D -- Language as Type System (revival after H10 REFUTED)
> Author: Liu Zewen (with Codex agent)
> Status: SPEC only, no implementation run yet.

---

## Context: why H12

Per `experiments_log/2026-07-29-H10-stratified-n5-result.md`, H10
(decoupled Monitor for LLM self-rewarding) is **direction-REFUTED**
at n=5/N=12 stratified pilot. Per the H11 pre-registration, H11 is
**moot** because H10 is REFUTED.

We are pivoting Project G to Project D: instead of using a small LM
to generate traces for a Monitor to classify, we use a small LM as
a **type checker** for DLR predicates. This combines:
- Project D language-as-type-system (the Y0 direction).
- Project E DLR (the cross-env validated verifier, 97.8% mean accuracy).

The pivot is documented in ROADMAP.md section 3.1 update.

## H12 hypothesis

**Statement**: A small LM (e.g., Qwen2.5-1.5B-Instruct) can serve
as a **type checker** for DLR predicates. Specifically, given a DLR
predicate prediction (continuous-valued) and a candidate label (e.g.,
"upright", "near_ground"), the small LM correctly classifies the
predicate as TRUE / FALSE / UNCERTAIN with accuracy comparable to
the DLR baseline (which uses a learned MLP head, 97.8% mean across
4 envs).

**Pre-registered decision rule**:

H12 is VALIDATED if ALL of:
1. Small LM type-checker accuracy >= 0.85 on held-out test set.
2. Small LM type-checker accuracy is within 0.10 of DLR baseline
   (i.e., the LM is competitive with the learned MLP head).
3. Welch t > 2.0 comparing LM accuracy to a random baseline
   (U[0,1] threshold) across n=5 seeds.

H12 is REFUTED if ANY of:
- Small LM accuracy < 0.70 (significantly worse than DLR).
- Random baseline beats small LM (negative control fails).
- Welch t < 2.0 in absolute value (not significant).

## Why this is novel

The DLR architecture (Project E) uses a learned MLP head to convert
slot-WM latents into predicate predictions. This requires labeled
training data for each environment. A small LM as type checker would:
1. **Zero-shot transfer**: LM can check predicates in environments
   without retraining (LM already has semantic knowledge).
2. **Compositional reasoning**: LM can chain predicates ("if upright
   AND near_ground, then...") using natural language.
3. **Explainable**: LM can output natural-language justifications.

If H12 holds, the DLR + LM combination is a more flexible and
explainable verification system than DLR alone.

## Pre-registered experimental design

**Environment**: LunarLander-v3 (8-dim state, 4 actions). The DLR
attention architecture is validated on this env at 95.5% mean
predicate accuracy.

**Predicates to check** (from DLR cross-env validation):
- "upright" (angle is small)
- "near_ground" (y_pos is small)
- "moving_slow" (velocity is small)
- "stable" (angle AND velocity are small)

**Small LM**: Qwen2.5-1.5B-Instruct (cached locally, CPU-runnable).

**LM prompt** (per predicate check):
```
You are a type checker for a robotics system.
Given the following state vector: [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_l, leg_r]
Is the predicate "<PREDICATE>" TRUE, FALSE, or UNCERTAIN?
Answer with exactly one of: TRUE / FALSE / UNCERTAIN.
```

**Architecture (3 arms)**:
1. **Small LM type checker**: uses Qwen2.5-1.5B-Instruct to check
   each predicate.
2. **DLR baseline**: uses the DLR learned MLP head (Project E).
3. **Random baseline**: predicts TRUE/FALSE/UNCERTAIN uniformly
   at random.

**Training**:
- DLR baseline: use existing trained DLR models (97.8% mean accuracy
  on 4 envs is the pre-registered baseline).
- Small LM: no fine-tuning; zero-shot.
- Random: no training.

**Evaluation**:
- 50 held-out LunarLander trajectories per seed.
- For each (state, predicate) pair, compare LM/DLR/Random predictions
  to ground truth.
- Report per-arm accuracy, then aggregate (mean, std, Welch t).

## Pre-registered sample size

n=5 seeds (matches H10 pre-reg).
Extension to n=15 if inconclusive (pre-registered extension).

## What this pre-registration commits to

1. **No silent extensions**: any change to n, the prompt template,
   or the predicate definitions will be documented in a follow-up
   pre-registration file before data collection resumes.
2. **Negative results are first-class**: if H12 is REFUTED, the
   paper will report it with the same precision as a positive result.
3. **No cherry-picking**: every seed's accuracy reported, including
   anomalies.

## What this pre-registration does NOT commit to

- The specific small LM (Qwen2.5-1.5B vs Phi-3-mini vs other).
  User may swap based on availability.
- The specific predicate definitions (the 4 above are pre-reg
  defaults; can be extended per env).
- The prompt template wording (the above is a starting point;
  can be tuned per env if documented).

## Relation to existing work

- **H10 (Project G)**: REFUTED at n=5/N=12. This pivot replaces
  Project G's "Monitor as LLM signal" direction with Project D's
  "LM as DLR type checker".
- **DLR (Project E)**: VALIDATED at 97.8% mean across 4 envs. H12
  tests whether a small LM can match this performance without
  per-env training.
- **Project D (Y0)**: outlined but not tested. H12 is the first
  experiment under the revived Project D.
- **Y1 paper**: documents Project D as a future direction; H12
  is the first concrete test of that direction.

## NO_SELF_DECEPTION.md compliance

This pre-registration follows the project's anti-self-deception
protocol:
1. **Decision rule is pre-committed** (above).
2. **Negative control is included** (Random baseline).
3. **Mechanism hypothesis is stated** (LM zero-shot transfer).
4. **Replication plan is stated** (n=5 -> n=15 if needed).
5. **Limitations are acknowledged** (CPU-only, small LM choice).
6. **Boundary on what is NOT claimed**: explicitly stated.

---

*Pre-registration filed 2026-07-29 BEFORE any data collection.
File: experiments_log/2026-07-29-PRE-REGISTERED-H12.md*