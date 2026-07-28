# Pre-Registration: H10 (Decoupled Monitor on LLM Self-Rewarding)

> Pre-registered: 2026-07-28 (before any data collection)
> Project: Project G -- LLM Self-Monitoring
> Author: Liu Zewen (with Codex agent)
> Status: SPEC only, no implementation run yet.

---

## H10 Hypothesis (frozen-Monitor on LLM traces)

**Statement**: In a self-rewarding LLM agent (RLHF / DPO / self-
rewarding loop), training a **decoupled** failure-prediction Monitor
on rollouts from a **frozen** LLM produces a more accurate failure
signal than training the Monitor **jointly** with the LLM.

## Decision rule (pre-registered before data collection)

**Success criterion (H10 VALIDATED)** -- ALL of:
1. Frozen Monitor AUROC > Joint Monitor AUROC by delta > 0.05
2. Welch t > 2.0 on n=5 seeds
3. Negative control: Random Monitor signal AUROC < Frozen Monitor
   AUROC by delta > 0.10

**Refutation criterion (H10 REFUTED)** -- ANY of:
- Frozen Monitor AUROC <= Joint Monitor AUROC (no decoupling
  effect in LLM domain)
- Frozen Monitor AUROC == Random Monitor AUROC (the frozen LLM
  trace is not informative for failure prediction)

**Inconclusive criterion** -- ANY of:
- Frozen > Joint by delta > 0.05 BUT t < 2.0 at n=5
- Mixed signal across seeds (some seeds positive, some negative)

If inconclusive: extend to n=15 (pre-registered extension protocol).

## Pre-registered experimental design

**Environment**: GSM8K-style math reasoning traces from a frozen
small LM (Qwen-1.5B or Phi-3-mini class). The LLM is NOT fine-tuned
during the experiment; it is used only to generate rollouts.

**Rollout count per seed**: 200 traces
**Seeds**: n=5 (extensible to n=15 if inconclusive)

**Trace representation**: slot attention on the last 20 (token,
logit) pairs (4 slots x 32 dim). This reuses the Project A
Slot-Monitor architecture, adapted to LLM traces.

**Failure label**: trace ends in incorrect final answer (GSM8K
ground truth). This is a clean, deterministic label with no
ambiguity.

**Architectures (3 arms)**:
1. **Frozen Monitor**: trained on rollouts from frozen LLM, then
   evaluated on held-out traces.
2. **Joint Monitor**: trained in parallel with a simulated LLM
   update (gradient step on LLM loss + Monitor loss jointly).
3. **Random Monitor**: untrained, random U[0,1] signal as
   negative control.

**Training**:
- Frozen Monitor: 50 epochs, BCE loss, Adam lr=1e-3, batch size 64
- Joint Monitor: same loss + simulated LLM gradient step every 5
  epochs (50 LLM steps total, lr=1e-5 on a frozen-embedding
  perturbation as a proxy for LLM update).
- Random Monitor: no training.

**Evaluation**:
- 50 held-out traces per seed.
- AUROC computed on (failure_prob, is_failure) pairs.
- Reported per-seed, then aggregate (mean, std, Welch t).

## Pre-registered sample size

n=5 for the primary test. Extension to n=15 is pre-registered
following the Y1.x protocol: if t < 2.0 at n=5 but direction is
positive, extend to n=15 (and again to n=20 if needed) WITHOUT
changing the decision rule.

## Pre-registered exclusions

NONE. We will report every seed's result, including any anomalies.
No seed will be silently dropped.

## What this pre-registration commits to

1. **No silent extensions**: any change to n (sample size), to the
   decision rule, or to the failure-label definition will be
   documented in a follow-up pre-registration file before data
   collection resumes.
2. **Negative results are first-class**: if H10 is REFUTED, the
   paper will report it with the same precision as a positive
   result.
3. **No cherry-picking**: if some seeds show frozen > joint and
   others don't, we report the aggregate honestly.

## What this pre-registration does NOT commit to

- A specific small LM choice (Qwen-1.5B vs Phi-3-mini vs other).
  The user may pick based on availability.
- A specific GSM8K-style dataset. The user may pick GSM8K,
  MATH, or another reasoning dataset.
- A specific training compute budget. CPU is acceptable; GPU is
  faster.

These choices will be documented when the experiment is run, but
they do not change the decision rule.

## Why this is novel

The H1 decoupling result on classical RL (frozen Monitor > joint
Monitor on LunarLander-v3, 5/5 seeds) has not been tested in the
LLM self-rewarding domain. The LLM domain has qualitatively
different failure modes (reasoning failures, logical contradictions,
length overflows) and a qualitatively different training signal
(policy gradient on tokens, not actions).

If H10 holds: the decoupling principle is a more general principle
that applies to LLM agents.
If H10 is REFUTED: the decoupling is classical-RL-specific and does
not transfer to LLM.

Either outcome is publishable.

## Relation to existing work

- **Project A** (H1 ablation, classical RL, 5/5 seeds): the source
  of the decoupling principle.
- **Project E** (DLR + GovBench): the verifier primitives that may
  integrate with H10 in a follow-up.
- **Project D** (language-as-type-system): the LM-as-type-checker
  direction that may benefit from H10.
- **Project F** (Phase 2 multi-agent): if H10 holds, the next
  direction is multi-agent LLM rollouts (H12).

## NO_SELF_DECEPTION.md compliance

This pre-registration follows the project's anti-self-deception
protocol:
1. Decision rule is pre-committed (above).
2. Negative control is included (Random Monitor).
3. Mechanism hypothesis is stated (joint Monitor gradient dragged
   by LLM update).
4. Replication plan is stated (n=5 -> n=15 if needed).
5. Limitations are acknowledged (no GPU may limit model choice).
6. Boundary on what is NOT claimed: explicitly stated.

---

*Pre-registration filed 2026-07-28 before any data collection.
File: experiments_log/2026-07-28-PRE-REGISTERED-H10.md*