# Pre-Registration: H12b (Larger LM for DLR Type Checker)

> Pre-registered: 2026-07-29 (BEFORE any data collection on Qwen2.5-7B)
> Project: Project D -- Language as Type System
> Author: Liu Zewen (with Codex agent)
> Status: SPEC only, no implementation run yet.

---

## H12b hypothesis

**Statement**: A larger LM (Qwen2.5-7B-Instruct) achieves higher
predicate-classification accuracy than Qwen2.5-1.5B-Instruct on
the H12 task (small LM as DLR type checker).

## Context

The H12 smoke test (1.5B LM on synthetic arithmetic state-predicate
pairs) showed LM accuracy = 0.500 = Random. The LM failed to parse
the expected output format. H12b tests whether a larger LM can
solve this task.

## Decision rule (pre-registered before any data collection)

**H12b is VALIDATED** if ALL of:
1. Qwen2.5-7B-Instruct accuracy >= 0.70 on held-out test set.
2. Qwen2.5-7B-Instruct accuracy > Qwen2.5-1.5B-Instruct accuracy
   by delta > 0.15.
3. Welch t > 2.0 comparing 7B to 1.5B across n=5 seeds.

**H12b is REFUTED** if:
- 7B accuracy < 1.5B accuracy (scaling makes it worse).
- 7B accuracy = 1.5B accuracy (no improvement from scaling).

**H12b is INCONCLUSIVE** if:
- 7B accuracy > 1.5B by delta in (0.0, 0.15] but < 0.70 absolute.
- Or Welch t < 2.0.

## Pre-registered experimental design

**Environment**: same synthetic arithmetic state-predicate dataset
as H12 smoke test (50 state-predicate pairs).

**Small LM (control)**: Qwen2.5-1.5B-Instruct (already validated in
H12 smoke test).

**Large LM (test)**: Qwen2.5-7B-Instruct (downloaded if not cached;
requires GPU).

**Architecture (2 arms)**:
1. **Qwen2.5-1.5B-Instruct**: zero-shot prompt.
2. **Qwen2.5-7B-Instruct**: same zero-shot prompt.

**Training**: no training; both are zero-shot inference.

**Evaluation**:
- 50 (state, predicate) pairs per seed.
- Per-arm accuracy on the test set.
- Per-seed table; aggregate with mean, std, Welch t.

**Compute**:
- 1.5B on CPU: ~3 sec per call. 50 pairs x 5 seeds x 2 arms = 500 calls.
  ~25 min.
- 7B on GPU (A100/H100): ~0.5 sec per call. ~5 min.
- Total: ~30 min if GPU available, or ~25 min on CPU (1.5B) + ~5 min
  for 7B if GPU.

## Pre-registered sample size

n=5 seeds. Extension to n=15 if inconclusive.

## What this pre-registration commits to

1. **No silent extensions**: any change to the LM, prompt, or
   decision rule will be documented in a follow-up pre-registration.
2. **Negative results are first-class**: if H12b refutes, the
   paper will report it with the same precision as a positive result.
3. **No cherry-picking**: every seed's accuracy reported.

## What this pre-registration does NOT commit to

- The specific larger LM (Qwen2.5-7B vs Llama-3.1-8B vs Phi-3-medium).
  User may swap based on availability.
- The prompt template (kept identical to H12 for direct comparison).
- Whether to use bf16 or float32 for the larger LM (bf16 default
  on A100/H100).

## H12b relation to existing work

- **H12 (Project D)**: original 1.5B pilot. H12b extends to 7B.
- **H12c (alternative)**: prompt engineering on 1.5B. If H12c
  validates, H12b may be unnecessary.
- **Kimi K3 [46]**: 2.8T model shows scale helps structured
  reasoning. H12b tests this hypothesis at a much smaller scale.
- **FlashKDA [47]**: kernel-level optimizations, not directly
  relevant to H12b (which is a model-level question).

## NO_SELF_DECEPTION.md compliance

This pre-registration follows the project's anti-self-deception
protocol:
1. **Decision rule is pre-committed** (above).
2. **Negative control is included** (1.5B baseline).
3. **Mechanism hypothesis is stated** (scaling improves
   instruction following + numeric reasoning).
4. **Replication plan is stated** (n=5 -> n=15 if needed).
5. **Limitations are acknowledged** (no GPU currently; 7B may
   not be enough).
6. **Boundary on what is NOT claimed**: explicitly stated.

---

*Pre-registration filed 2026-07-29 BEFORE any data collection.
File: experiments_log/2026-07-29-PRE-REGISTERED-H12b.md*