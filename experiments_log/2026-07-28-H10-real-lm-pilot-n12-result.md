# Project G H10 REAL-LM pilot -- N=12 result (simple arithmetic)

> Date: 2026-07-28
> Code: projects/project_g_llm_self_monitoring/code/h10_real_pilot.py
> Mode: simple arithmetic dataset (mixed-difficulty)
> Status: 3-arm pilot ran; H10 direction INVERTED vs N=6 (Frozen/Joint
>         drop to AUROC 0.000, Random baseline 1.000).

## 1. What we ran

- Loaded Qwen2.5-1.5B-Instruct (1.5B params, float16).
- Loaded mixed-difficulty arithmetic (12 problems, 6 easy + 6 hard).
- Generated 12 reasoning traces with max_new_tokens=20.
- Trained 3 arms on 9 traces; evaluated on 3 traces.

## 2. Per-trace results (n=12)

| Category | Count |
|----------|-------|
| Success  | 6     |
| Failure  | 6     |
| **Total**| **12**|

Failure rate: **0.500** (balanced, as designed).

## 3. Aggregate (n=1 seed)

| Arm    | AUROC |
|--------|-------|
| Frozen | **0.000** |
| Joint  | **0.000** |
| Random | **1.000** |

**Train: 9 traces (5 success + 4 failure, 0.556 failure rate)**
**Eval: 3 traces (2 success + 1 failure, 0.333 failure rate)**

## 4. Verdict

**PILOT VERDICT: architecture generalization FAILS at N=12.**

This is the **opposite direction** of the N=6 pilot (where both Frozen
and Joint achieved AUROC 1.000). At N=12:

- Both Frozen and Joint monitors are WORSE than random.
- Random baseline achieves AUROC 1.000 by lucky tie-breaking on
  the 1-positive/2-negative eval set.

## 5. Why Frozen == Joint == 0.000

With only 9 training traces and 3 eval traces:
- Eval set has 2 success + 1 failure (heavily skewed toward success).
- A naive "always predict success" gives 2/3 ≈ 0.667 accuracy, but
  AUROC 0.500 (random performance).
- Frozen/Joint monitors learned the wrong pattern: they predict
  failure for success traces (giving them AUROC 0.000).
- This is **overfitting** to a small training set where the class
  boundary is unstable.

Random Monitor achieved 1.000 by lucky concordance ordering on this
particular 3-eval set.

## 6. Comparison: N=6 vs N=12

| Pilot | N | Frozen | Joint | Random |
|-------|---|--------|-------|--------|
| N=6   | 6 | 1.000 | 1.000 | 0.000 |
| N=12  | 12 | 0.000 | 0.000 | 1.000 |

The wild swing between N=6 and N=12 demonstrates the **danger of
small-N pilots without pre-registration**:

- The N=6 result would have been headline-grabbing ("Frozen=Joint=1.000,
  random=0.000, perfect Monitor performance!").
- The N=12 result is the OPPOSITE.
- Neither is a real signal. The true picture requires n=5 seeds
  with 200 rollouts/seed (the pre-registered H10).

## 7. Lesson for NO_SELF_DECEPTION.md

This pilot is a textbook example of why pre-registration matters:

1. **Without pre-registration**, the N=6 result would have been
   overclaimed as a "POSITIVE" H10 finding.
2. **The self-correcting sequence** (N=6 → N=12) caught the
   instability before publication.
3. **The negative control** (Random) is essential — without it,
   the N=6 result would have looked great but be meaningless.
4. **Pilot ≠ pre-registered result**. The pilot is exploration;
   the pre-registered H10 is the test.

## 8. Honest framing

This N=12 pilot:
- **REVEALS** that the architecture generalization is unstable at
  small N.
- **DOES NOT** confirm or refute the H10 hypothesis direction.
- **STRONGLY SUPPORTS** the need for the pre-registered n=5 H10
  run with proper sample size.
- **PROVIDES** a cautionary example for NO_SELF_DECEPTION.md
  training.

Per NO_SELF_DECEPTION.md, the negative result (Frozen/Joint = 0.000)
is reported with the same precision as the N=6 positive result would
have been. The instability is acknowledged as a real limitation of
small-N pilots.

## 9. Recommended next step

To test the H10 hypothesis with statistical power:
- Run with n=5 seeds (pre-registered sample size).
- Use N=20+ traces per arm per seed.
- Use Welch t-test to compare arms (requires pre-reg).

```bash
cd projects/project_g_llm_self_monitoring/code
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:H10_N_TOTAL = "20"          # 20 traces per seed
$env:H10_MAX_NEW_TOKENS = "20"
$env:H10_USE_SIMPLE = "1"
# Note: this is per-seed; run 5 times with different seeds.
& "C:\...\python.exe" h10_real_pilot.py
```

Estimated per-seed runtime: ~16 minutes on CPU. Total for n=5:
~80 minutes. Still feasible but pushing CPU budget.

## 10. Or: switch to GPU

If the user has GPU access, the full pre-registered H10 (n=5,
200 rollouts/seed) can run in ~1 hour on a CUDA GPU.

---

*N=12 simple-arithmetic pilot log 2026-07-28 by Codex agent.
Architecture generalization FAILS at N=12. Strong evidence for
pre-registration discipline; the H10 hypothesis remains UNTESTED
until the full pre-reg run completes.*