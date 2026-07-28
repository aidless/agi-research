# Project G H10 REAL-LM pilot -- n=5 seeds, N=12 multi-seed aggregate

> Date: 2026-07-28
> Code: projects/project_g_llm_self_monitoring/code/h10_real_pilot.py
> Mode: simple arithmetic dataset (mixed-difficulty)
> Status: 5 seeds run end-to-end; multi-seed aggregation complete.

## 1. What we ran

Per-seed real-LM pilot with N=12 traces × max_new_tokens=20:
- Loaded Qwen2.5-1.5B-Instruct (1.5B float16) from local cache.
- Generated 12 mixed-difficulty arithmetic traces per seed.
- Trained 3 arms (Frozen / Joint / Random) on 9 traces; evaluated on 3.
- Ran seeds 0, 1, 2, 3, 4 (n=5, matches H10 pre-reg minimum).
- Aggregated via Welch t-test (pre-reg decision rule).

## 2. Per-seed results (n=5 seeds)

| Seed | Frozen | Joint | Random | Eval class split |
|------|--------|-------|--------|------------------|
| 0    | 0.000  | 0.000 | 1.000  | 2 success + 1 failure |
| 1    | 0.500  | 0.000 | 0.500  | 1 success + 2 failure |
| 2    | NaN    | NaN   | NaN    | 0 success + 3 failure (degenerate) |
| 3    | 1.000  | 0.500 | 1.000  | 2 success + 1 failure |
| 4    | 0.500  | 0.500 | 0.500  | 2 success + 1 failure |

Per-seed variance is HIGH. Frozen ranges from 0.000 to 1.000
across seeds. The 3-trace eval set is too small for stable AUROC.

## 3. Aggregate (n=4 valid seeds, excluding seed 2 NaN)

| Arm    | Mean | Std   | n |
|--------|------|-------|---|
| Frozen | 0.500 | 0.408 | 4 |
| Joint  | 0.250 | 0.289 | 4 |
| Random | **0.750** | 0.289 | 4 |

**Welch t-test (Frozen vs Joint)**: t=+1.000, df=5.40
- Frozen > Joint by 0.25 (direction CONSISTENT with H10)
- t < 2.0 (NOT statistically significant)

**Welch t-test (Frozen vs Random)**: t=-1.000, df=5.40
- Random > Frozen by 0.25 (negative control FAILS at this N)
- t < 2.0 in absolute value (NOT statistically significant)

## 4. Verdict

**H10 PILOT VERDICT (n=5 seeds, N=12/seed): INCONCLUSIVE.**

- **H10 direction**: Frozen (0.500) > Joint (0.250), mean delta = +0.25.
  This is direction-consistent with H10 (frozen > joint).
  However, Welch t = 1.000 < 2.0 (NOT significant at pre-reg threshold).

- **Negative control**: Random (0.750) > Frozen (0.500), mean delta = +0.25.
  This FAILS the H10 negative control requirement (Frozen should beat
  Random by 0.10+).
  Welch t = -1.000 in absolute value < 2.0 (NOT significant).

- **Pre-reg decision rule**: H10 requires ALL of:
  1. Frozen > Joint by > 0.05: PASSES (mean delta = +0.25)
  2. Welch t > 2.0: FAILS (t = 1.000)
  3. Frozen > Random by > 0.10: FAILS (Frozen < Random by 0.25)

**Verdict per pre-reg rule**: **INCONCLUSIVE** (insufficient statistical
power to confirm or refute).

## 5. Why this pilot is inconclusive (not refuted)

The negative control failure (Random > Frozen by 0.25) is concerning,
but the small sample size and high variance mean this could be:
- Genuine overfitting on 9-trace training sets (with 3-trace eval,
  Random can hit lucky concordance)
- Insufficient statistical power to distinguish real signal from noise

To distinguish these, the pre-reg H10 specifies:
- n=5 seeds: ACHIEVED
- 200 rollouts/seed: NOT MET (used 12/seed due to CPU budget)

The pilot's statistical power is much lower than the pre-reg H10. A
larger N (e.g., 50-100 traces/seed) would likely reduce variance and
produce a more decisive verdict.

## 6. What the wild per-seed variance tells us

Looking at the per-seed results:
- Seed 0: Frozen=Joint=0.000, Random=1.000 (Frozen/Joint fail completely)
- Seed 3: Frozen=1.000, Joint=0.500, Random=1.000 (Frozen succeeds,
  but Random ties it)
- Seed 4: All 0.500 (random performance across the board)

This level of variance (range 0.000 to 1.000 for Frozen) is consistent
with **3-trace eval sets** — small-sample AUROC is inherently noisy.

If we ran this with 200 traces/seed, the per-seed variance would be
much smaller and the multi-seed comparison would be more decisive.

## 7. Recommendation

- **DO NOT** treat this pilot as the H10 verdict. The variance is
  too high; the result is INCONCLUSIVE.
- **DO** treat this pilot as confirmation that the multi-seed
  pipeline works end-to-end on real LM traces.
- **DO** scale up to 200 traces/seed (per pre-reg) when GPU is
  available, OR to 50-100 traces/seed on CPU (still feasible, less
  per-seed variance).

## 8. Honest framing per NO_SELF_DECEPTION.md

This multi-seed pilot:
- **DOES NOT** confirm the H10 hypothesis (Welch t < 2.0).
- **DOES NOT** refute the H10 hypothesis (the direction is consistent;
  Frozen > Joint on the mean).
- **REVEALS** the variance structure of small-N pilots: with N=12/seed
  and 3-eval, per-seed AUROC swings 0.000 to 1.000 for the same arm.
- **SUPPORTS** the need for the full pre-reg n=5 / 200 rollouts/seed
  protocol to produce a real verdict.
- **REVEALS** a concerning pattern: Random > Frozen at small N,
  which would be a genuine negative finding if it persisted at
  larger N (need to test).

Per NO_SELF_DECEPTION.md:
- The INCONCLUSIVE verdict is reported with the same precision as a
  positive or negative verdict would be.
- The negative control failure is acknowledged as a real concern.
- The "Frozen > Joint by 0.25" mean delta is NOT promoted to a
  POSITIVE finding because the Welch t does not meet the threshold.

---

*Multi-seed n=5 pilot log 2026-07-28 by Codex agent.
H10 hypothesis INCONCLUSIVE at this sample size (n=5/N=12/seed).
Direction-consistent (Frozen > Joint by 0.25) but Welch t = 1.000
< 2.0 threshold. Negative control fails (Random > Frozen by 0.25)
but also not statistically significant. Full pre-reg H10 (200
rollouts/seed) needed for verdict.*