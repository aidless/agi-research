# Project G H10 multi-arm smoke test log

> Date: 2026-07-28
> Code: projects/project_g_llm_self_monitoring/code/h10_multi_arm_smoke.py
> Note: synthetic data, NOT the real H10 experiment.

## 1. What this log is for

This log records the 3-arm (frozen / joint / random) smoke-test
result for the Project G architecture. It validates that all 3
arms can be trained and evaluated end-to-end.

It is NOT the H10 pre-registered experiment; the real H10
experiment uses frozen-LLM rollouts and is defined in
`2026-07-28-PRE-REGISTERED-H10.md`.

## 2. Setup

- Synthetic data: 160 train traces, 40 eval traces per seed.
- 3 seeds (n=3, reduced from the planned n=5 for time budget).
- 3 arms: Frozen Monitor, Joint Monitor, Random Monitor.
- Frozen Monitor: 20 epochs (reduced from 50 for time).
- Joint Monitor: 4 LLM steps x 5 monitor epochs (reduced for time).
- Random Monitor: untrained U[0,1] signal.
- Failure signal: synthetic, lower logit variance in last 5
  positions for failure traces.

## 3. Per-seed results (n=3)

| Seed | Frozen AUROC | Joint AUROC | Random AUROC | Delta_F-J |
|------|--------------|-------------|--------------|------------|
| 0    | 0.878        | 0.820       | 0.330        | +0.058     |
| 1    | 0.780        | 0.823       | 0.587        | -0.043     |
| 2    | 0.803        | 0.830       | 0.655        | -0.027     |

## 4. Aggregate (mean +/- std, n=3)

| Arm    | Mean | Std  |
|--------|------|------|
| Frozen | 0.820 | 0.042 |
| Joint  | 0.824 | 0.004 |
| Random | 0.524 | 0.140 |

## 5. Decision rule check (smoke-test version)

- **Frozen > Joint**: -0.004 (does NOT pass +0.05 threshold)
- **Frozen > Random**: +0.296 (passes +0.10 negative control)

## 6. Verdict

**H10 direction NOT reproduced on synthetic data**:
- Frozen and Joint achieve similar AUROC (~0.82) on the synthetic
  signal.
- Random is much lower (0.524), confirming both Frozen and Joint
  pick up the synthetic signal.

**Why this is expected**: the synthetic data does NOT have
distribution drift between the frozen and joint arms. The "joint"
perturbation (Gaussian noise on features) does not create the kind
of failure-concept drift that a real LLM update would. Both arms
see similar data distributions; the decoupling advantage is
invisible.

**Implication for the real H10**:
- This smoke test confirms the 3-arm architecture works end-to-end.
- It does NOT validate or invalidate the H10 hypothesis.
- The real H10 requires a frozen LM, where the LLM update creates
  real distribution drift between the two arms.

## 7. Architecture validation

What this smoke test DOES validate:
- [x] All 3 arms can be trained (or generated for Random) end-to-end.
- [x] AUROC computation works on held-out traces.
- [x] Negative control (Random) is meaningfully lower than trained
      arms.
- [x] The Frozen and Joint training procedures are symmetric except
      for the perturbation.

What this smoke test does NOT validate:
- [ ] The H10 hypothesis direction (synthetic data has no drift).
- [ ] The LLM trace representation (using synthetic features).
- [ ] The failure-label generator (not used; labels come from the
      synthetic generator).

## 8. Honest framing

Per NO_SELF_DECEPTION.md, this is a **smoke test, not a result**.
The synthetic data is too easy for both Frozen and Joint; the
H10 hypothesis requires a frozen LM where the LLM update creates
real distribution drift.

The fact that the smoke test does NOT show the H10 direction is
NOT a refutation of H10. It is a limitation of the synthetic
data.

The real H10 experiment will run when:
1. The user picks a frozen LM (Qwen-1.5B / Phi-3-mini / other).
2. The user picks a reasoning dataset (GSM8K / MATH / other).
3. Compute is available (GPU preferred).

## 9. Next step

If the user wants to run the real H10:
1. Choose a frozen LM (Qwen-1.5B, Phi-3-mini, or other small LM).
2. Choose a reasoning dataset (GSM8K, MATH, or other).
3. Confirm compute budget (CPU may be too slow; GPU preferred).
4. Run the real experiment per pre-registered design (n=5 seeds).

Until then, this smoke test log + the H10 pre-registration are
the only Project G artifacts.

---

*Smoke test run 2026-07-28 by Codex agent. n=3 seeds, reduced
epochs for time budget. Synthetic data only.*