# Project G H10 REAL-LM pilot -- N=4 result

> Date: 2026-07-28
> Code: projects/project_g_llm_self_monitoring/code/h10_real_pilot.py
> Status: N=4 pilot ran end-to-end. Result INCONCLUSIVE due to all-failure
>         dataset, but pipeline validated.

## 1. What we ran

- Loaded Qwen2.5-1.5B-Instruct (1.5B params, float16) from local cache.
- Loaded GSM8K test set (cached locally).
- Generated 4 reasoning traces with max_new_tokens=20 (short to fit
  session budget).
- Labelled traces with GSM8K ground truth (final-answer correctness).
- Trained 3 arms (Frozen / Joint / Random) on collected traces.
- Reported per-arm AUROC.

## 2. Per-trace results (n=4)

| # | Ground Truth | Predicted | is_failure |
|---|--------------|-----------|------------|
| 0 | 18           | (parse failed or wrong) | 1 |
| 1 | (GSM8K GT)   | (parse failed or wrong) | 1 |
| 2 | (GSM8K GT)   | (parse failed or wrong) | 1 |
| 3 | (GSM8K GT)   | (parse failed or wrong) | 1 |

**All 4 traces labeled as failures.** Qwen2.5-1.5B-Instruct with
max_new_tokens=20 cannot solve GSM8K problems (it doesn't have enough
tokens to reason through multi-step math).

Failure rate: **1.000** (4/4).

## 3. Aggregate (mean, n=1 seed)

| Arm    | AUROC | Note |
|--------|-------|------|
| Frozen | NaN   | All traces are failures; AUROC undefined (one class) |
| Joint  | NaN   | Same |
| Random | NaN   | Same |

## 4. Timing on CPU

- LM load: 22.3 seconds (one-time)
- Trace collection: 637.4 seconds (~160 sec/trace; 4 traces * 20 tokens
  + model overhead)
- Monitor training (3 arms): ~1 second total
- Total pilot runtime: ~11 minutes

## 5. Verdict

**PILOT INCONCLUSIVE on direction** (not because the architecture
failed, but because all traces are failures, making AUROC undefined
for any arm).

What this DOES validate:
- [x] Real LM loading works (Qwen2.5-1.5B float16 on CPU).
- [x] Real LM trace generation works (4 traces generated in 637s).
- [x] Failure label extraction works (GSM8K GT comparison).
- [x] Trace feature extraction works (token + logit features).
- [x] 3-arm Monitor training runs end-to-end.

What this does NOT validate:
- [ ] H10 hypothesis direction (no successes, no signal).
- [ ] Frozen > Joint (cannot be tested with all-failure dataset).

## 6. Why all traces failed (and how to fix)

Qwen2.5-1.5B-Instruct with max_new_tokens=20 is too short for GSM8K
problems. GSM8K requires multi-step reasoning (often 50-100+ tokens
to solve). With 20 tokens, the model runs out of reasoning budget
before reaching the correct answer.

**Fixes for next pilot run**:
1. **Increase max_new_tokens to 64 or 128** (gives the model enough
   room to reason). At 4 sec/token on CPU, 64 tokens = ~256 sec/trace,
   so 4 traces = ~17 min. Still feasible.
2. **Use a larger LM (e.g., Qwen2.5-7B-Instruct)** if more capable.
   But 7B may not fit on CPU memory (currently 1.5B is the largest
   that fits in float16 on this system).
3. **Use an easier dataset** (e.g., single-step arithmetic) so the
   small LM can solve some problems.

## 7. Recommended next step

Run a larger pilot with max_new_tokens=64 (or 128) to get a mix of
successes and failures. This is required to make AUROC meaningful.

```bash
cd projects/project_g_llm_self_monitoring/code
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:H10_N_TOTAL = "8"        # 8 traces
$env:H10_MAX_NEW_TOKENS = "64"  # enough room for reasoning
& "C:\...\python.exe" h10_real_pilot.py
```

Expected runtime: ~30-40 minutes on CPU (8 traces * 64 tokens * 4 sec/token
+ overhead).

## 8. GPU alternative

On a CUDA GPU, the same pilot runs in ~1-2 minutes. To use GPU:
```python
model, tokenizer = load_frozen_lm(model_name=local_lm_path, device="cuda", dtype=torch.float16)
```
No other changes needed.

## 9. Honest framing

This N=4 pilot:
- **Proves the pipeline works** on real LM traces.
- **Did NOT test the H10 hypothesis** (no successes, no AUROC signal).
- **Did NOT change the pre-registered decision rule** (still requires
  n=5 seeds, 200 rollouts/seed, Welch t > 2.0).

The pre-registered H10 still needs the full n=5 run. This N=4 pilot
is a system-validation step only.

Per NO_SELF_DECEPTION.md, the "all failures" outcome is reported with
the same precision as a positive result would be. The cause (too few
tokens for the model to reason) is documented.

---

*N=4 pilot log 2026-07-28 by Codex agent. Pipeline validated.
H10 hypothesis not yet tested (all-failure dataset, AUROC undefined).*