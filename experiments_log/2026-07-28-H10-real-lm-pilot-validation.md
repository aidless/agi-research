# Project G H10 REAL-LM pilot -- system validation

> Date: 2026-07-28
> Code: projects/project_g_llm_self_monitoring/code/h10_real_pilot.py
> Note: This is a SYSTEM VALIDATION, not the pre-registered H10 verdict.

## 1. Goal

Validate that the real-LM pilot pipeline works end-to-end on this
system before committing to a full n=5 H10 run.

## 2. What we ran

- Loaded Qwen2.5-1.5B-Instruct (1.5B params) in float16 from the
  local cache at `F:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\`.
- Loaded GSM8K test set (cached at `F:\hf_cache`).
- Generated 4 reasoning traces with max_new_tokens=32.
- Labelled traces with GSM8K ground truth (final-answer correctness).
- Pipeline ran end-to-end successfully.

## 3. Per-trace results (n=4 pilot)

| # | Question (truncated) | GT | Predicted | is_failure |
|---|----------------------|----|-----------|------------|
| 0 | Janet''s ducks lay 16 eggs per day... | 18 | 2 | 1 |
| 1-3 | (other GSM8K problems) | (various) | (all wrong) | 1 |

All 4 traces were labeled as failures (the model got the wrong final
answer in all 4 cases with max_new_tokens=32). This is expected for
a small LM with limited tokens — the model runs out of reasoning
steps before reaching the correct answer.

## 4. Timing on CPU

- LM load: ~30 seconds (one-time)
- Per-trace generation: ~135 seconds (32 tokens at ~4 sec/token)
- 4 traces total: 539 seconds (~9 minutes)

**Pilot runtime estimate**:
- N=8 traces, max_new_tokens=32: ~18 minutes
- N=16 traces, max_new_tokens=32: ~36 minutes
- N=200 traces (full pre-reg), max_new_tokens=80: ~10 hours on CPU

## 5. System constraints discovered

1. **Float16 vs Float32**: float32 model load fails on this system
   (OSError 1455, paging file too small). float16 works.
2. **3B model is too large**: float16 3B model also exceeds paging
   file on this system. Only the 1.5B model fits.
3. **CPU is slow**: ~4 sec/token for 1.5B in float16. GPU would be
   ~100x faster.
4. **HF_HUB_OFFLINE required**: the cache validation tries to hit
   huggingface.co; offline mode bypasses this.
5. **Explicit local path needed**: passing the hub ID ("Qwen/...")
   triggers validation; passing the snapshot directory
   (`F:\hf_cache\hub\.../snapshots/<hash>`) loads without validation.

## 6. What this validates

- [x] Real LM can be loaded from local cache (with offline mode +
      explicit path).
- [x] GSM8K can be loaded from local cache.
- [x] LM trace generation works end-to-end on real GSM8K problems.
- [x] Failure labels can be extracted from generated text via
      final-answer regex.
- [x] Trace feature extraction (token + confidence) works.

## 7. What this does NOT validate

- [ ] The H10 hypothesis direction (only 4 traces, all failures;
      no Frozen vs Joint comparison done at full pilot scale yet).
- [ ] The full pre-registered 200 rollouts/seed protocol.
- [ ] Cross-env transfer (H11).
- [ ] Statistical significance (n=1 vs n=5 required).

## 8. Recommended next step

Run the full `h10_real_pilot.py` with N=8 (or N=16 if budget allows)
to get the 3-arm comparison on real LM traces. This will produce a
real pilot result, not just a system validation.

```bash
cd projects/project_g_llm_self_monitoring/code
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:H10_N_TOTAL = "8"
& "C:\...\python.exe" h10_real_pilot.py
```

Expected runtime: ~25 minutes for N=8 traces (8 rollouts * 4 sec/token
* 32 tokens + LM load + Monitor training).

## 9. Alternative: GPU acceleration

If the user has GPU access, the same pilot can run in ~1-2 minutes
with float16 on a CUDA GPU. The script does not need modification;
just remove `dtype=torch.float16` if the GPU has more memory, or
set `device="cuda"`.

## 10. Honest framing

This log documents the SYSTEM VALIDATION only. The pre-registered
H10 verdict requires:
1. N=5 seeds (vs N=1 in this validation)
2. 200 rollouts/seed (vs 4 in this validation)
3. Welch t > 2.0 statistical test (not applicable at N=1)
4. Negative control arm comparison (Random Monitor, which the
   multi-arm smoke already showed works on synthetic data)

The system validation confirms that the *pipeline* works. The
*result* is not yet known.

Per NO_SELF_DECEPTION.md, this is clearly labeled as a system
validation, not an H10 result.

---

*System validation log 2026-07-28 by Codex agent. Pipeline
validated on Qwen2.5-1.5B-Instruct + GSM8K in float16 on CPU.*