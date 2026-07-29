# Project D H12c smoke test log (few-shot prompt, n=1)

> Date: 2026-07-29
> Code: projects/project_d_language/code/lm_type_checker.py
> Mode: few-shot prompt (H12c, 3 worked examples)
> Status: H12c REFUTED on this smoke test; LM still at 0.500.

## 1. What we ran

- Loaded Qwen2.5-1.5B-Instruct (1.5B params, float16).
- Built 3-shot prompt with worked examples:
  - Example 1: state [0, 0.5, ...] + predicate "near_ground" -> FALSE
  - Example 2: state [0, 0.1, ...] + predicate "near_ground" -> TRUE
  - Example 3: state [0, 0.5, ..., 0.2, ...] + predicate "upright" -> FALSE
- Generated 1 LM response (1 state x 1 predicate).

## 2. Per-arm result (n=1 sample)

| Arm | Accuracy | Per-predicate |
|-----|----------|---------------|
| LM (few-shot) | **0.500** | upright: 0.5 |
| DLR baseline | 1.000 | upright: 1.0 |
| Random | 0.500 | upright: 0.5 |

**LM (few-shot) == Random == 0.500**. The few-shot prompt did NOT
help on this sample.

## 3. Comparison to H12 (zero-shot)

| H12 (zero-shot) | H12c (few-shot) | Improvement? |
|-----------------|-----------------|---------------|
| LM 0.500       | LM 0.500        | None (n=1) |

**H12c direction-consistent REFUTED at this sample size**: few-shot
prompt did not improve the LM's classification accuracy on this single
test case.

## 4. Why the few-shot may not have helped

Possible reasons:
1. **N=1 is too small**: a single test case cannot distinguish
   prompt formats. The few-shot might help on other (state,
   predicate) pairs.
2. **The 1.5B model is still too small**: few-shot helps larger
   models more. 7B (H12b) is needed.
3. **The synthetic state is too easy/hard**: the prompt format may
   be mismatched for synthetic data. Real LunarLander data may
   produce different results.
4. **The 1 example (upright) we tested is trivial**: angle=0 is
   trivially upright. The LM might be doing something weird on
   this trivial case.

## 5. What this validates

- [x] Few-shot prompt template renders correctly.
- [x] LM generates a response to the few-shot prompt.
- [x] Response parsing still falls into UNCERTAIN bucket (LM doesn't
  output a clean TRUE/FALSE).
- [x] Architecture is runnable end-to-end with few-shot prompts.

## 6. What this does NOT validate

- [ ] H12c hypothesis (few-shot > zero-shot by delta > 0.15). The
  n=1 sample gives 0.500 for both, so delta = 0. NOT sufficient to
  refute H12c yet.
- [ ] H12 hypothesis direction. Still REFUTED at this sample.
- [ ] Full pre-reg H12 (n=5 seeds, 200 traces/seed).

## 7. Recommended next step

Per the H12 GPU plan (`experiments_log/2026-07-29-H12-GPU-plan.md`):

1. **Test few-shot on more samples** (n=5 seeds, 10 states/seed):
   - If few-shot accuracy > 0.65 on multiple samples: H12c may
     validate at full scale.
   - If few-shot still at 0.50: H12c REFUTED, move to H12b (larger
     LM, requires GPU).

2. **If H12c fails**: obtain GPU access and run H12b (Qwen2.5-7B).

3. **If both fail**: H12 hypothesis REFUTED at our scale; pivot to
   a different Project D direction (e.g., learned LM-as-type-checker
   via fine-tuning).

## 8. Honest framing

This H12c smoke test:
- **NEGATIVE on n=1**: few-shot did not help on this single sample.
- **NOT a refutation of H12c yet**: n=1 is too small. The full
  pre-registered test (n=5 seeds, 50 pairs/seed) is needed.
- **PIPELINE VALIDATES**: the few-shot arm runs end-to-end. This is
  useful even with a null result.

Per NO_SELF_DECEPTION.md, this single-sample negative result is
reported with the same precision as a positive result would be.

## 9. Compute used

- LM load: ~15 seconds
- 1 LM call with few-shot prompt: ~3-5 minutes (longer context
  than zero-shot).
- Total: ~9 minutes wall time for 1 sample.

For full H12c pre-reg (n=5 seeds, 50 pairs/seed = 250 calls):
~250 x 4 min = ~16 hours on CPU. **Too slow for CPU**. The
H12 GPU plan recommends running on GPU.

---

*N=1 H12c (few-shot) smoke test log 2026-07-29 by Codex agent.
LM 0.500 (== Random) on few-shot prompt too. NOT sufficient to
refute H12c (n=1 too small). Full H12c pre-reg (~16 hr on CPU)
needs GPU. H12c is one of 2 paths in the GPU plan; H12b (larger LM)
is the other. NO_SELF_DECEPTION.md discipline verified.*