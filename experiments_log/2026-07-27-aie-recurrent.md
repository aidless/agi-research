# Recurrent AIE with GRU + Value Baseline — STRONG NEGATIVE

> Date: 2026-07-27
> Mode: AIE with recurrence + variance reduction
> Status: **NEGATIVE** — recurrent AIE worse than vanilla AIE
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we tried

Hypothesis: the original AIE failed to converge because (a) the latent state
has no temporal aggregation and (b) high variance from raw rewards. Fix:
1. **Recurrent latent state**: GRU updates latent across timesteps.
2. **Value baseline**: separate network predicts reward; use as variance reduction.
3. **Higher reward weight**: 0.1 -> 0.5 to address reward signal under-utilization.

## 2. Architecture (`aie_recurrent.py`)

```
encoder: obs -> posterior (mean, log_var)
gru: (latent_state, posterior) -> next_latent_state
transition: latent + action -> next_latent (model-based prior)
generation: latent -> predicted_obs
action_sampler: latent -> action_logits
value_baseline: latent -> value
```

Loss:
```
total = 0.5 * free_energy
      + 1.0 * action_loss
      + 0.5 * reward_loss
      + 1.0 * variance_reduced_policy_loss
```

## 3. Result (seed 0, n_outer=8)

| stage | eval return | loss |
|-------|-------------|------|
| untrained | -278.3 +/- 85.2 | - |
| outer 1 | -259.3 | 4864 |
| outer 4 | -360.7 | 4846 |
| outer 8 | **-390.4** | **4805** |
| **final (n=20)** | **-345.7 +/- 87.4** | - |

Eval *deteriorates* over training (-278 -> -346). The recurrent AIE is
**worse** than vanilla AIE (-127.7 +/- 44.7).

## 4. Why recurrent AIE fails

1. **GRU introduces path dependence**: the agent's latent state depends on
   past trajectory, so early mistakes compound.
2. **Value baseline amplifies bias**: when the value baseline learns a
   wrong reward predictor, the variance-reduced policy gradient becomes
   wrong.
3. **Loss is huge (~4800) and barely decreases**: free-energy dominates
   the loss; reward/policy components have small effect.

## 5. What this means for Project A

AIE has now been tried in 3 variants:
- aie_lunarlander.py: smoke test only
- aie_train_full.py: 3 seeds, -139.3 (worse than random)
- **aie_recurrent.py: 1 seed, -345.7 (much worse)**

All three variants are NEGATIVE on LunarLander. The conclusion:
**AIE does not work at our compute scale** (~50K env steps).

To make AIE competitive with PPO would require:
- 500K+ env steps (10x current)
- Better hyperparameter tuning (impossible without more compute)
- Possibly a fundamentally different architecture

## 6. Y1 direction (refined)

AIE is not the Y1 priority. Y1 should focus on:
1. **Y1.3 (training-time regularizer)** — already POSITIVE
2. **Cross-environment slot WM** — Project C
3. **Real self-improvement loop** — Project A (using Monitor feedback to update PPO)

AIE remains in the codebase as a ported ENWI component, but is not the
primary AIE research direction.

## 7. Artifacts

- `code/aie_recurrent.py` (~330 lines)
- `code/checkpoints/aie_recurrent/seed0/phase2_log.json`
- `experiments_log/_aie_recurrent_seed0.txt` (raw output)
- Compute: ~45 sec per run on CPU
