# AIE Full Training on LunarLander-v3 — 3-seed sweep

> Date: 2026-07-27
> Mode: Active Inference Engine (replaces PPO+Monitor with free energy minimization)
> Status: Modest learning, far from PPO baseline
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

Trained `ActiveInferenceEngine` (from `active_inference.py`) on LunarLander-v3
from scratch (no PPO bootstrap). The training loop is iterative:

1. Collect `n_episodes_per_outer` episodes using current AIE policy (stochastic).
2. Train AIE for `n_epochs_per_outer` epochs on the collected batch.
3. Repeat for `n_outer` iterations.

Loss function combines three terms:

```
loss = 1.0 * F.mean()            # variational free energy
      + 1.0 * cross_entropy(...)  # action prediction
      + 0.1 * MSE(r_pred, r_obs)  # reward prediction
```

## 2. Hyperparameters

```
n_outer = 8 (collect-train iterations)
n_episodes_per_outer = 8
n_epochs_per_outer = 8
batch_size = 4 (very small for fast iteration)
hidden = 64
learning_rate = 3e-4 (Adam)
```

Total environment steps per seed: ~64 episodes × ~150 steps = ~10K steps.

## 3. Results

| seed | final eval mean | final eval std | train mean (last) |
|------|-----------------|----------------|-------------------|
| 0    | -127.7          | 44.7           | -154.4             |
| 1    | -135.3          | 33.7           | -217.1             |
| 2    | -154.8          | 52.5           | -222.3             |
| **mean** | **-139.3**  | **~44**        | **-198**           |

The loss decreases monotonically across outer iterations (from ~21.7 to ~19.5),
indicating the AIE is *learning the data distribution* (perception accuracy),
but not yet learning to *land successfully* (which requires reward signal).

## 4. Comparison to baselines

| Method | Steps | Final return | Note |
|--------|-------|--------------|------|
| Random policy | N/A | -150 to -200 | baseline |
| AIE (this run) | ~10K | -139 | barely above random |
| PPO baseline | 100K | -100 to +50 | full budget |
| PPO with Monitor (Phase 1.5) | 100K | -100 to -50 | best so far |

**Honest assessment**: AIE is *not yet competitive* with PPO at our budget.
This is consistent with ENWI's Prediction 4 claim that AIE should match PPO
with at least 50% fewer samples — but that claim is about *asymptotic*
performance, not 10K-step performance.

## 5. Why AIE underperforms (and what would help)

1. **Reward signal under-utilized**: our `reward_weight = 0.1` is too low.
   Increasing to 1.0 might help but breaks the free-energy interpretation.
2. **Off-policy instability**: AIE trains on freshly-collected data, then
   re-collects. The data distribution shifts every outer iteration.
3. **Hidden-state bottleneck**: latent state is `obs_dim = 8`, same as
   observation. No temporal aggregation. Adding recurrence should help.
4. **No baseline subtraction**: AIE uses raw rewards. Adding a learned baseline
   would reduce variance.

## 6. What this means for the thesis

- AIE is a viable *alternative formulation* but not a *replacement* for PPO
  at our compute scale.
- The free-energy loss decreases as expected (perception accuracy improves),
  but the action-prediction loss dominates without enough data.
- For ENWI Prediction 4 to be testable, we need ~100K+ steps of AIE training,
  which is 10× our current budget.

## 7. Artifacts

- `code/aie_train_full.py` (~7K bytes, the training script)
- `code/checkpoints/aie_full/seed{0,1,2}/phase2_log.json`
- `experiments_log/_aie_train_full_seed{0,1,2}.txt` (raw output)
- Compute: ~42 sec per seed on CPU


## Long-budget AIE run (4x budget, seed 0)

To test if AIE converges with more compute, ran seed 0 with 4x budget:

```
n_outer=16, n_episodes_per_outer=16, n_epochs_per_outer=8, batch_size=8
Total: 16*16 = 256 episodes, ~50K env steps (5x short budget)
```

| metric | short budget (8×8) | long budget (16×16) | delta |
|--------|---------------------|---------------------|-------|
| Initial eval | -126.5 | -84.8 | +41.7 (random luck) |
| Final eval | -127.7 | -135.6 | -7.9 (no improvement) |
| Final loss | 19.451 | 17.402 | -2.05 (still decreasing) |
| Best mid-run eval | -94.2 (outer 5) | -99.9 (outer 13) | -5.7 |

**Conclusion**: Even at 4× budget, AIE final eval return is statistically
indistinguishable from the 1× run (within 1 standard deviation). The free-energy
loss continues to decrease, suggesting the model is still learning the data
distribution, but the policy does not converge to land successfully.

This is an **even stronger honest negative** for ENWI Prediction 4: at our
compute scale (~50K env steps), AIE does not match PPO. We would need
~500K+ steps (10× longer) for a fair test, which exceeds our Y0 budget.

## Implications for thesis

- Add to Addendum A: "AIE does not converge at our budget. ENWI Prediction 4
  requires either much more compute or a better AIE architecture."
- Move AIE from "viable alternative" to "open research direction" in Y1 plan.
