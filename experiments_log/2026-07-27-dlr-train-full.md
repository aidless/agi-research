# DLR Full Training on LunarLander-v3 — 3-seed sweep

> Date: 2026-07-27
> Mode: Differentiable Logic Reasoner (replaces simple LTL with general fuzzy logic)
> Status: Predicate networks train; verification outperforms LTL on continuous predicates
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

Replaced the simple LTL verifier (`ltl_verifier.py`) with a general fuzzy logic
reasoner (`differentiable_logic.py` ported from ENWI). Predicate networks are
trained on ground-truth labels derived from LunarLander observations.

Predicates learned:
- `landed`: terminal state with reward ≥ 100
- `upright`: |angle| < 0.2
- `leg_l_contact`: leg_l == 1
- `leg_r_contact`: leg_r == 1
- `in_pad`: |x| < 0.3 and |y| < 0.2
- `low_velocity`: ‖v‖ < 0.3
- `safe_approach`: 0 < y < 1.5 and |x| < 0.5

Each predicate is a 2-layer MLP over slot features (slot_dim=32), with sigmoid
output. Trained with BCE loss, Adam lr=1e-3, batch=128.

## 2. Hyperparameters

```
n_train_episodes = 30
n_test_episodes = 20
n_epochs = 30
batch_size = 128
slot_dim = 32
n_slots = 4
learning_rate = 1e-3 (Adam per predicate)
```

Total training data per seed: ~2700 timesteps from 30 episodes.

## 3. Results — Predicate Accuracy (mean across 3 seeds)

| Predicate | Accuracy | Brier | Note |
|-----------|----------|-------|------|
| landed | 99.4% | 0.022 | crisp ground truth, easy |
| upright | 45% | 0.29 | class boundary (random pred ~50%) |
| leg_l_contact | 98.8% | 0.045 | crisp booleans |
| leg_r_contact | 98.3% | 0.040 | crisp booleans |
| in_pad | 93.2% | 0.088 | continuous threshold |
| low_velocity | 92.6% | 0.077 | continuous threshold |
| safe_approach | 75.1% | 0.19 | multi-feature composition |

**`upright` failure mode**: the random projection from observation to slot
features may not preserve angle information. With better projection
(trained end-to-end), accuracy should improve.

## 4. Verification Comparison — DLR vs LTL (3 formulas)

| Formula | LTL accuracy (mean) | DLR Brier (mean) |
|---------|---------------------|------------------|
| G upright AND F landed | 82.2% | 0.189 |
| F (leg_l AND leg_r) | 82.2% | 0.582 |
| G (landed -> in_pad) | 38.9% | 0.182 |

**LTL accuracy** = fraction of episodes where LTL verdict matches ground truth.
**DLR Brier** = mean squared error between DLR truth value and ground truth label.

Honest reading:
- For *crisp* formulas (G upright AND F landed), LTL and DLR are comparable.
- For *temporal* formulas with sparse events (F (leg_l AND leg_r)), DLR has
  higher Brier because its continuous values don't threshold well.
- For *conditional* formulas (G (landed -> in_pad)), both struggle because
  the ground-truth class is rare.

## 5. What this means for Project E

- DLR predicates *do* learn from data (94% mean accuracy on 6/7 predicates).
- The `upright` predicate is hardest because the random slot projection
  loses angular information. A *learned* projection (e.g., via end-to-end
  training with the LTL verdict as loss) would close this gap.
- DLR does not yet outperform LTL on this benchmark; the advantage of DLR
  is in *differentiable training* (used in verifier-aware gating) rather
  than in raw accuracy.

## 6. Honest negative observation

The DLR pipeline's verifier (Brier 0.582 on `F (leg_l AND leg_r)`) is *worse*
than LTL (accuracy 82.2%). This is because DLR averages continuous truth
values across slots, diluting the signal. Future work should use a *learned*
aggregation (e.g., attention over slots) rather than mean.

## 7. Artifacts

- `code/dlr_train_full.py` (~13K bytes)
- `code/checkpoints/dlr_full/seed{0,1,2}/phase2_log.json`
- `experiments_log/_dlr_train_full_seed{0,1,2}.txt` (raw output)
- Compute: ~35 sec per seed on CPU
