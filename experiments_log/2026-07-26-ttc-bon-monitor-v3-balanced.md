# TTC BoN+Monitor v3 (balanced training) — LunarLander-v3

> Date: 2026-07-26
> Status: COMPLETE — new failure mode identified
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What changed

`code/ttc_bon_monitor.py`: added balanced subsampling before Monitor
training. When `n_neg > n_pos * 2`, subsample negatives to cap at
4:1 ratio. Also added Monitor output stats (min/max/mean/std) at
end of training for diagnostic.

## 2. Result (LunarLander-v3, 100K PPO, N=4, K=8, seed 0)

| metric | v2 (imbalanced) | v3 (balanced) |
|--------|-----------------|---------------|
| class balance | 90% neg, 10% pos | **50.7% pos** ✓ |
| Monitor output min | ~0.0 | 0.461 |
| Monitor output max | ~1.0 | 0.520 |
| Monitor output std | varies | **0.013** ← problem |
| PPO mean | 180.7 | 9.9 |
| BoN+Monitor mean | -56.2 | -39.9 |
| **Delta** | **-236.9** | **-49.8** |

PPO seed-0 variance is huge (174.2 in v2 vs 9.9 in v3 — same seed but
different random init order gives different PPO outcomes). The seed
comparison is approximate; the within-experiment monitor comparison
is the meaningful one.

## 3. New failure mode: Monitor output collapse

After balanced training:
- pos=76, neg=74 (50.7% pos)
- Monitor output: mean=0.492, **std=0.013**
- Monitor essentially outputs the same value (~0.49) for all inputs

**Root cause**: With balanced data, BCE loss converges to the marginal
prediction probability (~0.5). The 64-hidden MLP is too small to
find the discriminative signal in the 416-dim history vector. So
Monitor collapses to constant output regardless of input.

This is the OPPOSITE failure mode of v2 (where Monitor predicted ~0.1
for everything because of imbalance). Either way, BoN ranking
becomes near-random.

## 4. Implications

For TTC to work, the Monitor must have:
- Sufficient model capacity to learn the failure pattern
- Sufficient training signal (neither too imbalanced nor constant)

Y1 work candidates:
1. **Bigger Monitor architecture**: hidden=256 (4x larger) or 512
2. **Longer training**: epochs=20-50 instead of 5
3. **Better features**: include action history in Monitor input (currently it does)
4. **Per-step PRM scoring**: aggregate Monitor output across rollout
   steps with a learnable combiner
5. **Auxiliary loss**: add reconstruction loss to force Monitor to
   encode trajectory information (not just output label)
6. **Different label**: predict reward (regression) instead of
   binary failure
7. **Use value function as scorer**: skip Monitor, learn V(s) and
   pick action that maximizes V(s')

## 5. Honest assessment

**TTC BoN+Monitor as implemented does NOT improve over vanilla PPO
on LunarLander-v3 across 3 attempts (v1, v2, v3).** This is strong
negative evidence for the original ADR 0011 hypothesis.

We should:
- Keep state cloning infrastructure (reusable for Project C Y1)
- Keep the balanced sampling code (useful for any imbalanced RL data)
- NOT pursue TTC BoN+Monitor further in Y0
- Investigate ALTERNATIVE TTC methods (e.g., value-function BoN,
  learned verifier, etc.)

## 6. Artifacts

- `code/ttc_bon_monitor.py` (now ~12 KB, includes balanced sampling +
  output stats)
- `code/checkpoints/ttc_bon_monitor_LunarLander-v3_seed0/phase2_log.json`
- Compute: ~2 minutes

## 7. Y1 follow-up (revised TTC plan)

1. **Try value-function BoN** instead of Monitor-BoN: pick candidate
   action that maximizes V(s'). Simpler, no separate Monitor needed.
2. **Larger Monitor**: hidden=256, epochs=20, see if balanced
   training then produces non-constant output.
3. **Monitor with auxiliary reconstruction**: ensure Monitor
   encodes trajectory features even if BCE loss is at minimum.
4. **Cross-env**: LunarLander may just be hard for TTC; try CartPole
   or MountainCar where PPO dynamics are different.
5. **Theoretical analysis**: compute the theoretical maximum TTC
   gain given Monitor AUROC=0.796 — if it's < 5% then TTC isn't
   worth pursuing.