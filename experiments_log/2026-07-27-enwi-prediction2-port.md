# ENWI Prediction 2 port to Project C — initial result NEGATIVE

> Date: 2026-07-27
> Status: Composable WORSE than monolithic (opposite of ENWI's claim)
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

Ported `F:\TMLR\Fusion\enwi_prototype\composable_physics.py` to
E:\agi-research\projects\project_c_causal_world\code\. The 4 physics
modules (Gravity, Collision, Friction, Inertia) + Composer + gate
network are now in our codebase.

ENWI paper reports 94.22% improvement of composable over monolithic
on 5 physics scenes. We tried to replicate.

## 2. Method

`code/enwi_prediction2.py`: train both models on each of 5 scene
types, report final MSE on held-out test set.

Setup:
- Synthetic physics scenes (free_fall, collision, friction, inertia, compound)
- 1000 train scenes + 200 test scenes per type
- latent_dim=32 (smoke), 30 epochs (smoke), Adam lr=1e-3
- Composable: 4 modules + Composer + gate
- Monolithic: 4-layer MLP with same parameter count

## 3. Result (smoke test, latent_dim=32, 30 epochs)

| scene | monolithic MSE | composable MSE | ratio |
|---|---|---|---|
| free_fall | 2.09e-7 | 2.70e-6 | comp 10x worse |
| collision | 1.21e-6 | 3.70e-6 | comp 3x worse |
| friction | 6.69e-7 | 1.57e-6 | comp 2.5x worse |
| inertia | 2.25e-7 | 5.22e-7 | comp 2.3x worse |
| compound | 4.64e-7 | 1.25e-6 | comp 2.7x worse |
| mean | 5.55e-7 | 1.95e-6 | comp 3.5x worse |

Both MSEs are tiny (1e-6 range). Composable is consistently 3-10x
WORSE than monolithic, opposite of ENWI's 94% improvement claim.

## 4. Why negative?

Possible explanations:
1. **Insufficient training**: ENWI used 2000 epochs; we used 30
2. **Synthetic data too simple**: 1000 samples may not require modular
   decomposition; monolithic can memorize
3. **Port imperfect**: the gate net, encoder structure, or module
   architecture may differ from ENWI's reference
4. **No physics prior in synthetic data**: our scene generator uses
   simple linear perturbations; ENWI's uses closed-form physics
   (gravity t^2, momentum conservation, etc.)

## 5. Honest assessment

This is a **negative replication**. Our port does not reproduce
ENWI's headline result. Possible reasons:
- Insufficient training (most likely — 30 epochs is smoke level)
- Synthetic data too simple for composable to win
- Port may not match ENWI's exact architecture

ENWI's original result is on 2000 epochs with physics-based synthetic
data. Our smoke test is on 30 epochs with linear synthetic data.

**Next steps to actually replicate ENWI's 94%**:
1. Use 2000 epochs (not 30)
2. Use physics-accurate scene generator (gravity t^2, momentum conservation, etc.)
3. Use 128-dim latent (not 32-dim)
4. Match ENWI's exact module architectures

## 6. Artifacts

- `code/composable_physics.py` (11835 bytes, ported from F:\TMLR\)
- `code/enwi_prediction2.py` (~5000 bytes, replication script)
- `code/checkpoints/enwi_prediction2/phase2_log.json`
- `experiments_log/_p2_smoke2.txt` (full output)
- Compute: ~30 seconds for smoke test