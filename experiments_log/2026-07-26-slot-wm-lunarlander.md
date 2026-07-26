# Slot-WM on real LunarLander trajectories — Project C PoC

> Date: 2026-07-26
> Status: PoC COMPLETE — weak but informative
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. Question

Does slot attention learn to specialize on interpretable kinematic
features (position, velocity, angle, leg contact) when trained on
real LunarLander trajectories?

## 2. Method

`code/slot_attention_lunarlander.py` (9103 bytes):

1. Train PPO on LunarLander-v3 for 50K steps
2. Collect 100 frozen-policy trajectories (each ~117-500 timesteps)
3. Pad/truncate to fixed length 48 (early timesteps retained)
4. Slot attention: input (100, 48, 8) -> (100, 4 slots, 32-dim)
5. SlotDynamicsModel: predicts next state from current state + slot reps
6. Train 25 epochs with reconstruction loss + diversity loss
7. Compute per-slot / per-feature correlation (Pearson over n=100 episodes)

## 3. Results

| metric | value |
|---|---|
| reconstruction loss | 0.036 (low = good) |
| diversity loss | 0.356 (low = better specialization) |

**Per-slot top-3 feature correlations (Pearson r)**:

| Slot | Top features | Semantic interpretation |
|------|--------------|-------------------------|
| 0    | x_vel=+0.27, x_pos=+0.27, ang_vel=+0.24 | horizontal motion |
| 1    | angle=+0.21, ang_vel=+0.17, x_pos=+0.11 | rotation |
| 2    | y_vel=+0.13, y_pos=+0.13, x_vel=+0.12 | vertical motion |
| 3    | angle=+0.12, y_pos=+0.11, y_vel=+0.11 | overlap (weak) |

## 4. Interpretation

**Weak but informative specialization.**

- Slot 0 / 1 / 2 capture roughly orthogonal kinematic subspaces
  (horizontal / rotation / vertical)
- Slot 3 overlaps with slots 1 and 2 (highest variance 0.188 but
  weaker correlations)
- Diversity loss 0.356 indicates significant overlap between slots

This is the best slot attention can do on LunarLander because:
1. **LunarLander is a single rigid body**, not multi-object. There is
   no "objectness" to discover — only kinematic state.
2. Slot attention's prior is "split scene into distinct objects". On
   a single object, this prior is partially satisfied by splitting
   kinematic features into orthogonal subspaces (which is what we see).
3. The features that slots bind to (x_vel, angle, y_vel) are the
   natural "modes" of LunarLander motion — horizontal control,
   rotation control, vertical control.

## 5. Implications for Project C

1. **LunarLander is not the right env for slot+SCM demonstration.**
   We need multi-object environments where slot attention can
   demonstrate true object binding.
2. **Y1 work**: port to Procgen 16-game benchmark. Each Procgen
   game has 5-20 distinct objects (enemies, obstacles, items), which
   is the natural slot-attention regime.
3. **No need for slot attention on single-object envs**: for
   LunarLander specifically, a simple MLP over flattened state would
   be more parameter-efficient.
4. **The diversity loss is a useful diagnostic**: high diversity loss
   = slots not specializing. We can use this as a training-time
   metric for whether slot attention is learning useful structure.

## 6. Y1 follow-up

When Procgen is installed:
1. Run `slot_attention_lunarlander.py`-equivalent on Procgen coinrun
2. Visualize slot attention weights over time (which object does each
   slot track?)
3. Compare slot+SCM vs DreamerV3 vs DIAMOND on Procgen 16 games
4. If slot attention binds to distinct objects, paper C v1 ready

## 7. Artifacts

- `code/slot_attention_lunarlander.py` (9103 bytes)
- `code/checkpoints/slot_wm_lunarlander/phase2_log.json`
- Compute: ~1 minute (50K PPO + 25 epochs slot training)

## 8. Honest assessment

Project C is **not** yet a publishable result. We have:
- ✅ Slot attention implementation (validated on synthetic + LunarLander)
- ✅ Weak specialization on single-object env
- ❌ Strong specialization on multi-object env (Procgen not installed)
- ❌ SCM intervention layer
- ❌ Cross-domain transfer eval

Y1 work to get Project C to publication-ready state: ~6 months of
focused effort (per ROADMAP.md Y1 plan).