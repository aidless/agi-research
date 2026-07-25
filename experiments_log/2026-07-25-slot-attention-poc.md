# Project C Slot Attention PoC

> Date: 2026-07-25
> Status: PoC PASSED
> Author: Codex

## 1. Goal

Verify that a minimal slot-attention module (Locatello et al. 2020) can
be implemented in ~100 lines of PyTorch and trains end-to-end on CPU
without Procgen (which requires cmake + VS build tools unavailable
in the current environment).

This is a *feasibility check*, not the full Project C experiment. The
full Project C requires Procgen 16-game benchmark, which is Y1 work.

## 2. What I built

`code/slot_attention.py` (~140 lines, pure PyTorch CPU):

- `SlotAttention` module: input projection, learned slot initialization,
  iterative attention with softmax-over-slots (competition), GRU update,
  residual MLP.
- `slot_diversity_loss`: encourages distinct slots to bind to different
  input features.
- PoC `__main__` block: random batch of 8 samples × 6 features × 8 dim,
  4 slots, 3 attention iterations, 50 training steps.

## 3. Result

```
Input: torch.Size([8, 6, 8])
Slots: torch.Size([8, 4, 32])
Slot 0 mean: -0.063
Slot 1 mean: -0.002
Diversity loss: 0.396 (initial)
  step 0:  recon=1.209 div=0.387
  step 10: recon=1.123 div=0.282
  step 20: recon=1.069 div=0.240
  step 30: recon=1.045 div=0.217
  step 40: recon=1.030 div=0.225
PoC PASSED: slot attention trains and produces diverse slots.
```

Diversity loss decreased 0.39 -> 0.22 (slots became more distinct as
training progressed). Reconstruction loss decreased 1.21 -> 1.03.

## 4. Limitations

- **No real data**: input is random Gaussian. Real LunarLander states
  would have structure (position, velocity, angle, leg contact).
- **No semantic interpretability test**: we cannot verify that slot 0
  binds to "position" and slot 1 binds to "velocity" without labeled
  data.
- **No world model**: this is just the perception module. The full
  Project C needs dynamics + SCM on top.
- **No Procgen benchmark**: 16-game cross-domain evaluation deferred
  to Y1 (needs Procgen source build).

## 5. Next steps for Project C (Y1)

1. Install Procgen from source (requires cmake + Visual Studio).
   Alternative: use a smaller open-source benchmark like Atari or
   Crafter.
2. Add a Transformer-based dynamics module on top of slot outputs
   (predicts next slot states).
3. Add SCM intervention layer (do-calculus, Pearl L2).
4. Implement cross-domain transfer eval (train on game A, test on
   game B; measure transfer AUROC).

## 6. Connection to F:\TMLR H/I series

- **I03 World Models**: slot attention is the canonical object-centric
  perception front-end for DreamerV3-style world models.
- **I04 Multimodal**: slot outputs could be tied to language types
  via Project D's type system (slot -> typed entity).
- **DIAMOND (Alonso 2024)**: alternative is pixel-space diffusion,
  no slots. Y1 comparison.

## 7. Connection to Project A

- Project A's Monitor takes a flat history vector as input. Project C's
  slot outputs could be the structured input: each slot is a typed
  entity, and the Monitor operates on slot trajectories.
- This is a Y1 integration point: feed slot outputs into the Monitor
  instead of raw observations, measure AUROC improvement.

---

*PoC log, 2026-07-25. Total time: ~30 minutes (slot attention + tests + this log).*