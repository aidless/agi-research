# 2024_alonso_diamond_deep.md - DIAMOND: Diffusion World Models

> Paper: Alonso, E., et al. (2024). DIAMOND: Diffusion Models for
> Decision Making. arXiv:2402.03522.
> Affiliation: independent + academic collaboration
> Status: read 2026-07-25, deep note for Project C reading list (Y0 Q3)
> Related to: Project C (slot-WM baseline), F:\TMLR\前沿研究_03_世界模型.md

## 1. Problem

World models for decision-making (Dreamer V1/V2/V3, MuZero, IRIS) use
discrete-latent dynamics — VAE for perception, RNN/Transformer for
transition, Gaussian/Categorical for next-latent. This works but has
limitations:
- Discrete latent bottleneck loses fine perceptual detail
- Gaussian/Categorical next-state distribution may not capture
  multi-modal transitions (e.g. bouncing ball, contact dynamics)

DIAMOND replaces the discrete-latent dynamics with a **continuous
diffusion model** directly in pixel space.

## 2. Method

- **Observation model**: no encoder; DIAMOND operates in pixel space
- **Dynamics model**: conditional diffusion model (DDPM) that denoises
  the next observation given current observation + action
- **Reward/termination heads**: small CNNs applied to the predicted
  next observation (after denoising)
- **Actor-critic**: standard, trained on imagined trajectories

Key trick: DIAMOND uses **noise scheduling** that conditions on the
current observation, so the diffusion model only needs to predict the
delta (the residual), not the whole next frame from scratch.

## 3. Results

- Atari 100k benchmark: DIAMOND reaches 1.46x human-normalized score
  median, competitive with DreamerV3 (1.46x) and IRIS (~1.5x)
- DMControl: SOTA on continuous-control benchmarks
- **CSGO (a long-horizon task)**: DIAMOND significantly outperforms
  DreamerV3 — the diffusion-based dynamics handle long-horizon
  uncertainty better than Gaussian mixture heads

## 4. Why this matters for Project C

1. **DreamerV3 is no longer the only SOTA**: in 2024+ there's a
   serious alternative. Paper C's Related Work must position our
   slot-WM approach vs DIAMOND.
2. **Diffusion-WM may be simpler**: DIAMOND avoids the slot-attention
   + SCM complexity entirely. If a small DIAMOND baseline on Procgen
   matches or exceeds our slot-WM, that's a negative result for the
   "object-centric prior is necessary" claim.
3. **Long-horizon advantage**: DIAMOND's diffusion dynamics handle
   long horizons better than Gaussian heads. Project C's cross-domain
   goal (Crafter, Procgen 16 games) is long-horizon; DIAMOND is a
   natural baseline.

## 5. Comparison with our Project C direction

Our Project C uses **slot-attention world model + SCM**:
- Slot attention: object-centric representation
- SCM: causal structure (interventions, counterfactuals)
- Latent imagination for planning

DIAMOND uses **pixel-space diffusion**:
- No explicit object decomposition
- No explicit causal structure
- Diffusion dynamics handle stochasticity and long horizons naturally

The two approaches have different bets:
- **Our bet**: object-centric + causal structure is necessary for
  *transfer* across domains (Pearl L2+)
- **DIAMOND's bet**: pixel-space + diffusion is sufficient for
  *performance* on single domains but does not enable transfer

If DIAMOND-with-domain-randomization transfers, our bet is wrong. If
it doesn't, our bet is right.

## 6. Action items for Project C

1. **Paper C Related Work**: add DIAMOND as primary alternative baseline.
   Discuss tradeoffs of object-centric vs pixel-space approaches.
2. **Y1 experiment**: small DIAMOND baseline on Procgen coinrun to
   compare against our slot-WM. If DIAMOND < slot-WM on transfer
   metrics, our object-centric bet is validated. If DIAMOND > slot-WM,
   we need a v2 architecture.
3. **Hybrid architecture**: combine DIAMOND's diffusion dynamics with
   our slot-attention + SCM. DreamerV4-style. This is Y2+ work.

## 7. Critique / open questions

- DIAMOND's diffusion model is expensive (50-100 denoising steps per
  imagined step). On CPU this may be infeasible. Our setting is CPU-only
  so we need a small-scale DIAMOND or accept GPU.
- DIAMOND does not address **Pearl L3 (counterfactuals)**. Even with
  a great forward model, "what would have happened if X?" requires
  SCM-style interventions, not pixel-space diffusion.
- DIAMOND's transfer claim is implicit (DMControl variants) but not
  formally tested. This is exactly what Project C should test.

## 8. Connection to F:\TMLR H/I series

- I03 World Models: DIAMOND is one of the three 2024-2025 branches of
  world modeling (DreamerV3 / DIAMOND / Video-as-WM).
- I04 Multimodal: DIAMOND is a generative world model that uses vision
  natively; this aligns with video-generation-as-WM trend.

## 9. Cite in Paper C

Add to Paper C Related Work:
> Alonso et al. (2024) introduced DIAMOND, replacing DreamerV3's
> discrete latent dynamics with a continuous diffusion model in pixel
> space. DIAMOND achieves competitive performance on Atari 100k and
> SOTA on DMControl, but does not address cross-domain transfer or
> counterfactual reasoning (Pearl L3). Our slot-attention + SCM
> approach bets that object-centric + causal structure is necessary
> for these properties; we test this in the cross-domain evaluation
> of Section 5.

## 10. One-line takeaway

DIAMOND is the 2024 alternative to DreamerV3 — diffusion-dynamics in
pixel space, no object decomposition, no causal structure. Paper C
must position our slot+SCM approach against this serious baseline.