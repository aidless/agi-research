# Self-Critical Sequence Training (Rennie et al. 2017)

> Date read: 2026-07-25 (from Codex training-data memory)
> Confidence: **HIGH** -- classic image-captioning paper, well-cited
> One-line takeaway: SCST is the first widely-cited "decoupled critic"
> architecture: a separate reward model trained on the policy\'s outputs
> drives policy updates, without gradient flow between them.

---

## Why this paper matters for our program

This paper is **the conceptual ancestor** of Project A\'s decoupled
monitor, although applied to supervised sequence generation rather than
RL-on-actions. SCST was the 2017-style answer to "why can\'t we just
backprop the reward through the LSTM?".

Reading SCST tells us:
1. the historical precedent for decoupling
2. what kind of decoupling works / does not work
3. what reviewers expect when we make decoupling claims in a different
   domain (RL actions)

## Problem (in 2017 context)

Image captioning systems used the metric XE (cross-entropy) loss at
training time. At inference time, they were scored on CIDEr / METEOR /
BLEU. Train-inference mismatch is the central problem.

XE training maximises likelihood of ground-truth tokens; CIDEr rewards
n-gram overlap with reference. They correlate poorly.

Vanilla solution: policy gradient on CIDEr. But REINFORCE has high
variance. Need a low-variance baseline.

## Method

SCST:
1. Inference: sample two captions per image, one from current policy
   (sampled) and one greedy (baseline).
2. Reward: `R = CIDEr(c_sampled) - CIDEr(c_greedy)`. This is the
   advantage baseline itself.
3. Update: standard policy gradient on `log p(c | I) * R`.

Two key facts:
- The "critic" is implicit in the inference loop (greedy decode).
- The critic is **not** gradient-updated with the policy. It is fully
  decoupled.

(Strictly speaking: the critic is the policy itself in greedy mode,
not a separate network. But architecturally it is "another view of the
policy", and it is *not* gradient-updated jointly.)

## Empirical result

Rennie 2017:
- 7-point CIDEr improvement on MSCOCO image captioning over XE baseline
- SOTA at publication on the COCO test server

Compute cost: similar to XE baseline (no critic network needed).

## Criticisms

1. SCST\'s "critic" is just greedy decoding, not a learned network. So
   the decoupling is not a separate architecture; it is just
   "use a self-baseline".
2. SCST assumes the metric is differentiable-ish (CIDEr is a score, not
   gradient, but the resulting gradient is well-defined because
   `log p(c | I)` is).
3. SCST does NOT claim to predict failures ahead-of-time. The "self-
   critique" is in hindsight, not prospective.

## Connection to our program

Project A\'s decoupled monitor is **the next step past SCST**:
- architecturally separated critic (not just greedy decoding)
- prospective (predicts "will fail" before outcome)
- for RL actions, not caption tokens

We can write Project A paper\'s Related Work section around SCST + 6
years of follow-up. We can argue:
- the architecture of decoupling matters (our frozen-policy critic is
  more architecturally decoupled than SCST\'s self-baseline)
- prospective monitoring is something SCST never tried

We should cite this in section 2 of Project A paper outline.

## Confidence

HIGH. Well-documented and widely-cited image-captioning literature.

What to re-verify:
- exact implementation of the CIDEr gradient
- whether SCST carries over to RL unchanged or whether something
  similar in RL is the only relevant precedent

## Related papers

- MIXER (Ranzato 2015) -- similar PG on sequence scoring
- Actor-Critic (Konda 2000) -- foundational critic-as-baseline
- Self-Critical variants:  
  - Non-autoregressive SCST (Li 2019)  
  - SCST for video captioning (Chen 2018)

## Status

- [x] cite in Project A paper Related Work
- [ ] re-read Rennie 2017 figures in original paper
