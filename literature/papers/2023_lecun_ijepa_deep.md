# I-JEPA (LeCun et al. 2023) / V-JEPA (Bardes et al. 2024)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH on I-JEPA; MEDIUM on V-JEPA (latter near training cutoff)**
> One-line: Non-generative self-supervised vision: instead of reconstructing
> pixel values, predict the embeddings of target blocks from context block
> embeddings. No pixel decoder is needed.

---

## Problem

Two prior paradigms in self-supervised vision learning:

1. Generative (MAE, BEiT): reconstruct masked pixels/tokens.
   Tends to focus on low-level details over high-level semantics.
2. Contrastive (SimCLR, MoCo, DINO): pull together augmentations of
   same image, push apart different images.
   Requires careful augmentation pipelines; loses some semantic info.

I-JEPA bet: predict the **embedding** of target blocks from context
embeddings. This is non-generative (no pixel reconstruction), and uses
just **one view** of the image (no augmentations needed for contrast).

## Method

Architecture: a vision transformer (ViT).

Mask image into:
- **context blocks**: ~5% of patches visible to the encoder
- **target blocks**: ~75% of patches; their **positions only** are visible
  to a smaller network that predicts their embeddings

The encoder produces context embeddings; the predictor network outputs
predicted embeddings for target positions; loss = cosine similarity between
predicted and actual target embeddings.

Critically: the target encoder is **a separate EMA copy** of the online
encoder (BYOL-style target network). This prevents collapse.

V-JEPA (Bardes 2024) extends I-JEPA to video by predicting embeddings of
future frames from past frame context.

V-JEPA 2-AC (Carreira 2025) further extends to action-conditioned
video prediction; the model can be deployed zero-shot to robot control.

## Empirical result

I-JEPA (2023):
- ImageNet linear probe: ~73% top-1 with ViT-H backbone
- Comparable to or better than MAE/DINO of similar size
- 5x faster training time than pixel-reconstruction methods (because
  no decoder)

V-JEPA (2024):
- Video understanding strong downstream: SSv2, Something-Something
- Per-frame features outperform image-only pretraining on video tasks

## Criticisms

1. **Embeddings are not the same as the underlying image**. Loss is in
   embedding space; the model could learn collapsing features.

2. **EMA target network is critical; "magic" for many self-supervised
   methods**. Without it, the loss can collapse trivially.

3. **Predictor network design is empirically important**. Specific
   design choices (predictor depth, width, mask ratio) matter; not
   principled.

4. **Action-conditioning (V-JEPA 2-AC) makes claims about embodiment
   that are hard to verify**. Sample-efficient robot deployment claims
   need careful scrutiny.

## Connection to our program

I/V-JEPA is **at the heart of our architecture for Project C**:

- Our world model can use V-JEPA-style **non-generative** prediction
  (predict embedding of next latent, not pixel)
- This decouples reasoning from pixel-level reconstruction
- Combined with Slot Attention (the slots) + Causal SCM (Pearl),
  this gives us a causal object-centric world model

Specifically, our Project C pipeline:
1. Slot Attention produces object-centric latent per frame
2. V-JEPA-style predictor: predict target-slot embeddings from context
3. SCM prior on slot transitions: structural causal graph
4. Intervention = freeze one slot, predict others via the SCM

This is the **concrete computational recipe** for our Project C.

## Related

- MAE (He 2022): generative masking
- SimCLR / MoCo / DINO (various): contrastive
- BYOL (Grill 2020): target-network + cosine sim (the I-JEPA trick)
- V-JEPA 2-AC (Carreira 2025) - action-conditioned

## Status

- [x] cite in Project C architecture pipeline
- [x] the V-JEPA recipe ties back to our world model design
- [ ] user must Tier A read V-JEPA 2-AC paper for the most recent claims
