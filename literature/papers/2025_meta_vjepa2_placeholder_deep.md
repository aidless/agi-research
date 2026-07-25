# V-JEPA 2-AC (Carreira et al. 2025) — placeholder note

> Date: 2026-07-25 placeholder. **REQUIRES USER PRIMARY READ** before paper-grade citation.
> Confidence: **LOW** (only Tier A abstract-level knowledge)
> One-line: An action-conditioned video world model that learns from ~1M hours of video
> and 62 hours of robot data, then zero-shot transfers to new robot platforms
> (Franka arm) for pick-and-place tasks.

---

## What the user must primary-read for, before I cite it:

- exact perception backbone (I recall ViT-22B but uncertain)
- exact fine-tuning data composition (the 62 hours of robot data — what kinds?)
- exact zero-shot performance on the held-out Franka setups
- the AC (action-conditioning) mechanism details — is it concat to latents, or
  separate query tokens?

## What I can say from abstract-level memory:

- V-JEPA 2 (Bardes 2024): extends I-JEPA to video by predicting embeddings of
  future frames from past frame context.
- V-JEPA 2-AC (Carreira 2025): action-conditioned variant. Demonstrates that
  ~62 hours of robot fine-tuning data is enough to deploy to a *different* physical
  robot in a *new* lab setup, zero-shot. This is the kind of cross-embodiment
  transfer Project B aims to study in our 4-layer architecture.

## Connection to our program

V-JEPA 2-AC is **the most direct existing reference for Project B** (cross-domain
transfer to embodied agents). The 62-hour robot data + 1M-hour pretrain recipe
materially changes how we think about Project B's data budget: we may not need
millions of robot samples if we have a strong video-pretrained backbone.

We should cite this paper in:
- Project A Related Work (cross-domain mention)
- Project B primary reference (most direct)
- Project D language interface (V-JEPA 2-AC + PaLM-E is the closest thing to
  our language-as-type-system vision)

## What the placeholder needs from user

User to read V-JEPA 2-AC paper (2025-06 release, arXiv 2506.09985). After read,
replace this file's confidence flags with HIGH and fill in the missing details.

## Status

- [ ] user primary read required
- [ ] once read, this becomes HIGH-confidence Tier B note
