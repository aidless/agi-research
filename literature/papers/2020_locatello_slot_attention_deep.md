# Slot Attention (Locatello et al. 2020)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- well-documented, multiple follow-up papers
> One-line: A permutation-equivariant iterative attention mechanism that
> decomposes a scene into a fixed number of "slots", each representing
> an object, without any supervision about object identity.

---

## Problem

Standard CNN / ViT features for downstream tasks entangle objects: a
single feature map contains "all pixels blended together". Downstream
tasks must learn to disentangle which feature corresponds to which
object.

Object-centric representations (separate feature per object) have
theoretical advantages: better sample efficiency, fewer parameters,
compositional generalisation, interpretability. But unsupervised
object discovery is hard.

## Method

Slot Attention is an **iterative attention** procedure:

Inputs: feature grid F (e.g., CNN output of shape B x H x W x C = batch of
spatial features), and K slot initialisations (random or learned).

For T iterations:
  1. Compute attention weights: for each slot, softmax over input features
     (input features compete to belong to slot)
  2. Update slots: each slot is a normalised weighted sum of input features
     (slot updates by aggregating assigned features)
  3. Slot-to-slot interactions: optional normalisation or GRU-style update

After T iterations, you have K slots, each compactly representing a
different "object" of the scene.

Key trick: **shared normalization across slots in step 1**. The competition
ensures one slot doesn't dominate.

Loss function: reconstruction task that requires slot-based reasoning.
Typical: each slot is decoded independently; the decoder is a small
generative model; reconstruction loss is the per-pixel matching between
input and the **argmax-slot-softmax** decoded image.

## Empirical result

Scenes from CLEVR (synthetic 3D shapes) and CATER (complex object
dynamics): Slot Attention recovers near-perfect object localisation
without supervision. Reconstructions are visually separable into objects.

Generalization: a model trained on N slots generalises to images with
different numbers of objects up to a small range. Out-of-distribution
object counts fail gracefully.

## Criticisms

1. **Slot count K is fixed**. If a test image has more than K objects,
   the model has to "merge" some. This caps the model's scalability.

2. **No notion of slot permanence**. A "left chair" and a "right chair"
   after the camera pans are not guaranteed to correspond to the same
   slot. Solving this is HARD and partially open.

3. **Performance depends heavily on the decoder**. A simple per-slot
   autoencoder gives blurry reconstructions. The method requires
   non-trivial decoder architectures (Spatial Broadcast Decoder, etc.)
   to work well.

4. **Compute scaling is poor with T iterations**. Each iteration is
   O(K x HW x C). For a 224x224 image with K=10 slots, T=5 iterations,
   that's significant.

5. **Permutation equivariance is a theoretical nice-to-have but
   downstream tasks often ignore it**. Real applications may need
   labelled slot-id mappings anyway.

## Connection to our program

This is the **single most important architectural primitive** for our
Project C. Here's why:

Pearl's ladder says: L1 (prediction) -> L2 (intervention) -> L3
(counterfactual). To do L2/L3 we need **decomposable representations**
so we can intervene on one object without disturbing others.

Slot Attention gives us exactly that: each slot is one object's
"internal state". To compute `do(a_on_object_3)`, we change only slot
3's transition; slots 1, 2, 4 are unaffected. This is the **mechanical
foundation** for L2 lifting.

Our Project C should adopt slot-attention as the latent representation,
then layer causal structure on top: a causal graph over slots.

Specifically, our world model becomes:
  - encoder produces N slots (Slot Attention)
  - dynamics: SCM over slots (learned DAG)
  - intervention: simulate with one slot's transition frozen

This is the natural marriage of object-centric + causal.

## Concrete next move

Slot attention code: there are open-source implementations (e.g. from
the original authors). Reproduce on CausalWorld, then layer Project C's
causal structure on top.

## Confidence

HIGH. Multiple reproduction papers; the mathematics is straightforward
(attention + iteration + normalisation).

Re-verify:
- exact T iteration count (default 3 or 5)
- the exact decoder architecture used
- per-dataset numbers

## Related papers

- MONet (Burgess 2019): same goal via different architecture (attention over a VAE grid)
- IODINE (Greff 2019): iterative refinement approach
- GENESIS (Engelcke 2019): scene-graph generation via slot attention
- DINOSAUR (Sitzmann 2023): slot attention for video
- SAVi (Kipf 2022): slot attention for video, used as backbone for object-centric dynamics

## Status

- [x] cite in Project C Section 3 (Method)
- [x] cite as primitive for L2 lifting
- [ ] write Project C paper outline using slot attention as starting point
