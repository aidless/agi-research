# Neuro-Symbolic Concept Learner (Mao et al. 2019)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: An end-to-end NS-CL learns visual concepts via soft logic, parses
> scenes into structured programs, then answers questions by executing the
> programs symbolically over the learned concepts. State-of-the-art on CLEVR.

## Problem
Visual question answering (VQA) needs both perception (what objects) and
reasoning (combine). Deep models alone are black-box; symbolic systems alone
do not handle visual noise. We need both.

## Method
Three modules trained jointly:
- **Concept learner**: images -> object-level representations (deep, with
  learned features).
- **Parser**: images + concepts -> symbolic scene description.
- **Executor**: symbolic queries -> answers by executing programs over the
  scene description.

Soft logic makes the parser differentiable and end-to-end trainable. The executor
is purely symbolic (no gradients needed).

## Empirical result
- CLEVR: 98.7% accuracy (state-of-the-art at publication).
- Generalises to CLEVR-CoGenT (different attribute spaces per train vs test):
  >95% accuracy (deep CV baselines drop to ~50%).
- Strong interpretability: the parsed program can be inspected.

## Criticisms (specific)
1. **Limited visual diversity**: CLEVR is synthetic. Real-world CLEVR-class
   scenes are harder.
2. **Domain-specific**: parser syntax is hand-coded for CLEVR-like scenes.
3. **No action grounding**: the system does not act on the world.
4. **Symbolic assumption**: the parser assumes discrete-symbol output -
   mis-grasp to noisy scenes.

## Connection to our program
For Project D (language types):
- NS-CL is the canonical "structured output from perception" system.
- Our slot-WM produces slot latents; NS-CL shows how to extract structured
  predicates (concept learner) from those.

For Project E (verification):
- The executor is symbolic; ours should be similarly symbolic.
- Verification-as-execution is a direct analogue.

For Project A (Monitor):
- A Monitor could be NS-CL: parse observation into objects, check whether
  the parsed scene matches expected configuration.

## Confidence
HIGH.

## Related
- Visual Chain-of-Thought (V-CoT) (Ding 2022)
- NEUSTL (Atzeni 2024)
- CLEVR (Johnson 2017)
- Symbolic Concept Extraction (Loconte 2024)
- DreamCoder (Ellis 2023)
- Scallop (Manning 2020)

## Status
- cited in Project D Related Work (predicate extraction from perception)
- cited in Project E Related Work (verifier symbolic execution)
- cited in Project A future (Monitor as concept learner)
