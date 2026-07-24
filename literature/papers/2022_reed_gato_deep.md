# Gato (Reed et al. 2022)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- widely-discussed, code not open but methodology well-documented
> One-line: A single 1.2B-parameter transformer trained as a general agent --
> dialogue, image captioning, robotic arm control, Atari games, all via
> token prediction. One model, many tasks.

---

## Problem

The headline question of Gato (DeepMind): can one model be a *generalist*
across many modalities and tasks, with a single training algorithm?

Prior multi-task agents (e.g., Decision Transformer, Gato predecessors
like DT at scale) had task-specific encoders/decoders or suffered
catastrophic forgetting across very different tasks.

## Method

Everything is a **token sequence**:
- text: language tokens
- images: ViT-style patch tokens
- proprioceptive signals: discretised via k-means, treated as text tokens
- continuous action vectors: discretised, treated as text tokens
- rewards: discretised, treated as text tokens

Architecture: standard transformer decoder, 1.2B parameters, 24 layers,
context length ~1024 tokens.

Training: standard next-token-prediction cross-entropy. All tasks use
the same loss; the data mixture is the only "task-specific" knob.

Inference: conditional on task-specific "prompt tokens" so the model
knows what task to do.

## Empirical result

Gato at 1.2B parameters:
- 604 tasks across 23 domains (dialog, image, embodied, RL games, robotics)
- Above random on ALL 604 tasks in the test set
- Super-human on ~23% of tasks (mostly Atari at >=1 expert level)
- At-or-below human on the rest (varies)

The point is not the absolute performance -- specialised models beat it.
The point is the **uniform coverage**.

## Criticisms

1. **Absolute performance is often poor**. On harder tasks (e.g., realistic
   robot manipulation), Gato is far behind specialised methods. This is
   sometimes downplayed as "model is small (1.2B)" but is the real story.

2. **Continuations are discretised, which loses fidelity**. Actions
   discretised into 1024 bins (per DOF) have inherent error. Robotic arm
   tasks typically need high-precision control.

3. **The architecture is heavily biased toward language by data mixture**.
   ~85% of training tokens are language. Despite multi-task training,
   the model is essentially a smart text predictor with extra modalities.

4. **One model = competence average**. Gato is rarely best at anything
   except the tasks it was specifically trained for. The "general" framing
   obscures that.

5. **Generation diversity / planning is limited**. Standard transformers
   used this way are stochastic but myopic: Gato does not "plan ahead"
   in the sense of MCTS. So when paired with reasoning-required tasks
   (Block-stacking puzzles, etc.), it can fail.

## Connection to our program

Gato is **the closest published demonstration** of "one model for many
kinds of decisions", which is exactly our Project D's spirit. We should:

- Use Gato as the multi-task reference (architectural simplicity)
- Distinguish our contribution: Gato does token-prediction over
  everything; we want language-as-type-system over latent predicates,
  which is fundamentally different. Language tokens in Gato are first-
  class; in our project they describe latent state.

This distinction is critical for our positioning. Gato is "broader
predictor". We are "type-theorist".

## Concrete next move

When writing Project D paper, have a Section 2.2 dedicated to Gato:
"Gato demonstrates one-model-many-tasks feasibility. We argue that
lifting language to *type system* over latent predicates, rather than
extending token-prediction as Gato does, is the path to compositional
generalisation."

## Confidence

HIGH for architecture and headline results. Re-verify:
- exact 604 task count
- exact data-mixture proportions
- the specific discretisation scheme for proprioception / action

## Related papers

- Decision Transformer (Chen 2021): sequence-modelling RL, single task class
- Trajectory Transformer (Janner 2021): long-horizon planning via transformers
- RT-1 / RT-2 (Google DeepMind robotics): language-conditioned robot action
- Unified-IO (Lu 2022): multi-task multi-modal but more vision-language focus
- Flamingo (DeepMind 2022): multimodal but image+text only
- LLaVA (Liu 2023): open PaLM-E-like model

## Status

- [x] cite in Project D paper Related Work
- [x] highlight the distinguishing position in our contribution
- [ ] revisit per-task performance vs specialised methods in original paper
