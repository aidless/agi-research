# PaLM-E (Driess et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- paper widely documented, code released
> One-line: A 562B parameter language model trained jointly on text + continuous
> sensor embeddings (vision, proprioception). Demonstrates language as the
> natural grounding interface for embodied agents.

---

## Problem

Two prior paradigms in embodied AI:

1. Pure RL / behaviour cloning: small models, task-specific, no language grounding
2. LLM planners: call out to a planner, but no access to real sensor data

PaLM-E (Embodied Multimodal Language Model) unifies: **inject** continuous
sensor tokens directly into the LLM input. The LLM itself outputs actions.

## Method

Architecture:
- pretrained frozen PaLM-540B language model
- small per-modality **encoders** (vision: ViT-22B; proprioception: MLP)
- embeddings from these encoders are converted via a learnable linear projection
  into the LLM token embedding space
- LLM is fine-tuned on a mix of tasks across modalities

Training data mixture:
- T5 / PaLM text tasks (preserved to avoid losing language skill)
- VQA (vision + language QA)
- embodied sequence-to-sequence tasks ("pick the apple", "move to red cup")
- classification tasks from robot manipulation datasets
- captioning, navigation, etc.

Total multi-task mixture: dozens of task families across hundreds of envs.

## Empirical result

- Outperforms task-specific SOTA on most embodied tasks the paper tests:
  - OK-VQA (visual question answering)
  - some manipulation benchmarks
  - some navigation benchmarks (specifically R2R)
- Surprising transfer: a model trained on mostly robot data can do VQA
  reasonably well; a model trained on mostly VQA can do manipulation
- Notable sample efficiency result: PaLM-E requires orders of magnitude
  less robot training data than pure-RL baselines by leveraging the
  pretrained language prior

## Criticisms

1. **The LLM is frozen first then unfrozen**. Per-modality encoders are tiny
   relative to LLM. Most of the adaptation cost is on the LLM side, hidden
   behind "fine-tuning". This means much of the "transfer" can be hidden
   by the LLM's prior.

2. **Embodied results are mostly on simulation**. Real-robot results are
   subset, especially for larger tasks. PaLM-E is mostly a research /
   dmonstration system, not a deployable agent.

3. **It is unclear how much continuous-sensor "grounding" actually happens**.
   The architecture inputs sensor tokens, but the language side is dominant.
   Critics argue this is essentially "language model with extra inputs",
   not true grounding.

4. **The data mixture is opaque**. Exact proportions and per-task weights
   are not fully documented; reproducing the multi-task training is not trivial.

## Connection to our program (Project D)

PaLM-E is the **closest existing implementation** of our Project D's vision:
language-as-type-system over multimodal grounding. We should:

- Use PaLM-E as our reference architecture in Project D paper Related Work
- Distinguish our contribution: we want type-system over latent predicates
  not just text-conditioned actions. Specifically, we want the language
  to specify *function signatures* over the World Model's latent state,
  not just action sequences.
- The PaLM-E result that "language-only model + multimodal embedding
  functions" works is the proof-of-concept for our research direction.
  Without PaLM-E, the LLM+WM integration would have to be argued from
  first principles.

## Confidence

HIGH for architecture and headline numbers. Re-verify:
- exact ViT architecture used (I recall ViT-22B but variants exist)
- exact training mixture proportions
- exact numbers vs CLIP-style baselines

## Related

- Flamingo (DeepMind 2022): few-shot multimodal LM, similar design
- BLIP-2 (Li 2023): lighter multimodal LM
- LLaVA (Liu 2023): open multimodal conversational model
- Gato (Reed 2022): multi-task generalist agent (different architecture)

## Status

- [x] cite in Project D paper Related Work
- [x] cite in TASKBOOK architecture v2 mapping
- [ ] verify numbers vs CLIP-family baselines
