# Constitutional AI (Bai et al. 2022)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: Train an LLM to follow "principles" by self-critiquing its own
> outputs against a written constitution -- avoiding expensive human harm
> labelling while still improving safety/alignment.

---

## Problem

RLHF needs expensive human preference labels on every training example.
Constitutional AI bet: replace most human labelling with **LLM self-critique**
against a written constitution.

## Method

Two phases:

**Phase 1 - SL-CAI (Supervised Learning Constitutional AI)**:
For each (prompt, response), ask the model to:
1. critique its own response against a principle
2. revise the response
3. repeat

The revised responses become training data for supervised fine-tuning.

**Phase 2 - RL-CAI (RL with Constitutional AI)**:
For each (prompt, response A, response B), ask the model which better
follows the constitution. Train a preference model from these synthetic
labels. Use RLAIF (instead of RLHF).

The "constitution" is a list of ~50 principles; examples:
- "Choose the response that is least harmful"
- "Don't help with violence"
- "Be respectful of cultural differences"

## Empirical result

- Claude (Anthropic's main product) is trained with CAI principles
- Empirically, CAI produces models that match or beat RLHF on harmlessness
  benchmarks, with much less human labelling

## Criticisms

1. **The constitution's quality is critical**. If the principles are wrong
   or contradictory, the model learns conflicting feedback.

2. **Self-critique is only as good as the model's ability to assess its
   own output**. If the LLM is sycophantic, it may approve its bad
   responses.

3. **Long-term coherence vs short-term constitution matching**.
   Principles don't capture all of what humans value.

4. **Constitutional principles from whom and how determined**? The
   principles encode values. Whose values? This is a political question
   Anthropic has had to navigate.

5. **It does NOT solve the alignment problem**. It addresses certain
   surface behaviours while the underlying values/preferences of the
   LLM are not fully constrained.

## Connection to our program

**This is a HUGE conceptual reference for Project A**.

Constitutional AI is the LLM-world analogue of our Project A idea:
self-critique as a separate module. They trained the SAME model to
self-critique; we propose a separate architecture for self-critique.

Why our architecture might do better than CAI:
- CAI's critique module = same model as the policy. Same gradients,
  same biases. Could reinforce existing preferences.
- Our Monitor = architecture-frozen separate network. Independent
  parameter space, so capture-of-bias is less likely.

We can write Project A paper Discussion:
"Constitutional AI uses LLM self-critique for safety; our
frozen-policy decoupled critic extends this idea to RL agents with
explicit separation of parameters."

This is precisely the conceptual bridge from current LLM-self-critique
literature to RL-agent-self-critique.

## Related papers

- RLHF (Christiano 2017): the predecessor
- RLAIF (Lee 2023): AI feedback more generally
- Self-Refine (Madaan 2023): iterative refinement
- Constitutional AI harmlessness paper (Bai 2022)

## Status

- [x] cite in Project A Related Work (essential -- direct conceptual predecessor)
- [x] cite in self-mon critique bridge
