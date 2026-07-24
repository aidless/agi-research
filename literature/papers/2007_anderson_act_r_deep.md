# ACT-R (Anderson 2007)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** (canonical cognitive architecture textbook)
> One-line: Production-system cognitive architecture with declarative
> + procedural memory split. Models human cognition via a unified
> mathematical framework.

---

## Problem

Cognitive science needs a unified computational model of human cognition.
ACT-R is a long-running research program to provide one.

## Method

ACT-R's claim: human cognition can be modelled as:
- **Declarative memory**: chunks of facts (e.g., "Paris is the capital of France")
- **Procedural memory**: production rules (if-then)
- **Activation**: each declarative chunk has an activation level that
  decays; chunks above a threshold are retrievable
- **Matching**: production rules match against current buffer state

Learning happens by:
- Increasing activation of chunks that match current goals
- Adding new production rules
- Refining base-level learning rates

The architecture can fit human response-time and accuracy data across
hundreds of cognitive psychology experiments.

## Criticisms

1. **Limited applicability to AI engineering**. ACT-R is a model of
   HUMAN cognition; building similar architectures in software hasn't
   produced competitive AI systems.

2. **Symbol-grounding problem**. ACT-R's symbols (chunks) need to be
   grounded in environment. The "grounding" piece is not handled
   natively.

3. **Scale-up challenges**. ACT-R models of complex tasks (driver,
   aircraft pilot) require hand-coded procedures.

4. **Modern ML doesn't natively integrate with ACT-R**. Most deep
   learning architectures are NOT production-rule-based; bridging is
   non-trivial.

## Connection to our program

ACT-R gives us **a concrete memory architecture**:

- Our 4-layer architecture's "self-model" needs to remember past
  episodes and reason over them.
- ACT-R's declarative + procedural split suggests: **declarative memory =
  past experience index; procedural memory = skill execution rules**.

We can write Project A paper Section 2.1 Architecture:
"Our self-model follows ACT-R's declarative/procedural split. Past
failures are stored declaratively; the failure-prediction logic
is procedural, triggered when relevant conditions are met."

This adds theoretical grounding to our architecture.

## Related

- SOAR (Laird 2012)
- LIDA (Franklin 2006) - global workspace variant
- CLARION (Sun 2001) - explicit/implicit dual-process
- Sigma (2008) - graphical models for cognition

## Status

- [x] cite in self-model architectural justification
- [x] reference declarative/procedural memory split in Project A
