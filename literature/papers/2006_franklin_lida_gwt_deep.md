# Global Workspace Theory & LIDA (Baars 1988; Franklin 2006)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** (theoretical frameworks, well-documented)
> One-line: A cognitive architecture based on the idea that consciousness
> is a global workspace where information from specialised modules is
> integrated and broadcast back. LIDA is a software implementation of this
> framework.

---

## Problem

Classical cognitive architectures (production systems, declarative
memory) lack a unified theory of how specialised modules interact to
produce coherent cognition. GWT proposes a specific mechanism for this.

## Method

GWT's central claim: at any moment, one coalition of specialised
processors competes to be the "global workspace" -- the content that
becomes available to all other modules. The winning coalition's
information is broadcast globally; this is what we experience as
consciousness.

Key mechanisms:
- **Coalition competition**: modules compete for attention
- **Broadcast**: the winning content is distributed to all modules
- **Integration**: broadcasted content modifies subsequent processing
- **Cyclic operation**: this happens in ~100ms cycles (the "psychological
  present")

LIDA implements GWT as a software architecture:
- **Sensory modules**: low-level feature extraction
- **Perceptual modules**: object recognition
- **Workspace**: a shared blackboard of broadcast content
- **Motor modules**: action selection
- **Memory modules**: episodic, semantic, procedural

Each "cognitive cycle" runs through these, broadcasting through the
workspace.

## Criticisms

1. **GWT is theoretical, not mechanistic**. Critics argue it describes
   what consciousness looks like, not what it IS. The lack of a
   physical substrate is a feature for some (consciousness is the
   workspace; substrate is irrelevant) but a bug for others.

2. **LIDA implementations are large and brittle**. Real-world
   deployment of LIDA-style systems is limited.

3. **The 100ms cycle is empirical, not principled**. Brain rhythms
   do match, but the connection to "psychological present" is loose.

4. **Doesn't directly inform modern ML**. GWT may inform attention
   mechanisms in transformers, but LIDA specifically has had minimal
   direct impact on deep learning research.

## Connection to our program

GWT/LIDA are **the cognitive-architecture foundation for our Project
A's self-model**:

- Our "Self-Model" block is essentially a global workspace over the
  agent's own prediction components
- The Monitor's failure prediction competes for attention with the
  policy's action decision
- When the Monitor broadcasts high-failure-probability, the planner
  should integrate this and respond accordingly

This gives us a **principled justification** for our architecture:
the monitor is the global-workspace of self-awareness.

We can write Project A paper Section 1 (Introduction):
"Following GWT, we propose that failure-awareness is the global
workspace where the agent's self-model integrates."

## Related papers

- Baars 1988 (GWT theory)
- Baars, Franklin, Ramsoy 2013 (updated GWT)
- ACT-R (Anderson 2007) - alternative cognitive architecture
- SOAR (Laird 2012) - production systems
- CLARION (Sun 2001) - dual-process theory

## Status

- [x] cite in Project A Section 1 (architectural justification for self-model)
- [x] cite in TASKBOOK cognitive-architecture reference
