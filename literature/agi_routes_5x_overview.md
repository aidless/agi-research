# AGI Technical Routes - Comprehensive Map (2026-07-25)

> Source: user-provided document on the 5 main AGI routes and their
> representative works. Saved as a canonical reference for the whole programme.

---

## The 5 routes (Russell/Norvig-Chollet-aligned view)

| # | Route | Core claim | Champion | Limits |
|---|---|---|---|---|
| 1 | Scaling (LLM) | Bigger model + data + compute -> general intelligence | OpenAI/Google/Anthropic | Real understanding vs pattern match debate; data wall approaching |
| 2 | Neuro-Symbolic | Neural perception + symbolic reasoning | AlphaGo, DreamCoder, Scallop | Interface between paradigms is unsolved |
| 3 | World Models | Internal world simulator, enables planning + counterfactual | LeCun (JEPA/V-JEPA), DreamerV3 | Pearl L3 not solved; object discovery unsolved |
| 4 | Embodied | Intelligence needs physical interaction | Gato, robot foundation models | Long-horizon weak; sim-to-real open |
| 5 | Cognitive Architecture | Unified framework (memory, attention, learning) | SOAR, ACT-R, LIDA | Scaling unclear; rarely competitive with deep nets |

## Key insight (the document's own conclusion)

> AGI is unlikely to come from a single technology breakthrough.
> It is more likely to be **the fusion of multiple technical routes**:
> - LLM provides language understanding + reasoning base
> - World Model provides causal understanding
> - Agent architecture provides autonomous action
> - Embodied intelligence provides physical interaction experience
> - Cognitive architecture provides a unified framework

This is **exactly our 4-layer architecture**, just spelled out across a wider space.

## Chollet quote (this reframes everything)

> "AGI的本质不是拥有大量技能，而是获取新技能的能力。
> 真正的通用智能不在于你知道多少，而在于你面对未知时能多有效地学习。"
> -- Francois Chollet (DeepMind / ARC-AGI)

This is the holy framing for our programme:
- We measure Project A not by "monitor AUROC on CartPole",
  but by **how quickly monitor transfers to a new task**.
- We measure Project B by "time-to-acquire on novel env (zero-shot)"
- We measure Project C by "how many causal mechanisms can be learned
  from N interventions for unseen objects"
- We do NOT measure LLM by next-token perplexity. We measure it
  by **type-scaffolding efficiency** for new predicates.

## Mapping 5 routes -> our 4-layer architecture

| 4-layer component | Maps to which route | Where in our program |
|---|---|---|
| LLM (semantic + type system) | Route 1 (Scaling) + Route 2 (Symbolic) | Project D |
| World Model | Route 3 | Project C |
| VLA / executor | Route 4 (Embodied) | Project B |
| Self-Model | Route 5 (Cognitive Arch) | Project A |
| Neuro-symbolic verification | Route 2 (Symbolic) | Project E (still missing) |

All 5 routes are present in our program. We are not betting on a single route.

## Roadmap implications (delta from current)

1. **Reorient KPI**: from "task A performance" to "learning curve on
   novel task" (per Chollet). Need to define "novel task" carefully.
2. **Cognitive Architecture (Route 5) becomes elevated**: SOAR/ACT-R
   heritage adds the concept of production systems + declarative memory
   to our architecture. Worth one literature review.
3. **Embodied (Route 4) becomes explicit in our roadmap**: we cannot
   skip VLA grounding just because Project B is "P1".

## Suggested reading-list additions

- Francois Chollet's writings on measure of intelligence (ARC-AGI paper)
- SOAR (Laird 2012) - production systems for cognitive architectures
- ACT-R (Anderson 2007) - declarative + procedural memory split
- LIDA (Franklin 2006) - global workspace theory in software
