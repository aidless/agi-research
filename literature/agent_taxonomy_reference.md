# AI Agent Taxonomy Reference (2026-07-25)

> Source: user-provided overview document (Russell & Norvig-aligned).
> Saved as workspace reference. Not a research artefact; it is the
> shared vocabulary all future papers / code in this workspace speak.

---

## Quick index

- **Definition**: Agent = perceiver + actor in environment (Russell & Norvig)
- **6 core traits**: autonomy, perception, reactivity, pro-activeness,
  social ability, learning, reasoning/planning
- **6 agent types** (Russell/Norvig):
  1. Simple Reflex  (condition->action)
  2. Model-based Reflex (perception + internal state)
  3. Goal-based (planning)
  4. Utility-based (multi-objective)
  5. Learning (4 components: learning / performance / critic / problem-generator)
  6. LLM-based (Profile + Memory + Planning + Action)

## Mapping to our architecture (4-layer v2)

| Layer | Maps to which agent type | Notes |
|---|---|---|
| World Model (Project C) | Model-based Reflex (extended) + Goal-based | State + planning |
| LLM-as-type-system (D) | LLM-based Agent | Profile + Memory + Action |
| VLA / executor | Performance element of Learning Agent | Output of policy |
| Self-Monitor (Project A) | Critic element of Learning Agent | The 4th component |
| Neuro-symbolic verify (E) | Utility-based (formal) | Verification layer |

Our 5-year program is essentially building a **custom 4-element
Learning Agent at AGI scale**, where the 4 elements are themselves
each research sub-projects (A, B, C, D, E).

## The closing paragraph is the key

> Current LLM-based Agent, while powerful, is limited by the
> statistical substrate of LLM — limited reasoning, hallucination,
> unreliable long-term planning. That is why academia + industry is
> exploring deep fusion of formal reasoning (symbolic AI), planning
> (PDDL), and RL with LLM --- to build reliable next-gen agents.

This is exactly what our 4-layer architecture does. We are aligned
with the published direction. The unresolved bits are: causal-Pearl-L3
(Project C) and the verification (Project E).

## "Or you can first optimise current tools"

This is the recursive pivot: apply Project A\'s monitor principle
TO OURSELVES first. The current workspace IS a custom Agent (you +
Codex). Improving the tooling of this Agent is the lowest-cost
demonstration of Project A.

Concrete tools to build (decision pending):
1. Status CLI (read all root docs and digest)
2. Experiment template generator (write experiments_log/<date>.md)
3. Review-marker tracker (which files need user review)
4. Citation index (BibTeX-style for the papers we read)
