# Voyager (Wang et al. 2023) - Agentic Learning in Minecraft

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- widely-discussed, code available
> One-line: An LLM-driven agent that incrementally writes its own code as a
> growing "skill library", retrieving relevant skills on each new challenge.
> The skill library IS the agent memory -- no fine-tuning required.

## Problem this solves

Minecraft is open-ended: tasks stretch from "collect wood" to "defeat Ender Dragon".
Static RL agents excel at small sub-tasks but fail to grow. The question:
how do you make an agent that ACCUMULATES capabilities over time?

Prior approaches (RL, behaviour cloning, scripted curricula) struggle with
the combinatorial exploration problem.

## Method

Three components:
1. **Curriculum proposer**: GPT-4 proposes what to learn next based on current
   state and skill library.
2. **Code generator**: GPT-4 writes a Python function for the new skill, which
   gets executed in the Minecraft interface (Mineflayer).
3. **Skill library**: persistent, indexed by similarity to current state. New
   skills are added after a self-check pass.

Loop:
```
while not finished:
  state = env.observe()
  curriculum = llm(state, library) -> next goal
  if curriculum needs new skill:
    code = llm.write_skill(curriculum)
    if execute(code) works:
      library.add(code)
  action = llm.retrieve_and_select(state, library)
  env.step(action)
```

No gradient updates to the LLM weights. The skill library is the only memory.

## Empirical result

- **Minecraft tech tree**: Voyager obtained 3.3x more items and unlocked
  the diamond pickaxe 15x faster than baselines.
- **Skill library**: 250+ reusable skills by end of curriculum.
- **Generalisation**: skills transfer to new tasks without re-learning.

## Criticisms (specific)

1. **No persistent memory across sessions**. The skill library is in-RAM.
   Lose it on restart, unless explicitly serialised.
2. **GPT-4 bottleneck**. Without GPT-4 quality, the generated code becomes
   buggy. Skill quality ceiling is the LLM.
3. **Skill library may not compose**. Two individually-working skills may
   fail when chained because of side effects on world state.
4. **No principled safety**. The agent writes arbitrary code in Minecraft,
   no policy layer.

## Connection to our program

Trend #3 (Agentic Learning) maps directly to Voyager. The architectural idea
that applies to our research assistant:

- **`.experience_log/` is the equivalent of Voyager's skill library**.
  Each entry is a (situation, what-tried, what-worked) record. Future
  sessions retrieve relevant experiences and apply.
- **Codex does NOT update its weights** -- but it DOES update the workspace
  (PROGRESS.md, code files, etc.) so future sessions see the latest state.
- Voyager's weaknesses (no safety, no composition, LLM-bottlenecked) are
  *exactly* what Project I (Cedar-like policy) and Project F (workspace tools)
  try to fix for our case.

We should cite Voyager as the canonical "agentic learning" example, while
applying the lesson that skill libraries need durable persistence + safety
layer + validation.

## Confidence

HIGH. Wang et al. paper widely cited; the code released.

Re-verify:
- the exact GPT-4 calls per step
- the diagamming of the skill library.

## Related

- Auto-GPT (Significant Gravitas 2023): early agentic loop with GPT-4
- BabyAGI (Nakajima 2023): task-driven agentic loop
- Generative Agents (Park et al. 2023): LLM agents with memory stream
- Reflexion (Shinn et al. 2023): self-reflective agents
- AWM (agent workflow memory) follow-up work, 2024-2025

## Status

- [x] cite in Trend #3 rationale (Agentic Learning)
- [x] cite in Project F (skill library analog to .experience_log)
- [ ] future: implement skill-library on top of .experience_log
