# Reflexion (Shinn et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: HIGH -- widely-discussed, code available
> One-line: agents that verbally reflect on failure traces and store the
> reflection in an episodic memory buffer for next-attempt retrieval; no
> weight updates.

## Problem

LLM agents fail on multi-step tasks. Pure trial-and-error retries repeat the
same mistakes. Without persistence, the agent does not learn between
attempts.

## Method

Three core components:
1. Actor: standard ReAct / chain-of-thought agent producing actions.
2. Self-reflection module: an LLM, prompted with the failed trajectory,
   generates a textual reflection ("I should have checked X earlier").
3. Episodic memory: stores (task, reflection) pairs in vector store.
4. Memory retrieval: next attempt retrieves top-K reflections for the task.

Loop:
for attempt in range(max_attempts):
    trajectory = actor.act(task, memory.retrieve(task))
    if evaluator(trajectory) == "success":
        return trajectory
    reflection = reflector(trajectory, evaluator.feedback)
    memory.add(task, reflection)
return None  # try again

## Empirical result

- HumanEval: 88.4% pass@1 (vs 75% w/o Reflexion, vs 67% ReAct).
- HumanEval is not benchmark saturated; for many tasks Reflexion goes
  from 0% to over 50% via single reflection.
- For complex multi-step tasks, gain from Reflexion is largest.

## Criticisms (specific)

1. Reflection quality is LLM-dependent. Bad reflector = bad memory.
2. Memory growth is unbounded. Long runs accumulate noise.
3. No formal theory of when reflection helps.
4. The reflection text is opaque; downstream tools cannot always act on it.

## Connection to our program

Direct map to Trend #3 (Agentic Learning) for our research assistant:
- .experience_log/ plays the role of episodic memory.
- The reflection text in Reflexion IS the "what worked / what broke" sections
   in our retro entries.
- The actor code IS Codex. It sees the workspace state (PROGRESS.md, commits),
   and acts on it.

What Reflexion does for LLM agents, our workspace tools do for human-agent
collaboration: surface failure traces across sessions. Our tooling is more
cumbersome (manual markdown files vs. an episodic memory store) but it is
available in the same year.

For Project A (RL-agent Monitor):
- Our frozen-policy Monitor IS the RL analog of Reflexion reflector.
- Both learn from failure traces without weight updates.
- Both deposit a learned signal into a memory (Monitor weights vs. episodic text).
- Difference: Monitor is a small NN, Reflexion is large LLM.

## Confidence
HIGH. Code released. Results documented.

## Related
- Voyager (skill library)
- Generative Agents (Park 2023, with reflection and memory stream)
- Auto-GPT / BabyAGI (early agentic loops, no reflection)
- Self-Refine (Madaan 2023, single-step refinement, no memory)

## Status
- cited in Trend #3 (Agentic Learning) rationale
- cited in Project F (skill library analog) and Project A (Monitor analog)
- future: implement auto-reflection in our retro script
