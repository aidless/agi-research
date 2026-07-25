# Generative Agents (Park et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: 25 LLM-driven agents in a sandbox town maintain a memory stream,
> retrieve relevant memories at each step, and reflect on long-term patterns.
> Demonstrates emergent social behaviours.

## Problem this solves

How do LLM agents maintain coherent behaviour over long time scales?
Naive LLM prompting has no real memory. Each turn starts fresh.

## Method

Three components:
1. **Memory stream**: each agent stores every observation, dialogue, action in
   a time-ordered log.
2. **Retrieval**: at each step, the LLM retrieves top-k memories by recency,
   importance (LLM-judged), and relevance (embedding similarity to current state).
3. **Reflection**: every ~50 events, LLM summarizes recent memories into higher-
   level observations ("I have been spending time with X, who is friendly").

Agents then plan next actions conditioned on the retrieved+reflected context.

## Empirical result

- 25 agents in Smallville (a Sims-like town)
- 2 days of simulated time
- Emergent behaviour: agents formed friendships, organised a Valentine party
   (unprompted), spread gossip, made plans
- Ablation: removing reflection made agents behave more randomly

## Criticisms (specific)

1. **The emergent behaviour is in the simulation, not real agency**.
   Agents do not actually understand what they are doing.
2. **LLM cost is enormous** (one paper reported ~$1000/day of API calls).
3. **No formal guarantees** about emergent properties.
4. **Memory growth is unbounded**; retrieval is heuristic.

## Connection to our program

Trend #1 + #2 + #3 all at once. For our research assistant:
- Memory stream <- a richer .experience_log/ than markdown
- Retrieval <- read-time selection of relevant historical commits
- Reflection <- periodically summarise .experience_log into PROJECT patterns
- Multi-agent <- spawn per-project sub-agents (one for AGI papers, one for tool code)

For Project A on monitored RL agents:
- Monitor logs every "failure prediction" with the trajectory state - like
  an agent memory stream.
- Reflection over many such logs can surface systematic failure patterns.

## Confidence
HIGH.

## Related
- Voyager (skill library, not memory stream)
- Reflexion (textual reflection, episodic memory)
- Auto-GPT / BabyAGI (no memory stream)
- MemGPT (virtual context as memory)
- CrewAI / AutoGen (multi-agent orchestration)
- LangChain Memory modules (production analog)

## Status
- cited in Trend #1/#2/#3 (long-horizon + multi-agent + agentic learning)
- cited in Project F (workspace automation plan)
- future: implement memory-stream on top of .experience_log/
