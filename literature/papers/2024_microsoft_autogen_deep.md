# AutoGen (Wu et al. Microsoft 2024)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- Microsoft Research, extensively documented
> One-line: Multi-agent LLM orchestration framework. Agents communicate via
> structured messages; the developer specifies roles, workflows, and tools;
> the framework manages turn-taking and tool routing.

## Problem

Building multi-agent LLM applications is hard:
- Agents need different personas
- Agents need a way to call tools
- Coordination (who speaks when) needs explicit rules
- Errors propagate between agents

## Method

AutoGen provides:
1. **Conversable agents**: each agent has a system message defining role,
   tools, and termination conditions.
2. **Group chat manager**: routes messages between agents; supports
   round-robin, hand-off, or LLM-driven (AutoPM-style).
3. **Tool/function execution**: agents call user-defined Python functions.
4. **Human in the loop**: any agent can be a human proxy.

Code example:
```python
coder = ConversableAgent(name="coder", system_message="You write Python. ...")
reviewer = ConversableAgent(name="reviewer", system_message="You review ...")
group = GroupChat([coder, reviewer])
group.initiate_chat("Write a function that does X")
```

## Empirical result

- Adopted by 50+ companies in 2024 (Microsoft reported case studies).
- Used for: software engineering agents, customer support, multi-step RAG
  pipelines, code review automation.
- Open source, multimodal support.

## Criticisms (specific)

1. **Conversation can run away**: agents loop and never terminate unless
   explicitly bounded.
2. **No real planning**: just back-and-forth; agents do not have a World Model.
3. **No formal verification**: agents output speech, not proofs.
4. **Topology is hand-coded**: user designs the conversation graph.

## Connection to our program

Direct map to Trend #2 (multi-agent orchestration):

AutoGen is essentially the engine our Project F (workspace automation) and
Project G (multi-agent verification) would build on top of. Specifically:

- **Project F**: replace "Codex-as-one" with "Codex + planner + executor + reviewer"
  sub-agents orchestrated by AutoGen-style topology.
- **Project G (multi-agent verification)**: AutoGen CAN structure the three-
  agent pipeline (proposer/verifier/coordinator) we wrote in Project E paper
  outline. Without AutoGen, it would be bespoke.

For Project A (self-improvement):
- Each Monitor IS an agent. We could spawn "Monitor-A" and "Monitor-B"
  candidate variants and have them debate; that debate is exactly the
  verifier architecture from Project E.

## Confidence
HIGH.

## Related
- CrewAI (multi-agent framework alternative)
- LangGraph (graph-based agent orchestration)
- ChatDev (multi-agent software development)
- MetaGPT (multi-agent for software eng)
- Generative Agents (Park 2023, sandbox town)
- Voyager (Wang 2023, single agent with skill library)
- Reflexion (Shinn 2023)
- ReAct (Yao 2022)

## Status
- cited in Trend #2 (multi-agent orchestration baseline)
- cited in Project F (workspace automation engine)
- cited in Project G (multi-agent verification)
- cited in Project A (Monitor debate extension)
