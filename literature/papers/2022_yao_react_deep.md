# ReAct (Yao et al. 2022)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- foundational LLM-agent paper
> One-line: Combine Chain-of-Thought (reasoning) and Action (tool use) in a
> single prompt trajectory; the LLM generates both interleaved.

## Problem

LLM agents typically do one of two things:
1. Reason (CoT: "let me think step by step") - no environment interaction
2. Act (use tools) - no explicit reasoning about why

Neither alone is great. CoT hallucinates; act is opaque.

## Method

ReAct prompt format interleaves three element types:
Thought: ... // reasoning step (used by next loop)
Action: search[x] // call tool
Observation: ... // tool output

The LLM generates each step conditioned on the prefix of the trajectory.
Tools can be anything: search, calculator, code exec, SQL.

Few-shot examples teach the format.

## Empirical result

- HotpotQA: ReAct beats CoT-only and Act-only. CoT hallucinates actions;
  Act is opaque.
- Fever fact-checking: ReAct verifies claims with retrieval.
- ReAct+CoT hybrid: best of both (reasoning AND action).
- Better interpretability: human can read Thoughts to understand decisions.

## Criticisms (specific)

1. **Reasoning errors compound**. A wrong Thought leads to wrong Action leads to
   wrong Observation. No guardrail.
2. **Token cost**: every step produces a tool output + reasoning. Long
   trajectories expensive.
3. **No global planning**. ReAct is greedy on each step. Fails on multi-step
   planning tasks.
4. **Memory: trajectory grows unboundedly within a session**; persistence
   across sessions is assumed by external tools.

## Connection to our program

Direct map to Trend #2 (multi-agent) AND Project A (Monitor):

- The **Monitors reasoning loop**: "is this rollout in failure mode?" is
  structurally a ReAct prompt:
    Thought: current state has slots {a, b}; prior history suggests failure.
    Action: query critic-M for prediction.
    Observation: P(failure) = 0.65.
    Thought: above threshold (0.5); recommend safe action.
    Action: emit safe-action override.

- For Project D (language types): the planned-action output of an LLM is a
  typed predicate, not raw text. ReAct + types = typed agent.

- For Trend #2 (multi-agent): each ReAct step is a sub-agent call. Spawning
  review/safety actors = multi-agent ReAct.

## Confidence
HIGH.

## Related
- Chain-of-Thought (Wei 2022)
- Toolformer (Schick 2023) - learning when to call tools
- Self-Ask (Press 2022) - LLM asks itself clarifying questions
- Tree of Thoughts (Yao 2023) - extends ReAct to search
- Reflexion (Shinn 2023) - extends ReAct with reflection
- Voyager (Wang 2023)
- Auto-GPT, BabyAGI (2023)

## Status
- cited in Project A Related Work (Monitor reasoning structure)
- cited in Project D (LM-on-types architecture baseline)
- cited in Trend #2 (multi-agent orchestration baseline)
