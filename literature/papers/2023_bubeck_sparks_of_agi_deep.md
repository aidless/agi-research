# Sparks of Artificial General Intelligence (Bubeck et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- Microsoft Research / OpenAI-aligned; widely-discussed
> One-line: An early systematic exploration of GPT-4 capabilities, arguing
> "sparks of AGI" are visible in many domains; the paper is intentionally
> qualitative.

## Problem

Is GPT-4 a step towards AGI? Empirical capability assessment, not benchmark
leaderboard. The paper documents capabilities that emerge without explicit
training for them.

## Method

No formal method. Authors run diverse tasks (math, coding, reasoning, planning,
medicine, law, common sense) and describe the surprising capabilities observed.
Examples:
- Math: solved 10/10 AMC problems correctly
- Coding: passes LeetCode hard problems
- Reasoning: solves logic puzzles zero-shot
- Medicine: writes accurate differential diagnoses
- Visual reasoning: solves ARC tasks partially

The paper argues: emergence is real and visible. Scaling has unlocked
capabilities that were not present even in GPT-3.

## Empirical result

- Many examples of multi-step reasoning, code, planning that approach or
  exceed human average performance.
- The paper is qualitative; no aggregated score.

## Criticisms (specific)

1. **Cherry-picked examples**. The paper highlights GPT-4 successes, not failures.
2. **No quantitative metric**. "Sparks of AGI" is rhetorical, not measured.
3. **Many failures still**. The same paper acknowledges GPT-4 fails many times.
4. **No architectural insight**. The paper describes behaviour, not mechanism.

## Connection to our program

For Project F (workspace automation):
- Our research assistant uses a model with strong emergent reasoning.
- "Sparks" framing matters: we can use the model for planning tasks
  (paper outlines, decision summaries) but should test concretely.

For our KPI framework:
- Chollet (ARC-AGI) argues GPT-4 is not AGI because it lacks the *ability to
  acquire new skills from few examples*. Bubeck disagrees: emergent
  capabilities matter.
- Our KPIs (in TASKBOOK Section 9) lean Chollet-style: skill-acquisition
  efficiency. Sparks cautions us to also track emergent capabilities as a
  supplementary signal.

## Connection to arc 2026 / model ecosystem

Since publication (March 2023), the AI landscape has fragmented:
- OpenAI: GPT-4o, GPT-4.5, o1, o3 (reasoning)
- Anthropic: Claude 3, 3.5, 4 (with tools and citations)
- Google: Gemini 1.0, 2.0, 2.5
- Meta: Llama 3.1, 3.2 (open weights), Llama 4
- Chinese: MiniMax-M3 (the model powering me, Codex), Qwen, DeepSeek

The "sparks of AGI" debate has intensified: o3 (2024) won gold medal equivalent
on IMO-like math; ARC-AGI 2025 top score ~87% (human); SWE-bench shows
~70% on real coding. AGI remains undefined but proximate models are closer.

## Confidence
HIGH for the paper itself. MEDIUM for current model ecosystem state
(my knowledge cutoff may be stale for late 2025/2026).

## Related
- ARC-AGI leaderboard (Chollet 2024, 2025)
- GPT-4 Technical Report (OpenAI 2023)
- Sparks++ (Hassabis 2024 talk, qualitative follow-up)
- Chollet 2019 On the Measure of Intelligence
- AlphaProof / AlphaGeometry 2024

## Status
- cited in TASKBOOK architecture v2 motivation
- cited in agent_futures_plan.md as the AI-economics backdrop
- future: re-survey late 2026 model capabilities
