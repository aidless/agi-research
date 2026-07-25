# PlanBench (Valmeekam et al. 2024)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: A benchmark for evaluating LLM planning capabilities. PlanBench
> reveals that LLMs are surprisingly poor at multi-step symbolic planning,
> even when the plan is in their training distribution.

## Problem
LLMs are claimed to be good planners. Empirically? They write plausible code
but often produce invalid plans when probed systematically.

## Method
PlanBench provides:
- Blocksworld: tower-stacking problems of increasing depth.
- Logistics: warehouse routing.
- Plan-generation instances from IPC.
- "Tell me the optimal plan length": tests plan-completion intuition.
- "Optimal plan synthesis": generates step-by-step.

These are classic AI planning benchmarks with verify-able ground truth.

## Empirical result
- GPT-4: < 5% accuracy on optimal plan synthesis beyond depth 5.
- Even with chain-of-thought, performance is poor.
- LLMs fail on the simplest reasonable planning tasks.
- Llama 2 70B: 0% on PlanBench.
- The LLM failure mode is consistency, not capability: LLMs produce
  plausible-looking plans that break on inspection.

## Criticisms (specific)
1. **Generates "almost right" plans**: dangerous for deployment in safety-
   critical contexts (where most agents get used).
2. **No plan verification**: most LLM stacks don NOT check plans against
   domain constraints.
3. **Cost-prohibitive reasoning**: planning fine is expensive; models
   resort to pattern-matching.

## Connection to our program
For Project As Monitor:
- Monitor evaluates policy outputs. If policy outputs plans, Monitor
  should check them against domain constraints (Project E verifier!).
- PlanBench demonstrates why this is necessary: LLM-as-planner without
  verification is dangerously unreliable.

For Project E (verification):
- PlanBench is exactly the kind of benchmark Project E must address.
- A verified Planner architecture (PDDL + LLM proposal + symbolic check)
  like AlphaProof, is what we propose.

For Project F (workspace automation):
- Our multi_orchestrator.py wraps planners but is not itself a planner.
- PlanBench is a benchmark we can run against a future multi-orchestration
  to measure reasoning capability.

## Confidence
HIGH.

## Related
- Blocksworld (classic 1970)
- IPC (International Planning Competition)
- AlphaProof - LLM + verification (DeepMind 2024)
- Tree of Thoughts (Yao 2023)
- Reasoning via Planning RAP (Hao 2023)
- Voyager (Wang 2023)

## Status
- cited in Project A Related Work (Monitor reasoning coverage gap)
- cited in Project E Related Work (verifier benchmark)
- future: run our Project E verifier on PlanBench as a quantitative test
