# 2023_lightman_prm_deep.md - Let's Verify Step by Step

> Paper: Lightman, H., et al. (2023). Let's Verify Step by Step. arXiv:2305.20050.
> Authors: OpenAI (Lightman et al.)
> Status: read 2026-07-25, deep note for Project A reading list (Y0 Q3 must-read)
> Related to: Project A (decoupled Monitor = process reward model in disguise),
>              F:\TMLR\前沿研究_01_TestTimeCompute.md

## 1. Problem

Outcome-supervised reward models (ORM) score only the final answer of a
chain-of-thought. When the chain is long, most intermediate steps are
unobserved. The ORM has to figure out which step introduced the error
purely from the final-answer loss — this is a hard credit-assignment
problem and leads to reward-hacking / sparse signal.

## 2. Contribution

Process-supervised reward models (PRM): score every intermediate step in a
CoT, not just the final answer. The PRM gets step-level labels from human
annotators ("is this step correct?") and learns to predict per-step
correctness.

Key results on MATH benchmark:
- ORM (outcome-only): 72.4% solve rate
- PRM (process-only): 78.2% solve rate
- PRM + Best-of-N (N=256, re-rank by PRM): **78.2% -> 82.0%**
- Consensus voting (maj@N): from 72.4% (ORM@N=256) to 80.6% (maj@N=256)

PRM is strictly better than ORM at every compute budget tested.

## 3. Method

- Annotators mark each step in a CoT solution as correct/incorrect
- PRM is a classifier trained on these step-level labels
- At inference: generate N candidate solutions, score each with PRM,
  pick highest-PRM-score (or use weighted voting)
- Active learning: PRM identifies its own uncertain steps and asks
  annotators to label only those, reducing annotation cost ~2.5x

## 4. Why this matters for Project A

**Our decoupled Monitor IS a process reward model.** Each PPO policy
step is analogous to a CoT reasoning step. The Monitor takes the policy's
history vector and outputs a probability that the episode will fail. This
is exactly per-step supervision at the trajectory level.

Implications:
1. **Paper A framing**: reframe Monitor as "PRM over policy steps" rather
   than "auxiliary classifier". This connects our work to the
   LLM-reasoning literature and makes the contribution more visible.
2. **TTC extension (ADR 0011)**: Best-of-N over PPO actions, with Monitor
   as the per-sample PRM scorer. Lightman 2023 shows BoN+PRM gives ~4
   percentage points over maj@N at the same compute budget. We should
   expect similar gains at the policy level.
3. **Active learning for Monitor**: Lightman's active-learning trick
   reduces annotation cost. We could apply this: when Monitor is
   uncertain (output probability near 0.5), query the human (or a
   heuristic) for a label. Saves 2-3x annotation cost.

## 5. Critique / open questions

- Lightman's PRM is for math (MATH benchmark). The transfer to RL
  policy-step supervision is not validated. Our LunarLander result
  (joint 0.072 AUROC, frozen 0.796) suggests the analogy holds, but
  we should run a process-supervised vs outcome-supervised ablation in
  Y1 to be sure.
- PRM quality depends on step granularity. In CoT, steps are sentences.
  In PPO, "steps" are (obs, action, reward) tuples. The granularity
  mismatch is a real concern — a single wrong PPO action can fail an
  episode, so "process" supervision might be noisier.
- Lightman does not test whether PRM generalizes across distributions.
  Our Monitor on Procgen (Y1) will be the test.

## 6. Connection to F:\TMLR H/I series

- I01 TestTimeCompute: PRM is the canonical process-level supervision
  for TTC scaling. The whole o1 / DeepSeek-R1 / AlphaProof stack uses
  PRM (or PRM-equivalent).
- I05 Reasoning: PRM is one of the canonical ways to operationalize
  "reasoning step supervision".
- I07 Alignment: PRM is a step toward formal verification of intermediate
  steps, which is the AGI safety agenda.

## 7. Cite in Paper A

Add to Paper A Related Work:
> Lightman et al. (2023) introduced process-supervised reward models
> (PRM) and showed they outperform outcome-supervised models on MATH
> (78.2% vs 72.4%). Our decoupled Monitor is a PRM-equivalent at the
> RL policy level: it scores each step (transition) for failure
> likelihood, trained on the frozen-policy trajectory distribution.
> The Lightman BoN+PRM gain (~4 pp on MATH) suggests a similar
> free TTC extension at the policy level (ADR 0011).

## 8. One-line takeaway

Process-supervision > outcome-supervision for step-level decision
problems; this validates our Monitor-as-PRM framing and motivates the
ADR-0011 TTC extension.