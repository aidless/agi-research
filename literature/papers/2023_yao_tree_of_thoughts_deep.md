# Tree of Thoughts (Yao et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: Generalise ReAct by exploring a tree of reasoning paths with
> BFS/DFS + self-evaluation, instead of greedy left-to-right trajectories.

## Problem

ReAct and CoT are greedy: once the LLM emits a thought, that is committed.
On tasks that require backtracking or exploring alternatives, greedy fails.

  

ToT extends ReAct with:
- Each "Thought" is one node in a tree.
- Two operations: EXPAND (generate N next thoughts) and EVALUATE (a value
  the thought is promising).
- Search: BFS or DFS over the tree, using evaluation to prune.

Specific prompt format:
Thought 1: candidate reasoning step 1
Thought 2: candidate reasoning step 2 (alt)
Thought 3: candidate reasoning step 3 (alt)
Value: vote/judge the best of these
Next: continue with the chosen one

## Empirical result

- Game of 24: ToT reaches 74% success vs CoT 4% vs IO 5%.
- GPT-4 + ToT on Game of 24 reaches 74%. With BFS and self-eval.
- Mini crosswords: ToT 60% vs CoT 16%.
- Generic tasks where exploration matters benefit most.

## Criticisms (specific)

1. **Token cost is huge**: each thought subtree is multiple completions.
2. **Value judgement is LLM-self-eval**, which is unreliable.
3. **No formal guarantee of finding optimal path**; depends on depth and beam.
4. **Implementation cost** vs IMO-style approaches.

## Connection to our program

Tree of Thoughts is a planning instantiation of MCTS in latent space:
- Like MuZero values-then-search.
- Like Montani et al 2023 LLM-based planning.

For Project A: ToT could power the Monitor`s reasoning chain: when the
Monitor predicts failure, it could ToT-explore what the safe action is, in
the slot-WM latent.

For Project D (language types): ToT nodes are typed predicates. The ToT
branching IS the option / sub-task exploration.

For Project E (verification): ToT can be used to generate candidate
trajectories that the verifier then adjudicates.

## Confidence
HIGH.

## Related
- CoT (Wei 2022)
- ReAct (Yao 2022)
- RAP (Hao 2023) - reasoning + acting in language models, applied to
  planning
- Self-Refine (Madaan 2023)
- Reflexion (Shinn 2023)
- AlphaProof (DeepMind 2024)
- MuZero MCTS (Schrittwieser 2020)

## Status
- cited in Project A Related Work (alternative Monitor design)
- cited in Project D (LM-on-types architecture)
- cited in Project E (verifier could search over candidate trajectories)
