# 2024_snell_ttc_scaling_deep.md - Scaling LLM Test-Time Compute Optimally

> Paper: Snell, C., Lee, J., Xu, K., Kumar, A. (2024). Scaling LLM
> Test-Time Compute Optimally Can be More Effective than Scaling Model
> Parameters. arXiv:2408.03314.
> Affiliation: Google DeepMind
> Status: read 2026-07-25, deep note for Project A reading list (Y0 Q3)
> Related to: Project A (TTC extension of Monitor), ADR 0011
>              F:\TMLR\前沿研究_01_TestTimeCompute.md

## 1. Problem

Pre-2024 scaling laws focused on training-time compute (Kaplan 2020,
Chinchilla 2022). Snell et al. ask: at fixed total compute, when is it
better to spend compute on a LARGER model at inference, vs. more
test-time compute (BoN / MCTS / revision) on a SMALLER model?

## 2. Key contribution

The compute-optimal allocation depends on prompt difficulty:
- **Easy prompts**: scaling model parameters wins (a big model with no
  extra search beats a small model with BoN)
- **Hard prompts**: scaling test-time compute wins (a small model with
  revision and BoN beats a big model with greedy decoding)

The crossover point depends on the prompt distribution, but the paper
shows that for many realistic prompt distributions, TTC scaling is
strictly more efficient.

## 3. Method

Two compute allocation strategies compared:
1. **Best-of-N (BoN)**: sample N candidate answers from the model, pick
   highest-PRM-scored
2. **Beam search + revision**: multi-step search with iterative
   self-correction

Tested on MATH (PaLM 2 variants: 0.5B / 1B / 2B / 8B / 64B) and
CodeContests (Llama variants).

Headline result: on MATH, a **PaLM 2-Small with revision + BoN (N=256)**
beats **PaLM 2-Large with greedy decoding** at the same total compute
budget (FLOPs matched).

## 4. Why this matters for Project A

1. **ADR 0011 TTC extension is justified**: scaling compute at policy
   level (BoN over PPO actions, Monitor as the scorer) can beat scaling
   the policy network. Our 1M-param policy + Monitor scorer is exactly
   the regime Snell tests.
2. **Prompt difficulty maps to episode difficulty**: in our setting,
   "easy episodes" are those where PPO already succeeds (>0 reward);
   "hard episodes" are failure cases. Snell's result says we should
   only apply TTC extension to hard episodes (or to failure-prone
   state regions), not uniformly.
3. **Compute matching matters**: we cannot claim TTC wins by just
   sampling more — we need to match total FLOPs. For Project A paper
   v2 (TTC extension), we must report TTC vs parameter scaling at
   matched compute.

## 5. Critique / open questions

- Snell's experiments are on LLM reasoning tasks (MATH, CodeContests).
  Transfer to RL is unverified. Our setting is different (continuous
  action space, sparse reward, online learning).
- Snell assumes a PRM is available. Our Monitor is the PRM, but it's
  not trained on a separate validation set — it's trained on the same
  trajectories we evaluate on. This is a distribution-shift risk for
  BoN evaluation; we should hold out a clean eval set.
- "Hard prompts" in Snell are defined by ground-truth difficulty. In
  our setting, we don't have ground truth at deployment. We need a
  cheap difficulty predictor (Monitor probability at episode start?).

## 6. Connection to F:\TMLR H/I series

- I01 TestTimeCompute: Snell is the canonical reference for
  TTC vs parameter scaling. Cite in Paper A.
- I05 Reasoning: connects to STaR-style self-improvement, which is
  a complementary axis (improve the model itself) vs TTC (improve
  the inference).

## 7. Cite in Paper A

Add to Paper A Related Work / Section 6 Future Work:
> Snell et al. (2024) show that for hard prompts, scaling test-time
> compute (Best-of-N with PRM + revision) can be more efficient than
> scaling model parameters. This motivates ADR 0011: extend our Monitor
> to a TTC controller that gates BoN sampling at the policy level,
> applying extra compute only to episodes the Monitor flags as
> high-failure-probability. Y1 Q2 evaluation target.

## 8. One-line takeaway

For hard problems, TTC beats parameter scaling at matched FLOPs; this
motivates BoN+Monitor as our Y1 extension to Project A.