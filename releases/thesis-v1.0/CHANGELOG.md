# CHANGELOG -- Thesis v1.0

> Release history for the Archimedes thesis.

## v1.1 -- 2026-07-29 (this release)
- Added Chapter 1.4 "SOTA context (mid-2026 update)" discussing
  Kimi K3 (Moonshot AI, 2.8T MoE) and FlashKDA (kernel
  implementation of Kimi Delta Attention). Relevance to Archimedes
  is discussed: KDA's channel-wise forget gate parallels our
  decoupled Monitor's policy/freeze decision; AttnRes is a
  complementary selective retrieval; partial rollout is relevant
  to Project F (multi-agent). We do NOT claim competitiveness with
  frontier LLMs on raw capability benchmarks; the thesis is about
  a different research direction.
- Added references [46] Kimi K3, [47] FlashKDA, [48] Gated DeltaNet
  predecessor. Total references: 48 (was 45).
- Bumped CHANGELOG + manifest + citation accordingly.

## v1.0 -- 2026-07-27

### Initial release

The first complete thesis draft, including:

- **Abstract + Introduction**: 5-year program framing.
- **Background and Related Work**: ENWI framework, world models,
  causality, self-critics.
- **Method**: Project A (decoupled Monitor + Y1.3), Project C (slot
  world model), Project D (language-as-type-system), Project E
  (Differentiable Logic Reasoner), Project F (multi-agent DMC).
- **Results**: Y0 + Y1 empirical findings, including 8 pre-registered
  H tests with hard decision rules.
- **Discussion**: architectural lessons, mechanism analysis.
- **Limitations**: statistical, generalization, methodological,
  reproducibility, multi-agent.
- **Conclusion**: AGI-Substrate (not AGI-Strong) framing.
- **References**: ~45 papers.
- **Appendices**: 8 addenda + 4 appendices.

### Highlights

- **H1 decoupling ablation**: 5/5 seeds on LunarLander-v3, frozen
  Monitor AUROC 0.796 vs joint 0.072, delta 0.724. Frozen > joint
  Monitor on classical RL is real.
- **Slot-Monitor**: AUROC 0.989 vs raw-history 0.796, +0.193
  relative improvement.
- **Y1.3 training-time regularizer**: 15 seeds on LunarLander, +50
  mean over PPO baseline, p<0.001. (Note: this was later retracted
  on the Real-vs-Random test, n.s. on n=15; documented in thesis
  addenda.)
- **DLR attention**: 4 envs, 19 predicates, 97.8% mean accuracy.
- **8 pre-registered H tests**: 0 supported at strict t>2.0 rule.
  Honest null synthesis.

### Limitations

- Single-author independent work; no institutional review.
- Limited compute (CPU only; GPU grants pending).
- Single-environment headline result for Y1.3 (LunarLander only).
- No peer review.

---

*v1.0 packaged for arXiv-ready distribution 2026-07-29 by Codex
agent. See README.md for full claims and non-claims.*