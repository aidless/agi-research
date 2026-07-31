# Thesis v3.0 Abstract (Markdown extract from v2.0 body)

This is the abstract extracted from the v2.0 thesis body section, which
appears in the v3.0.tex source. The .tex source is committed; this
Markdown extract is for Obsidian and quick reference.

---

abstract}
This thesis presents the Archimedes Project, a 5-year independent research
program toward a self-improving AGI substrate. The central hypothesis is that
decoupling the failure-prediction Monitor from the policy gradient enables
stable self-monitoring in RL agents. We validate this hypothesis on five
random seeds of PPO-trained LunarLander-v3, observing a mean AUROC improvement
of 0.724. The Archimedes architecture is a four-layer integration (decoupled
Monitor, slot-attention world model, template-based language interface, neuro-
symbolic verifier). The Y2 follow-up systematically investigated 6 architectures
for using failure-prediction signals in cooperative MARL: 5 of 6 are REFUTED
at $p<0.05$, and the single publishable result is DLR cross-agent predicates
in the critic ($+0.06$ at $n=100$, $p<0.05$ with Bonferroni). The Y2 LLM self-
monitoring pilot (H10) is also REFUTED. The failure-prediction Monitor does
not transfer from single-agent RL to other contexts -- it is a context-
specific signal whose verified shipping use is verification, not training.
\end{abstract}

---

**Source:** papers/thesis_draft_v3.0.tex (line 34-46 area)
**Generated:** 2026-08-01 (P3 hybrid pre-reg execution day 1)
**Status:** v3.0 .tex source committed; PDF rebuild deferred until v2.0 LaTeX errors are repaired
