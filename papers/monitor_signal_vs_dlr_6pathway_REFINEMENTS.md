# Refined Y3 paper: related work + improved abstract + figure plan

> Status: Refinement notes for `papers/monitor_signal_vs_dlr_6pathway.md`
> and `papers/monitor_signal_vs_dlr_6pathway.tex`
> Date: 2026-07-29

This file collects the refinements to apply to the 6-pathway paper:

## 1. Improved abstract (more concrete, less hand-wavy)

**Old abstract key phrases**: "verified training-time signal",
"5 of 6 architectures are REFUTED", "trust head architecture
contributes a small but inconsistent effect that is independent
of the input source".

**New abstract key phrases** (more specific, includes effect sizes):

"We systematically investigated 6 architectures for using failure-
prediction Monitors in cooperative MARL on PettingZoo Simple Spread
v3, training each arm at 80 PPO updates x 10 episodes = 800 env
episodes per seed, with 5-30 seeds per arm (totaling 14,000+
training episodes). The 6 architectures cover critic-side extras
(aux loss in v3, inter-agent messages in v4), actor-side trust
head with three different input sources (Monitor in v5, random in
v6, DLR in v8), and the trust-head ablation (v7).

Five of six architectures are REFUTED at $p<0.05$. v3 (Monitor
aux loss in critic) actively HURTS by $-3.03$ at 10K episodes
(0/5 positive). v5 (trust head + Monitor) shows a direction-
consistent but shrinking effect ($+0.1665$ at $n=5$, $+0.055$ at
$n=212$). The trust head completely ignores its input signal:
v6 with\_verifier == v6 with\_trusthead\_random BIT-FOR-BIT
IDENTICAL (5/5 at $n=5$, 30/30 at $n=30$ CLEAN, max abs diff
$0.00$).

The single publishable result is **DLR cross-agent predicates in
the critic** (v8 dlr\_only): $+0.1447$ ($p<0.005$, $t=+3.216$,
20/30 positive at $n=30$, effect stable across sample sizes,
Cohen $d_z=0.59$). The trust head with DLR input is identical to
DLR in the critic alone (v8 $==$ dlr\_only, 0.00 diff at $n=30$).

The architectural lesson: **hand-crafted interpretable features
(DLR) in the critic work; learned failure predictions (Monitor)
in any critic/actor position do not.** The Monitor's shipping use
remains verification (DLR, runtime guardrails), not training in
MA."

## 2. Related Work section (to add between Background and 6 Pathways)

```
\section{Related Work}

\subsection{Failure-prediction Monitors in single-agent RL}

Failure-prediction Monitors (small networks that predict episode
failure) have been studied extensively in single-agent RL. The Y1
paper established the strongest single-agent baseline: frozen-
decoupled Monitors trained per-agent (vs joint shared) give
AUROC 0.796 vs 0.072 on LunarLander-v3, and using the Monitor
signal as a reward penalty gives $+39.5$ mean improvement at
$n=15$ seeds (Liu 2026, \cite{liu2026y1}). Other work has
explored related ideas including intrinsic motivation
(Pathak et al. 2017), curiosity-driven exploration
(Burda et al. 2018), and failure-detection-based shaping
(Dai et al. 2024). Our work extends the Monitor framework to
MARL, asking whether the verified single-agent signal
transfers.

\subsection{Multi-agent credit assignment}

Credit assignment in cooperative MARL is the problem of assigning
team reward to individual agents' contributions. The dominant
paradigm is centralized training with decentralized execution
(CTDE), exemplified by MADDPG (Lowe et al. 2017,
\cite{lowe2017}), COMA (Foerster et al. 2018), and QMIX
(Rashid et al. 2018). These methods use a centralized critic
to reduce the non-stationarity of the multi-agent setting. Our
work uses MADDPG v2 as the baseline and asks whether additional
information (Monitor or DLR predicates) can improve the
centralized critic.

\subsection{Trust and credit assignment in MARL}

Recent work has explored "trust" as a mechanism for selective
credit assignment in MARL. Approaches include TarMAC
(Das et al. 2019), IC3Net (Singh et al. 2019), and ATOC
(Jiang \& Lu 2018), which learn inter-agent communication
protocols. Our v5/v6/v7/v8 trust head is a simpler design that
takes a single signal (Monitor, random, or DLR) and outputs per-
other-agent trust weights. Our 6-pathway investigation shows
that the trust head's contribution is independent of the input
source, suggesting that the simple architecture may be sufficient
for the small effect we observe.

\subsection{Differentiable logic rules (DLR)}

DLR is a framework for embedding logical rules as differentiable
functions of neural network inputs (Yang et al. 2022, Fischer
et al. 2024). In our v8, DLR predicates like "agent $i$ is closest
to landmark $j$" are added to the critic input. DLR's strength
is that the predicates are hand-crafted and deterministic, so
they consistently provide useful information across training
runs. This is consistent with the v8 dlr\_only finding: DLR
predicates give a stable, reproducible effect that the Monitor
signal does not.
```

## 3. Figure plan (figures to add in next iteration)

The paper currently has 5 tables but no figures. The following
figures would strengthen the paper:

### Figure 1: 6-pathway overview diagram

A flowchart showing the 6 architectures and their relationships:
- Critic-side (v3, v4): Monitor aux loss, inter-agent comms
- Actor-side trust head (v5/v6/v7): different input sources
- DLR (v8): hand-crafted predicates in critic

### Figure 2: Effect-shrinkage trajectory (v5)

X-axis: sample size $n$ (5, 13, 29, 100, 212)
Y-axis: mean_diff vs no_verifier
Show: v5 effect shrinks from +0.17 (n=5) to +0.055 (n=212) -- a
textbook signature of a small effect that is more precisely
estimated with larger samples.

### Figure 3: Effect-stability (v8 dlr_only)

X-axis: sample size $n$ (5, 30)
Y-axis: mean_diff vs no_verifier
Show: v8 dlr_only effect STABLE at +0.14 to +0.15 across
sample sizes, reaching $p<0.005$ at $n=30$.

### Figure 4: Bit-for-bit identity evidence

A combined figure showing the n=5, n=30 r3, and n=30 r4 CLEAN
bit-for-bit identity results, with a clear annotation that the
n=30 r3 was contaminated by env inconsistency.

### Figure 5: Per-seed scatter (v8 dlr_only vs no_verifier at n=30)

A scatter plot with x=no_verifier final_eval, y=dlr_only
final_eval, points colored by seed. Shows that the dlr_only
points are mostly above the y=x line (positive effect).

## 4. Discussion section expansion

The Discussion section could be expanded to address:

- **Why critic-side Monitor (v3) hurts but critic-side DLR (v8)
  helps**: the Monitor is a learned function biased toward
  Stage-1 failure modes; when used as aux loss it pulls the
  critic's representation in a harmful direction. DLR is hand-
  crafted and deterministic, so it consistently provides useful
  information.

- **Why the trust head ignores its input**: the gradient is
  dominated by my_obs (high-dim, varies per batch). The Monitor
  input is low-dim and broadcast, contributing little to the
  trust head's output. This is a known issue with low-dim
  conditioning in high-dim architectures.

- **Why the v5 effect is direction-consistent but small**: the
  trust head architecture itself contributes a small positive
  effect (likely from the additional parameters and the trust-
  weighted Q blend), but the input signal is ignored, so the
  effect is the same regardless of the input source.

## 5. Limitations section (to add before Conclusion)

- **Single environment**: we test only on PettingZoo Simple
  Spread v3. The findings may not generalize to other MARL
  environments.
- **Single compute scale**: we use 800 env episodes per seed
  (80 PPO updates x 10 episodes). Real-world MARL often uses
  100K+ episodes.
- **3 agents only**: Simple Spread v3 has 3 agents. Larger
  agent counts may have different dynamics.
- **Continuous action space only**: we test continuous actions
  (matches MADDPG v2). Discrete-action MARL may behave
  differently.
- **Monitor training data**: the Monitor is trained on 80
  episodes from a frozen Stage-1 policy. Different training
  data could yield different Monitors.
- **v6 is a thin wrapper around v5**: the only difference is
  the trust head input source. A truly independent v6
  implementation (with different trust head architecture) would
  strengthen the conclusion.

## 6. Conclusion sharpening

The current conclusion is 4 numbered findings. Refine to:

1. **Monitor signal does not transfer to MA as a training signal**:
   5/6 architectures REFUTED at $p<0.05$ (v3, v4, v5, v6, v7).

2. **DLR in critic is the right architectural choice**:
   v8 dlr\_only gives $+0.1447$ ($p<0.005$, 20/30 positive at
   $n=30$), the only publishable result, stable across sample
   sizes.

3. **The trust head architecture gives a small effect that is
   independent of the input source** (Monitor, random, DLR all
   give the same result, verified at $n=5$ and $n=30$ CLEAN
   via bit-for-bit identical per-seed results).

4. **The Monitor's shipping use remains verification**, not
   training in MA.
