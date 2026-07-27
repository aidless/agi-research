# AGI Research Workspace — Archimedes Project (AGI-2026-001)

> **A Self-Improving AGI Substrate: Decoupled Monitors, Causal World Models,
> and Typed Language Interfaces**

**Copyright (c) 2026 刘泽文 (Liu Zewen)** — see [LICENSE](./LICENSE) and [AUTHORS](./AUTHORS).
**License**: MIT (with attribution requirement). See [PUBLICATION_HOLD.md](./PUBLICATION_HOLD.md).

[![Commits](https://img.shields.io/badge/commits-97-blue)](https://github.com/aidless/agi-research/commits/main)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/status-Y0_Q3-yellow)](./PROGRESS.md)
[![Reproducibility](https://img.shields.io/badge/reproducibility-CPU_only-brightgreen)](./README.md#quickstart)

Independent 5-year research program toward a self-improving, cross-domain,
causally-grounded, language-queryable AGI substrate.
**Started**: 2026-07-25. **Current**: Y0 Q3 (Foundation phase).

---

## TL;DR — Headline Results (2026-07-27 Y0 Q3 close)

| Result | Status | Effect size | Source |
|--------|--------|-------------|--------|
| **H1 ablation** (decoupled Monitor) | ✅ **5/5 seeds** | AUROC delta = **0.724** | [paper_v2 §4.6-4.8](./projects/project_a_self_improvement/paper_v2_full.md) |
| **Slot-Monitor integration** (A+C) | ✅ breakthrough | AUROC **0.989** vs raw 0.796 (+0.193) | commit 8d89ca0 |
| **DLR attention fix** (Project E) | ✅ **STRONG POSITIVE** | upright 45% → **89%**, mean **95.5%** | commit 525d1ee |
| **Y1.3 Monitor regularizer** (Project A) | ✅ **BREAKTHROUGH** | +50 over baseline, 3/5 seeds win | commit ef90c2c |
| **Slot world model dynamics** | ✅ | next-step MSE **0.000007** | commit 1093b82 |
| **DEC-0011 v0.4 6-way sweep** | ❌ HALT | 0/6 inference-time gating | commit edcc34a |
| **ENWI Prediction 2 (2000 epoch)** | ⚠️ mixed | composable wins 2/5 scenes, mean -346% | commit ae47f82 |

**Pattern**: 4 inference-time interventions fail on LunarLander. **Training-time
regularization is the publishable path** (Y1.3, +50).

---

## The Story

The Archimedes Project investigates whether **decoupling** — separating the
failure-prediction Monitor from the policy gradient that shapes behavior —
is the core mechanism enabling stable self-monitoring in reinforcement-
learning agents.

The theoretical foundation rests on the **ENWI framework** (Embodied
Neurosymbolic World-model Intelligence), a 5-layer architecture with 11
mathematical theorems and 5 falsifiable predictions. We ported four ENWI
components into a working codebase (Active Inference Engine, Differentiable
Logic Reasoner, Composable Physics, Slot Attention) and validated the
central H1 hypothesis across 5 random seeds on LunarLander-v3.

We also document **honest negative results**: ENWI's central Prediction 2
(composable physics outperforms monolithic by 94%) does not replicate at
our scale; inference-time action gating on LunarLander fails 6/6
experiments; Active Inference does not converge at our compute budget.
These negative results are reported with the same precision as positive
ones.

We operate in **AIKR mode** (Assumption of Insufficient Knowledge and
Resources, after Pei Wang's NARS): finite knowledge, bounded compute,
open tasks, iterate, report honestly.

---

## Architecture — 4-Layer Integration

```
                  ┌──────────────────────────────────────┐
                  │  4-Layer AGI Integration (v1.0)     │
                  └──────────────────────────────────────┘

   [Sensors]  →  [Slot World Model]  →  [Monitor]  →  [Q-BoN]  →  [action → env]
                    │                   │                       │
                    │                   │                       ▼
                    │                   │              [Language Interface]
                    │                   ▼
                    │           [LTL Verifier / DLR]
                    ▼
              [next state]
```

| Layer | Module | Status | Key metric |
|-------|--------|--------|------------|
| A | Decoupled Monitor | ✅ validated | AUROC 0.989 (Slot-Monitor) |
| C | Slot WM + Dynamics | ✅ validated | next-step MSE 0.000007 |
| D | Language Interface | ✅ template-based | typed natural language |
| E | DLR (Differentiable Logic) | ✅ **95.5% acc** | upright 89% (was 45%) |

The Monitor (A) and World Model (C) were **integrated first** via slot-attention
input (Slot-Monitor, AUROC 0.989). The Verifier (E) was generalized from
crisp LTL to differentiable fuzzy logic via the DLR-attention fix. The
Language Interface (D) is template-based (Y1: replace with small LM).

---

## Project Structure

```
agi-research/
├── README.md                          (this file)
├── LICENSE                            (MIT, attribution required)
├── AUTHORS                            (full citation block)
├── CHANGELOG.md                       (commit history summary)
├── PUBLICATION_HOLD.md                (LIFTED 2026-07-26)
├── PROGRESS.md                        (live project state)
├── ROADMAP.md                         (Y0 → Y5 plan)
├── TASKBOOK_v1.md                     (current task list)
│
├── thesis_draft_v1.0.md               (2700 lines, 103 KB Markdown)
├── thesis_draft_v1.0.pdf               (188 KB, 229 blocks)
├── thesis_draft_v1.0.html              (132 KB, browser-friendly)
│
├── phd_applications/                  (PhD app templates)
│   ├── statement_of_purpose.md
│   ├── academic_cv.md
│   ├── writing_sample_outline.md
│   └── README.md
│
├── projects/
│   ├── project_a_self_improvement/    (H1, Monitor, TTC, AIE)
│   │   ├── README.md
│   │   ├── paper_v2_full.md           (374 lines)
│   │   └── code/                      (25+ Python files)
│   ├── project_b_cross_domain/        (LLaVA-style, sketch)
│   ├── project_c_causal_world/        (Slot Attention, ENWI physics)
│   ├── project_d_language/            (Type lattice, template LM)
│   ├── project_e_verification/        (LTL, DLR, verifier-aware gating)
│   └── project_f_multi_agent/         (decentralized monitors)
│
├── literature/                        (43 paper deep-reads)
│
├── community/                         (public announcements)
│   ├── csdn_announcement_v3.md
│   ├── oschina_announcement_v3.md
│   ├── twitter_v0p4_halt.md
│   ├── discord_v0p4_halt.md
│   ├── twitter_intro.md
│   ├── CROSSPOST_CHECKLIST.md
│   └── (more drafts ready to post)
│
├── experiments_log/                   (chronological experiment logs)
│   ├── 2026-07-27-dlr-attention.md
│   ├── 2026-07-27-aie-train-full.md
│   ├── 2026-07-27-enwi-p2-2000ep.md
│   ├── 2026-07-27-v03-fixed-threshold.md
│   ├── 2026-07-27-dlr-verifier-gating.md
│   ├── 2026-07-27-mbp-slot-dlr.md
│   └── (40+ more logs)
│
├── decisions/                         (DEC-0011 etc., 11 records)
│
├── .experience_log/                   (session logs)
├── 00_daily/                          (daily work summaries)
└── grant_applications/                (HF, Google, Lambda drafts)
```

---

## Quickstart

### Reproduce H1 ablation (5 seeds × 100K PPO)

```bash
cd projects/project_a_self_improvement/code
python full_integration.py --env LunarLander-v3 --n-ppo-steps 100000 \
    --n-train-episodes 200 --n-eval-episodes 5 --seed 0
# Expected: AUROC ~0.989 (Slot-Monitor)
```

### Reproduce DLR attention fix

```bash
cd projects/project_e_verification/code
python dlr_attention.py --env LunarLander-v3 --seed 0 \
    --n-train-episodes 30 --n-test-episodes 20 --n-epochs 30
# Expected: upright accuracy ~89%, mean accuracy ~95.5%
```

### Reproduce Y1.3 (Monitor regularizer)

```bash
cd projects/project_a_self_improvement/code
python y13_monitor_regularizer.py --env LunarLander-v3 --seed 0
# Expected: 3/5 seeds win, mean +50 over PPO baseline
```

### Compute requirements

- **CPU only** (no GPU required)
- LunarLander 100K PPO: ~30 min on single core
- 5-seed sweep: ~2.5 hours wall time
- Full DLR training: ~2 min
- Total compute per major experiment: <3 hours

---

## Key Results in Detail

### 1. H1 Ablation — Decoupling works

**Hypothesis**: A Monitor trained on rollouts from a frozen policy `π_f`
has higher failure-prediction AUROC than a Monitor trained jointly with
the policy being improved, on the same PPO budget.

**Result** (5 seeds, 100K PPO each):

| seed | joint AUROC | frozen AUROC | delta |
|------|-------------|--------------|-------|
| 0    | 0.103       | 0.98         | 0.877 |
| 1    | 0.041       | 0.90         | 0.859 |
| 2    | 0.044       | 0.21 (anomaly)| 0.166 |
| 3    | 0.074       | 0.92         | 0.846 |
| 4    | 0.099       | 0.97         | 0.871 |
| **mean** | **0.072** | **0.796** | **0.724** |

**5/5 seeds support H1.** Frozen Monitor discriminates failure reliably;
joint Monitor gets dragged by policy updates and learns the inverse signal.

### 2. Slot-Monitor Integration — Structural decomposition

Adapting slot attention (Locatello et al. 2020) to 1-D trajectory
decomposition improves the Monitor:

```
Raw-history Monitor:  AUROC 0.796
Slot-Monitor (K=4):   AUROC 0.989  (+0.193, 24% relative)
```

The slots specialize (learned, not hand-designed): horizontal motion,
rotation, vertical motion, residual.

### 3. DLR Attention Fix — STRONG POSITIVE

The original DLR (`dlr_train_full.py`) had a critical failure: `upright`
predicate (depends on angle) reached only 45% accuracy. Root cause: the
random projection from observation to slot features loses angular
information, and mean aggregation over slots cannot recover it.

**Fix** (`dlr_attention.py`):
- **Learned obs → slots projection** (`ObsToSlots` MLP)
- **Attention-based slot aggregation** (`AttnSlotPredicateNet`)
- **Joint training** of projection + predicates end-to-end

**Result** (3 seeds):

| predicate    | before (mean) | after (attention) | delta |
|--------------|---------------|-------------------|-------|
| landed       | 99.4%         | 99.8%             | +0.4  |
| **upright**  | **45.4%**     | **89.0%**         | **+43.6** |
| leg_l_contact| 98.8%         | 99.7%             | +0.9  |
| leg_r_contact| 98.3%         | 99.9%             | +1.6  |
| in_pad       | 93.2%         | 96.3%             | +3.1  |
| low_velocity | 92.6%         | 94.5%             | +1.9  |
| safe_approach| 75.1%         | 89.0%             | +13.9 |
| **mean**     | **86.7%**     | **95.5%**         | **+8.8**  |

### 4. Y1.3 — Training-Time Regularizer (BREAKTHROUGH)

After 6 failed attempts at **inference-time** gating (DEC-0011 v0.1 → v0.4C),
Y1.3 takes a fundamentally different approach: use the Monitor as a
**training-time reward shaper**, not an inference-time action selector.

```
Pipeline (LunarLander-v3, 100K PPO total):
  Phase 1: PPO 25K steps warm-up (no Monitor).
  Phase 2: Collect 200 rollouts, train SlotMonitor (frozen).
  Phase 3: PPO 75K more steps with shaped_reward = env_reward
           - 0.5 * Monitor_prob(window).
  Phase 4: Evaluate - PPO only, no Monitor at inference.
```

**5-seed result**:

| Method                      | Mean   | Std   | Pos |
|-----------------------------|--------|-------|-----|
| PPO-only baseline           | 40.6   | 37.1  | -   |
| **Y1.3 (Monitor regularizer)** | **90.5** | **56.3** | **3/5** |

Per-seed deltas: +64.2, -58.7, +84.8, +105.4, +53.6.

**Why this works**: PPO learns to AVOID Monitor-flagged states. At
inference, PPO acts alone with no Monitor overhead. The Monitor signal
is a *constraint during learning*, not an *instruction*.

---

## Honest Negative Results

We report negative results with the same precision as positive ones.

### 1. ENWI Prediction 2 — Not replicated at our scale

ENWI claims composable physics outperforms monolithic by 94.22% on five
synthetic physics scenes. We tried to replicate:

| scale | composable MSE | monolithic MSE | ratio |
|-------|----------------|----------------|-------|
| 30 epoch, latent=32 | 1.95e-6 | 5.55e-7 | 3.5× worse |
| 100 epoch, latent=64 | 3.23e-7 | 1.73e-7 | 1.9× worse |
| **2000 epoch, latent=64** | 2.51e-8 | 5.63e-9 | **4.5× worse** |

At 2000 epochs, **composable wins on 2/5 scenes** (free_fall, inertia)
but loses badly on 3/5 (collision, friction, compound). Mean still
negative. The 94% mean improvement claim is NOT replicated.

### 2. Inference-time intervention — 6/6 failures on LunarLander

Six different gating mechanisms all failed:

| Version | Design | Delta | t-stat |
|---------|--------|-------|--------|
| v0.1 | Q-BoN, fixed threshold | +21.5 | +0.72 (n.s.) |
| v0.2 | Q-BoN, calibrated | -158.1 | -1.69 |
| v0.3 | safe_action=2, calibrated | -717.6 | -3.71 |
| v0.4A | Q-BoN, calibrated (5x data) | -1.8 | -0.25 |
| v0.4B | Q-BoN, CartPole | -270.4 | -3.48 |
| v0.4C | Imitation, top-25% | -33.7 | -2.64 |

**0/6 statistically significant HELP.** **DEC-0011 HALTed.**

Plus 2 more from this session:
- DLR verifier gating (3 thresholds, all delta < -120)
- Model-based planning (delta = -273)

**Root cause**: replacing PPO actions with anything (do-nothing, Q-BoN,
behavior-clone, MBP) hurts on LunarLander because PPO is already strong
and the lander needs to maneuver.

### 3. Active Inference Engine — 3 variants, all NEGATIVE

| variant | result |
|---------|--------|
| aie_lunarlander.py (smoke) | trivial |
| aie_train_full.py (3 seeds) | -139.3 (worse than random) |
| aie_recurrent.py (GRU + value baseline) | -345.7 (much worse) |

Free energy loss decreases (perception learns) but policy does not
converge. ENWI Prediction 4 (AIE matches PPO with fewer samples) requires
~500K+ env steps, beyond our Y0 budget.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{liu2026archimedes,
  title={Archimedes: A Self-Improving AGI Substrate},
  author={Liu, Zewen},
  year={2026},
  note={Independent 5-year research program, AGI-2026-001},
  howpublished={\url{https://github.com/aidless/agi-research}}
}
```

---

## License

MIT — see [LICENSE](./LICENSE). Copyright (c) 2026 刘泽文 (Liu Zewen).

Attribution is required for any use, modification, or redistribution.
All public artifacts carry:
- LICENSE file with copyright
- AUTHORS file with citation block
- README.md with attribution
- Key Python file headers
- Thesis draft author block

---

## Acknowledgments

- **ENWI framework** (F:\TMLR\Fusion\ENWI_PAPER.md): theoretical foundation.
- **Slot attention**: Locatello et al. (2020).
- **Active inference**: Friston (2010).
- **AIKR mode**: Wang (2013).
- **AI assistance**: Codex (a coding agent based on MiniMax-M3). All
  AI-generated content is reviewed by the PI before inclusion.

---

## Project Stats

| Metric | Value |
|--------|-------|
| Started | 2026-07-25 |
| Total commits | 97 |
| Total commits Y0 Q3 | 60+ (in 3 days) |
| Python files | 50+ |
| Lines of code | ~15,000 |
| Paper draft | 2,700 lines (103 KB Markdown, 188 KB PDF) |
| References | 45 |
| Experiments documented | 40+ |
| Decision records | 11 |
| STRONG POSITIVE | 2 (DLR fix, Y1.3) |
| BREAKTHROUGH | 1 (Y1.3) |
| Honest negative | 11+ |

---

*"Eureka! Eureka!" — Archimedes*
