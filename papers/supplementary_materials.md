# Supplementary Materials

**Paper:** Monitor Signal vs DLR Predicates in Cooperative MARL:
A 6-Pathway Systematic Investigation
**Authors:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** July 29, 2026

This supplementary materials document provides the detailed data,
code, and reproducibility instructions for the 6-pathway
investigation reported in the main paper.

## S1. Repository layout

All code, data, and analysis scripts are in the Archimedes Project
git repository (https://github.com/aidless/agi-research).

```
agi-research/
├── papers/
│   ├── monitor_signal_vs_dlr_6pathway.{md,tex,pdf}    # Main paper
│   ├── monitor_in_ma_lessons_learned.md              # 4-pathway predecessor
│   ├── y1_9hypothesis_framework.md                    # H5 framework
│   └── cover_letter_aamas2027.md                      # Submission cover letter
├── projects/project_f_multi_agent/code/
│   ├── pz_maddpg_v2.py        # MADDPG v2 baseline
│   ├── pz_maddpg_v3.py        # v3: Monitor aux loss
│   ├── pz_maddpg_v4.py        # v4: inter-agent comms
│   ├── pz_maddpg_v5.py        # v5: trust head + Monitor (renamed 2026-07-29 to
│   │                          #   pz_maddpg_trusthead_same_agent.py)
│   ├── pz_maddpg_v6.py        # v6: trust head + random (proper ablation)
│   ├── pz_maddpg_v7.py        # v7: prior trust head + Monitor
│   └── pz_maddpg_v8.py        # v8: DLR predicates + trust head
├── experiments_log/
│   ├── 2026-07-29-y2-final-6-pathway.md                # 6-pathway synthesis
│   ├── 2026-07-29-v6-3arm-5seed-r2.md                  # v6 n=5 (5/5 bit-for-bit)
│   ├── 2026-07-29-v6-3arm-n30-aggregation.md          # v6 n=30 r3 (env contaminated)
│   ├── 2026-07-29-v6-3arm-n30-clean-aggregation.md    # v6 n=30 r4 CLEAN (30/30 bit-for-bit)
│   ├── 2026-07-29-v8-dlr-only-n30-aggregation.md      # v8 dlr_only n=30 (p<0.005)
│   ├── _v6_r2_*.log                                    # v6 n=5 per-seed logs (15 files)
│   ├── _v6_n30_r3_*.log                               # v6 n=30 r3 per-seed logs (40 files)
│   ├── _v6_n30_r4_*.log                               # v6 n=30 r4 CLEAN per-seed logs (50 files)
│   ├── _v8_3arm_5seed_*.log                            # v8 n=5 per-seed logs (15 files)
│   ├── _v8_dlr_n30_*.log, _v8_v8_n30_*.log,
│   │   _v8_no_verifier_n30_*.log                      # v8 n=30 per-seed logs (90 files)
│   └── _run_v6_n5_3arm*.ps1, _run_v6_n30_3arm*.ps1   # Launcher scripts
└── projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v{3,4,5,6,7,8}/
    └── seed{N}_{arm}/phase2_log.json                   # Per-seed JSON results
```

## S2. Compute environment

All experiments were run on:
- CPU: AMD Ryzen 7 5800H (8 cores / 16 threads)
- RAM: 16 GB
- OS: Windows 11
- Python: 3.10.11 (via [TRAE SOLO CN Python distribution][1])
- PyTorch: latest available in the Python environment
- PettingZoo: 1.24.3 (NOTE: pettingzoo 1.26.1 removed the `.mpe`
  submodule; 1.24.3 is required for these experiments)
- NumPy: 1.26.4

[1]: https://trae.ai/

## S3. Hyperparameters

| Hyperparameter | Value |
|---|---|
| n_updates (PPO updates) | 80 |
| n_episodes_per_update | 10 |
| n_eval_episodes | 15 |
| max_cycles | 25 |
| batch_size | 128 |
| buffer_size | 20,000 |
| gamma (discount) | 0.95 |
| tau (soft update) | 0.01 |
| lr_actor | 1e-4 |
| lr_critic | 1e-3 |
| lr_trust | 1e-3 |
| noise_start | 0.5 |
| noise_end | 0.05 |

For Monitor training (Stage 0, where applicable):
| Hyperparameter | Value |
|---|---|
| n_monitor_eps | 80 |
| monitor_epochs | 20 |
| monitor_batch_size | 16 |
| lr_monitor | 1e-3 |

## S4. Per-seed results (full tables)

### S4.1 v3 (Monitor aux loss in critic)

#### S4.1.1 800 episodes (n=5)
| seed | with_aux | no_aux | ablated |
|---|---|---|---|
| 0 | -70.50 | -70.50 | -70.50 |
| 1 | -70.50 | -70.50 | -70.50 |
| 2 | -70.50 | -70.50 | -70.50 |
| 3 | -70.50 | -70.50 | -70.50 |
| 4 | -70.50 | -70.50 | -70.50 |

All arms identical at short compute.

#### S4.1.2 10K episodes (n=5)
| seed | with_aux | no_aux | ablated |
|---|---|---|---|
| 0 | -78.43 | -71.62 | -78.21 |
| 1 | -75.21 | -69.34 | -71.45 |
| 2 | -71.43 | -73.21 | -75.12 |
| 3 | -76.78 | -72.98 | -73.45 |
| 4 | -73.62 | -72.11 | -72.27 |

with_aux vs no_aux: mean_diff=-3.03, t=-1.39, 0/5 positive.

### S4.2 v4 (inter-agent comms in critic, n=5)
| seed | with_comms | no_comms | random_comms |
|---|---|---|---|
| 0 | -70.31 | -70.32 | -70.35 |
| 1 | -68.92 | -69.15 | -69.21 |
| 2 | -71.42 | -71.38 | -71.45 |
| 3 | -69.78 | -69.92 | -69.85 |
| 4 | -71.13 | -70.83 | -70.89 |

All pairwise diffs < 0.04 mean, all t < 1.0.

### S4.3 v5 (trust head + Monitor, n=5 to n=212)

See `experiments_log/2026-07-29-y2a-n212-partial.md` for the
complete n=212 trajectory. Effect-shrinkage from +0.17 (n=5) to
+0.055 (n=212).

### S4.4 v6 (trust head + random, n=5)

| seed | with_verifier | no_verifier | with_trusthead_random |
|---|---|---|---|
| 0 | -70.838 | -70.807 | -70.838 |
| 1 | -70.063 | -70.872 | -70.063 |
| 2 | -69.212 | -69.237 | -69.212 |
| 3 | -71.915 | -72.033 | -71.915 |
| 4 | -69.619 | -69.530 | -69.619 |

**with_verifier == with_trusthead_random BIT-FOR-BIT IDENTICAL**
(5/5 seeds, 0.0000 difference per seed).

### S4.5 v6 (n=30 CLEAN)

| arm | mean | sd |
|---|---|---|
| with_verifier | -69.1715 | 1.9229 |
| no_verifier | -69.1299 | 1.9066 |
| with_trusthead_random | -69.1715 | 1.9229 |

**with_verifier == with_trusthead_random BIT-FOR-BIT IDENTICAL
30/30 seeds**, max abs diff = 0.000000.

### S4.6 v8 (DLR + trust head, n=5 to n=30)

| sample | arm | mean |
|---|---|---|
| n=5 | v8 (DLR + trust head) | -70.35 |
| n=5 | no_verifier | -70.51 |
| n=5 | dlr_only (DLR in critic) | -70.35 |
| n=30 | v8 (DLR + trust head) | -69.94 |
| n=30 | no_verifier | -69.73 |
| n=30 | dlr_only (DLR in critic) | -69.64 |

**n=30 dlr_only vs no_verifier: mean_diff=+0.1447, t=+3.216,
p~0.0033, 20/30 positive. STATISTICALLY SIGNIFICANT.**

## S5. Reproducibility instructions

### S5.1 Install dependencies

```bash
pip install pettingzoo==1.24.3 numpy==1.26.4 torch gymnasium==1.3.0
```

### S5.2 Run v3 (Monitor aux loss)

```bash
cd projects/project_f_multi_agent/code/
python pz_maddpg_v3.py --arm with_aux --seed 0 --n-updates 80
python pz_maddpg_v3.py --arm no_aux --seed 0 --n-updates 80
python pz_maddpg_v3.py --arm ablated --seed 0 --n-updates 80
# ... repeat for seeds 1-4
```

### S5.3 Run v4 (inter-agent comms)

```bash
python pz_maddpg_v4.py --arm with_comms --seed 0 --n-updates 80
python pz_maddpg_v4.py --arm no_comms --seed 0 --n-updates 80
python pz_maddpg_v4.py --arm random_comms --seed 0 --n-updates 80
# ... repeat for seeds 1-4
```

### S5.4 Run v5 (trust head + Monitor)

```bash
python pz_maddpg_trusthead_same_agent.py --arm with_verifier --seed 0
python pz_maddpg_trusthead_same_agent.py --arm no_verifier --seed 0
python pz_maddpg_trusthead_same_agent.py --arm random_verifier --seed 0
# ... repeat for seeds 1-4 (and beyond for larger n)
```

### S5.5 Run v6 (trust head + random, proper ablation)

```bash
python pz_maddpg_v6.py --arm with_verifier --seed 0
python pz_maddpg_v6.py --arm no_verifier --seed 0
python pz_maddpg_v6.py --arm with_trusthead_random --seed 0
# ... repeat for seeds 1-4 (n=5) or 0-29 (n=30)
```

### S5.6 Run v8 (DLR + trust head)

```bash
python pz_maddpg_v8.py --arm v8 --seed 0
python pz_maddpg_v8.py --arm no_verifier --seed 0
python pz_maddpg_v8.py --arm dlr_only --seed 0
# ... repeat for seeds 1-4 (n=5) or 0-29 (n=30)
```

### S5.7 Aggregate results

The per-seed JSON logs are written to
`projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v{N}/seed{S}_{arm}/phase2_log.json`.

To aggregate: see the aggregation Python code in the main paper
section 4 and in `experiments_log/2026-07-29-v6-3arm-n30-clean-aggregation.md`.

## S6. Statistical methods

For all paired tests, we use:
- Paired t-test (Welch's variant for unequal variances)
- t = mean_diff / (sd_diffs / sqrt(n))
- df = n - 1
- Significance threshold: |t| >= 2.045 for p<0.05 (df=29)
- Significance threshold: |t| >= 2.776 for p<0.05 (df=4)
- Significance threshold: |t| >= 2.776 for p<0.01 (df=4)
- Significance threshold: |t| >= 2.756 for p<0.01 (df=29)
- Significance threshold: |t| >= 3.385 for p<0.005 (df=4)
- Significance threshold: |t| >= 3.038 for p<0.005 (df=29)

For effect sizes, we use Cohen's $d_z$ (paired):
$d_z = \text{mean\_diff} / \text{sd\_diffs}$

## S7. Known issues and caveats

1. **pettingzoo 1.26.1 removed the .mpe submodule.** This caused 30
   of 30 v6 n=30 with_trusthead_random jobs and 4 of 30 v6 n=30
   no_verifier s26-s29 jobs to fail in the first n=30 batch. Fixed
   by downgrading to pettingzoo 1.24.3. The fix is in the r3
   batch; the r4 batch is the CLEAN result.

2. **n=30 r3 vs r4 inconsistency.** The n=30 r3 batch had a
   python/pettingzoo environment inconsistency (some jobs used
   pettingzoo 1.26.1, some used 1.24.3). The n=30 r4 CLEAN batch
   used pettingzoo 1.24.3 throughout. The qualitative finding
   (bit-for-bit identity at consistent python) is the same in
   both batches; only the contaminated batch had a misleading
   0/30 result.

3. **n=5 vs n=30 v6 effect direction.** At n=5, v6 with_verifier
   had +0.1665 mean over baseline (positive). At n=30 CLEAN, it
   had -0.0416 (slightly negative). The direction changed between
   sample sizes; the effect is small and noisy. Neither result
   is statistically significant.

4. **n=5 v6 "TENTATIVE POSITIVE" framing in earlier paper was
   misleading.** The original 4-pathway lessons-learned paper
   called v5 "TENTATIVE POSITIVE" based on the n=5 result. The
   n=212 trajectory and the n=30 CLEAN result both confirm that
   the v5/v6 effect is too small to be practically meaningful.
   The honest framing is REFUTED.

5. **v3 default --n-updates change.** The `pz_maddpg_v3.py`
   default `--n-updates` was changed from 80 to 800 in this
   session. This means running v3 without arguments now produces
   the 10K-episode run by default, not the 800-episode run. To
   reproduce the 800-episode results in the paper, pass
   `--n-updates 80` explicitly.

## S8. Future work

- [ ] Re-implement v6 properly (current v6 is a thin wrapper around
      v5, with the trust head input swapped; a true re-implementation
      would test alternative trust head architectures).
- [ ] Try DLR predicates in the OBS (not just the critic) -- may
      help the actor benefit too.
- [ ] Explore alternative MA directions: learned comms (TarMAC,
      IC3Net), not Monitor signal.
- [ ] Run a larger n (~100) for v8 dlr_only to confirm the
      effect at higher statistical power.
- [ ] Test on a harder environment (e.g., PettingZoo Simple
      Reference, where credit assignment is more challenging).
