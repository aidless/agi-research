---
license: mit
tags:
  - reinforcement-learning
  - multi-agent
  - prompt-optimization
  - monitor
  - archimedes-project
  - pre-registered
---

# Archimedes Project -- Proposition 3 Hybrid Pre-Reg Launcher

This repository contains the **pre-registered MADDPG v8 launcher** for the
Archimedes Project's Proposition 3 hybrid test. The test runs a 3-arm
cooperative multi-agent reinforcement learning experiment:

- **Monitor alone** (`--arm monitor_only`): a single Monitor signal as
  training-time regularizer, no DLR predicates
- **DLR alone** (`--arm dlr_only`): hand-crafted DLR cross-agent predicates
  in the critic, no Monitor signal
- **Hybrid** (`--arm v8`): both Monitor (trust head) AND DLR predicates in the
  critic

The pre-registered hypothesis is that the Hybrid arm produces a larger positive
effect on the policy reward than either Monitor alone or DLR alone. This is
the framework's Proposition 3 (Y5 \u00a77.6.2).

## Paper

This launcher is referenced in:

- **Y5 v1.3 master synthesis** (COLM 2026, 89 pages): the master synthesis
  paper that places this test in the 11-comparison cross-context record
- **Y3 v1.0 6-pathway paper**: the multi-agent investigation that motivates this
  test (5 of 6 pathways REFUTED, 1 positive v8 dlr_only result)
- **Pre-Reg PROP3-HYBRID.md**: the formal pre-registration document

The 3-arm hypothesis is registered in
`experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md` with a clear
decision rule:

- VALIDATED if Hybrid - DLR >= +0.05 with p<0.05 (Bonferroni)
- REFUTED otherwise

## Quick start

### Prerequisites

- Python 3.11+ with torch and pettingzoo installed:
  ```bash
  pip install torch pettingzoo numpy
  ```
- A POSIX shell (Linux, macOS, or WSL on Windows)
- ~50 GPU-hours of compute for the full pre-reg (n=100 seeds x 3 arms x 800 updates)
- ~5 GPU-hours for the sandbox-feasible tightened config (n=20 x 3 arms x 200 updates)

### Run the smoke test (~30 sec total)

```bash
# 3 arms x 1 seed x 8 PPO updates (~10 sec per arm)
for arm in no_verifier dlr_only v8; do
    python projects/project_f_multi_agent/code/pz_maddpg_v8.py \
        --arm $arm --seed 0 \
        --n-updates 8 --n-episodes-per-update 4 --n-eval-episodes 4
done
```

### Run the tightened production config (~5 hours CPU)

```bash
# 3 arms x 20 seeds x 200 PPO updates, MAX_PARALLEL=6
powershell -File experiments_log/_run_p3_hybrid_production.ps1
```

### Run the full pre-reg (~50 GPU-h, 2026-08-01 to 2026-08-15 window)

```bash
# 3 arms x 100 seeds x 800 PPO updates, MAX_PARALLEL=6
powershell -File experiments_log/_run_v8_10k_n50_3arm.ps1
```

## Repository structure

```
agi-research/
  papers/
    y5_v1_3_master_synthesis.pdf          # Master synthesis (89 pages)
    y3_v1_0_6pathway.{md,pdf}            # 6-pathway Y3 paper
    monitor_signal_vs_dlr_6pathway.md     # Y3 paper source
    y1_v1_0_paper.{md,pdf}               # Single-agent Y1 paper
    project_g_v0_5_h10_paper.md            # LLM Y4 paper
  projects/
    project_f_multi_agent/
      code/
        pz_maddpg_v8.py                   # Training script (3 arms supported)
  experiments_log/
    2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md  # Pre-reg
    _run_p3_hybrid_production.ps1         # Tightened launcher
    _run_v8_10k_n50_3arm.ps1              # Full pre-reg launcher
    _smoke_test_results.md                # Smoke test documentation
```

## Architecture

The P3 hybrid test runs the MADDPG v8 cooperative multi-agent algorithm
on `pettingzoo.mpe.simple_spread_v3`. The 3 arms differ in what auxiliary
signal is used:

- **no_verifier**: baseline, no auxiliary signal
- **dlr_only**: 24-dimensional DLR cross-agent predicate vector (e.g., "agent
  i is closest to landmark j") is fed into the critic as additional input
- **monitor_only**: a small failure-prediction network (the "Monitor") is
  trained on the frozen reference policy's rollouts and used as a trust
  head at the actor
- **v8** (Hybrid): both Monitor and DLR predicates are used

The Proposition 3 hypothesis predicts that v8 > dlr_only AND v8 > monitor_only
with effect size >= +0.05 at n=100 paired seeds.

## Compute budget

The full pre-reg (n=100 x 3 arms x 800 updates) takes ~50 GPU-h wall-clock on
CPU-equivalent. The tightened production config (n=20 x 3 arms x 200 updates)
takes ~5 GPU-h. The smoke test (1 seed x 3 arms x 8 updates) takes ~30 sec.

We recommend starting with the smoke test to verify the pipeline on your
hardware, then the tightened config to estimate the full pre-reg cost, then
the full pre-reg for the actual hypothesis test.

## Results format

Each job produces a log file at `experiments_log/_p3_hybrid_<arm>_s<seed>.log`
with the format:

```
============================================================
MADDPG v8 - arm=<arm>
============================================================
Phase 1: Random baseline = <mean> +/- <std>
    update 1/200: mean_episode_return=<val>, buffer=<size>
    ...
Phase 3: Final eval...
  MADDPG v8 (<arm>) eval: <final_mean> +/- <final_std>  (delta vs random: <delta>)
  Log: <checkpoint_path>
```

After all jobs complete, the bootstrap aggregator should be run to compute
the 3 pairwise contrasts (Monitor vs DLR, Monitor vs Hybrid, DLR vs Hybrid)
with confidence intervals and p-values. The aggregator is not included in
this repo (see Y3 v1.0 6-pathway paper for the standard aggregator).

## Citation

```bibtex
@misc{archimedes2026p3,
  title = {Pre-Registration: Proposition 3 Hybrid Test
           (Monitor + DLR > Either Alone in Cooperative MARL)},
  author = {Liu, Zewen and {Archimedes Project}},
  year = {2026},
  howpublished = {GitHub repository, AGI-2026-001},
  note = {Pre-registered 2026-07-31; execution window 2026-08-01 to 2026-08-15}
}
```

## License

MIT License. See `LICENSE` in the repository root.

## Contact

Liu Zewen (Archimedes Project, AGI-2026-001)
Repository: https://github.com/aidless/agi-research