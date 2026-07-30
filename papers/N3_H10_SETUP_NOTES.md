# H10 n=20 setup notes for next session

**Goal**: Run H10 n=20 (3 arms x 20 seeds = 60 jobs) to confirm
direction-consistent REFUTATION at larger n.

**Problem in current session**: The h10_real_pilot.py requires
the `llm_monitor` module which is not installed in any of the
available Python environments (ModuleNotFoundError). The Y4
paper's n=5 results were run in a specific environment that we
cannot access now.

## Required setup for next session

### 1. Locate the working Python environment that has

- `llm_monitor` (Project G code)
- `pettingzoo` (with mpe submodule, version <= 1.24.3)
- `transformers` (for Qwen2.5-1.5B-Instruct)
- `torch`

### Possible environment locations (this session's failures)

- `C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe`
  -- has pettingzoo 1.26.1 (no `.mpe`)
- `C:\Users\Administrator\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`
  -- no pettingzoo
- `C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
  -- has pettingzoo 1.26.1

### 2. Install required modules in the chosen environment

```bash
pip install pettingzoo==1.24.3
pip install torch transformers
```

### 3. Verify modules

```python
import sys
sys.path.insert(0, r'E:\agi-research\projects\project_g_llm_self_monitoring\code')
import llm_monitor
import joint_monitor
import real_llm_rollout_collector
import simple_arithmetic_dataset
import h10_smoke
```

### 4. Run the launcher (already exists)

```powershell
cd E:\agi-research\experiments_log
Start-Process powershell -ArgumentList '-File', 'E:\agi-research\experiments_log\_run_h10_n20.ps1' -WindowStyle Hidden
```

The launcher runs 60 jobs (3 arms x 20 seeds) at 6-parallel.
Each job uses H10_N_TOTAL=8 rollouts, simple arithmetic, CPU,
stratified split.

### 5. Monitor progress

```powershell
Get-Content E:\agi-research\experiments_log\_h10_n20_*.done
```

### 6. Aggregate when done

```python
# Load H10 results, compute Frozen vs Joint vs Random paired tests
# with Bonferroni correction, confidence intervals
# Compare to existing n=5 result (Joint > Frozen by 0.10,
# t=-0.516 NOT sig)
```

### 7. Update Y4 paper with the n=20 result

Similar to how n=5 was integrated earlier. Add:
- Power analysis with actual n=20 effect
- Bonferroni correction with 3 paired tests
- 95% confidence intervals
- Effect-stability discussion (is the n=20 result similar to n=5?)

## Current Y4 paper status

H10 REFUTED at n=5, direction-consistent (Joint > Frozen by
0.10, t=-0.516 NOT sig). Underpowered (need n=36 for 80%
power).

## Goal of n=20

Confirm direction-consistent REFUTATION at larger n, ideally
with stronger statistical evidence. If n=20 gives p<0.05 with
Bonferroni, the Y4 paper has a stronger REFUTATION result. If
n=20 reverses direction (Frozen > Joint), the n=5 result was
noise and the Y4 paper should be reframed as "n=5 was underpowered
and direction is not stable".

## Estimated timing

At ~3 min per job (H10_N_TOTAL=8, simple arithmetic, CPU), 60
jobs at 6-parallel = 10 batches x 3 min = ~30 min. The Y4 n=5
took ~10 min total, so the n=20 should be similar (since
N_TOTAL is the same).

## Outcome (2026-07-30)

- Full 60-job H10 n=20 launch completed (seed 100-119, 3 arms x 20 seeds).
- Two launchers failed in this session because the wrapper scripts
  did not set `sys.path` to include the Project G `code` directory
  (so `llm_monitor` was reported as missing). The launcher
  `experiments_log/_run_h10_n20.ps1` was patched to insert
  `sys.path.insert(0, r"E:\agi-research\projects\project_g_llm_self_monitoring\code")`
  before `exec` of the pilot script.
- Added `H10_MAX_NEW_TOKENS` env override to keep CPU wall-clock
  bounded (16 tokens vs 80 default).
- Reduced `MAX_PARALLEL` to 1 after observing severe CPU thrash at
  6-way parallelism (1.5B model load + 6 concurrent inferences).
- Seed 111 collapsed to a single-class eval under the stratified
  split; patched `h10_real_pilot.py` to fall back to a rebalanced
  (1 success, 1 failure) eval set when this happens.
- Aggregated results in `experiments_log/_h10_n20_summary.json`;
  updated Y4 paper (Section 7) with the n=20 follow-up.
