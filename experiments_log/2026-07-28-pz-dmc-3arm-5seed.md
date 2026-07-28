# DMC 3-arm 5-seed sweep: real vs random vs no Monitor shaping

> Date: 2026-07-28
> Goal: separate two effects confounded in the 5-seed DMC run:
>  1) Does reward shaping itself help, regardless of Monitor quality?
>  2) Is the *trained* per-agent Monitor a better shaping signal than a random signal?
> Code: `projects/project_f_multi_agent/code/pz_dmc.py --shaping-mode {real,random,none}`

## 1. Setup

- 5 seeds (0-4), PettingZoo Simple Spread v3 (discrete, N=3, max_cycles=25)
- Stage 1: 5 SHARED-PPO updates x 10 episodes = 50 episodes (shared across all 3 arms per seed)
- Monitor training: 50 frozen-PPO rollouts (real Monitor only, same for all 3 arms)
- Stage 2: 5 per-agent PPO updates x 10 episodes with shaped reward
  - arm=real:   r_total = r_env - 0.5 * monitor_prob_i
  - arm=random: r_total = r_env - 0.5 * U[0,1] per agent
  - arm=none:   r_total = r_env (no shaping; matches per-agent PPO Stage 2)
- Eval: 15 episodes deterministic

## 2. Per-seed results

| seed | real   | random | none   | real-rand | real-none |
|---|---|---|---|---|---|
| 0 |  -96.33 |  -93.01 |  -90.31 | -3.32 | -6.02 |
| 1 | -125.93 | -130.67 | -132.98 | +4.75 | +7.05 |
| 2 | -197.90 | -193.13 | -192.51 | -4.78 | -5.39 |
| 3 |  -98.09 | -105.75 | -103.73 | +7.65 | +5.63 |
| 4 | -108.46 | -117.53 | -118.77 | +9.07 | +10.31 |
| mean | **-125.34** | **-128.02** | **-127.66** | +2.67 | +2.32 |
| sd | 42.23 | 38.98 | 39.63 | - | - |

## 3. Paired t-tests

| comparison | mean_diff | se | t | positive |
|---|---|---|---|---|
| real vs random | +2.67 | 2.84 | +0.94 | 3/5 |
| real vs none   | +2.32 | 3.36 | +0.69 | 3/5 |
| random vs none | -0.36 | 0.95 | -0.38 | 2/5 |

None of the paired t-tests reach significance (|t|<2.776 with df=4).

## 4. Honest interpretation

**Three findings, all NEGATIVE on the original hypotheses:**

1. **real monitor shaping is NOT significantly better than no shaping**
   (mean +2.32, t=+0.69). The Y1.3-style reward penalty does not transfer
   to multi-agent PPO at this compute scale. Same conclusion as H1.4 in
   single-agent: the real Monitor is not a useful shaping signal.

2. **real monitor shaping is NOT significantly better than random shaping**
   (mean +2.67, t=+0.94). At this compute scale, a *trained* per-agent
   Monitor (AUROC 0.99!) gives essentially the same reward perturbation
   as uniform noise. This is a stronger negative result than H1.4: in
   MA the real Monitor is *not* better than random (it was worse in
   single-agent), but it is also not better than nothing.

3. **random monitor shaping = no shaping** (mean -0.36, t=-0.38). Random
   noise as a reward penalty averages to a constant offset which PPO
   adapts to, leaving policy unchanged. Consistent with theory.

## 5. Why the 5-seed DMC showed +16.2 but 3-arm shows +2.3

The earlier 5-seed DMC report showed DMC vs Stage-1 shared PPO = +16.2
mean. This 3-arm study shows DMC vs no-shaping = +2.3. The discrepancy
is because Stage-1 shared PPO is unstable (it diverges to -141 vs the
later Stage-2 PPO of -125); the apparent DMC gain over Stage 1 was
actually mostly Stage-2 PPO recovering from Stage-1 collapse, not the
Monitor shaping itself. Honest re-framing: **DMC shaping contributes
at most +2-3 mean; the +16.2 was an artifact of comparing against an
unstable Stage 1**.

## 6. Implications for H5 and Y1 paper

- H5 status remains PARTIAL: per-agent Monitor learns very well (AUROC
  0.99) but does not yield a statistically significant reward-shaping
  gain at this compute scale.
- The decoupling assumption is still validated (AUROC matches Y1.3 in
  single-agent) but the Y1.3-style reward transfer does NOT generalise
  to multi-agent at this scale.
- This sharpens the Y1 paper claim: Y1.3 is a *single-agent* finding.

## 7. Action items

- [x] Add `--shaping-mode` arg to pz_dmc.py (real/random/none)
- [x] 3-arm 5-seed sweep
- [x] Honest re-framing of earlier +16.2 finding
- [ ] Update H5 section in 9-hypothesis framework
- [ ] C/B (better MADDPG + continuous DMC) will revisit this comparison
      at matched compute. Real verdict on H5 needs that.

## 8. Lessons

- 3-arm ablation is *essential* whenever you compare to a baseline that
  might be unstable. A single +16.2 number is meaningless without a
  no-shaping control.
- The decoupling assumption generalises to MA envs (Monitor AUROC 0.99).
  The Y1.3 reward-shaping recipe does NOT generalise at this scale.
- 5-seed n is too small for paired t-tests of small effects (se~3 gives
  power for effects >7, but observed effects are 2-3). Y2 should use
  10-15 seeds if effect sizes stay this small.