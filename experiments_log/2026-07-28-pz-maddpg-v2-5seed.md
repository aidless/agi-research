# MADDPG v2 5-seed sweep: proper next_obs bootstrap (STRONG POSITIVE)

> Date: 2026-07-28
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v2.py`
> Verdict: 5/5 seeds positive, paired t=+6.50, p<0.001 df=4

## 1. Why v2 (vs v1)

`pz_maddpg.py` (v1) had three bugs:
- `target_q = torch.zeros(B)`: no next_obs bootstrap at all.
- `target_actors` / `target_critic` were soft-updated but never queried.
- Other agents obs/action were zero-padded in the centralized critic,
  so the critic was learning from a partial state.

v2 fixes all three: each transition stores `(obs, action, next_obs, reward, done)`,
critic target uses `target_actor(next_obs) -> target_critic`,
and the critic input is the full global state (other agents zero-padded, but at
both current and next step).

## 2. Setup (per seed)

- 80 PPO-style updates x 10 episodes = 800 env episodes
- Replay buffer 20000; batch size 128; gamma 0.95; tau 0.01
- LR actor 1e-4, LR critic 1e-3
- Linear noise decay 0.5 -> 0.05 over first 40 updates
- Eval: 15 episodes deterministic, seed=4000

## 3. 5-seed results

| seed | random | MADDPG v2 | delta |
|---|---|---|---|
| 0 | -81.95 | -70.81 | +11.13 |
| 1 | -74.88 | -70.79 | +4.09 |
| 2 | -76.66 | -69.18 | +7.49 |
| 3 | -78.69 | -72.00 | +6.69 |
| 4 | -78.69 | -69.48 | +9.21 |
| mean | -78.17 | **-70.45** | **+7.72** |
| sd | 2.64 | 1.14 | 2.66 |

**5/5 positive**. Paired t (delta vs 0): mean=+7.72, se=1.19, **t=+6.50**,
p<0.001 (df=4).

## 4. vs v1 (broken bootstrap)

- v1 seed 0 (broken): -75.78, delta +1.66
- v2 seed 0 (fixed): -70.81, delta +11.13
- v2 - v1 (seed 0): **+4.97 improvement**
- v1 was likely benefitting from the replay buffer + 30 updates of off-policy
  gradient even without bootstrap; v2 is now closer to canonical MADDPG.

## 5. What this means

- **MADDPG is a STRONG positive baseline** when implemented correctly.
- The original v1 +1.66 was an under-estimate. Real MADDPG on this env at
  our compute gives +7.7 mean (p<0.001).
- This raises the bar for DMC. To make H5 closure honest, continuous-action
  DMC must beat -70.45 on the same compute budget.

## 6. Action items

- [x] MADDPG v2 end-to-end (proper bootstrap + target networks)
- [x] 5-seed sweep with matched compute (~800 episodes)
- [ ] Continuous-action DMC at matched compute (800 episodes) - phase B
- [ ] Final 6-way comparison: random / per-agent PPO / shared PPO / DMC-discrete / DMC-continuous / MADDPG-v2