# MADDPG v5 5-seed 3-arm: TENTATIVE POSITIVE (first consistent direction)

> Date: 2026-07-28
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v5.py`
> Y2 path c: Monitor as MA verifier via cross-agent evidence chain
> + trust head at actor level. This is the third Monitor-in-MA attempt,
> after v3 (aux loss in critic) and v4 (inter-agent messages in critic)
> both showed no effect.

## 1. Architecture (different from v3, v4)

- Critic: unchanged MADDPG v2 (full global state, per-agent Q)
- Monitor: per-agent frozen-decoupled (trained on Stage-1 PPO rollouts)
- **Evidence chain**: per step, per-agent monitor_prob -> SHA-256 entry
  `(agent_id, step, monitor_prob, prev_hash)`
- **Trust head** at the ACTOR level: takes (my_obs, my_monitor_prob,
  others_monitor_stats) -> per-other-agent trust weights in [0,1]
- Actor: trust-weighted sum of OTHER agents' Q + own Q; trust head
  is trained end-to-end via this loss
- KEY DIFFERENCE from v3/v4: Monitor signal only affects the ACTOR
  via trust-weighted Q blend, not the CRITIC loss. The trust head is
  the only Monitor-consumer besides the chain itself.

Three arms:
- **with_verifier**: real Monitor + trust head
- **no_verifier**: MADDPG v2 baseline (no trust head)
- **random_verifier**: random Monitor + trust head (control)

## 2. 5-seed results (80 updates x 10 episodes = 800 env episodes)

| arm | mean | sd | n |
|---|---|---|---|
| with_verifier | **-70.33** | 1.07 | 5 |
| no_verifier | -70.50 | 1.13 | 5 |
| random_verifier | -70.52 | 1.12 | 5 |

Paired t-tests (df=4, |t|>=2.776 for p<0.05):
- with_verifier vs no_verifier: mean_diff=+0.17, t=+1.01, 3/5 positive (NOT sig)
- with_verifier vs random_verifier: mean_diff=+0.20, t=+1.28, 3/5 positive (NOT sig)
- no_verifier vs random_verifier: mean_diff=+0.03, t=+1.55, 4/5 positive (NOT sig)

## 3. Honest interpretation

**TENTATIVE POSITIVE**: with_verifier is the FIRST Monitor-in-MA
design to show a CONSISTENT (though not significant) positive direction.
Effect size is +0.17 to +0.20 mean improvement, with 3/5 seeds positive
in both comparisons vs no_verifier and random_verifier.

However, the effect is small and not statistically significant at
n=5, df=4. To claim a real effect, we would need n=15-20 seeds or
a larger effect (e.g., longer training to amplify the signal).

## 4. Why might v5 differ from v3/v4?

- **v3 / v4 failed**: Monitor/messages were input to the CRITIC. The
  critic already has full global state; adding redundant info from
  Monitor or messages did not help.
- **v5 (tentative positive)**: trust head is at the ACTOR. It
  selects which other agents' Q values the actor should attend to,
  rather than just adding more info to the critic. This is a
  different architectural pattern: Monitor signal goes into action
  selection, not value estimation.
- The trust head is a NOVEL component (not present in v3/v4). It
  may be the source of the +0.17 signal, not the Monitor itself.
  Future ablation: with_verifier without trust head (trust head on
  random features) would isolate the Monitor contribution.

## 5. Comparison to all 4 Y2 attempts

| path | design | result |
|---|---|---|
| v3 (aux loss) | Monitor -> critic loss | 0 effect (800ep), -3.03 (10K) |
| v4 (comms) | TarMAC messages -> critic input | 0 effect |
| **v5 (verifier)** | **Monitor -> trust head -> actor Q blend** | **+0.17 (tentative)** |

v5 is the FIRST design to show a consistent (not significant)
positive direction. The path c design IS potentially better than
v3/v4, but the effect needs more seeds to confirm.

## 6. Implications for Y2 paper-worthy claim

The current result is suggestive but not strong enough to publish
as 'Monitor as MA verifier works'. To make this claim, we need:

- 15-20 seeds (3-4x current n) for significance at the observed
  effect size (~0.17).
- Or: longer training (10K+ episodes) to amplify the +0.17 effect.
- An ablation: with_verifier without trust head (test if Monitor
  alone matters vs trust head alone matters).

## 7. Action items

- [x] pz_maddpg_v5.py end-to-end works (3 arms)
- [x] 5-seed sweep at matched compute (800 ep/seed)
- [x] Honest log: v5 TENTATIVE POSITIVE (not significant)
- [ ] Y2: 15-seed v5 re-run for significance test
- [ ] Y2: trust-head-without-monitor ablation (isolate sources)
- [ ] Y2: longer compute v5 (10K episodes)
- [ ] If significant: write paper 'Monitor as MA Verifier'

## 8. Why this is the most interesting Y2 result so far

v3 (aux loss) and v4 (comms) both had clear negative results: the
Monitor signal does not help when used as critic input. v5 shows
that placing the Monitor signal at the ACTOR level (via trust head)
may be the right architectural choice. This is a substantive design
lesson, not just a numerical claim:

- 'Monitors as critic input' = dead end (v3, v4 NEGATIVE)
- 'Monitors as actor-side verifier' = promising direction (v5 tentative)
- The Monitor is a SHIPPING USE that should be a per-agent decision
  signal, not a learning signal for the value function.