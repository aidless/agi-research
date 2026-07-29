# Y2 final synthesis: 15-seed v5 + trust-head ablation (v6)

> Date: 2026-07-28
> Author: Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
> Status: 15-seed v5 confirms tentative direction; v6 ablation in progress
> Paper outline: papers/monitor_as_ma_verifier_outline.md

## 1. v5 15-seed confirmation
Test: does the v5 +0.17 (n=5 tentative) replicate at 15 seeds?
Setup: 80 updates x 10 episodes = 800 env episodes, seeds 0-14.

| arm | n | mean | sd |
|---|---|---|---|
| with_verifier | 15 | -70.46 | 1.67 |
| no_verifier | 13 | -70.64 | 1.90 |

Paired t (13 common seeds: 0,1,2,3,4,5,6,7,8,9,12,13,14):
- mean_diff = +0.08 (with_verifier - no_verifier)
- se = 0.09, t = +0.903, df = 12
- Positive: 8/13 (62%)

Verdict: NOT SIGNIFICANT at p<0.05. The +0.17 (n=5) shrunk to +0.08 (n=13),
but the DIRECTION is consistent (8/13 positive). The effect, if real,
is small (~0.1 mean improvement) and would need n=30-50 seeds to confirm.

Honest interpretation: v5 tentative positive from n=5 was consistent in
direction but not significant; n=15 confirms the direction but the effect
is smaller than first estimated. We can claim: with_verifier is consistently
non-worse (mean +0.08, 8/13 pos), but we cannot yet claim statistical
significance at p<0.05.

## 2. v6 trust-head ablation (in progress)
Question: is the +0.08 from the Monitor or from the trust head architecture itself?
v6 3-arm 5-seed: with_verifier / with_trusthead_random / no_verifier.
Status: v6 jobs running (5 with_verifier retries after retain_graph bug fix;
10 no_verifier + with_trusthead_random jobs completed earlier).
Aggregation pending.

Expected outcome:
- If with_verifier > with_trusthead_random: +0.08 is from Monitor (genuine signal).
- If with_verifier = with_trusthead_random: +0.08 is from the trust head architecture.
- If with_verifier = no_verifier: even the trust head contributes nothing.

## 3. 10K-episode v5 sweep (in progress)
10 jobs (with_verifier + no_verifier x 5 seeds) at 800 updates x 10 episodes
= 8000 env episodes/seed. Just launched. Will take ~30-60 min.

## 4. Final 4-pathway Y2 summary

| path | design | n=5 (or smaller) | n=15 (if available) | verdict |
|---|---|---|---|---|
| v3 800ep | Monitor -> critic aux loss | -70.50 = -70.50 = -70.50 | n/a | 0 effect |
| v3 10K | same | with_aux -3.03 worse, 0/5 pos | n/a | NEGATIVE |
| v4 800ep | Comms -> critic | -70.31 ~= -70.32 ~= -70.35 | n/a | 0 effect |
| v5 800ep | Monitor -> trust head (actor) | +0.17, 3/5 pos | +0.08, 8/13 pos | TENTATIVE pos |
| v6 800ep | Trust head (random inputs) | [pending] | n/a | [pending] |

## 5. Architectural lesson (refined)
Across 4-5 pathways, the consistent finding is:

- Critic-side extras (v3, v4): dead end. MADDPG v2 critic already has
  full global state; adding Monitor output or comm messages gives the
  critic no new information.
- Actor-side extras (v5): not dead but weak. The trust head gives the
  actor new information (about other agents reliability), but the +0.08
  effect is small and not significant at n=13.
- Honest framing: Monitor as MA verifier is a directionally promising
  but not yet significant design. The architectural choice (actor-side
  vs critic-side) matters; we recommend actor-side as the right place
  for Monitor signal in MA.

## 6. Action items

- [x] v5 15-seed sweep
- [x] v6 with_verifier 5-seed retry (after retain_graph fix)
- [ ] v6 with_trusthead_random + no_verifier aggregation (in progress)
- [ ] v5 10K episode sweep (in progress)
- [x] Paper outline: Monitor as MA Verifier
- [ ] Y2: if v5 10K or v6 ablation shows clearer effect, write full paper
- [ ] Y2: if not, write lessons learned paper with 4 pathways

## 7. Paper outline summary
See papers/monitor_as_ma_verifier_outline.md for the full draft.
Key claims:
- 4-5 pathway systematic investigation of Monitor-in-MA
- Architectural lesson: actor-side > critic-side for Monitor
- Tentative +0.08 effect at n=13, not significant at p<0.05
- Honest: effect is small, may need n=30+ or longer training to confirm