# MADDPG v4 5-seed 3-arm: NEGATIVE (inter-agent comms has near-zero effect)

> Date: 2026-07-28
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v4.py`
> Y2 follow-up: try inter-agent message broadcast (TarMAC-lite) as
> alternative to v3's Monitor aux loss. Same compute as v2/v3 baseline.

## 1. Architecture

- per-agent MessageEncoder: obs -> 32-dim message (broadcast to all)
- critic input: (all_obs, all_actions, all_messages) -> Q_i
- actor unchanged: still obs -> action
- msg encoder updated via -Q gradient (msg that helps Q is reinforced)

Three arms:

- **with_comms**: real per-agent learned message encoder
- **no_comms**: MADDPG v2 baseline (no messages)
- **random_comms**: random per-agent messages (control)

## 2. 5-seed results (80 updates x 10 episodes = 800 env episodes)

| arm | mean | sd | n |
|---|---|---|---|
| with_comms | -70.31 | 1.14 | 5 |
| no_comms | -70.32 | 1.22 | 5 |
| random_comms | -70.35 | 1.22 | 5 |

Paired t-tests:
- with_comms vs no_comms: mean_diff=+0.00, t=+0.05 (NOT sig)
- with_comms vs random_comms: mean_diff=+0.04, t=+0.80 (NOT sig)
- no_comms vs random_comms: mean_diff=+0.03, t=+0.75 (NOT sig)

## 3. Honest interpretation

Inter-agent comms have near-zero effect. Per-seed with_comms is
sometimes slightly better, sometimes worse, sometimes equal to
no_comms. The 3 arms produce results that are essentially the same
(within 0.04 mean difference).

This is the SECOND confirmation that adding extra critic inputs
(Monitor aux loss in v3, inter-agent messages in v4) does NOT
improve MADDPG v2 at this compute scale.

## 4. Why might this be?

- PettingZoo Simple Spread v3 may be too 'easy' for the centralised
  critic + full global state to be near-saturating at 800 env
  episodes; the marginal value of extra info is small.
- The 32-dim messages may be too low-dimensional to encode useful
  inter-agent info beyond what the full global state already gives.
- The message encoder is trained end-to-end with the critic, which
  is a hard credit assignment problem on its own.
- The actor is unchanged, so the messages can only help via the
  critic; the actor cannot act on them.

## 5. What this means for Y2

The unifying finding: extra critic inputs (Monitor, messages) do
NOT help at our compute scale. The remaining Y2 directions that
might still help:

- **Path c (Monitor as MA verifier)**: instead of training critics
  to consume Monitor signal, use Monitor output as a post-hoc
  verification signal in a cross-agent evidence chain. The trust
  score from the chain gates inter-agent cooperation. This is a
  different architectural pattern: monitors VERIFY, not TRAIN.
- **Longer compute**: v3 10K re-run is in progress. If even 10K
  episodes don't show an effect, we conclude the negative result
  is robust to compute scale.
- **Other envs**: the current result is on Simple Spread only. SMAC
  lite, Hanabi (2-player), or Level-Based Foraging might show
  different behaviour.

## 6. Action items

- [x] pz_maddpg_v4.py end-to-end works (3 arms)
- [x] 5-seed sweep at matched compute
- [x] Honest log: v4 NEGATIVE
- [ ] v3 10K re-run in progress (waiting for completion)
- [ ] Y2: path c (Monitor as MA verifier) implementation
- [ ] Y2: other MA envs (SMAC lite, Hanabi)

## 7. Why publish this as a negative result

A clean, well-controlled 3-arm ablation showing that the
'obvious' extension (inter-agent comms in critic) does not help
is publishable. We document:

- The architecture (TarMAC-lite message broadcast in critic)
- The matched-compute comparison
- The result: with_comms = no_comms = random_comms
- The implication: critic-side extras are not the bottleneck

This narrows the design space for MA-RL: the credit assignment
problem on Simple Spread is already solved by MADDPG v2's full
global state, and adding more global info does not help.