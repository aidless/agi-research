# MADDPG v4 design: inter-agent comms follow-up to v3

> Date: 2026-07-28
> File: `projects/project_f_multi_agent/code/pz_maddpg_v4.py`
> Y2 follow-up: try inter-agent message broadcast as alternative to
> Monitor aux loss. v3 showed aux loss has zero effect (3 arms identical).

## 1. Architecture

Each agent has a small MessageEncoder (obs -> 32-dim message). All
agents broadcast their message to all others. The critic sees the FULL
global state including all messages, similar to how it sees all obs and
all actions. The actor is unchanged: still obs -> action (no message in
actor input; we test if giving the critic extra information helps).

Three arms:

- **with_comms**: real per-agent learned message encoder (gradient
  flows through Q to message encoder via -Q as msg loss)
- **no_comms**: MADDPG v2 baseline (no messages)
- **random_comms**: random per-agent messages (control)

## 2. Smoke test (seed 99, 5 update x 3 episodes)

- random baseline: -70.02 +/- 16.19
- with_comms (5 update): -81.08 +/- 21.95 (delta -11.07 vs random)

This is a SHORT smoke (5 updates only). Real comparison needs the full
5-seed 3-arm sweep at 80 updates matched to v2 (currently running in
background).

## 3. Full sweep status

15 jobs (3 arms x 5 seeds x 80 updates) launched. Each ~10-15 min.
Total: ~30-60 min wall time, parallelized to 15.

## 4. What we expect

- If with_comms > no_comms and with_comms > random_comms: comms help.
- If with_comms == no_comms == random_comms: message information is
  not useful (similar to v3 finding).
- If with_comms < no_comms: extra critic input HURTS (overfitting).

## 5. v3 10K-episode re-run (path a) status

15 jobs also running in parallel (3 arms x 5 seeds x 800 updates).
Each ~5-10x slower than the 80-update baseline. Will report once done.

## 6. Path c (Monitor as MA verifier)

Designed but not implemented in this session (deferred to next session):
- Per-agent Monitor output is concatenated into a cross-agent evidence
  chain (hash over (timestamp, agent_id, monitor_prob, prev_hash))
- Trust head on top of critic: takes (my obs, my monitor, others' Q,
  others' monitor) -> trust weight per other agent
- Q_i is computed using only trusted Q_j weighted by trust_j
- Claim: this is the SHIPPING use of Monitors in MA (verifier, not
  reward signal), building on H5 lesson
- This would be a new paper-worthy claim distinct from H5/v3/v4.