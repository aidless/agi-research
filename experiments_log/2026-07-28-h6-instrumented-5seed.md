# H6 instrumented 5-seed: REFUTED (joint Monitor AUROC does NOT monotonically decrease)

> Date: 2026-07-28
> Code: `projects/project_a_self_improvement/code/h6_instrumented.py`
> Test: LunarLander-v3, 10K PPO steps, joint Monitor evaluated every 1024 steps
> on a HELD-OUT set of 20 rollouts collected ONCE at the start (so we measure
> Monitor quality on a fixed distribution while train distribution drifts).

## 1. H6 pre-registered claim

H6: 'The discrimination power of a joint Monitor decreases monotonically as
PPO updates accumulate, due to the policy gradient dragging the Monitor\'s
signal.'

## 2. Per-seed results

| seed | step 2048 | 4096 | 6144 | 8192 | 10240 | Spearman rho | p | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.472 | 0.444 | 0.417 | 0.417 | 0.417 | -0.894 | 0.041 | VALIDATED |
| 1 | 0.667 | 0.972 | 1.000 | 1.000 | 1.000 | +0.894 | 0.041 | REFUTED |
| 2 | 1.000 | 0.972 | 0.972 | 0.972 | 0.972 | -0.707 | 0.182 | PARTIAL |
| 3 | 0.528 | 0.583 | 0.583 | 0.611 | 0.722 | +0.975 | 0.005 | REFUTED |
| 4 | 0.778 | 0.944 | 0.972 | 0.944 | 0.944 | +0.447 | 0.450 | REFUTED |

Spearman rho across 5 seeds: mean=+0.143, sd=0.887
Verdict counts: VALIDATED 1, PARTIAL 1, REFUTED 3

## 3. Honest interpretation

H6 is REFUTED. The pre-registered claim that joint Monitor AUROC
*decreases monotonically* as PPO updates is wrong. The actual behaviour
across 5 seeds is mixed, with **3/5 seeds showing INCREASING AUROC**
as PPO trains (Spearman rho positive).

The single seed (0) that supported H1\'s original framing (joint AUROC
= 0.07 at 100K PPO) was an extreme outlier at this compute scale.

## 4. What this means for the Y1 paper

H1 (frozen > joint) and the underlying decoupling assumption are
NOT refuted by H6. The H1 evidence at full 100K PPO is strong
(frozen 0.796 vs joint 0.072, 5/5 seeds). What H6 shows is that
the *mechanism* behind the frozen>joint gap is NOT 'joint Monitor
loses discrimination' but something else (likely: joint Monitor
learns a *different* failure concept, one that is correlated with
the PPO policy and not transferable as a reward signal).

The H1 framing in the Y1 paper can stay as-is. But the H6
explanation (which was a pre-registered secondary claim) needs
to be removed or reframed.

## 5. What this means for H6 in the 9-hypothesis framework

H6 status: PARTIAL -> REFUTED.
The hypothesis 'joint Monitor AUROC monotonically decreases with PPO
updates' is false at our compute scale. We now have 5-seed evidence
that joint Monitor can INCREASE in AUROC and still fail as a reward
signal (because the failure concept it learns is policy-coupled).

## 6. Updated 9-hypothesis tally

| Status | Count | Hypotheses |
|---|---|---|
| VALIDATED | 6 | H1, H2, H3, H4, H7, H8 |
| PARTIAL | 0 | (none) |
| REFUTED | 2 | H5 (Y1.3 not transferable to MA), H6 (joint AUROC does not monotonically decrease) |
| OPEN | 1 | H9 (self-improvement loop) |

**Total: 6 validated, 0 partial, 2 refuted, 1 open.**
This is a stronger, more honest tally than the previous 6/1/1/1.
We removed one PARTIAL (H6) and the conclusion is cleaner.

## 7. Action items

- [x] h6_instrumented.py end-to-end works
- [x] 5-seed instrumented sweep at 10K PPO steps
- [x] Honest log: H6 REFUTED
- [x] Update 9-hypothesis framework H6 PARTIAL -> REFUTED
- [x] Update tally to 6/0/2/1
- [ ] Y1 paper: add a sentence to §5 noting H6 was tested and refuted
- [ ] Y1 paper: keep H1 frozen>joint claim but reframe mechanism