# CHANGELOG -- Y1 Paper

> Release history for the Y1 paper.
> Format: keep-a-changelog inspired, adapted for single-paper releases.

## v3.8 -- 2026-07-29 (this release)
- Added Y2 Project G section under Future Work. Pre-registered H10
  pilot (n=5 stratified seeds, N=12/seed) direction-REFUTED at this
  sample size (Joint 0.650 > Frozen 0.550; Welch t = -0.516 n.s.).
  Negative control PASSES (Frozen > Random by 0.30). Full pre-reg
  H10 (n=5 x 200 rollouts/seed) not run due to CPU budget; GPU-bound.
- Source: experiments_log/2026-07-29-H10-stratified-n5-result.md
- Reference: experiments_log/2026-07-28-PRE-REGISTERED-H10.md

## v3.7 -- 2026-07-28

### Status
- **Pre-arXiv**: ready for PI final review, not yet submitted.

### Headline result
- Y1.3 (decoupled Monitor as PPO training-time reward shaper):
  +50 mean over PPO baseline on LunarLander-v3, n=15 seeds,
  p<0.001 (t=6.76, df=14). 13/15 seeds positive.
- 9-hypothesis framework: 6 VALIDATED, 2 REFUTED, 1 OPEN.
- DLR attention: 97.8% mean over 4 envs (LunarLander, CartPole,
  Acrobot, Pendulum), 19 predicates total.

### Honest null synthesis (Section 4.10.25-4.10.27)
- 8 pre-registered H tests, 0 supported at the strict t>2.0 rule.
- 6/6 inference-time interventions (DEC-0011 v0.1-v0.4C) FAILED.
- 6 inference-time + 2 model-based planning (DLR gating, MBP) FAILED.
- H1.4 (Monitor as exploration bonus): REFUTED, 1/5 positive,
  delta=-25.6.
- H3 (500K PPO budget with Monitor): NOT supported, ACTIVE HARM,
  delta=-53.1 vs random signal.
- H5 (Monitor-as-reward-shaper on multi-agent): REFUTED, 1/5
  positive, t=-2.53 on continuous actions.
- H6 (joint Monitor AUROC monotonically decreases): REFUTED on
  instrumented 5-seed sweep, mean Spearman rho=+0.14 (3/5 seeds
  REFUTED).

### New in v3.7 vs v3.6
- Section 4.10.26 (6-test comprehensive review of online PPO
  interventions) added.
- Section 4.10.27 (H2.0 n=10 extension) added.
- H6 instrumented 5-seed experiment (REFUTED) added to discussion.
- Phase 2 closure (DMC continuous, MADDPG v2 +7.7 vs random,
  p<0.001) added.
- 9-hypothesis framework tally updated to 6/0/2/1 (was 6/1/1/1).
- No overclaim language; every positive has a stated null counter-
  part in the same paragraph.

## v3.6 -- 2026-07-28 (Y1 paper polish)

- 搂4.5/4.6 + 搂5.4 + 搂6.5 added.
- Phase 2 closure section added (DMC continuous REFUTED).
- Thesis addendum M (Phase 2 closure) cross-referenced.

## v3.5 -- 2026-07-28 (Y1 paper 搂4.5/4.6 + thesis addendum M)

- Internal release only; superseded by v3.6.

## v3.0 -- 2026-07-28 (Y1 paper polish)

- 9-hypothesis framework introduced.
- 搂4.5/4.6 sections added.
- Phase 2 outline integrated.

## v2.x -- 2026-07-27 (earlier drafts)

- Multiple intermediate versions.
- The v1.0 commit (ef90c2c, 2026-07-27) declared "+50, t=6.76"
  which was later retracted (it was a comparison vs PPO baseline,
  not vs random signal). The retraction sequence (v1.0 -> v1.1 ->
  v1.2 -> v1.3 -> v1.4) is documented in commits ef90c2c, e515565,
  8faf30b, 78b6044, 40c570f and in the paper's Section 4.10.25.3.

## v1.0 -- 2026-07-27 (initial Y1 paper draft)

- 搂1-3 draft + 4 figures + 2 LaTeX tables.
- First honest framing throughout.

## v0.x -- 2026-07-25 (Project A paper v1_full)

- Earlier 5-seed H1 ablation paper.
- Section 4.6-4.11 with 5-seed results.
- Adversarial perturbation test (Monitor robust to input noise).

---

*CHANGELOG maintained by Codex agent under NO_SELF_DECEPTION.md.*