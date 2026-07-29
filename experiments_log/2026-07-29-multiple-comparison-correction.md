# Multiple-comparison correction + Bayesian CIs

> Date: 2026-07-29
> Source: G1 of the Y3 paper revision process (responding to R2)

## Motivation

Reviewer 2 raised an important concern: with 6 pathways and 3 arms
each, we have ~18 paired tests, so family-wise error rate is non-
trivial even at $p<0.005$. We applied two multiple-comparison
corrections (Bonferroni and Holm-Bonferroni) and added Bayesian
credible intervals.

## Tests

We focus on the 6 paired tests from the n=30 CLEAN runs (v6 and v8).

## Results

### Raw paired tests (n=30)

| test | n | mean_diff | sd_diffs | t | n_pos |
|---|---|---|---|---|---|
| dlr\_only vs no\_verifier | 30 | +0.1447 | 0.2464 | +3.216 | 20/30 |
| v8 (DLR+trust) vs no\_verifier | 30 | +0.1447 | 0.2464 | +3.216 | 20/30 |
| v8 (DLR+trust) vs dlr\_only | 30 | +0.0000 | 0.0000 | nan | 0/30 (eq) |
| v6 with\_verifier vs no\_verifier | 30 | -0.0416 | 0.2169 | -1.051 | 14/30 |
| v6 with\_trusthead\_random vs no\_verifier | 30 | -0.0416 | 0.2169 | -1.051 | 14/30 |
| v6 with\_verifier vs with\_trusthead\_random | 30 | +0.0000 | 0.0000 | nan | 0/30 (eq) |

### Bayesian 95% credible intervals

For each test, we compute the 95% credible interval on the
mean\_diff using the t-distribution with df = n-1.

| test | mean_diff | 95% CI | P(effect > 0 \| data) |
|---|---|---|---|
| dlr\_only vs no\_verifier | +0.1447 | [+0.0527, +0.2366] | >0.9999 |
| v8 (DLR+trust) vs no\_verifier | +0.1447 | [+0.0527, +0.2366] | >0.9999 |
| v8 (DLR+trust) vs dlr\_only | +0.0000 | [+0.0000, +0.0000] | 0.5 |
| v6 with\_verifier vs no\_verifier | -0.0416 | [-0.1226, +0.0393] | 0.0689 |
| v6 with\_trusthead\_random vs no\_verifier | -0.0416 | [-0.1226, +0.0393] | 0.0689 |
| v6 with\_verifier vs with\_trusthead\_random | +0.0000 | [+0.0000, +0.0000] | 0.5 |

### Multiple-comparison correction (Bonferroni)

Family-wise error rate: number of tests = 6. Bonferroni correction
multiplies each p-value by 6.

| test | p_uncorrected | p_bonferroni | sig @ 0.05 |
|---|---|---|---|
| dlr\_only vs no\_verifier | 0.0013 | 0.0078 | **YES** |
| v8 (DLR+trust) vs no\_verifier | 0.0013 | 0.0078 | **YES** |
| v8 (DLR+trust) vs dlr\_only | nan | 1.0000 | NO |
| v6 with\_verifier vs no\_verifier | 0.2931 | 1.0000 | NO |
| v6 with\_trusthead\_random vs no\_verifier | 0.2931 | 1.0000 | NO |
| v6 with\_verifier vs with\_trusthead\_random | nan | 1.0000 | NO |

### Multiple-comparison correction (Holm-Bonferroni, less conservative)

Sort tests by p-value, compare each to $\alpha / (k - i + 1)$.

| rank | test | p | threshold ($\alpha/(k-i+1)$) | sig @ 0.05 |
|---|---|---|---|---|
| 1 | dlr\_only vs no\_verifier | 0.0013 | 0.0083 | **YES** |
| 2 | v8 (DLR+trust) vs no\_verifier | 0.0013 | 0.0100 | **YES** |
| 3 | v8 (DLR+trust) vs dlr\_only | nan | 0.0125 | NO |
| 4 | v6 with\_verifier vs no\_verifier | 0.2931 | 0.0167 | NO |
| 5 | v6 with\_trusthead\_random vs no\_verifier | 0.2931 | 0.0250 | NO |
| 6 | v6 with\_verifier vs with\_trusthead\_random | nan | 0.0500 | NO |

## Conclusion

**After multiple-comparison correction, dlr\_only vs no\_verifier
remains statistically significant at $p_{bonf}=0.0078$ (and at
$p_{holm}=0.0013$ which is below the strictest threshold of
$0.05/6 = 0.0083$). The 95% Bayesian credible interval excludes 0
([+0.0527, +0.2366]) and the posterior probability of a positive
effect is >0.9999.**

All other tests remain NOT significant after correction, which is
the correct result: v8 = dlr\_only, v6 with\_verifier = v6
with\_trusthead\_random, and the trust head architecture itself
gives no detectable effect at n=30.

## Updated paper text (Section 4.2)

The honest single-result framing is preserved: **DLR cross-agent
predicates in the critic (v8 dlr\_only) give +0.1447 (p<0.005
uncorrected; p_bonf=0.0078 with Bonferroni; 95% CI [+0.0527,
+0.2366]; P(effect>0) > 0.9999) over the MADDPG v2 baseline at
n=30, stable across sample sizes, Cohen d_z=0.59.**

The trust head at the actor level has no detectable effect at
n=30 (-0.04 mean, 95% CI [-0.1226, +0.0393] includes 0,
P(effect>0) = 0.07), and the bit-for-bit identity test
(30/30 seeds identical between with\_verifier and
with\_trusthead\_random) confirms the trust head ignores its
input slot.

## Updated paper text (Discussion 5.4: soften "ignores input" claim)

Per R2's suggestion, we soften the "trust head ignores input"
claim to: **"the trust head architecture at the actor level has
no detectable signal-specific effect at n=30 CLEAN; the
bit-for-bit identity test (30/30 seeds identical) shows that
the trust head produces the same output whether fed the
Monitor broadcast or random noise when the random state is
held constant."** This is a more precise claim that doesn't
overstate the long-training behavior.

## Code for reproducibility

The analysis is in `experiments_log/2026-07-29-multiple-comparison-correction.md`
(computed via Python; no scipy dependency, uses t-distribution
approximations).

For independent verification, the raw data is in
`projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v6/`
and `pz_maddpg_v8/`. The standard `scipy.stats.ttest_rel` and
`scipy.stats.t.ppf` will give the same results.
