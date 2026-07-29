# Reviewer Simulator Output

**Paper:** "Monitor Signal vs DLR Predicates in Cooperative MARL: A
6-Pathway Systematic Investigation"
**Simulated review process:** Three independent reviewers (R1, R2, R3)
provide feedback as if for an AAMAS 2027 submission.

---

## Reviewer 1 (R1): Experienced MARL researcher

### Summary

The paper systematically investigates 6 architectures for using
failure-prediction signals in cooperative MARL, spanning 14,000+
episodes of training on PettingZoo Simple Spread v3. The central
finding is that 5/6 architectures are REFUTED at $p<0.05$, with
DLR cross-agent predicates in the critic being the single
publishable result. The trust head is shown to ignore its input
signal via bit-for-bit identical per-seed results at $n=5$ and
$n=30$ CLEAN.

### Strengths

1. **Comprehensive systematic investigation**: 6 architectures
   spanning critic-side and actor-side positions, with proper
   ablation protocols. The 4-pathway lessons-learned paper at
   commit 7bbc363 was a great start; this paper extends it
   cleanly.

2. **Honest post-hoc audit**: The authors openly report the
   discovery that the v5 "cross-agent evidence chain" was defined
   but never read by the trust head, and rename the file to
   `pz_maddpg_trusthead_same_agent.py`. This kind of honest
   disclosure is rare and welcome.

3. **Clean bit-for-bit identity test**: The 5/5 and 30/30
   bit-for-bit identical results across Monitor, random, and
   DLR inputs to the trust head is a striking finding. The
   consistency check (catching the n=30 r3 contamination) is
   rigorous.

4. **Statistically significant positive result**: v8 dlr_only
   gives $+0.1447$ ($p<0.005$, 20/30 positive at $n=30$),
   reproducible across sample sizes. While small in magnitude,
   this is a real and useful contribution.

### Weaknesses

1. **Small effect size**: The publishable effect is +0.1447 mean
   on a baseline of -69.7, i.e., a relative improvement of ~0.2%.
   This is small enough that some readers may question whether
   it's practically meaningful. The authors should add a more
   detailed discussion of the practical implications.

2. **Limited environment coverage**: The paper only tests on
   PettingZoo Simple Spread v3. Other MARL environments (predator-
   prey, traffic junction, harder cooperative tasks) could
   yield different results. The Limitations section does
   acknowledge this.

3. **v6 is a thin wrapper around v5**: The "architecture-only
   ablation" uses the same trust head architecture as v5, with
   only the input source changed. A truly independent v6
   implementation would strengthen the bit-for-bit identity
   claim. (Acknowledged in Limitations.)

4. **Effect-shrinkage trajectory for v5 could be analyzed
   more deeply**: The n=5 to n=212 trajectory is shown, but a
   Bayesian analysis (e.g., posterior on the true effect size)
   would be more informative than just the sample means.

5. **DLR predicates could be more diverse**: Only "closest
   agent to landmark j" is used. Other cross-agent relationships
   (pair-wise distances, coverage areas, etc.) might give
   larger effects.

### Questions for authors

1. How sensitive is v8 dlr_only to the choice of DLR predicates?
   Would different predicates (e.g., "agent i has visited
   landmark j" or "agent i is moving away from agent k") give
   similar or different effects?
2. Have you considered running the dlr_only arm at n=100 to
   confirm the effect at higher statistical power?
3. The trust head's input slot is "ignored" -- but the trust
   head output IS used in the actor loss. Is it possible that
   the trust head outputs are very similar across arms because
   the gradient is small (not because the input is ignored)?
4. Why does Monitor aux loss (v3) HURT the baseline? Is the
   Monitor's bias toward Stage-1 failure modes the only
   explanation, or are there other factors?

### Recommendation

**Weak Accept (with revisions)**. The 6-pathway systematic
investigation is valuable, the honest post-hoc audit is
commendable, and the bit-for-bit identity evidence is striking.
The publishable result is small but real. Major revisions:
- Add a more detailed discussion of practical implications
- Strengthen the DLR predicates diversity
- Consider Bayesian analysis of the effect-shrinkage trajectory
- Add v8 n=100 (or similar) for higher statistical power

---

## Reviewer 2 (R2): Methods-focused, statistically rigorous

### Summary

The paper presents a 6-pathway systematic investigation of
failure-prediction signal architectures in cooperative MARL.
Uses paired t-tests with multiple sample sizes (n=5, n=30, n=212)
to test the effect of each pathway. Reports that 5/6 are REFUTED
at $p<0.05$ and identifies DLR-in-critic as the single
publishable pathway.

### Strengths

1. **Paired tests with multiple sample sizes**: The paper reports
   not just n=5 results but also n=13, n=29, n=100, n=212 for
   v5, and n=5, n=30 for v8. This is rare and important for
   effect-size estimation.

2. **Effect-shrinkage vs effect-stability trajectories**: The
   contrast between v5 (effect shrinks with n) and v8 dlr_only
   (effect is stable across n) is well-presented in Figure 3
   and Figure 4. This is the key insight of the paper.

3. **Bit-for-bit identity test**: This is an unusually rigorous
   ablation test. The 30/30 bit-for-bit identical result at
   $n=30$ CLEAN is the strongest possible evidence that the
   trust head ignores its input.

### Weaknesses

1. **Statistical power is limited for small effects**: Even at
   n=30, the v8 dlr_only result has $d_z=0.59$ and
   $p<0.005$. The v5 result at n=212 has $d_z=0.065$ and
   $p=0.34$. The paper's interpretation that v5 effect is
   "small" is correct, but a more rigorous power analysis would
   be useful.

2. **The bit-for-bit identity test only proves what the trust
   head does in 800 episodes of training**: At longer training,
   the trust head might learn to use its input. The n=30 result
   shows this happens (0.06 mean_diff, NOT sig). The
   interpretation that the trust head "ignores" its input may be
   too strong.

3. **No correction for multiple comparisons**: The paper tests
   multiple paired comparisons (v5 vs no_verifier, v6 with vs
   no_verifier, v8 dlr_only vs no_verifier, etc.) without
   Bonferroni or FDR correction. With 6 pathways and 3 arms each,
   there are ~18 paired tests. Even at $p<0.005$, the family-wise
   error rate is non-trivial.

4. **Sample size justification is missing**: Why n=5, n=30, n=212?
   These are not standard sample sizes for MARL papers. The
   paper should justify the choice of sample sizes, ideally with
   a power analysis.

### Questions for authors

1. Have you considered multiple-comparison correction (Bonferroni,
   Holm, FDR)? If so, what is the result?
2. The trust head ignores its input at n=5 and n=30 CLEAN. Is
   this also true at n=100 or n=212? If not, the "ignores input"
   claim is over-stated.
3. The dlr_only effect is $+0.1447$ at n=30. What is the
   predicted n needed to reach $p<0.01$? What about $p<0.001$?
4. Have you considered a Bayesian analysis (e.g., credible
   interval on the true effect) for the dlr_only result?

### Recommendation

**Weak Accept (with revisions)**. The 6-pathway investigation is
well-designed and the bit-for-bit identity test is rigorous. The
publishable result is statistically significant but small. Major
revisions:
- Add multiple-comparison correction
- Justify sample sizes with power analysis
- Add Bayesian credible intervals
- Soften the "trust head ignores input" claim to "trust head
  ignores input at short training"

---

## Reviewer 3 (R3): Applied, focused on practical impact

### Summary

The paper systematically tests 6 architectures for using failure-
prediction signals in cooperative MARL, finding that 5/6 are
REFUTED and that DLR-in-critic gives a small but statistically
significant positive result. The investigation is methodologically
sound and the findings are valuable for the field.

### Strengths

1. **Practical actionable findings**: The paper identifies DLR
   in critic as the right architectural choice for cross-agent
   signal. This is a useful recipe for practitioners.

2. **Honest negative result**: The 5/6 REFUTED result saves the
   field from repeating these investigations. This is the most
   valuable contribution of the paper.

3. **Comprehensive coverage**: The 6 architectures cover the
   natural design space for failure-prediction signals in MA
   (critic-side vs actor-side, Monitor vs DLR vs random, with
   vs without trust head).

### Weaknesses

1. **Single environment limits generalizability**: PettingZoo
   Simple Spread v3 is a relatively simple environment. The
   findings may not hold in more complex MARL tasks (e.g.,
   StarCraft Multi-Agent Challenge, Hanabi, Diplomacy).

2. **Effect size is small in practical terms**: +0.1447 mean on
   a baseline of -69.7 is a 0.2% improvement. In real-world
   applications, this is often below the noise floor. The paper
   should be more honest about the practical implications.

3. **No comparison to other MARL methods**: The paper compares
   the 6 pathways to the MADDPG v2 baseline, but doesn't
   compare to other state-of-the-art MARL methods (QMIX, COMA,
   MAPPO, etc.). This makes it hard to know how the DLR
   improvement stacks up against other advances.

4. **No compute scaling analysis**: The 800-episode training
   regime is short. What happens with 10K or 100K episodes?
   Does v3 still hurt? Does dlr_only still help?

5. **DLR predicates are hand-crafted**: While this is a strength
   (they consistently provide useful information), it also
   means the result is limited to the specific predicates
   chosen. Generalization to other predicates is not tested.

### Questions for authors

1. Have you tested the DLR-in-critic approach on other MARL
   environments (e.g., Predator-Prey, Cooperative Navigation,
   StarCraft)?
2. How does the DLR improvement compare to other MARL
   improvements (e.g., QMIX over MADDPG, MAPPO over MADDPG)?
3. Does the DLR improvement hold at longer training (10K, 100K
   episodes)?
4. Is there a learned alternative to hand-crafted DLR
   predicates that gives a similar effect?

### Recommendation

**Weak Accept (with revisions)**. The 6-pathway investigation is
valuable and the findings are honest. The publishable result is
small but real. Major revisions:
- Test on more environments
- Compare to other MARL methods
- Test at longer training
- Discuss practical implications more honestly

---

## Meta-review summary

**All three reviewers**: Weak Accept (with revisions).

**Common themes**:
- The 6-pathway systematic investigation is valuable
- The honest post-hoc audit (v5 cross-agent chain not wired) is
  commendable
- The bit-for-bit identity test is rigorous
- The publishable result (v8 dlr_only) is small but real
- Multiple revisions suggested:
  - Multiple-comparison correction (R2)
  - Bayesian analysis (R1, R2)
  - Other environments (R3)
  - Longer training (R3)
  - Comparison to other MARL methods (R3)
  - More DLR predicate diversity (R1)
  - Soften "trust head ignores input" claim (R2)

**Decision**: Accept with major revisions.
