# Reviewer Simulator Output (Y4 paper)

**Paper:** "Project G v0.5: Stratified Split for H10 LLM Self-Monitoring
Pilot: When Decoupling Doesn't Help LLM Self-Monitoring Either"
**Simulated review process:** Three independent reviewers (R1, R2, R3)
provide feedback as if for an ICML 2027 Workshop submission.

---

## Reviewer 1 (R1): Applied ML researcher

### Summary

The paper pre-registers H10 ("decoupling transfers to LLM self-
monitoring") and runs an n=5 pilot. The result is H10 REFUTED:
Joint Monitor AUROC 0.650 vs Frozen Monitor AUROC 0.550
(Joint > Frozen by 0.10, t=-0.516 NOT sig). The Project G v0.5
stratified train/eval split fixes a v0.4 degenerate eval issue.
The paper concludes that decoupling does not transfer to LLM
self-monitoring, consistent with the Y3 finding that
decoupling does not transfer to MA.

### Strengths

1. **Pre-registered hypothesis and decision rule**: The H10
   hypothesis is pre-registered with a clear decision rule
   (Frozen > Joint by >0.05 AND Welch t > 2.0 AND Frozen >
   Random by >0.10). This is the right standard for LLM
   self-monitoring research.
2. **Honest pre-registration of stratified split fix**: The
   v0.4 to v0.5 transition is honest about a silent failure
   (seed 2 had undefined AUROC). The stratified split is a
   principled fix.
3. **Consistency with Y3 finding**: The direction (Joint >
   Frozen) is consistent with the Y3 finding (Frozen > Random
   in some cases but Joint > Frozen in others). The cross-
   context synthesis is a valuable contribution.

### Weaknesses

1. **n=5 is small**: The result is direction-consistent (Joint
   > Frozen in 2/5 seeds) but not statistically significant.
   The paper correctly notes this caveat but does not provide a
   power analysis.
2. **Simple arithmetic tasks only**: The H10 result is on
   "simple arithmetic tasks" (3+4=7, 12+5=17, etc.). This is
   a very simple task distribution. The result may not
   generalize to harder LLM tasks (e.g., multi-step reasoning,
   code generation, math olympiad).
3. **The "frozen" Monitor implementation is not described in
   detail**: The paper says "frozen LM-based Monitor" but
   doesn't describe the architecture, training data, or
   inference procedure in detail. This makes it hard to
   reproduce.
4. **No comparison to other LLM self-monitoring methods**:
   The paper doesn't compare the Monitor to other LLM self-
   monitoring approaches (e.g., ensemble methods, self-
   consistency, verifier-based methods).

### Questions for authors

1. What is the Monitor's architecture, training data, and
   inference procedure? Please describe in detail for
   reproducibility.
2. Have you tested H10 on harder LLM tasks (multi-step
   reasoning, code, math)?
3. How does the Monitor compare to other LLM self-monitoring
   approaches (e.g., self-consistency, verifier-based)?
4. What is the power analysis for n=5? How many more seeds
   would be needed to detect the observed effect (Joint >
   Frozen by 0.10) at p<0.05?

### Recommendation

**Weak Accept (with revisions)**. The pre-registration and
stratified split fix are good practices. The result is
direction-consistent but not statistically significant at
n=5. Major revisions:
- Add detailed Monitor architecture description for
  reproducibility
- Test on harder LLM tasks
- Add power analysis
- Compare to other LLM self-monitoring approaches

---

## Reviewer 2 (R2): Methods-focused, statistically rigorous

### Summary

The paper tests H10 (decoupling transfers to LLM self-
monitoring) on simple arithmetic tasks with n=5 stratified
seeds. Result: H10 REFUTED (Joint > Frozen by 0.10, t=-0.516
NOT sig). The paper correctly applies the pre-registered
decision rule.

### Strengths

1. **Pre-registered decision rule**: The H10 decision rule
   (Frozen > Joint by >0.05 AND Welch t > 2.0 AND Frozen >
   Random by >0.10) is a good standard.
2. **Stratified split fix**: The v0.4 to v0.5 transition is
   honest and the stratified split is a principled fix.
3. **Negative control**: Random baseline is included as a
   negative control. Frozen > Random by 0.30 (not sig but
   directionally correct).

### Weaknesses

1. **No multiple-comparison correction**: With 3 paired tests
   (Frozen vs Joint, Frozen vs Random, Joint vs Random), the
   family-wise error rate is non-trivial. The paper doesn't
   apply Bonferroni or Holm-Bonferroni correction.
2. **No power analysis**: n=5 is small. A power analysis
   would clarify what sample size is needed to detect the
   observed effect.
3. **Welch t-test assumption**: The t-test assumes
   approximately normal distribution of differences. With
   n=5, this is hard to verify. A non-parametric test (e.g.,
   Wilcoxon signed-rank) would be more robust.
4. **No confidence intervals on the mean differences**: The
   paper reports point estimates without CIs. CIs would give
   a better sense of the uncertainty.

### Questions for authors

1. Have you applied multiple-comparison correction (Bonferroni
   or Holm-Bonferroni) to the 3 paired tests? If so, what is
   the result?
2. What is the power analysis? How many more seeds would be
   needed to detect the observed effect (Joint > Frozen by
   0.10) at p<0.05?
3. Have you considered a non-parametric test (Wilcoxon signed-
   rank) for the small-sample case?
4. What are the 95% confidence intervals on the mean
   differences?

### Recommendation

**Weak Accept (with revisions)**. The pre-registration and
stratified split are good practices. The result is
direction-consistent but not statistically significant at
n=5. Major revisions:
- Add multiple-comparison correction
- Add power analysis
- Consider non-parametric tests for small samples
- Add confidence intervals

---

## Reviewer 3 (R3): LLM safety / AI alignment researcher

### Summary

The paper tests H10 (decoupling transfers to LLM self-
monitoring) on simple arithmetic tasks. Result: H10 REFUTED
(Joint > Frozen by 0.10, t=-0.516 NOT sig). The paper
concludes that decoupling does not transfer to LLM self-
monitoring.

### Strengths

1. **Pre-registered hypothesis and decision rule**: The H10
   pre-registration is a good standard.
2. **Consistent with Y3 finding**: The Joint > Frozen
   direction is consistent with the Y3 finding (decoupling
   doesn't transfer to MA). The cross-context synthesis is
   valuable.
3. **Practical implications**: The paper correctly notes
   that the verified shipping use of the Monitor is
   **verification** (runtime guardrails, DLR), not training
   in LLM self-monitoring.

### Weaknesses

1. **Simple arithmetic tasks only**: The H10 result is on
   "simple arithmetic" tasks (3+4=7). This is a very
   restricted task distribution. LLM self-monitoring is most
   needed for harder tasks where the LLM is more likely to
   fail (multi-step reasoning, code generation, math olympiad,
   etc.).
2. **No comparison to LLM-specific self-monitoring methods**:
   The paper doesn't compare the Monitor to LLM-specific self-
   monitoring methods (e.g., self-consistency, self-
   verification, ensemble methods, calibration-based methods).
3. **The "Monitor" definition for LLMs is not well-motivated**:
   The paper uses a small LM as the Monitor backbone. For LLM
   self-monitoring, the Monitor is typically a smaller model
   that predicts the larger LLM's failures. The paper doesn't
   discuss this asymmetry.
4. **The practical implications are limited**: The "shipping
   use is verification" framing is too narrow. LLM self-
   monitoring has many other potential uses (early stopping,
   selective prediction, calibration).

### Questions for authors

1. Why simple arithmetic tasks? What is the failure rate of
   the underlying LLM on these tasks?
2. Have you tested H10 on harder LLM tasks where failure is
   more common and self-monitoring more useful?
3. How does the Monitor compare to LLM-specific self-
   monitoring methods like self-consistency?
4. What is the asymmetric relationship between the Monitor
   (small LM) and the target LLM (which the paper doesn't
   specify)?
5. What are the broader implications for LLM self-monitoring
   beyond the Monitor architecture?

### Recommendation

**Weak Accept (with revisions)**. The pre-registration is
good. The result is direction-consistent but not statistically
significant at n=5. The simple arithmetic tasks limit
generalizability. Major revisions:
- Test on harder LLM tasks where self-monitoring is more
  useful
- Compare to LLM-specific self-monitoring methods
- Discuss the asymmetric relationship between Monitor and
  target LLM
- Broaden the practical implications discussion

---

## Meta-review summary

**All three reviewers**: Weak Accept (with revisions).

**Common themes**:
- The pre-registration and stratified split are good practices
- The result is direction-consistent but not statistically
  significant at n=5
- The simple arithmetic tasks limit generalizability
- The Monitor implementation details are insufficient
- The comparison to other LLM self-monitoring methods is
  missing

**Decision**: Accept with major revisions.
