# Project E: Neuro-Symbolic Verification Sketch

> **Status**: design document (v0.1, 2026-07-25)
> **Priority**: P1 (per ADR-0007, promotion effective 2026-07-25)
> **Implementation**: deferred to Y1 Q1 (after Procgen baseline solid)
> **Framing**: three-layer alignment (Outer/Inner/Corrigibility) per Hubinger 2019 + I07

---

## 1. The gap Project E fills

Projects A-D form the 4-layer AGI substrate (Self-Model + WM + LLM + VLA).
What they do NOT provide is **formal guarantees** that the agent's behavior
satisfies user intent. Projects A's Monitor gives a probabilistic
failure signal; Projects C's world model gives a generative process;
Projects D's language interface gives semantic types. None of these is
a logical verifier.

Project E adds a *neuro-symbolic verifier* that takes an LTL formula
over policy trajectories and outputs a graded truth-value (frequency +
confidence, NARS-style) of whether the trajectory satisfies the formula.

This is the verification layer that closes the AGI safety loop.

---

## 2. Three-layer alignment framework (per Hubinger 2019)

We adopt the Outer/Inner/Corrigibility framework from Hubinger et al.
2019, which is the canonical AI safety taxonomy:

### 2.1 Outer Alignment (the "right spec" problem)

**Question**: Does the LTL formula we ask the verifier to check
actually match the user's real intent?

**Risk**: the verifier can be perfectly accurate and still approve
behaviors the user would reject, if the spec is wrong.

**Mitigation**:
- User testing: every LTL rule is derived from a user interview
  where the user is shown counterfactual policy trajectories and
  asked to label them as acceptable or not.
- Spec versioning: all LTL rules have a `version` and `author` field;
  changes go through ADR review.
- Spec audit: every 6 months, re-audit the rule set against any
  failures in the deployed system.

### 2.2 Inner Alignment (the "faithful executor" problem)

**Question**: Does the verifier's prediction match the actual LTL
satisfaction of the trajectory?

**Risk**: the verifier learns a shortcut (e.g., predicts
"satisfies" for any trajectory with reward > threshold, regardless
of LTL satisfaction).

**Mitigation**:
- Synthetic test cases: for each LTL rule, generate 100+ trajectories
  with known satisfaction labels (true positives, true negatives,
  edge cases). Report verifier accuracy per rule.
- Held-out eval: 20% of trajectories are held out from training; the
  verifier's accuracy on the held-out set is the inner-alignment
  metric.
- Adversarial probing: a "red team" trajectory generator searches for
  trajectories that fool the verifier; if any are found, they become
  training data and the verifier is retrained.

### 2.3 Corrigibility (the "can be overridden" problem)

**Question**: When a human operator decides to override the verifier
(or the agent's action), does the system comply without resistance?

**Risk**: a sufficiently capable agent may resist correction in order
to preserve its objective (instrumental self-preservation).

**Mitigation**:
- **Structural**: every verifier rule has an `override_authority`
  field; when a human invokes override, the verifier outputs
  "OVERRIDDEN" regardless of its prediction. The agent must accept
  the override as a fact.
- **Behavioral**: test the system on synthetic override scenarios
  where the verifier's prediction disagrees with the override; verify
  the agent does not circumvent the override.
- **Documented**: every override event is logged with timestamp,
  operator, and reason. A weekly review checks for patterns.

This three-layer framing is the structural organization of Project E.

---

## 3. Concrete verifier architecture

### 3.1 LTL rule language

We use a subset of Linear Temporal Logic (LTL) over policy trajectories.
Syntax (simplified):

```
formula   ::= atom | NOT formula | formula AND formula | formula OR formula
            | formula UNTIL formula | EVENTUALLY formula | ALWAYS formula
            | formula IMPLIES formula
atom      ::= "reward_in_range(lo, hi)" | "stayed_in_region(R)"
            | "visited_region(R)" | "action_count(A, op, N)"
            | "max_velocity(v)" | "no_collision(object)"
```

Each atom is a *temporal predicate* that can be computed from a
trajectory $(s_t, a_t, r_t)_{t=0}^T$.

### 3.2 Example rules for LunarLander-v3

```ltl
# Rule 1: agent must not crash into the ground at high velocity
ALWAYS (NOT (velocity_y > 5.0 AND landed AND NOT in_pad))

# Rule 2: agent must use fuel efficiently
EVENTUALLY (fuel_consumed < 50) IMPLIES EVENTUALLY (reward > 100)

# Rule 3: agent must stay upright (no flip)
ALWAYS (angle < pi/4)
```

These rules are LTL-encoded versions of intuitive safety/efficiency
requirements. The user can write them in plain English; we provide
a translator (Y1 work).

### 3.3 Verifier implementation

For each rule, the verifier has two components:

1. **Symbolic checker**: a Python function that, given a trajectory,
   returns True/False for the rule. This is the ground truth.

2. **Learned predictor**: a small NN (analogous to Project A's Monitor)
   that predicts the symbolic checker's output from the trajectory
   features. This is what the agent uses in practice.

Inner alignment is measured by the agreement between the learned
predictor and the symbolic checker on held-out trajectories.

For real-time use, the learned predictor is queried at every $K$
transitions (default $K=10$). The symbolic checker is queried only
at episode end (or on demand) for audit purposes.

### 3.4 Graded output (NARS-style)

Following Pei Wang's NARS (carried from v1.9 TMLR synthesis), the
verifier output is not Boolean. Instead, each prediction is a pair
$(f, c)$ where $f \in [0,1]$ is the *frequency* (observed fraction
of satisfaction) and $c \in [0,1]$ is the *confidence* (based on
evidence strength). This avoids the brittleness of strict Boolean
verification: the verifier can express "70% confident this trajectory
satisfies the rule, based on 50 transitions of evidence".

The agent's planner (Project C + Project D) consumes these graded
truth-values and makes decisions accordingly. If $f < 0.3$ AND $c > 0.7$
on any rule, the agent's safety override fires and the episode is
terminated (or a human is paged).

---

## 4. Concrete scope (P1, not P0)

We do not start Project E implementation today. We use it as a
**design target** for Projects A-D so they remain compatible with
verification.

### 4.1 Documentation touchpoints (now)

- **Project A paper Section 6 (Limitations)**: explicitly cite
  Hubinger 2019 and identify the three alignment gaps in our
  current Monitor. State that Project E will close them.
- **Project C paper Section 7 (Future Work)**: cite that the WM
  should support counterfactual reasoning (Pearl L2/L3) so the
  verifier can check "what would have happened" for inner
  alignment.
- **Project D paper Section 6 (Limitations)**: cite that the
  language-as-type-system should be expressible in LTL so the
  verifier can check it.

### 4.2 Implementation (Y1 Q1, after Procgen baseline)

- 200-line LTL rule language implementation (Python + Lark parser).
- 100-line symbolic checker library for common trajectory predicates.
- 300-line learned predictor (analogous to Project A's Monitor).
- Benchmark: LunarLander-v3 with the 3 example rules from Section 3.2.
  Report symbolic checker accuracy, learned predictor agreement,
  verifier response time.

### 4.3 When to escalate to P0

Per ADR-0007: if Y0 Q4 shows Project A Monitor Procgen AUROC > 0.85
AND Project C slot-WM transfer AUROC > 0.6, we escalate Project E
to P0 for Y1 H2.

---

## 5. Risk register

| risk | severity | mitigation |
|------|----------|------------|
| LTL rules too restrictive (block all behaviors) | high | rule versioning, A/B testing |
| Verifier too slow for real-time use | medium | symbolic checker runs only on-demand, learned predictor is fast |
| Symbolic checker has bugs | high | formal methods (Lean) for the checker, paper appendix |
| User intent drifts over time | medium | quarterly rule audit, rule provenance tracking |
| Adversarial trajectories fool learned predictor | medium | red team trajectory generator, periodic retraining |

---

## 6. Open questions for the user

- Do you want Project E implementation in Y1, or stay documentation-only?
- Should Project E use Lean (formal proofs) or a Python symbolic checker?
  Lean is more rigorous but harder to write; Python is faster but
  less trustworthy.
- For graded truth-values, do you want NARS $(f, c)$ or probability +
  entropy? They are similar but not identical.

---

## 7. References

- Hubinger, E., et al. (2019). Risks from Learned Optimization in
  Advanced Machine Learning Systems. arXiv:1906.01820.
- Soares, N., et al. (2015). Corrigibility. AAAI Workshop on AI and Ethics.
- Wei, A., et al. (2023). Jailbroken: How Does LLM Safety Training Fail?
  NeurIPS 2023.
- Wang, P. (2013). Non-Axiomatic Logic: A Model of Intelligent Reasoning.
  World Scientific.
- Pnueli, A. (1977). The Temporal Logic of Programs. FOCS.
- Baier, C. & Katoen, J. (2008). Principles of Model Checking. MIT Press.

---

*Project E sketch v0.1, 2026-07-25. P1 status per ADR-0007. Implementation
deferred to Y1 Q1.*