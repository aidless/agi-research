# AlphaGeometry (DeepMind 2024)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **MEDIUM** (post-training-cutoff)
> One-line: Symbolic geometry reasoning system combining a Neuro-Symbolic
> LLM loop with a deductive closure engine. Solved IMO 2024 geometry
> problems that are NOT amenable to pure search.

---

## Problem

Geometry problems have been historically hard for ML. Search alone
fails because the space of constructions is unbounded. Proof-based
approaches fail because geometric reasoning requires both visual
intuition (synthetic) and algebraic manipulation.

## Method

Architecture:
- LLM (Gemini-based) generates candidate geometric constructions
  given the problem text
- Numerical engine evaluates each candidate construction as if it
  were a step in a proof
- Deductive closure (similar to forward-chaining theorem proving)
  searches for a valid proof using the generated constructions
- Loop until: proof found OR timeout

Crucial mechanism: the LLM is conditioned on the original geometry
problem AND on the running proof state. The LLM essentially proposes
new "objects to construct" given the current state.

## Empirical result

IMO 2024: AlphaGeometry solved the geometry problem (P4?) which
contributed to DeepMind's silver medal.

Benchmark on 30 classic geometry problems: AlphaGeometry solves
25/30, comparable to or beating human gold medalist performance.

## Criticisms

1. **Specialised to geometry**. Cross-domain transfer of this approach
   is unclear.

2. **Compute is non-trivial**. Several hours of GPU per problem.

3. **The deductive closure component is brittle**. Hard-coding geometry
   rules into it.

4. **Doesn't generalise to other proof domains**. Requires engineering
   specific to geometry.

5. **Evaluation on synthetic benchmarks is not IMO-equivalent**. Real
   IMO problems have specific creative elements that synthetic ones miss.

## Connection to our program

AlphaGeometry + AlphaProof together = the two proofs from IMO 2024.
Both map to our Project E (Neuro-Symbolic verification).

For Project E paper intro:
"DeepMind's AlphaProof and AlphaGeometry demonstrate that LLM
generation + formal verification + RL search is a working pattern
for hard reasoning tasks. We argue this same pattern generalises
beyond mathematics to general reasoning."

## Related papers

- Lean theorem prover (Lean community / de Moura)
- Symbolic geometry reasoners (prior art)
- LLM + formal methods literature (increasing)

## Status

- [x] cite in Project E paper v0 intro
- [x] pair with AlphaProof for the IMO 2024 silver context
