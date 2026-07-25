# Options Framework (Sutton 1999; Precup 2000)

> Date: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** (well-established textbook material)
> One-line: Formal framework for hierarchical RL where an "option" is a closed-loop
> policy with its own termination condition; options compose across timescales,
> enabling multi-step credit assignment.

---

## Problem the paper solves

Flat RL agents learn primitive actions and value functions at fixed time scales.
This creates two problems:
- **Deep credit assignment**: large negative rewards at episode end require backprops
  through hundreds of small primitives; high variance.
- **Skill reuse**: a useful behaviour learned for one task cannot transfer to another
  that would benefit from it.

Options: temporal abstractions that simplify both.

## Method

An option `o = (I_o, pi_o, beta_o)`:
- `I_o`: initiation set, states where option can start
- `pi_o`: internal policy, picks primitive actions
- `beta_o(s)`: termination probability at state s

The agent's policy over options, mu(o | s, o_prev). The semi-Markov Decision Process
(SMDP) has reward accumulated within an option until termination.

Q-learning is extended to options: Q(s, o) gives value of starting option `o` at `s`.
Bellman equations for SMDPs are well-defined for options.

## Empirical result

Preprint era results (1999/2000) showed:
- Substantial speedups on toy tasks (rooms + subgoals)
- Found options that match human-designed sub-skills
- Convergence proofs for SMDP Q-learning

## Criticisms

1. **Initiation set design**. Options must be hand-designed or learned. Learning is hard.
2. **Termination learning**. Termination condition affects how often options end mid-task;
   bad termination => poor credit assignment.
3. **Sample efficiency**. With options learned end-to-to-end, results are less impressive;
   the original "speedups" rely on having a good option set.
4. **Single-task vs multi-task**. Options shine when sub-skills re-occur across tasks. If
   not, options add overhead.

## Connection to our program

Options are **the natural extension of Project A to hierarchical settings**.

Project A's decoupled Monitor looks at an episode history and predicts failure.
With options:
- Monitor can be *per-option*: predicting failure of specific sub-skill
- Monitor can be *cross-option*: predicting failure of the option composition
- Decoupled training works for each

For Project B (cross-domain transfer), options are also relevant: skills learned in one
game are often transferable (move-toward-object, avoid-enemy). Options framework lets us
package them.

For Project C (causal world model), options add layers of abstraction: each option
becomes a higher-level action; the world model can predict transitions between option
states, similar to options as latent variables in MuZero (which uses options internally
without naming them).

## Confidence

HIGH. Standard textbook material.

Re-verify:
- exact theorem statements for SMDP Q-learning convergence
- the specific options used in the original rooms example

## Related

- Precup 2000 (Temporal Abstraction in Reinforcement Learning, the textbook reference)
- Bacon 2017 (option-critic, end-to-end option learning)
- Hierarchical RL with feudal networks (Vezhnevets 2017)
- Decision Transformer / Trajectory Transformer — modern alternatives to options

## Status

- [x] cite in Project A future "hierarchical Monitor" extension
- [x] cite in Project B cross-domain (options as transfer units)
- [ ] not primary deliverable of any single paper in our program; we wait for
  where it becomes load-bearing
