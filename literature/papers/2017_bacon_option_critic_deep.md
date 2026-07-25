# Option-Critic Architecture (Bacon 2017)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** - widely-cited hierarchical-RL paper
> One-line: End-to-end differentiable discovery of temporal abstractions:
> policy-over-options (intra-option policies) and termination function
are learned simultaneously via actor-critic on a SMDP.

## Problem

Hierarchical RL: how to learn sub-skills (options) without hand-design?
Existing approaches either:
1. Hard-code options (Sutton 1999)
2. Use bottleneck states (Stolle 2002, McGovern 2001)
3. Use skill-discovery heuristics

These do not scale; they require domain knowledge.

## Method

Three learned components (in addition to the primitive-action policy):
1. **Intra-option policies** pi(a | s, omega): given an option omega, choose
   the primitive action.
2. **Termination function** beta_omega(s): probability the option ends here.
3. **Policy-over-options** pi_Omega(omega | s): select which option to enter.

All three are differentiable (intra-option via policy gradient, termination
via continuous relaxation, policy-over-options via categorical).

Loss:
L_beta = E[ log beta_omega(s) for s where option continued ] +
        E[ log(1 - beta_omega(s)) for s where option terminated ] +
        advantage-weighed objective for intra-option policy

Trained end-to-end with actor-critic on the SMDP objective.

## Empirical result

- Tested on 4 Atari games + 4-room navigation.
- Discovered useful options in several games (e.g., "go-to-corridor" in
  Ms. Pacman).
- Options improve sample efficiency vs primitive-action PPO.

## Criticisms (specific)

1. Options discovered often do not match human-interpretable sub-skills.
   The continuous relaxation can converge to trivial short-horizon options.
2. Termination learning is finicky.
3. No principled mechanism for transfer across tasks: each env learns its
   own option set from scratch.

## Connection to our program

For Project A (self-improvement), Options extends the Monitor idea:
- Each option can have its own Monitor head.
- Decoupled training still applies: each option-policy critic can be frozen
   independently.

For Project B (cross-domain), transferable options would be the dream.
Our 4-layer architecture could lift transferable options to slot-WM types.

For Project D (language-as-types), the type vocabulary itself IS the option
space. Options are predicates; types classify them.

## Confidence
HIGH.

## Related
- Sutton 1999 - Options Framework (hand-designed)
- Kulkarni 2016 - Hierarchical Deep Reinforcement Learning
- Vezhnevets 2017 - Feudal Networks (differentiable sub-goals)
- Harb 2017 - When Waiting is not an Option
- Klissarov 2022 - MOC: Multi-Option Critic

## Status
- cited in Project A future work (per-option Monitor ablation)
- cited in Project D (type vocabulary is option space framing)
