# PDDL - Planning Domain Definition Language (McDermott et al. 1998)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **VERY HIGH** -- foundational AI planning language; standard textbook material.
> One-line: A declarative language for specifying classical planning problems;
> state, action, preconditions, effects; many solvers (FastDownward, Metric-FF, etc.).

## Problem
Classical planning: given initial state, goal state, and action set, find a
sequence of actions that achieves the goal. Pre-1998: ad-hoc representations;
each planner had its own format. PDDL was the standardisation.

## Method
PDDL separates **domain** from **problem**:
- Domain file: object types, predicates, action schemas (parameters,
  preconditions, effects).
- Problem file: specific objects, initial state, goal state.
- Planner reads both and emits a plan (sequence of grounded actions).

Example domain action:
```
(:action pickup
  :parameters (?x - block)
  :precondition (and (clear ?x) (on-table ?x) (arm-empty))
  :effect (and (holding ?x) (not (clear ?x)) (not (arm-empty))))
```

Example problem:
```
(define (problem blocks-1) (:domain blocksworld)
  (:objects a b c - block)
  (:init (clear a) (on a b) (on b c) (ontable c) (arm-empty))
  (:goal (and (on a b) (on b c)))
)
```

## Empirical result
- International Planning Competition (IPC) standardised on PDDL.
- FastDownward, Metric-FF, LPG, OPT, OPT+ solvers all support PDDL.
- Variants: PDDL 1.2 -> 2.1 (numeric) -> 2.2 (timed) -> 3.0 (continuous, stochastic).

## Criticisms (specific)
1. **Planning explosion**: action sequence grows exponentially with goals.
2. **Closed world assumption**: hard for partial observability.
3. **No continuous actions**: until PDDL 3.0 (limited).
4. **No perception coupling**: PDDL is symbolic only - needs an external
   mechanism to ground predicates from sensors.

## Connection to our program
Direct fundamental for Project E (Neuro-Symbolic Verification):
- Our verifier approach uses LTL rules over symbols.
- PDDL extends LTL to action-based planning.
- For a future revision: a verifier that uses PDDL-style action models
  to reason about candidate world-model rollouts would be a strict superset
  of what we currently do.

For Project A:
- Monitor could use a simple PDDL planner to find a "safe action" override
  given a known goal and environment constraints.

## Confidence
VERY HIGH.

## Related
- STRIPS (Fikes/Nilsson 1971) - predecessor to PDDL
- ADL (Pednault 1989) - extension of STRIPS
- Hierarchical PDDL / HDDL (HTN planners)
- LTL (Pnueli 1977) - we use this for verifier rules
- STRIPS to PDDL converters (e.g. Pyperplan)
- Action languages / Golog

## Status
- cited in Project E Related Work (the planning-language foundation)
- future: evaluate PDDL as the verifier language for Project E v2
