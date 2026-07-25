# Hierarchical PDDL / HDDL (H?llerer et al. 2020, others)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **MEDIUM** (recent, not all variants primary-read by me)
> One-line: Hierarchical Task-Network planning extension to PDDL: tasks are
> decomposable; primitive actions are leaves; compound tasks form a network.
> Planner finds the right task decomposition (vs action sequence).

## Problem
Classical PDDL scales poorly as plan depth grows. Long-horizon problems
are exponential. HTN planners decompose abstract tasks into sub-tasks
using methods (recipes), then find action sequences for primitives.

## Method
Add two new constructs to PDDL:
- **Task**: a high-level action that must be accomplished via methods.
- **Method**: a recipe for accomplishing a task, parameterised by variables.
  Each method contains a task network (partial order of subtasks).

E.g., a "serve-meal" task has methods that decompose to subtasks
"prepare-food", "set-table", "bring-food". Each of which has its own
decomposition down to primitive actions.

Examples:
- SHOP2 (Nau et al. 2009) - state-space HTN planner
- PyHIPP / Unified Planning framework

## Empirical result
- SHOP2: order-of-magnitude faster than partial-order classical planners on
  benchmarks with structure (blocksworld, transportation, etc.).
- HDDL was introduced at IPC 2020 as a standard HTN language.
- Industrial deployment: automated manufacturing, space mission planning.

## Criticisms (specific)
1. **Method authoring is hard**: encoding the decomposition recipes
   requires domain expertise.
2. **HTN mis-specification**: if methods are wrong, planning fails silently.
3. **Mixed-initiative difficult**: hard for a human to override a method.
4. **HTN vs RL**: HTN planning is symbolic, RL is learned - they cover different
   problems.

## Connection to our program
For Project E (verification): the verifier could be an HTN planner over
WM rollouts:
- Verify that a rollout respects the abstract task structure.
- Decompose candidate plans into subtasks to check them hierarchically.

For Project A (Monitor + Options):
- Monitor can leverage HTN-style abstractions to predict failure.
- Project As per-option Monitor (cf Options Framework, Bacon 2017)
  resembles HTN decomposition.

## Confidence
MEDIUM.

## Related
- SHOP2 (Nau 2009)
- HDDL 2.0 (H?llerer 2020)
- Optic planner (McCluskey 2017) - HTN + PDDL hybrid
- HPlan (McDermott)
- Constraint-based methods (Linho 2019)

## Status
- cited in Project E Related Work (planning-language alternative for v2)
- future: prototype HTN-style task decomposition alongside LTL verifier
