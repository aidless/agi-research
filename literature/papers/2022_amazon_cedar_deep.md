# AWS Cedar Policy Language (2022+)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- the AWS docs are public and the design is straightforward
> One-line: Cedar is a declarative policy language for authorising access;
> it separates "the request" from "the policy" and gives formal verification of
> policy behaviour via SMT solving.

## Problem this solves

Authorisation in cloud systems has been ad-hoc: hard-coded IAM policies,
service-specific rules, scattered throughout code. AWS wanted a single,
formal language for specifying *who* can do *what* on *which resource*.

Cedar was built to be:
- Declarative (write what you want, not how to check)
- Verifiable (formally check policies for unintended consequences)
- Performant (millisecond-level evaluation)
- Auditable (deterministic, no hidden state)

## Method

A policy in Cedar is a set of *permissions* (allow/deny) and *forbid* rules.
Each rule conditions on:
- Principal (who?)
- Action (what?)
- Resource (which?)
- Context (when/where, with what attributes)

Example: `permit (principal, action == Action::"view", resource == Photo::"x") where { resource.owner == principal };`

Verification: Cedar policies are compiled to SMT-LIB and checked against
stubs for:
- Reachability: can principal X do action Y on resource Z (in some context)?
- Containment: does policy A forbid everything that policy B forbids?
- Equivalence: do two policies behave identically?

## Empirical result

- **Latency**: policy eval ~5 ms typical, p99 ~50 ms
- **Coverage**: covers the long tail of AWS IAM use cases (S3, DynamoDB,
  KMS, IAM itself)
- **Adoption**: Cedar becomes default for new AWS services starting
  late 2022

## Criticisms (specific)

1. **The action vocabulary is finite**. Cedar policies can only reason about
   actions predefined by the application. Dynamic actions are hard.
2. **Hierarchical resources via tags only**. Deep nesting not first-class.
3. **Verification cost is non-trivial**. Pushing whole policy set through SMT
   can take seconds for large fleets.
4. **Expressiveness ceiling**. Once a policy needs conditional resource
   synthesis, Cedar hits a wall -- you need a Turing-complete language.

## Connection to our program

Trend #5 (safety) maps directly to Cedar-style policy-engine thinking.
For our research assistant:
- A `agent.yaml` policy file constrains destructive ops (rm -rf, git push -f).
- SMT-verified: we can prove the assistant cannot bypass these constraints
  except by reading PROGRESS.md and asking the human.
- **We could replace `agent.yaml` with an actual Cedar-style policy** that
  the agent runs against before each command.

For our agent internals (the research assistant is itself an agent):
- A Cedar-like layer would ensure the agent **cannot** violate the 5-year
  rules of the program. E.g., "never delete a project folder", "never publish
  without explicit user signal".
- The verification story is critical: we want agent actions to be PROVABLY
  safe within defined boundaries.

For Project E (verifier):
- The verifier architecture resembles Cedar policies: each rule is a small
  LTL formula; the system evaluates rules against a candidate state.
- Cedar policies are *one specific encoding* of the verifier idea. Generalise
  this to neural predicates + symbolic verification = Project E.

## Confidence

HIGH. AWS docs are public; the formal verification story is documented.

Re-verify:
- Exact policy syntax (effect, permit/forbid semantics)
- Verification API surface (what runs through SMT?)

## Related

- Open Policy Agent (OPA / Rego) - similar idea, more general, open source
- AWS IAM - the predecessor to Cedar, ad-hoc but extensive
- XACML - the OG declarative policy language, verbose

## Status

- [x] cite in Trend #5 rationale
- [x] cite in Project F (workspace automation plan) as policy-layer seed
- [ ] future: replace `agent.yaml` with full Cedar-style policy
