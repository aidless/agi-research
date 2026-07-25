# Open Policy Agent / Rego (OPA, 2016+)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: HIGH -- mature OSS project, widely used in production
> One-line: OPA is a policy engine that evaluates declarative Rego rules
> against structured data; de-facto standard for fine-grained policy-as-code
> in cloud-native systems.

## Problem the project solves

Before OPA, authorisation logic was scattered: IAM policies in AWS, RBAC
in Kubernetes, ad-hoc checks in code. Each was a separate DSL, with no
unification. OPA provides one place to write policy, evaluate it, and reason
about it.

## Method

OPA policy language is Rego (sometimes called Rego-as-policy):

- declarative (allow if X)
- Datalog-like reasoning (transitive closure, partial evaluation)
- evaluable against any structured input (JSON, YAML, ...)
- policy eval JSON output: result field true or false

Example Rego snippet:
package http.authz
allow if input.method == GET and input.path == health
allow if input.user == alice and input.method == POST

OPA bundles support: HTTP API server, CLI evaluator, Go library. The
decision latency is sub-millisecond for typical inputs.

## Empirical result

- ~10 million users as of 2024 (CNCF survey)
- Adoption: Kubernetes admission control, service mesh, supply chain,
  cloud resources, custom enterprise gates.
- Battle-tested on rulesets with thousands of rules.

## Criticisms (specific)

1. Rego has a learning curve for non-Datalog users.
2. Performance under complex rules: deep transitive closure can be slow.
3. No formal verification (unlike Cedar): bad rules can be silently wrong.
4. Granularity conflicts: multiple packages can read/write; coordination
   is fragile.

## Connection to our program

Trend #5 (safety) maps strongly to OPA + Rego. For our research assistant:

- We can write Rego rules that gate agent actions.
- Rules are testable, deterministic, auditable.
- Rules compose: a Cedar-like or Rego-like policy file IS the artifact.

OPA pairs naturally with Cedar: Rego is more general, Cedar is more
verified. Either could be the actual engine that our .policy/agent.yaml
compiles down to.

For Project E (verification): OPA policy eval is essentially "is this
candidate world-model prediction consistent with my rules?" - the same
question Project E answers with neural+symbolic combinations.

## Confidence
HIGH. Public CNCF project.

## Related
- AWS Cedar (related but proprietary)
- Open Policy Engine (Styra commercial, OPA open-source)
- Topaz (CNCF authorization)
- Casbin (Go policy engine with multiple DSLs)

## Status
- cited in Trend #5 (safety) rationale
- cited in Project F (workspace automation plan) as policy engine option
- future: test OPA decision latency on Rego policy that mirrors our .policy/agent.yaml
