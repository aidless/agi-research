# Project D: Language-as-Type-System Sketch

> **Status**: design document (v0.1, 2026-07-25)
> **Priority**: P1 (paired with Project E per TASKBOOK v1.0)
> **Implementation**: deferred to Y1 (after Project C baseline)
> **Framing**: language acts as a *type system* over slot latents

---

## 1. The interface problem

Projects A, C, and E produce structured outputs:
- A: failure probability (scalar)
- C: slot latents, SCM interventions (structured tensors)
- E: LTL satisfaction (graded truth-values)

A human user (or another LLM-based agent) needs to *interpret*
these outputs. The current best interface is natural language, but
natural language is too permissive: a Monitor saying "this might
fail" could mean anything.

Project D introduces a **language-as-type-system**: a constrained
natural language where each predicate is a *typed edge* over the
slot latents from Project C. The LLM emits only type-consistent
utterances; downstream tools can statically check type safety.

---

## 2. Architecture

### 2.1 The type lattice

We define a hierarchy of types over slot latents:

```
Top (Any slot)
  +-- Object (a physical entity: position, velocity, shape)
  |    +-- Movable (can be acted upon: ball, agent, opponent)
  |    +-- Static (cannot be acted upon: wall, ground, marker)
  +-- Region (a spatial area: bounds, occupancy)
  +-- Event (a discrete occurrence: collision, pickup, death)
  +-- Predicate (a typed function: distance(A, B) -> Real, in_region(O, R) -> Bool)
```

Each slot from Project C's slot-attention world model is typed as
Object or Region. Predicates are functions over these types.

### 2.2 Language constraints

The LLM is constrained to emit only *type-consistent* statements.
Concretely:

```python
# Valid:
"The red ball is in the left region."
# parse: Ball(Object) in(Left(Region))
# type-check: Ball in Region = OK

# Invalid:
"The red ball is in the velocity."
# parse: Ball(Object) in(Velocity(Real?))
# type-check: Ball in Real = TYPE ERROR
```

The type checker rejects invalid statements at parse time. The LLM
is fine-tuned (DPO, per H05) to prefer type-consistent outputs.

### 2.3 Implementation sketch

- **Parser**: Lark or LARK grammar with ~50 production rules.
- **Type checker**: simple Hindley-Milner-style inference over the
  AST. ~200 lines.
- **LLM**: any 7B+ open model (Llama 3, Mistral). Fine-tune with
  DPO on ~1000 type-consistent vs inconsistent pairs.
- **Integration**: Project C's slot latents are converted to typed
  entities; Project A's Monitor output is converted to typed events
  (e.g., "high failure probability" -> Event(failure_imminent, conf=0.7)).

---

## 3. Concrete benefits

1. **Type safety**: invalid statements are rejected, not just
   unlikely. This is a stronger guarantee than prompt engineering.
2. **Composable**: type-consistent statements compose. "If the ball
   is in the left region AND velocity < 5, then..." is well-typed.
3. **Auditable**: every emitted statement can be parsed and
   type-checked, giving a clean audit trail for safety review.
4. **Cross-domain**: types are domain-agnostic. The same type
   lattice works for LunarLander, Procgen, Crafter, or any other
   slot-attention world model.

---

## 4. Connection to F:\TMLR H/I series

- **H01 Prompt Engineering**: DSPy's eval-driven prompt iteration
  is our default workflow. We use DSPy to generate the DPO training
  pairs.
- **H03 Agent frameworks**: ReAct-style observation/action/thought
  interleaving is naturally expressed in our type system. Each
  ReAct step emits a type-consistent statement.
- **H04 RAG**: Self-RAG's reflection tokens (IsREL, IsSUP, IsUSE)
  can be re-expressed as typed events in our lattice.
- **H05 Evaluation**: DPO is the alignment method we use for the
  LLM (per H05).
- **I02 SAE / Interpretability**: types give us a coarse
  interpretability layer over the LLM's outputs. SAE-style fine
  interpretability is Y2 work.

---

## 5. Connection to AGI safety

Project D is not directly an alignment mechanism, but it provides
a *type-theoretic* basis for safety arguments:
- Every statement the agent emits can be type-checked.
- Invalid statements are rejected by construction.
- Valid statements have a formal meaning (Hindley-Milner semantics).

Combined with Project E's LTL verifier, Project D gives us
*typed + verified* outputs: statements that are both type-consistent
and satisfy user-specified LTL rules.

---

## 6. Concrete scope (P1, not P0)

Project D is documentation-only for Y0. Y1 implementation if
Project C baseline is solid.

### 6.1 Documentation touchpoints (now)

- **Project A paper Section 6 (Limitations)**: note that the
  Monitor's scalar output could be enriched with typed events for
  human-interpretable diagnostics.
- **Project C paper Section 7 (Future Work)**: note that slot
  latents are the natural substrate for typed predicates.
- **TASKBOOK v1 Section 4.2 (Mapping to 5 routes)**: link Project D
  to Scaling (LLM type system) and Neuro-Symbolic (LTL verification
  via Project E).

### 6.2 Implementation (Y1, after Project C baseline)

- 200-line Lark grammar for the type lattice.
- 200-line type checker.
- 1000-pair DPO dataset (generated via DSPy + golden test set).
- LLaMA-3-8B fine-tune with DPO on the dataset.
- Benchmark: LunarLander-v3 scenario descriptions; measure type
  consistency rate and human-rated usefulness of the typed outputs.

---

## 7. Open questions

- Do we want Hindley-Milner (rigorous but complex) or a simpler
  bidirectional type checker? Y1 decision.
- Should the LLM be trained from scratch on types, or fine-tuned
  from a base model? Fine-tune is cheaper.
- Should we adopt existing typed languages (e.g., SHACL for RDF) or
  invent our own? Invent for domain fit.

---

## 8. References

- Hindley, R. (1969). The Principal Type-Scheme of an Object in
  Combinatory Logic. Trans. AMS.
- Milner, R. (1978). A Theory of Type Polymorphism in Programming
  Languages. JCSS.
- Wei, J., et al. (2022). Chain-of-Thought Prompting. NeurIPS.
- Rafailov, R., et al. (2023). Direct Preference Optimization. NeurIPS.
- Asai, A., et al. (2024). Self-RAG. ICLR 2024.
- Khattab, O., et al. (2023). DSPy. arXiv:2310.03714.

---

*Project D sketch v0.1, 2026-07-25. P1 status. Implementation
deferred to Y1.*