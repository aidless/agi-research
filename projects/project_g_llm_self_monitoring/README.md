# Project G -- LLM Self-Monitoring (decoupled Monitor as LLM self-rewarding)

> New direction kickoff: 2026-07-28
> Author: Liu Zewen (with Codex agent)
> Status: SPEC + pre-registration, no experiments run yet.

---

## 1. Why this is a NEW direction (not a continuation)

The 6 existing projects (A-F) all share the same substrate assumption:
the Monitor is trained on rollouts from a *frozen policy* in a
classical RL loop. Project G breaks that assumption by moving the
Monitor signal into the **LLM self-rewarding** domain:

- **Frozen policy** becomes **frozen LLM weights**.
- **Rollouts** become **LLM-generated trajectories** (chain-of-thought,
  tool use, multi-turn dialogue).
- **Failure prediction** becomes **error detection on reasoning
  traces**.
- **Reward shaping** becomes **auxiliary reward in RLHF / DPO /
  self-rewarding loops**.

The H1 decoupling result (frozen-Monitor > joint-Monitor on classical
RL) has a natural LLM analogue: **a self-rewarding LLM with a frozen
critic should outperform one with a joint critic**, because the joint
critic's gradient gets dragged by the LLM update.

## 2. Central hypothesis (H10)

**H10**: In a self-rewarding LLM agent (RLHF/DPO with an LLM-as-judge
or learned reward model), training a **decoupled** failure-prediction
Monitor on rollouts from a **frozen** LLM produces a more accurate
failure signal than training the Monitor **jointly** with the LLM,
where both accuracy is measured by AUROC on a held-out failure dataset.

**Decision rule**:
- Frozen Monitor AUROC > Joint Monitor AUROC by delta > 0.05
- AND Welch t > 2.0 on n=5 seeds
- AND negative control (random Monitor signal) AUROC NOT equal to
  frozen Monitor AUROC

If all three hold: H10 is VALIDATED.
If frozen == joint (no decoupling effect in LLM domain): H10 is
REFUTED.
If frozen > joint but t < 2.0: H10 is INCONCLUSIVE (extend to n=15).

**Pre-registration**: see `../../experiments_log/2026-07-28-PRE-REGISTERED-H10.md`.

## 3. Why this is interesting (the framing)

The Y1.x + H2.0 closure (8 pre-registered H tests, 0 supported at
the strict t>2.0 rule) showed that decoupled Monitors are **useful
for offline analysis** (DLR 97.8% cross-env accuracy, GovBench H1+H2
tamper detection 1.000) but **not generically useful as online RL
interventions** (Y1.3 +50 only on LunarLander; H5 REFUTED on
multi-agent; H6 mechanism REFUTED).

H10 asks the natural follow-up: **does the decoupling result transfer
to a different domain (LLM self-rewarding) where the failure concept
is qualitatively different** (reasoning failures vs trajectory
failures)? This is a fresh test of the decoupling principle, not an
extension of the classical-RL experiments.

## 4. Why now (after Y1.x closure)

The Y1.x closure means the classical-RL Monitor work is **done** in
the sense that 8 pre-reg tests have been run. The next logical move
is to test the decoupling principle in a new domain. LLM self-
rewarding is a high-leverage target because:

- **Hot research area**: self-rewarding LLM agents (RLHF, DPO,
  constitutional AI, RLAIF, self-refine, Reflexion) is an active
  research direction in 2026.
- **Direct applicability**: if H10 holds, the Archimedes
  decoupling principle can be added to the LLM training toolkit.
- **Direct governance fit**: a decoupled Monitor on LLM rollouts
  can be a verifier in the evidence-chain substrate (Project E /
  GovBench).

## 5. What is NOT being claimed

- **NOT claiming** H10 will hold. The classical-RL result was
  LunarLander-specific in part; the LLM domain may not show the
  decoupling effect for entirely different reasons.
- **NOT claiming** this is the only "new direction" worth exploring.
  Y2 multi-agent (Project F) is also a valid next direction; the
  user picked H10 by saying "新方向".
- **NOT claiming** a positive result on H10 would make Y1.x + H2.0
  closures any less significant. The 8-test null synthesis is a
  real contribution regardless.

## 6. Substrate components (reused from Projects A + E)

H10 reuses three validated primitives:

1. **Slot-Monitor architecture** (Project A, AUROC 0.989 vs raw
   0.796 on LunarLander). Adapt to LLM trace input: slot attention
   on the last 20 (token, logit) pairs.
2. **DLR (Differentiable Logic Reasoner)** (Project E, 97.8%
   cross-env accuracy). Use DLR to verify the Monitor's failure
   prediction against hand-coded "this should be true" predicates.
3. **GovBench evidence chain** (Project E, H1+H2+H3 VALIDATED).
   Use the evidence chain to log the Monitor's predictions in a
   tamper-detectable way.

## 7. Substrate components (NEW for Project G)

Three new components:

1. **LLM trace tokenizer**: token-level representation of LLM
   reasoning traces, slot-attention compatible.
2. **Frozen-LLM rollout collector**: deterministic rollouts from a
   frozen LLM (e.g., Qwen-1.5B base, no fine-tuning).
3. **Failure-label generator**: weak supervision for "this trace is
   a failure". Candidates:
   - Trace ends in incorrect answer (GSM8K-style math).
   - Trace contains a logical contradiction (DLR verifier output).
   - Trace exceeds a length budget.

## 8. Starter code skeleton

See `code/` directory:
- `llm_monitor.py` -- Slot-Monitor adapted to LLM traces.
- `frozen_rollout_collector.py` -- deterministic LLM rollout
  collector.
- `failure_label_generator.py` -- weak supervision for failure
  labels.
- `h10_smoke.py` -- end-to-end smoke test (1 seed, 100 rollouts).

## 9. Expected timeline

- **2026-07-28**: kickoff (this document + pre-registration).
- **2026-07-29 to 2026-08-15**: implement components 1-3, smoke
  test passes.
- **2026-08-15 to 2026-09-15**: H10 pre-registered run (n=5 seeds).
- **2026-09-15**: H10 verdict (VALIDATED / REFUTED / INCONCLUSIVE).
- **2026-09 to 2026-12**: depending on H10 outcome, either extend
  to n=15 or pivot to a follow-up hypothesis (H11).

## 10. Risk and honest framing

Risks:
- **No GPU**: LLM training is GPU-intensive. The user has CPU only.
  H10 may need to use small models (Qwen-1.5B-class) or API access
  (e.g., a paid Claude / GPT endpoint).
- **Compute budget**: even with small models, 5 seeds × 100 rollouts
  per seed may take days on CPU. H10 may be too slow without GPU.
- **Frozen-LLM scope**: which LLM to freeze? The choice matters
  (Qwen-1.5B vs Llama-3.2-1B vs Phi-3-mini) and may affect H10.
- **Negative control**: a *random* failure signal is the natural
  negative control, but the random signal may not be a fair
  comparison (different failure concept).

Honest framing:
- This is a **direction**, not a **result**.
- The pre-registered H10 hypothesis may be REFUTED. That is a valid
  outcome and a publishable finding.
- H10 is independent of the Y1.x + H2.0 closure; a null result on
  H10 does not change the 8-test null synthesis.

## 11. Related projects

- **Project A**: original H1 ablation on classical RL. H10 is the
  LLM analogue.
- **Project E**: DLR + GovBench governance primitives. H10 may
  use these for verification.
- **Project D**: language-as-type-system. H10 may benefit from a
  small LM as the type checker.
- **Project F (Phase 2)**: multi-agent. If H10 holds, the next
  direction is H12 (decoupled Monitor on multi-agent LLM rollouts).

---

*Project G kickoff: 2026-07-28 by Codex agent under
NO_SELF_DECEPTION.md protocol. Pre-registration in
`experiments_log/2026-07-28-PRE-REGISTERED-H10.md`.*