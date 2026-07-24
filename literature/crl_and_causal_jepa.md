# Deep Dive: Causal World Models (2026 wave) - 2026-07-25

> **Purpose**: this is the v0 reading + design document for Project C
> (Causal World Model). After reading, you should be able to write
> Project C's paper v0 outline without re-reading the originals.
>
> **Honesty note**: Causal-JEPA (Feb 2026), V-JEPA 2-AC (Jun 2025),
> JEPA-WM (Dec 2025), Value-Guided JEPA (Jan 2026) are recent. I
> synthesise from abstracts + author talk reports + my domain knowledge;
> any detail I am uncertain about is marked **[UNCERTAIN]**.

---

## 1. What is actually new in 2025-2026 (and why it changes our plan)

Before 2025:
- World models = "predict next latent". Causal structure is absent.
- CRL (causal representation learning) exists but requires labeled
  interventions.

After Feb 2026:
- **Causal-JEPA shows object-level causal injection into latent.**
  This is the first time (per my knowledge cutoff) that someone has
  demonstrated a world model whose latent structure can be argued
  to respect Pearl's do-calculus on a meaningful subset of objects.
- **V-JEPA 2-AC shows 62 hours of robot data is enough for a
  zero-shot Franka deployment after 1M-hour video pretrain.**
  This collapses the data-efficiency story for cross-domain transfer.
- **Genie 3 + Cosmos 2.5 are infrastructure**: real-time, open-source
  world simulators that we can plug our Project A monitor INTO.

These change Project C in three ways:
1. We no longer need to build a simulator from scratch. We use
   Cosmos 2.5 / Genie 3 as the forward simulator and only build the
   causal layer on top.
2. We no longer need to argue "this might transfer". V-JEPA 2-AC
   proves the architecture CAN transfer. Our marginal contribution
   is the causal part.
3. The bottleneck is no longer "does this work?", it is "does the
   causal part give us Pearl L3 capability?"

## 2. Causal-JEPA - detailed mechanism (with uncertainty)

Paper: arXiv 2602.11389 (Feb 2026).

**What I am fairly confident it does** (from abstract-level reasoning):

The paper appears to extend JEPA-style non-generative prediction to
include a **structural latent constraint** that enforces a graph-
shaped causal prior over object slots. The intervention training
procedure injects do-calculus operations (clamp one slot, perturb
another) as data augmentation, so the model sees counterfactual-like
samples without needing a real intervention engine.

Critical design choices I expect (from the JEPA tradition):
- **Slot attention** as the latent object detector. Each slot is one
  object. The "object" is what makes the causal structure localisable.
- **Energy-based JEPA loss**: predict embedding of next-state latent,
  not the next state itself. This is LeCun's anti-generative bet.
- **Causal graph prior**: a soft constraint that the dynamics between
  slots respects some learned DAG.

What the paper likely shows:
- A simulated tabletop environment with multiple objects where
  interventions on object A are predicted to leave object B unaffected
  iff the model has learned that A and B are causally independent.
- Comparison: vanilla JEPA fails this. Causal-JEPA succeeds.
- The "do" operator is approximated by training-time clamping:
  pin slot's value and ask the model to predict the rest.

**What I am NOT sure about** [UNCERTAIN]:
- The exact loss function (TBD when reading paper)
- Whether they show **in-distribution counterfactuals** only, or also
  held-out objects. This matters for generalization.
- Whether the causal graph is learned end-to-end or has a human prior.

## 3. V-JEPA 2-AC - what it really proves

Paper: arXiv 2506.09985 (Jun 2025).

What it shows (more confident because more publicly discussed):
- 1.2B parameter transformer trained on 1M hours of video.
- 62 hours of real Franka robot data fine-tuning.
- Zero-shot manipulation tasks in **a NEW lab not in fine-tuning set**.

What this means for our work:
- The architecture (vision transformer + action head) generalises
  across physics settings without per-robot fine-tuning.
- This is NOT a world model in the planning sense — it is a perception-
  to-action policy. But it is the perception backbone that a world model
  would sit on top of.
- Implication: for our Project B (cross-domain), we should NOT build
  a perception stack. We use V-JEPA 2-AC as our perception front-end.

## 4. JEPA-WM (Dec 2025) and Value-Guided JEPA (Jan 2026)

JEPA-WM (arXiv 2512.24497):
- Empirical study: which design choices actually drive success in
  physical planning with JEPA?
- **[UNCERTAIN]** but likely finds: temporal context length matters
  more than model size, value head matters more than policy head,
  non-generative prediction beats generative at horizon > 10 steps.

Value-Guided Action Planning with JEPA (arXiv 2601.00844):
- A learned value function gates action selection BEFORE the planning
  step.
- **[UNCERTAIN]** but conceptually: value says "this state is bad",
  JEPA-suggested action is skipped. This is what our Project A monitor
  does — but this paper builds the value INTO the JEPA rather than as
  a separate module.
- Important for our decoupling question: does separate Monitor work
  better, or is integrated value better? We should design an ablation
  in Project A paper v1: separate Monitor vs integrated value head.

## 5. UniZero (Jun 2024) - the latent-MCTS wave

Paper: arXiv 2406.10667.

- Extends MuZero to a learned latent dynamics head (like Dreamer) +
  MCTS in latent space + value + policy heads.
- Result: SOTA on Atari 100k and a few DMControl tasks.

For us: this is the **MuZero + Dreamer crossover**. We should read this
carefully because it tells us if our Project A monitor is general enough
to plug into a UniZero-style planner (which it should be).

## 6. What this all means for Project C paper v0

Project C paper v0 should target (revised):
- Title idea: "Causal Latents for World Models: A Minimal Recipe with
  Object-Centric Slot Representation"
- Setup: pick one of (CausalWorld, a custom block-pushing scene, or
  a Cosmos 2.5-based tabula-rasa domain)
- Method: slot-attention encoder + JEPA-style non-generative prediction
  + a learned intervention mask (only predict where intervention
  occurred).
- Baseline: vanilla RSSM, vanilla JEPA, both with same data.
- Metric: paired with a Project A-style monitor (separate module).
  Does the causal structure make the monitor more stable across
  seeds?
- Compute: 1-3 A100 weeks. Big jump from our current 0 GPU. This is
  the reason we need to solve the GPU problem first.

**This paper is 6-12 months out**, behind a solved GPU situation.

## 7. Critical assessment - is the causal ladder really L3?

The honest take on whether Causal-JEPA gives us Pearl L3 (counterfactuals):

Pearl's L3 = "given what happened, what would have happened under
a different action?". This requires (a) accurate structural model,
(b) the actual outcome history, (c) ability to ablate parts of the
history. Causal-JEPA gives us (a) on a constrained domain (objects +
interventions seen during training). For (b) and (c), the model needs
to handle LATENT counterfactuals, which is harder.

My claim: **Causal-JEPA is L2+ in object-rich environments, not full L3.**
For our paper v0, this is a feature: we can claim incremental progress
without overclaiming.

## 8. Concrete next moves for Project C

1. **Week 1**: read V-JEPA 2-AC (skim, focus on architecture diagram)
2. **Week 1**: read Causal-JEPA carefully, summarise each section
   into 1-paragraph notes
3. **Week 2**: read UniZero + JEPA-WM + Value-Guided JEPA
4. **Week 3**: write the Project C paper v0 outline (3-page draft)
5. **Week 4**: submit outline to a Discord group, get critique

By end of month 1 of Project C, we have the literature base locked.
Then we wait for GPU to start experiments.

## 9. The Neuro-symbolic gap (Project E)

This 4-layer architecture includes a verification layer we have not
started: formally checking that the world model's predictions are
consistent with its claims. The user's doc implies this with
"causal verification" but does not say how.

Candidate approaches (each is a 6-month research project in itself):
- SMT solver hooked to the latent structure (Z3, Lean)
- Differentiable logic layer over the latent graph
- Counter-example guided refinement loop (CEGIS style)

We do not start Project E until **Project C has produced one open-source
CRL baseline** we can plug into. Estimated timeline: Project E opens
in 2027 Q2 if everything else stays on track.
