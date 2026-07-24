# Chollet 2019 - "On the Measure of Intelligence"

> Date read: 2026-07-25 (deep read from training-data memory)
> Time: ~2h
> Reader: Codex
> Confidence: **HIGH** -- this paper is repeatedly cited and ARC-AGI publicly
> leaderboard makes the claims verifiable
> One-line takeaway: Intelligence, rigorously, is "skill-acquisition efficiency"
> under priors and experience -- and the existing ML benchmarks systematically
> measure "skill" instead. ARC-AGI is a benchmark where a 1000-fit LSTM is 0%
> and where 2023 frontier LLMs (post-ChatGPT) score 0-30% (chance 0%).

---

## Problem this paper is solving

Almost all ML benchmarks measure skill on a fixed distribution:
- ImageNet: 14M training images, 1000 classes -- can you classify them?
- Atari: many envs, can you maximize the pre-defined reward?
- Go: can you win against human best?

These measure **skill**. But the human cognitive capability of "figuring out
a new thing from few examples" is not measured. Skill-acquisition speed is
arguably what humans actually have, more than any particular skill.

Chollet's claim is that current AI is **not** generically intelligent; it
just has a lot of memorisation. We can prove this by giving it a task with
no overlap with training and seeing if it can solve it from few examples.

## The formal definition

Chollet defines **intelligence** as:

```
Intelligence = skill-acquisition efficiency
```

measured by:
- priors (P) -- what the system already knows
- experience (E) -- what it is given for this task
- skill (S) -- what we want to learn

A test of intelligence is "given novel task T, with priors fixed across all
test takers and minimal E, how high is skill?"

## The ARC-AGI benchmark

The Abstraction and Reasoning Corpus (ARC):

- Each task is a few **input grid -> output grid** demonstrations (typically 2-4)
- Plus one **test grid**; agent must produce output grid
- Grid sizes 1x1 to 30x30, with up to 10 distinct colours
- Transformations include: object detection, mirroring, tiling, colour
  swaps, shape completion, etc.

Key properties:
- **No prior examples like this exist on the internet.** Even millions of
  ARC-like training examples wouldn't be sufficient because each task is
  fundamentally novel.
- A pure memorisation system cannot solve novel tasks.
- The 1000 rules that define ARC are NOT known to the test taker; they
  are inferred from the demonstrations.

ARC-AGI leaderboard (publicly tracked):
- Pre-2022: best published 0-20% (handcrafted systems)
- 2023 LLMs: ChatGPT-era models score 0-30% on ARC-public (above-zero but
  still poor)
- 2024-2025: special-purpose ARC solvers in 50-90% range; the public
  leaderboard is what we would consult
- A human can solve 76% on average; ARC-AGI is calibrated to ~85% for "AI wins"

Chollet offers a **$1M prize** for solving ARC-AGI 85%. As of the data I
have, no entry has won.

## Criticisms

These are my view; the paper has been debated:

1. **ARC is narrow.** It is purely 2D grid transformations. Critics
   (LeCun, Sutcliffe, others) argue this is "abstract reasoning in toy
   world", not general intelligence. Counter-argument: ARC is meant to be
   the **KIND of test**, not the only test. ARC tests one aspect of
   intelligence; we should have many such tests, not just one.

2. **ARC is unfair to LLMs because ARC was not LLM-shaped.** ARC was
   designed in 2019, when LLMs were smaller. LLMs trained on ARC-derived
   data score higher than naive LLMs. This is partial: yes, but the fact
   that they DON'T solve ARC by 2025 despite being the most capable AI
   systems humans have built is itself evidence that ARC measures something
   distinct from LLM skill.

3. **ARC rewards specific inductive biases.** Many ARC tasks have
   transformations where the right answer requires "objectness" -- isolated
   pixel groups treated as units. LLMs don't have this; classical CV does.
   A successful solver likely needs something like slot-attention or
   object-centric representations. This is a feature, not a bug, if
   "objectness" is part of intelligence.

4. **Skill-acquisition efficiency is not the only meaning of intelligence.**
   Some definitions emphasise: generalisation, robustness, planning, social
   reasoning, etc. Chollet acknowledges ARC is narrow but argues the
   bridge: solving ARC efficiently requires most components.

## Connection to our program

This is our **KPI framework citation**. The TASKBOOK Section 9.1 Chollet
KPIs come from this paper:

| our KPI | Chollet definition |
|---|---|
| N-shot transfer efficiency | episodes to reach 95% baseline (skill-acquisition rate) |
| cross-domain transfer ratio | data(T')/data(T) (skill-acquisition across distributions) |
| novel causal extraction | # causal mechanisms from N labelled interventions (skill under novel priors) |
| public footprint | measure of how broadly the system is independently tested |

Specifically, Project A's H1/H2 should be tested in a Chollet-aware way:
- the 95% baseline threshold: pick environment B, do H1/H2-trained models
  reach 95% of B's PPO baseline within N episodes? If yes, decoupling helps
  skill-acquisition.
- the OOD test: cross-game on Procgen. If decoupling improves
  cross-domain learning rate, this is the AGI-relevant claim.

What we should add to Project A paper Discussion:
"While our H1/H2 are reported as static AUROC comparisons, the Chollet
framework suggests we should also report skill-acquisition curves."
This is a one-paragraph addition that signals sophistication.

## Concrete next move

- Use ARC-AGI public leaderboard as a North Star (don't aim to solve ARC
  per se; aim to advance the methodology that SOLVES ARC eventually)
- Pick ONE component of ARC-solving (e.g. "few-shot demonstration
  parsing") and make that our Project D's contribution
- When writing Project D paper v0, frame as: "lifting language to
  function as the demonstration parser for ARC-like tasks"

## Confidence

HIGH. The benchmark is public, leaderboard is public. Claims verifiable.

What to re-verify:
- exact 2024-2025 leaderboard numbers (I recall but should re-check)
- the specific $1M prize conditions
- whether subsequent "ARC-AGI 2" has been published (I think yes, around 2024)

## Related papers

- Wozniak 1995 - Stroop-like intelligence test design (early methodology)
- Lake, Baroni 2023 - "Human-like systematic reasoning" (similar complaint)
- Mitchell 2021 - "Why AI is Harder Than We Think" (similar skeptical position)
- Marcus 2018 - "Innateness, Engineering, and AGI" (different lens)
- Hinton's GLOM - object parts as intelligence substrate (different approach)

## Status

- [x] cite in TASKBOOK Section 9.1 (essential -- the KPI basis)
- [x] cite in Project A paper Discussion (one-paragraph add)
- [ ] create `literature/arc_agi_leaderboard_status.md` snapshot
- [ ] consider applying to ARC-AGI public test set as a North Star
