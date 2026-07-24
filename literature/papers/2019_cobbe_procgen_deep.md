# Procgen Benchmark (Cobbe et al. 2019 / 2020 update)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- widely-used benchmark, official site public
> One-line: A 16-game procedurally-generated benchmark designed to test
> generalisation in RL. Each game's levels are generated from random seeds,
> so the train and test distributions have no level-level overlap.

---

## Problem

The original 49 Atari games had problems for generalisation:
- Some games are highly overfit-able (specific level patterns)
- Train and test distributions overlap exactly
- 49 games = narrow scope

For an agent that *learns generally*, you want a benchmark where:
- Train and test are different (no level memorisation)
- Many visually diverse envs (test whether representations generalise)
- Difficulty is tuneable (hard mode vs easy mode)

## Method

Procgen uses **procedural content generation**: each env instance (level)
is generated from a random seed. The generator produces variation along
multiple axes (layout, asset placement, colours, etc.).

16 games, including:
- CoinRun: simple platformer; collect coins, avoid obstacles
- Maze: navigate to a goal in a generated maze
- DodgeBall: 1v1 arena game
- Heist: collect gems after breaking walls
- BigFish: eat smaller fish, grow bigger
- CaveFlyer: navigate 3D cave
- ... and 10+ more

Two difficulty modes:
- "Easy": 200 levels per game
- "Hard": infinite levels per game, more generation diversity

Train: 200 levels. Test: held-out 200 levels (different seeds). The
test set has disjoint level layouts from the train set.

## Empirical result

The original Procgen paper tested multiple RL algorithms (PPO, Rainbow,
DQN, IMPALA, etc.) on 16 games:

- All baselines perform much worse on test vs train, even with hundreds
  of levels of training
- Average generalisation gap (test - train) is ~50% normalized return
- Generalisation improves with more levels in training but plateaus

Even with millions of steps in train, agents do not match easy-level
performance on hard-mode test levels.

## Criticisms

1. **The "generalisation gap" is partly a level-completion problem**. Some
   generated levels are simply not solvable by the random seeds. Not a
   fair test of agent capability.

2. **Some games are sensitive to specific asset recognition**. If a test
   level uses a new asset arrangement, agents may fail on visual feature
   detection, not policy learning.

3. **The 16 games are not necessarily diverse**. They share generator
   patterns (similar 2D side-scrollers, similar visual styles). A method
   that overfits to "Procgen-similarity" rather than "general RL" could
   win.

4. **Compute burden is non-trivial on hard mode**. Realistic experiments
   need 25M-100M steps per agent per game; 16 games * 5 seeds = realistic
   range is 2-15 GPU-months.

5. **Sample efficiency is the bottleneck for non-frontier labs**. Most
   academic labs cannot run the full benchmark within reason. The "mini"
   versions (e.g., CoinRun-only, easy difficulty) are more common in papers.

## Connection to our program (Project A paper env)

Procgen is **our chosen paper env per DEC-0008**. Why:

- Designed exactly for cross-game generalisation (the H2 transfer claim)
- 16 games x multiple seeds = 80+ paired observations for statistical test
- Each game has different dynamics, so transfer is hard (challenging claim)
- Public benchmark and baseline numbers available for comparison

For our H1 (decoupling helps), we measure: decoupled monitor AUROC vs
joint-trained monitor AUROC on each Procgen game, paired Wilcoxon test.

For our H2 (transfer), we measure: cross-game generalisation. Train
monitor on 8 games, test on 8 held-out games.

**Practical concern**: We need at minimum a few million env steps per
game. On CPU this is intractable. We will use the small Procgen setting
or pre-train on a reduced subset first.

## Concrete next move

- Pick "CoinRun" + "Maze" as our first 2 Procgen games (lowest compute)
- Train PPO baseline with 1M steps each (CPU slowly; GPU in future)
- Train Monitor with frozen-policy critic
- Compare decoupled vs joint-trained critic AUROC across games
- If positive signal (p<0.01), extend to all 16 games

## Confidence

HIGH. Numbers are widely-reproduced.

Re-verify:
- exact generalisation gap percentages from the paper
- the human performance baselines
- the difficulty mode definitions

## Related papers

- CoinRun (original Cobbe paper)
- Generalisation benchmarks: Craft, RTFM, ALFWorld
- "Avoiding Side Effects" benchmarks from AI Safety field
- Atari 100k (for sample-efficient comparison, separate from Procgen)

## Status

- [x] selected as Project A paper env (DEC-0008)
- [x] cite in TASKBOOK / paper outline
- [ ] install procgen Python env (numpy-based, CPU-friendly subset exists)
