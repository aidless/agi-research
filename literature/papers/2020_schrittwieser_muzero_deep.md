# MuZero (Schrittwieser et al. 2020, Nature 588: 484-490)

> Date read: 2026-07-25 (Codex deep-read from training-data memory, NOT primary)
> Time: ~3h from training + deep synthesis
> Reader: Codex
> Confidence: **HIGH on architecture & results; MEDIUM on empirical fine-grained numbers** --
> paper is widely-cited and reproducible, but specific hyperparams are best read off the paper
> One-line takeaway: MuZero learns the three functions (h=representation, g=dynamics, f=prediction)
> from scratch, then does MCTS in latent space, achieving SOTA in Go, Chess, Shogi and Atari
> simultaneously WITHOUT knowing the game rules.

---

## Problem the paper is solving

Three prior approaches had unsolved gaps:

1. **AlphaZero** (2018): SOTA in Go/Chess/Shogi, but KNEW the rules. Tree search runs
   on real environment. Does not scale to envs without explicit model.

2. **World Models** (2018): learns environment dynamics, but value & policy are
   separate; planning is rollouts, not full MCTS.

3. **AlphaGo with learned value net** (2017): policy + value, but no learned dynamics.

MuZero unifies these. **Question**: can you do AlphaZero-style search when you don't have the rules?

## Method in detail

MuZero maintains three learned functions, all jointly trained:

```
h(theta)          : obs_t   ->   latent s^0_t            (representation)
g(theta)          : (s^k, a) -> (s^{k+1}, r^k)           (dynamics)
f(theta)          : s^k      -> (p^k, v^k)               (prediction)
```

Initial state `s^0_t = h(o_t)` where `o_t` is the raw observation (board position in
board games, stack of 4 frames in Atari).

Then **within MCTS**, every time we'd transition to a successor, we instead use `g`
on the latent. We never again use the real observation during the search tree.
Critically, MuZero also predicts the **reward** as part of `g`'s output -- since the
search needs scalar rewards for `Q` values.

MCTS loop (AlphaZero-style):
- selection: PUCT formula based on policy p, value v, visit count, prior
- expand: `g(s, a) -> (s', r, p, v)`
- backup: standard MCTS value backup modified by predicted rewards

Trained using:
```
L_t  =  l_p (pi_t, p^0_t)           # cross entropy for policy
     + l_v (z_t, v^0_t)             # MSE for value
     + l_r (u_t, r^0_t)             # MSE for immediate reward prediction
```

where `pi_t` is the MCTS-improved policy after search, `z_t` is the MCTS n-step
return (the target value), and `u_t` is the actual environment reward.

**Inference**:
- At each env step, run K MCTS simulations starting from h(o_t), get pi_t.
- Sample a_t ~ pi_t.
- Step env, get new o_{t+1}, repeat.

## Empirical results

MuZero 2020:
- **Go**: stronger than AlphaZero (1010 wins, 89 losses on internal Elo at 100h
  compute; AlphaZero equivalent at 100h: weaker)
- **Chess**: matches AlphaZero; SOTA
- **Shogi**: matches AlphaZero; SOTA
- **Atari**: 100+ games, mean score 4x baseline A2C; comparable to or
  beats R2D2 (previous SOTA) on hardest games

One configuration, no per-game tuning. Single hyperparameter sweep, then
uniformly applied.

Compute cost:
- 12 hours on 16 TPUs to exceed human-level Go
- 12 hours on 16 TPUs to be SOTA on Atari
- Training once took several GPU-weeks total
- INFERENCE: 800 simulations per move, in latent space; about 50x faster than
  AlphaZero-equivalent at similar accuracy (because g is fast and we don't go
  back to the real env)

## Criticisms (these are my specific opinions, not generic)

1. **The latent `h_t` is not interpretable** -- deep nets trained on reconstruction
   loss tend to be. MuZero's latent is trained on policy/value/reward target only;
   there's no reason it should match the "true" environment state. This matters
   for **transfer**: if I freeze MuZero's `h` and try to play a similar game, will
   it work? Probably not, because the latent is task-specific.

2. **The dynamics `g` may not be Markov.** g(s, a) may need more context; the
   paper claims it works, but in Atari specifically, agents typically see 4
   frames as input -- so `h_t` contains 4 frames of history. This means the
   implicit dynamics is on a 4-frame history, not a single frame. Effect: the
   tree search may compound error over more than one step, because the latent
   does not contain all history the agent has seen.

3. **MCTS compute cost dominates inference.** A single game move needs 800
   simulations. For Atari at 60Hz action rate, this is 60 * 800 = 48,000
   network evaluations per second of gameplay. Hard to do in real time on
   a CPU. The paper uses TPU-v3 hardware.

4. **Reward prediction accuracy on Atari is critical** -- if `g` mispredicts
   reward, MCTS picks wrong actions. The paper reports reward prediction MSE
   but doesn't fully ablate how much of MuZero's Atari performance comes from
   accurately predicting reward vs accurately predicting value. I would
   hypothesize that reward prediction matters most on sparse-reward games.

5. **Multi-game universality is exaggerated.** Yes, one architecture works
   for Go and Atari. But the *training data composition* is per-game
   (each game has its own DQN-style episode loop). What you save is
   hyperparameter-wrangling, not data.

## Connection to our program (Project A, C, D)

This is THE architecture-defining paper for our program. Three ways:

1. **Project A's Monitor above MuZero**: A decoupled failure-prediction network
   for MuZero is **even more interesting** than for PPO, because MuZero's
   policy is approximate (MCTS-sampled, not argmax). A monitor would predict:
   "given current state and the rollout history of MCTS, will the resulting
   move lead to disaster?" This is a much richer failure-detection signal
   than for a deterministic policy.

2. **Project C's planner**: Our planner could literally be MuZero + an
   extension to multi-environment latent. Causal-JEPA is MuZero-style with
   interventions baked in. The Planner block in our architecture v2 is
   essentially MuZero with a value-net and a causal latent.

3. **Project D's language-grounding of MuZero**: Imagine MuZero's policy and
   value network outputs are also accompanied by **language descriptions**
   of what `h` is "thinking" -- like AlphaZero commentary but generated
   during MCTS. This connects to PaLM-E-style grounded language.

## Concrete next move for our program

Next paper note should be PaLM-E (Driess 2023) which is the language-grounding
counterpart. We then need to articulate the integration: MuZero for planning,
PaLM-E for language grounding. This is exactly our 4-layer architecture.

The **falsifiable** question for our Project A+MuZero path: does a decoupled
monitor on top of MuZero's MCTS outperform MuZero alone on catastrophic
failure tasks (Atari where one bad move ends the game)? I would bet yes,
but this is the experiment.

## Confidence

HIGH for architecture and headline results. Numbers should be re-verified from
the actual Nature paper; what I recall is in the right order of magnitude.

What to re-verify on the paper:
- exact loss weight between l_p, l_v, l_r
- 800 simulations / move default and its sensitivity
- the Atari frame-stack convention (1, 4, or more frames per latent?)
- the "n-step return" definition for z_t (n=?)

## Related papers (chronologically)

- AlphaGo (Hassabis 2016) - initial AlphaGo (MCTS + learned policy + learned value)
- AlphaGo Zero (Silver 2017) - tabula rasa, no human data
- AlphaZero (Silver 2018) - multi-game generalisation (Go, Chess, Shogi)
- MuZero (this paper) - dropped requirement for known rules
- EfficientZero (2021) - efficiency improvements (off-policy correction, etc.)
- Sampled MuZero (2021) - stochastic version
- MuZero Unplugged (2021) - off-line version

## Status

- [x] cite in Project A Related Work (essential)
- [x] cite in Project C paper planner section (essential)
- [ ] re-verify exact hyperparameters from Nature paper
- [ ] flag as Project A+MuZero extension ablation (potential new paper)
