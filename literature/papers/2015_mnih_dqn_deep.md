# DQN (Mnih et al. 2013/2015)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **VERY HIGH** -- the original 2015 Nature paper is foundational and every RL course references it
> One-line: Train a deep CNN to play Atari via Q-learning with experience replay
> and a target network, achieving human-level performance on 49 games from
> raw pixels.

---

## Problem

Before DQN, model-free RL with neural networks had been unstable. The
combination of:
- function approximation + bootstrapping (off-policy temporal difference)
- highly correlated sequential samples
- non-stationary target

gives the "deadly triad" of Q-learning instability. Two independent
innovations addressed this: (1) experience replay buffer, (2) target
network with periodic update.

## Method

Three key tricks:

1. **Experience Replay**: store agent transitions in a buffer; sample
   mini-batches uniformly to compute Q-learning updates. Decorrelates
   sequential samples and reuses data.

2. **Target Network**: maintain a copy of the Q-network updated periodically;
   use it to compute Q-learning targets. This stabilises the bootstrap
   target.

3. **Reward clipping** (for Atari): clip rewards to [-1, +1] so gradients
   across games are on comparable scales. (Clip the actions too if needed.)

Loss:
```
L = E_(s,a,r,s') [ (r + gamma * max_a' Q_target(s', a') - Q(s,a))^2 ]
```

Architecture: 3 conv layers + 2 FC layers, ~1.5M parameters. Input:
4 stacked grayscale frames (so the agent has velocity information).

## Empirical result

- 49 Atari games from raw pixels
- Same hyperparameters across all games (no per-game tuning)
- Above human expert level on 23 games
- 75% of human on average

This was the first convincing demonstration that end-to-end deep RL
could solve diverse perception tasks.

## Criticisms

1. **Sample inefficient**: requires millions of frames per game. Vanilla
   DQN needs days of GPU training.

2. **Reward clipping discards information**. Distance from ideal, degree
   of progress, etc. are lost when clipped.

3. **Greedy on-policy behavior**. After training, the policy is greedy
   w.r.t. Q, but exploration during training is epsilon-greedy. This
   separates learned Q from deployed policy in subtle ways.

4. **Doesn't actually do well on all games**. Many Atari games require
   long-term planning that DQN's 4-frame stack cannot capture.

5. **Function approximation can still diverge** despite target network.
   Subsequent work (Double DQN, Dueling DQN, Rainbow) added many hacks
   to fix this. The original DQN is not stable by 2025 standards.

## Connection to our program

DQN is foundational but NOT used in Project A's primary path. We use
PPO. **Why PPO over DQN for Project A**:

- PPO is on-policy: the policy the monitor observes is the policy that
  runs at inference (modulo dropout). For decoupling claims, this
  cleanliness matters.
- DQN would require the monitor to be trained on the Q-network's
  recommendations, not the policy's actions; this is messier to argue.

So DQN is in our background reading list but not in our primary method.

## Concrete next move

None for Project A directly. We may extend Project A to add a DQN
baseline in year 2, but not now.

## Confidence

VERY HIGH. Numbers are widely re-verified.

## Related

- Double DQN (van Hasselt 2015)
- Dueling DQN (Wang 2016)
- Rainbow (Hessel 2018): ensemble of DQN improvements
- QR-DQN (Dabney 2018): distributional RL
- Categorical DQN (Bellemare 2017): C51

## Status

- [x] cite in TASKBOOK background reading
- [ ] not in Project A primary method
