"""
envs.py — Environment wrappers for Project A.

We use three classic-control tasks:
- CartPole-v1    (very easy, baseline sanity check)
- Acrobot-v1     (medium, sparse-ish reward)
- LunarLander-v2 (harder, used as "real" experiment in v1)

All CPU-runnable. All gymnasium-registered.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Tuple


# Review-needed:
# (1) The threshold values for "failure" are heuristic.
#     See ``is_failure_episode``. Codex can tune these, but you
#     need to confirm they make sense for the paper's claim.
# (2) The observation history length (HISTORY_LEN) controls
#     how much the Monitor sees. REVIEW-ME: should it be 32, 64, 128?


# Heuristic thresholds for what counts as a "failure episode" during Monitor training.
# These will appear in the paper's Appendix A, so they must be justified.
FAILURE_THRESHOLDS = {
    "CartPole-v1":     100.0,    # very short episodes if we fail
    "Acrobot-v1":     -500.0,    # acrobot converges around -80 to -100
    "LunarLander-v2":  100.0,    # landing roughly < 100 is unstable
}


@dataclass
class Transition:
    """A single (state, action, reward) record stored in episode history."""
    obs:  np.ndarray
    action: int
    reward: float


@dataclass
class EpisodeLog:
    """Full episode trajectory used as Monitor input."""
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    env_name: str = ""

    def history_vector(self, history_len: int = 32) -> np.ndarray:
        """
        Flatten the last `history_len` transitions into a single fixed-size vector.
        The Monitor takes this as input.

        Observation is zero-padded if fewer than history_len transitions exist.
        Action and reward are zero-padded likewise.
        """
        obs_dim = self.transitions[0].obs.shape[0] if self.transitions else 1
        # Each transition contributes: obs + action_onehot + reward = obs_dim + 1 + 1 floats
        per_step = obs_dim + 2
        vec = np.zeros(history_len * per_step, dtype=np.float32)
        # Take the most recent `history_len` transitions
        recent = self.transitions[-history_len:]
        for i, tr in enumerate(recent):
            base = i * per_step
            vec[base:base + obs_dim] = tr.obs
            # Action as one-hot (or just its index normalised, here one-hot for clarity)
            vec[base + obs_dim + tr.action] = 1.0
            vec[base + obs_dim + 1] = tr.reward
        return vec


def is_failure_episode(episode: EpisodeLog) -> bool:
    """
    Decide whether an episode counts as a failure for Monitor training labels.

    REVIEW-ME: This is a heuristic definition. In the paper Appendix A we
    need to (a) justify this definition and (b) show a sensitivity analysis
    over different thresholds. The current values are chosen so ~25% of
    early-training PPO episodes are labelled failures — that gives the
    Monitor a balanced binary classification problem.
    """
    threshold = FAILURE_THRESHOLDS.get(episode.env_name, 0.0)
    # Treat episodes much worse than "average" as failure
    return episode.total_reward < threshold


def make_env(env_name: str, seed: int | None = None) -> gym.Env:
    """Factory. Defaults to CPU render_mode=None for speed."""
    env = gym.make(env_name)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def rollout_one_episode(env: gym.Env, policy, max_steps: int = 1000) -> EpisodeLog:
    """Run a single episode under `policy` (callable obs -> action)."""
    log = EpisodeLog(env_name=env.spec.id if hasattr(env, "spec") and env.spec else "")
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = int(policy(obs))
        log.transitions.append(Transition(obs=obs, action=action, reward=0.0))
        obs, reward, terminated, truncated, _ = env.step(action)
        log.transitions[-1].reward = float(reward)
        log.total_reward += float(reward)
        done = terminated or truncated
        steps += 1
    return log
