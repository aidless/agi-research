"""
envs.py â€” Environment wrappers for Project A.

We use three classic-control tasks:
- CartPole-v1    (very easy, baseline sanity check)
- Acrobot-v1     (medium, sparse-ish reward)
- LunarLander-v2 (harder, used as "real" experiment in v1)

Plus Procgen games for the paper environment (multi-task generalisation).
All CPU-runnable. All gymnasium-registered.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
try:
    # Procgen 0.10.x registers envs only with the legacy ``gym`` package,
    # not with gymnasium. Import it as an alias so we can use its make().
    import gym as _legacy_gym  # noqa: F401
except ImportError:
    _legacy_gym = None
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

    def history_vector(self, history_len: int = 32, n_actions: int = 2) -> np.ndarray:
        """
        Flatten the last `history_len` transitions into a single fixed-size vector.
        The Monitor takes this as input.

        n_actions is the action-space size; action is encoded as a 1-hot
        vector of length n_actions (default 2 for backward-compatibility with
        CartPole).
        """
        obs_dim = self.transitions[0].obs.shape[0] if self.transitions else 1
        # Each transition contributes: obs (obs_dim) + action_onehot (n_actions)
        # + reward (1) = obs_dim + n_actions + 1 floats
        per_step = obs_dim + n_actions + 1
        vec = np.zeros(history_len * per_step, dtype=np.float32)
        # Take the most recent `history_len` transitions
        recent = self.transitions[-history_len:]
        for i, tr in enumerate(recent):
            base = i * per_step
            vec[base:base + obs_dim] = tr.obs
            # Action one-hot
            if 0 <= tr.action < n_actions:
                vec[base + obs_dim + tr.action] = 1.0
            vec[base + obs_dim + n_actions] = tr.reward
        return vec


def is_failure_episode(episode: EpisodeLog) -> bool:
    """
    Decide whether an episode counts as a failure for Monitor training labels.

    REVIEW-ME: This is a heuristic definition. In the paper Appendix A we
    need to (a) justify this definition and (b) show a sensitivity analysis
    over different thresholds. The current values are chosen so ~25% of
    early-training PPO episodes are labelled failures â€” that gives the
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


# ©¤©¤ Procgen (optional) ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤

PROCGEN_TRAIN_GAMES: tuple[str, ...] = (
    "bigfish", "bossfight", "caveflyer", "chaser",
    "coinrun", "dodgeball", "fruitbot", "jumper",
)

PROCGEN_TEST_GAMES: tuple[str, ...] = (
    "starpilot", "climber", "ninja", "plunder",
    "leaper", "maze", "heist", "miner",
)

ALL_PROCGEN_GAMES: tuple[str, ...] = PROCGEN_TRAIN_GAMES + PROCGEN_TEST_GAMES

PROCGEN_GAME_GROUPS = {
    "bigfish":    "T1",  "bossfight": "T1",  "caveflyer":  "T1", "chaser":    "T1",
    "coinrun":    "T1",  "dodgeball": "T1",  "fruitbot":   "T1", "jumper":    "T1",
    "starpilot":  "T2",  "climber":   "T2",  "ninja":      "T2", "plunder":   "T2",
    "leaper":     "T3",  "maze":      "T3",  "heist":      "T3", "miner":     "T3",
}


def percentile_failure_threshold(
    returns: list[float], percentile: float = 10.0  # lowered for more failure cases
) -> float:
    """Adaptive failure threshold â€” bottom `percentile` of training returns."""
    if not returns:
        return 0.0
    return float(np.percentile(returns, percentile))


class ProcgenWrapper(gym.Env):
    """Lightweight gymnasium-compatible wrapper for a legacy-gym procgen env.

    procgen 0.10 only registers with the legacy `gym` package, so we
    cannot subclass gymnasium.ObservationWrapper (which would type-check
    the underlying env). This wrapper mimics the gymnasium Env API
    we need and applies an obs_encoder to every observation.

    The resulting object supports:
        .reset() -> (obs, info)
        .step(a) -> (obs, r, term, trunc, info)
        .observation_space, .action_space, .close()
    """

    metadata = {"render_modes": []}

    def __init__(self, env, obs_encoder):
        self.env = env  # legacy-gym env (e.g. OrderEnforcing wrapper)
        self._encoder = obs_encoder
        # Probe one obs to size the observation_space.
        _reset = self.env.reset()
        sample_obs = _reset[0] if isinstance(_reset, tuple) else _reset
        if isinstance(sample_obs, dict):
            rgb = sample_obs.get('rgb')
            if rgb is None:
                rgb = sample_obs[list(sample_obs.keys())[0]]
            sample_obs = rgb
        sample_obs = np.asarray(sample_obs)
        sample_encoded = self._encoder(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_encoder.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = env.action_space
        self.spec = getattr(env, 'spec', None)

    def reset(self, *, seed=None, options=None):
        try:
            if seed is not None:
                self.env.action_space.seed(seed)
        except Exception:
            pass
        _reset = self.env.reset()
        obs = _reset[0] if isinstance(_reset, tuple) else _reset
        return self._extract_obs(obs), {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, r, term, trunc, info = result
        else:  # legacy gym 4-tuple
            obs, r, done, info = result
            term, trunc = bool(done), False
        return self._extract_obs(obs), float(r), bool(term), bool(trunc), info

    def _extract_obs(self, obs):
        if isinstance(obs, dict):
            rgb = obs.get('rgb')
            if rgb is None:
                rgb = obs[list(obs.keys())[0]]
            obs = rgb
        return self._encoder(np.asarray(obs))

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def render(self):
        try:
            return self.env.render()
        except Exception:
            return None


def create_procgen_env(
    game: str, seed: int, obs_encoder,
    start_level: int = 0, num_levels: int = 0, distribution_mode: str = "easy",
) -> gym.Env:
    """Factory â€” requires `pip install procgen`.

    Note: procgen 0.10.x only registers with the legacy ``gym`` package,
    not with ``gymnasium``. We use ``gym.make`` via that legacy module,
    but the resulting env exposes standard gymnasium-style
    .observation_space and .action_space, so all downstream code works.
    """
    import procgen  # noqa: ensure registered
    if _legacy_gym is None:
        raise ImportError(
            "create_procgen_env needs the legacy `gym` package because procgen"
            " 0.10.x only registers with it. Try: pip install gym==0.23.0"
        )
    env = _legacy_gym.make(
        f"procgen-{game}-v0",
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=distribution_mode,
    )
    # Seed the spaces from the legacy env (uses gymnasium-compatible API)
    try:
        env.action_space.seed(seed if seed is not None else 0)
        env.observation_space.seed(seed if seed is not None else 0)
    except Exception:
        pass  # not all procgen envs support gym.space.seed
    return ProcgenWrapper(env, obs_encoder=obs_encoder)


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


