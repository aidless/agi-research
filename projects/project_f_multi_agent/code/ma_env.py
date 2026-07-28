"""ma_env.py - Minimal multi-agent coverage env (Phase 2 base).

This is a hand-coded minimal multi-agent env for testing the DMC
architecture. It is NOT a full PettingZoo-compatible implementation.

Honest framing:
- 3-agent coverage env, 5x5 grid, 3 landmarks
- Each agent moves 1 step (up/down/left/right/no-op) per turn
- Joint reward = -mean distance to nearest unclaimed landmark
- Episode ends after 20 steps OR all landmarks covered
- This is a SKELETON, not a real benchmark

Why hand-coded:
- PettingZoo not available in our Python env (path issues)
- Building a minimal env is faster than fighting package management
- Honest: this is a starting point, not a real benchmark

The env follows a similar interface to PettingZoo''s AEC API for
compatibility (agents list, action_spaces, observation_spaces, step/reset).
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class CoverageEnv:
    """3-agent coverage env on a 5x5 grid with 3 landmarks."""
    N_AGENTS = 3
    N_LANDMARKS = 3
    GRID_SIZE = 5
    N_ACTIONS = 5
    OBS_DIM = 2 + 2 * N_LANDMARKS
    MAX_STEPS = 20

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.agents = [f"agent_{i}" for i in range(self.N_AGENTS)]
        self.agent_positions = None
        self.landmark_positions = None
        self.claimed = None
        self.steps = 0
        self.action_spaces = {a: list(range(self.N_ACTIONS)) for a in self.agents}
        self.observation_spaces = {
            a: {"dim": self.OBS_DIM, "low": -self.GRID_SIZE, "high": self.GRID_SIZE}
            for a in self.agents
        }

    def reset(self):
        positions = set()
        while len(positions) < self.N_AGENTS:
            pos = (self.rng.randint(0, self.GRID_SIZE),
                   self.rng.randint(0, self.GRID_SIZE))
            positions.add(pos)
        self.agent_positions = {a: list(p) for a, p in zip(self.agents, positions)}

        landmarks = set()
        while len(landmarks) < self.N_LANDMARKS:
            pos = (self.rng.randint(0, self.GRID_SIZE),
                   self.rng.randint(0, self.GRID_SIZE))
            if pos not in positions:
                landmarks.add(pos)
        self.landmark_positions = [list(p) for p in landmarks]

        self.claimed = [False] * self.N_LANDMARKS
        self.steps = 0
        return self._get_observations()

    def _get_observations(self):
        obs = {}
        for agent in self.agents:
            ax, ay = self.agent_positions[agent]
            lm_rel = []
            for lx, ly in self.landmark_positions:
                lm_rel.extend([lx - ax, ly - ay])
            obs[agent] = np.array([ax, ay] + lm_rel, dtype=np.float32)
        return obs

    def step(self, action_dict):
        for agent, action in action_dict.items():
            ax, ay = self.agent_positions[agent]
            if action == 1:
                ay = min(ay + 1, self.GRID_SIZE - 1)
            elif action == 2:
                ay = max(ay - 1, 0)
            elif action == 3:
                ax = max(ax - 1, 0)
            elif action == 4:
                ax = min(ax + 1, self.GRID_SIZE - 1)
            self.agent_positions[agent] = [ax, ay]

        for i, (lx, ly) in enumerate(self.landmark_positions):
            if not self.claimed[i]:
                for agent in self.agents:
                    ax, ay = self.agent_positions[agent]
                    if (ax, ay) == (lx, ly):
                        self.claimed[i] = True
                        break

        distances = []
        for i, (lx, ly) in enumerate(self.landmark_positions):
            if not self.claimed[i]:
                min_dist = min(
                    abs(ax - lx) + abs(ay - ly)
                    for ax, ay in self.agent_positions.values()
                )
                distances.append(min_dist)
        if distances:
            joint_reward = -float(np.mean(distances))
        else:
            joint_reward = 0.0
        reward_dict = {a: joint_reward for a in self.agents}

        self.steps += 1
        all_claimed = all(self.claimed)
        timed_out = self.steps >= self.MAX_STEPS
        term_dict = {a: all_claimed or timed_out for a in self.agents}
        trunc_dict = {a: timed_out and not all_claimed for a in self.agents}
        info_dict = {
            "claimed": sum(self.claimed),
            "all_claimed": all_claimed,
            "steps": self.steps,
            "joint_reward": joint_reward,
        }
        return self._get_observations(), reward_dict, term_dict, trunc_dict, info_dict

    def get_state(self):
        return {
            "agent_positions": [list(p) for p in self.agent_positions.values()],
            "landmark_positions": self.landmark_positions,
            "claimed": list(self.claimed),
            "steps": self.steps,
        }


def smoke_test():
    print("=" * 50)
    print("CoverageEnv smoke test")
    print("=" * 50)
    env = CoverageEnv(seed=42)
    obs = env.reset()
    print(f"agents: {env.agents}")
    print(f"obs shapes: {[obs[a].shape for a in env.agents]}")
    total_reward = 0.0
    for t in range(env.MAX_STEPS):
        actions = {a: env.rng.randint(0, env.N_ACTIONS) for a in env.agents}
        obs, r, term, trunc, info = env.step(actions)
        total_reward += r[env.agents[0]]
        if t < 3 or t == env.MAX_STEPS - 1:
            n_claimed = info["claimed"]
            print(f"  t={t}: joint_reward={r[env.agents[0]]:.2f}, claimed={n_claimed}/{env.N_LANDMARKS}")
        if all(term.values()):
            break
    print(f"Total joint reward: {total_reward:.2f}")
    print("OK")


if __name__ == "__main__":
    smoke_test()
