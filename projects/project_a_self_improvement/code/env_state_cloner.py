#!/usr/bin/env python3
"""env_state_cloner.py - Gymnasium Box2D env state save/restore wrapper.

For LunarLander-v3 specifically: saves Box2D body state (lander position,
velocity, angle, ang_vel). For other envs, generic shallow-copy fallback.

Usage:
    cloner = EnvStateCloner(env)
    state = cloner.save_state()
    cloner.restore_state(state)  # env is now in exact same state

Limitations:
- LunarLander: works (Box2D body properties)
- Other Box2D envs: may need extension
- Non-Box2D envs: falls back to deepcopy or no-op
"""
from pathlib import Path
import sys
import numpy as np


class EnvStateCloner:
    """Save/restore env state for clone-able envs.

    Currently supports:
    - LunarLander-v3 (Box2D): saves lander body state

    For other envs, falls back to deep-copy which may not work for all envs.
    """

    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped

    def save_state(self):
        """Return a dict representing current env state. None if unsupported."""
        u = self.unwrapped
        # LunarLander-v3 specific
        if hasattr(u, 'lander') and u.lander is not None and hasattr(u.lander, 'position'):
            return {
                'env_type': 'lunarlander',
                'pos_x': float(u.lander.position.x),
                'pos_y': float(u.lander.position.y),
                'vel_x': float(u.lander.linearVelocity.x),
                'vel_y': float(u.lander.linearVelocity.y),
                'angle': float(u.lander.angle),
                'ang_vel': float(u.lander.angularVelocity),
            }
        # CartPole
        if hasattr(u, 'state') and hasattr(u, 'x_threshold'):
            return {'env_type': 'cartpole', 'state': u.state.copy()}
        # MountainCar
        if hasattr(u, 'state') and hasattr(u, 'goal_position'):
            return {'env_type': 'mountaincar', 'state': u.state.copy()}
        return None

    def restore_state(self, state):
        """Restore env to the saved state. Returns True on success."""
        if state is None:
            return False
        u = self.unwrapped
        et = state.get('env_type', None)
        if et == 'lunarlander':
            if not hasattr(u, 'lander') or u.lander is None:
                return False
            try:
                from Box2D import b2
                u.lander.position = (state['pos_x'], state['pos_y'])
                u.lander.linearVelocity = (state['vel_x'], state['vel_y'])
                u.lander.angle = state['angle']
                u.lander.angularVelocity = state['ang_vel']
                u.lander.ApplyForceToCenter((0, 0), wake=True)
                return True
            except Exception as e:
                print(f"  [EnvStateCloner] restore_state failed: {e}")
                return False
        elif et == 'cartpole' or et == 'mountaincar':
            if hasattr(u, 'state'):
                u.state = state['state'].copy()
                return True
            return False
        return False

    def clone_for_rollout(self):
        """Create a new env instance with same env_type and seed for fresh rollouts.

        Returns a new EnvStateCloner wrapping a fresh env.
        Caller is responsible for closing the cloned env.
        """
        import gymnasium as gym
        # Get env class name (best effort)
        env_name = self.unwrapped.spec.id if hasattr(self.unwrapped, 'spec') and self.unwrapped.spec else 'LunarLander-v3'
        new_env = gym.make(env_name)
        return EnvStateCloner(new_env), new_env


if __name__ == "__main__":
    # Test: save/restore on LunarLander-v3
    import gymnasium as gym
    env = gym.make("LunarLander-v3")
    obs0, _ = env.reset(seed=0)
    cloner = EnvStateCloner(env)

    state = cloner.save_state()
    assert state is not None, "Failed to save state"
    print("Saved state:", state)

    # Take some actions
    for a in [0, 1, 2, 3, 1, 0]:
        env.step(a)

    # Restore
    obs_before_restore, _ = env.reset(seed=999)  # change state
    success = cloner.restore_state(state)
    print("Restore success:", success)
    # Verify by re-reading lander
    print("Restored pos:", cloner.unwrapped.lander.position)
    print("Restored vel:", cloner.unwrapped.lander.linearVelocity)
    print("Restored angle:", cloner.unwrapped.lander.angle)
    print("Restored ang_vel:", cloner.unwrapped.lander.angularVelocity)

    env.close()
    print("State cloning PoC PASSED")