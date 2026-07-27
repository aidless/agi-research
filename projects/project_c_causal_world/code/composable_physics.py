#!/usr/bin/env python3
"""composable_physics.py - ENWI physics modules ported to Project C.

Ported from F:\TMLR\Fusion\enwi_prototype\composable_physics.py.
4 specialized physics modules + Composer + gate network.

Architecture:
  obs -> encoder -> latent_state (128-dim)
  latent_state + action -> gate_net -> gate_weights (4 weights)
  For each module: latent_state + physics_params -> predicted next state
  Composer: gate-weighted sum of module outputs

ENWI Prediction 2 result: 94.22% improvement over monolithic baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import math


# ============================================================================
# 4 physics modules
# ============================================================================

class GravityModule(nn.Module):
    """Gravity module: predict state under gravity effect.

    Input: state + physics_params (g, mass, air_resistance, dt)
    Output: state after gravity
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.g = 9.81

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, params], dim=-1)
        return self.net(inp)


class CollisionModule(nn.Module):
    """Collision module: predict state after pairwise object collisions.

    Input: state (B, num_objects, latent_dim) + physics_params
    Output: state after collision (B, num_objects, latent_dim)
    """
    def __init__(self, latent_dim: int, num_objects: int = 4):
        super().__init__()
        self.num_objects = num_objects
        pair_dim = latent_dim * 2 + 4
        self.pair_net = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        B, N, D = states.shape
        results = []
        for i in range(N):
            for j in range(i + 1, N):
                pair = torch.cat([states[:, i], states[:, j], params], dim=-1)
                delta = self.pair_net(pair)
                if i < len(results):
                    results[i] = results[i] + delta * 0.5
                else:
                    results.append(delta * 0.5)
                if j < len(results):
                    results[j] = results[j] + delta * 0.5
                else:
                    results.append(delta * 0.5)
        return torch.stack(results, dim=1)


class FrictionModule(nn.Module):
    """Friction module: predict state after friction effect."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, params], dim=-1)
        return self.net(inp)


class InertiaModule(nn.Module):
    """Inertia module: predict state after inertia effect (Newton's 2nd law)."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, params], dim=-1)
        return self.net(inp)


# ============================================================================
# Composable physics system
# ============================================================================

class ComposablePhysics(nn.Module):
    """ENWI-style composable physics system.

    Pipeline:
      1. Encode obs to latent state
      2. Compute gate weights (which modules to use)
      3. Each module predicts next state independently
      4. Composer aggregates module outputs via gate-weighted sum

    Trained on synthetic physics scenes (free_fall, collision, etc.)
    per ENWI Prediction 2 setup.
    """
    def __init__(self, obs_dim: int = 8, latent_dim: int = 128,
                 action_dim: int = 4, num_objects: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_objects = num_objects

        # Encoder: obs -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # 4 physics modules
        self.gravity = GravityModule(latent_dim)
        self.collision = CollisionModule(latent_dim, num_objects)
        self.friction = FrictionModule(latent_dim)
        self.inertia = InertiaModule(latent_dim)

        # Composer: aggregates module outputs
        self.composer = nn.Sequential(
            nn.Linear(latent_dim * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Gate net: per-state-action weight over modules
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                physics_params: Optional[torch.Tensor] = None) -> Dict:
        B = obs.shape[0]
        if physics_params is None:
            physics_params = torch.tensor(
                [[9.81, 1.0, 0.1, 0.01]] * B,
                device=obs.device,
            )

        # Encode
        state = self.encode(obs)

        # Gate weights
        gate_input = torch.cat([state, action], dim=-1)
        gate_weights = self.gate_net(gate_input)  # (B, 4)

        # Per-module predictions
        gravity_out = self.gravity(state, physics_params)
        collision_out = self.collision(
            state.unsqueeze(1).expand(-1, self.num_objects, -1),
            physics_params,
        ).mean(dim=1)
        friction_out = self.friction(state, physics_params)
        inertia_out = self.inertia(state, physics_params)

        # Stack module outputs
        module_outputs = torch.stack(
            [gravity_out, collision_out, friction_out, inertia_out], dim=1
        )  # (B, 4, latent_dim)

        # Gate-weighted combination
        weighted = module_outputs * gate_weights.unsqueeze(-1)  # (B, 4, latent_dim)
        # Concatenate for composer
        concat = weighted.reshape(B, -1)  # (B, 4*latent_dim)

        # Final composer output
        next_state = self.composer(concat)

        return {
            "next_state": next_state,
            "gate_weights": gate_weights,
            "module_outputs": module_outputs,
            "state": state,
        }


# ============================================================================
# Monolithic baseline (for comparison)
# ============================================================================

class MonolithicWorldModel(nn.Module):
    """Single MLP world model (baseline for ENWI Prediction 2).

    Same input/output as ComposablePhysics but no modular decomposition.
    """
    def __init__(self, obs_dim: int = 8, latent_dim: int = 128, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([obs, action], dim=-1)
        return self.net(inp)


# ============================================================================
# Synthetic scene generator (for testing)
# ============================================================================

def generate_scene(scene_type: str, n_scenes: int = 100,
                   latent_dim: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic physics scenes for training.

    Returns: (obs, action, target_next_state) tensors.
    """
    obs_list, action_list, target_list = [], [], []

    for _ in range(n_scenes):
        if scene_type == "free_fall":
            obs = torch.randn(8) * 0.1
            obs[1] = torch.rand(1).item() * 5  # high y position
            action = torch.zeros(4)
            target = obs.clone()
            target[3] += 0.5  # velocity increases
            target[1] -= 0.5  # y position decreases
        elif scene_type == "collision":
            obs = torch.randn(8) * 0.5
            obs[0] = torch.rand(1).item() * 5
            obs[2] = torch.rand(1).item() * 2 - 1
            action = torch.zeros(4)
            target = obs.clone()
            target[0] = -target[0] * 0.8  # bounce
            target[2] = -target[2] * 0.8
        elif scene_type == "friction":
            obs = torch.randn(8) * 0.3
            obs[2] = torch.rand(1).item() * 2  # moving
            action = torch.zeros(4)
            target = obs.clone()
            target[2] *= 0.5  # friction slows
        elif scene_type == "inertia":
            obs = torch.randn(8) * 0.2
            obs[2] = 0.0
            obs[5] = 0.0
            action = torch.zeros(4)
            target = obs.clone()
            target[5] = 0.5  # angular vel from torque
        elif scene_type == "compound":
            obs = torch.randn(8) * 0.3
            action = torch.zeros(4)
            target = obs.clone()
            # combine all
            target[1] -= 0.3
            target[2] *= 0.7
            target[5] = 0.2
        else:
            raise ValueError(f"Unknown scene: {scene_type}")

        obs_list.append(obs)
        action_list.append(action)
        target_list.append(target)

    obs = torch.stack(obs_list)
    actions = torch.stack(action_list)
    targets = torch.stack(target_list)
    return obs, actions, targets


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("ENWI Composable Physics — Project C port")
    print("=" * 60)

    # Smoke test
    obs_dim = 8
    latent_dim = 128
    action_dim = 4

    comp = ComposablePhysics(obs_dim, latent_dim, action_dim, num_objects=4)
    mono = MonolithicWorldModel(obs_dim, latent_dim, action_dim)

    n_params_comp = sum(p.numel() for p in comp.parameters())
    n_params_mono = sum(p.numel() for p in mono.parameters())
    print(f"  Composable params: {n_params_comp:,}")
    print(f"  Monolithic params: {n_params_mono:,}")

    # Test forward
    obs = torch.randn(2, obs_dim)
    action = torch.eye(action_dim)[[0, 1]].float()
    out = comp(obs, action)
    print(f"  Composable next_state: {out['next_state'].shape}")
    print(f"  Gate weights: {out['gate_weights'][0].tolist()}")

    # Test scene generation
    for st in ["free_fall", "collision", "friction", "inertia", "compound"]:
        o, a, t = generate_scene(st, n_scenes=10)
        print(f"  Scene '{st}': obs={o.shape}, action={a.shape}, target={t.shape}")
    print()
    print("ENWI Composable Physics — ported to Project C, smoke test PASSED")