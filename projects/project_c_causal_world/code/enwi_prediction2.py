#!/usr/bin/env python3
"""enwi_prediction2.py - Replicate ENWI Prediction 2 experiment.

Compares composable physics (4 specialized modules) vs monolithic
single MLP. Trains both on 5 physics scene types, reports MSE.

ENWI's reported result: 94.22% improvement on average.
"""
import argparse
import json
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PC_CODE) if (PC_CODE := HERE / ".." / "project_c_causal_world" / "code") else HERE)
sys.path.insert(0, str(HERE.parent / "project_c_causal_world" / "code"))

from composable_physics import (
    ComposablePhysics, MonolithicWorldModel,
    generate_scene,
)


def train_model(model, scene_type, n_epochs=200, batch_size=32, lr=1e-3):
    """Train model on a scene type. Returns final MSE."""
    obs, actions, targets = generate_scene(scene_type, n_scenes=1000)
    # Targets are 8-dim; we need to project them to latent_dim for loss
    # For simplicity, use 8-dim targets and project via encoder
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        idx = np.random.permutation(len(obs))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(obs), batch_size):
            bi = idx[start:start + batch_size]
            obs_b = obs[bi]
            act_b = actions[bi]
            tgt_b = targets[bi]

            opt.zero_grad()
            if isinstance(model, ComposablePhysics):
                out = model(obs_b, act_b)
                pred_latent = out["next_state"]
            else:
                pred_latent = model(obs_b, act_b)
            if isinstance(model, ComposablePhysics):
                tgt_latent = model.encode(tgt_b)
            else:
                tgt_latent = model.net[:5](torch.cat([tgt_b, act_b], dim=-1))
            if pred_latent.shape[-1] != tgt_latent.shape[-1]:
                target_dim = max(pred_latent.shape[-1], tgt_latent.shape[-1])
                if pred_latent.shape[-1] < target_dim:
                    pred_latent = F.pad(pred_latent, (0, target_dim - pred_latent.shape[-1]))
                if tgt_latent.shape[-1] < target_dim:
                    tgt_latent = F.pad(tgt_latent, (0, target_dim - tgt_latent.shape[-1]))
            loss = F.mse_loss(pred_latent, tgt_latent)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}: avg loss = {total_loss / max(1, n_batches):.6f}")
    # Compute final MSE on test set
    obs_test, act_test, tgt_test = generate_scene(scene_type, n_scenes=200)
    model.eval()
    with torch.no_grad():
        if isinstance(model, ComposablePhysics):
            out = model(obs_test, act_test)
            pred = out["next_state"]
        else:
            pred = model(obs_test, act_test)
        if isinstance(model, ComposablePhysics):
            tgt = model.encode(tgt_test)
        else:
            tgt = model.net[:5](torch.cat([tgt_test, act_test], dim=-1))
        if pred.shape[-1] != tgt.shape[-1]:
            target_dim = max(pred.shape[-1], tgt.shape[-1])
            if pred.shape[-1] < target_dim:
                pred = F.pad(pred, (0, target_dim - pred.shape[-1]))
            if tgt.shape[-1] < target_dim:
                tgt = F.pad(tgt, (0, target_dim - tgt.shape[-1]))
        mse = F.mse_loss(pred, tgt).item()
    model.train()
    return mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    print("=" * 60)
    print("ENWI Prediction 2: Composable vs Monolithic")
    print("=" * 60)
    print(f"  latent_dim: {args.latent_dim}, epochs: {args.n_epochs}")

    obs_dim = 8
    action_dim = 4
    scenes = ["free_fall", "collision", "friction", "inertia", "compound"]

    results = {"composable": {}, "monolithic": {}}

    for model_name in ["composable", "monolithic"]:
        print(f"\n--- Training {model_name} model ---")
        for scene in scenes:
            print(f"  Scene: {scene}")
            torch.manual_seed(42)
            if model_name == "composable":
                model = ComposablePhysics(obs_dim, args.latent_dim, action_dim, num_objects=4)
            else:
                model = MonolithicWorldModel(obs_dim, args.latent_dim, action_dim)
            mse = train_model(model, scene, n_epochs=args.n_epochs, batch_size=args.batch_size)
            results[model_name][scene] = mse
            print(f"    Final MSE: {mse:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION 2 RESULTS")
    print("=" * 60)
    print(f"  {'Scene':<12} {'Monolithic':<15} {'Composable':<15} {'Improvement':<12}")
    print("  " + "-" * 56)
    for scene in scenes:
        mono = results["monolithic"][scene]
        comp = results["composable"][scene]
        ratio = mono / max(comp, 1e-9)
        imp = (1 - comp / max(mono, 1e-9)) * 100
        print(f"  {scene:<12} {mono:<15.4e} {comp:<15.4e} {imp:.1f}% (ratio {ratio:.1f}x)")

    mean_mono = np.mean(list(results["monolithic"].values()))
    mean_comp = np.mean(list(results["composable"].values()))
    mean_imp = (1 - mean_comp / mean_mono) * 100
    print(f"  {'mean':<12} {mean_mono:<15.4e} {mean_comp:<15.4e} {mean_imp:.1f}%")

    # Save
    log_path = HERE / "checkpoints" / "enwi_prediction2" / "phase2_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "env": "synthetic_physics",
        "model": "composable_vs_monolithic",
        "n_epochs": args.n_epochs,
        "latent_dim": args.latent_dim,
        "scenes": scenes,
        "results_monolithic": results["monolithic"],
        "results_composable": results["composable"],
        "mean_improvement": mean_imp,
        "note": "ENWI Prediction 2 replication. Original ENWI: 94.22% improvement.",
    }, indent=2))


if __name__ == "__main__":
    main()