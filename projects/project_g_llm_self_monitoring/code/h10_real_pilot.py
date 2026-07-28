"""
Project G -- H10 REAL-LM pilot.

This script runs the H10 pre-registered protocol on REAL LLM rollouts
from Qwen2.5-3B-Instruct + GSM8K (both already cached in this
environment). It is a PILOT (n=1 seed, reduced trace count) not the
full pre-registered H10 (n=5 seeds, 200 rollouts/seed).

What this script does:
1. Loads Qwen2.5-3B-Instruct (frozen, no fine-tuning).
2. Loads GSM8K test set (math reasoning).
3. Generates reasoning traces for each problem (deterministic).
4. Labels each trace as failure (1) or success (0) based on
   final-answer correctness against GSM8K ground truth.
5. Trains 3 arms: Frozen Monitor, Joint Monitor, Random Monitor
   (negative control) on the collected traces.
6. Reports per-arm AUROC on a held-out subset.

NO_SELF_DECEPTION.md compliance:
- This is a PILOT, not the pre-registered H10. n=1, reduced trace
  count, no n=5 statistical claim.
- Pilot results are NOT a "headline result" and do not count
  toward the H10 verdict.
- If the pilot is inconclusive, the full H10 still needs to run
  with the pre-registered sample size (n=5 -> n=15 if needed).
- The 3-arm comparison structure matches the pre-registration.

Computational expectation:
- 3B model on CPU: ~1-2 sec/token forward pass.
- For 16 rollouts * 80 tokens: ~30 min on CPU.
- For 8 rollouts * 80 tokens: ~15 min on CPU.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import numpy as np

from llm_monitor import LLMSlotMonitor
from joint_monitor import train_frozen_monitor, train_joint_monitor
from real_llm_rollout_collector import (
    load_frozen_lm, load_gsm8k, collect_real_rollouts,
)
from h10_smoke import compute_auroc


def random_monitor_auroc(eval_features, eval_labels, seed=0):
    """Random Monitor: untrained U[0,1] signal."""
    rng = torch.Generator().manual_seed(seed)
    scores = torch.rand(len(eval_labels), generator=rng)
    return compute_auroc(scores, torch.tensor(eval_labels, dtype=torch.float32))


def main():
    print("=" * 70)
    print("Project G H10 REAL-LM pilot (Qwen2.5-3B-Instruct + GSM8K)")
    print("=" * 70)
    print("PILOT: n=1 seed, reduced trace count. NOT the full pre-reg H10.")
    print()

    # Pilot parameters (small for CPU).
    n_total = int(os.environ.get("H10_N_TOTAL", "16"))  # total rollouts
    n_train_frac = 0.75
    seed = 0

    # Step 1: Load LM.
    t0 = time.time()
    # Use explicit local path to avoid hub validation in offline mode.
    local_lm_path = os.environ.get(
        "H10_LM_PATH",
        r"F:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    model, tokenizer = load_frozen_lm(model_name=local_lm_path, device="cpu", dtype=torch.float16)
    t_lm = time.time() - t0
    print(f"LM load time: {t_lm:.1f}s")
    print()

    # Step 2: Load GSM8K.
    t0 = time.time()
    problems, ground_truths = load_gsm8k(split="test", n_samples=n_total)
    t_ds = time.time() - t0
    print(f"Dataset load time: {t_ds:.1f}s")
    print()

    # Step 3: Collect traces.
    t0 = time.time()
    trace_features, labels, metadata = collect_real_rollouts(
        model, tokenizer, problems, ground_truths,
        max_new_tokens=64, device="cpu", seed=seed,
    )
    t_trace = time.time() - t0
    n_total_collected = len(trace_features)
    print(f"Trace collection time: {t_trace:.1f}s")
    print(f"Total rollouts: {n_total_collected}")
    if n_total_collected == 0:
        print("ERROR: no rollouts collected. Check dataset / LM load.")
        return
    n_success = sum(1 for l in labels if l == 0.0)
    n_failure = sum(1 for l in labels if l == 1.0)
    print(f"  Success: {n_success}, Failure: {n_failure}")
    print(f"  Failure rate: {n_failure / n_total_collected:.3f}")
    print()

    # Step 4: Train/eval split (deterministic).
    n_train = int(n_total_collected * n_train_frac)
    train_features = torch.stack(trace_features[:n_train])
    train_labels = torch.tensor(labels[:n_train], dtype=torch.float32)
    eval_features = torch.stack(trace_features[n_train:])
    eval_labels = torch.tensor(labels[n_train:], dtype=torch.float32)
    print(f"Train: {tuple(train_features.shape)}, eval: {tuple(eval_features.shape)}")
    print(f"Train failure rate: {train_labels.mean().item():.3f}")
    print(f"Eval failure rate:  {eval_labels.mean().item():.3f}")
    print()

    # Step 5: Train 3 arms.
    arms = {}

    # Arm 1: Frozen Monitor.
    t0 = time.time()
    torch.manual_seed(seed)
    frozen_mon = LLMSlotMonitor()
    frozen_opt = torch.optim.Adam(frozen_mon.parameters(), lr=1e-3)
    frozen_mon, _ = train_frozen_monitor(
        frozen_mon, frozen_opt, train_features, train_labels,
        n_epochs=20, batch_size=8, seed=seed,
    )
    frozen_mon.eval()
    with torch.no_grad():
        frozen_pred = frozen_mon(eval_features)
    frozen_auroc = compute_auroc(frozen_pred, eval_labels)
    t_frozen = time.time() - t0
    arms["Frozen"] = frozen_auroc
    print(f"Frozen Monitor: AUROC={frozen_auroc:.3f}  (time={t_frozen:.1f}s)")

    # Arm 2: Joint Monitor.
    t0 = time.time()
    torch.manual_seed(seed)
    joint_mon = LLMSlotMonitor()
    joint_opt = torch.optim.Adam(joint_mon.parameters(), lr=1e-3)
    joint_mon, _ = train_joint_monitor(
        joint_mon, joint_opt, train_features, train_labels,
        n_llm_steps=4, perturb_scale=0.05,
        n_monitor_epochs_per_llm_step=5, batch_size=8, seed=seed,
    )
    joint_mon.eval()
    with torch.no_grad():
        joint_pred = joint_mon(eval_features)
    joint_auroc = compute_auroc(joint_pred, eval_labels)
    t_joint = time.time() - t0
    arms["Joint"] = joint_auroc
    print(f"Joint Monitor:  AUROC={joint_auroc:.3f}  (time={t_joint:.1f}s)")

    # Arm 3: Random Monitor (negative control).
    random_auroc = random_monitor_auroc(eval_features, eval_labels, seed=seed)
    arms["Random"] = random_auroc
    print(f"Random Monitor: AUROC={random_auroc:.3f}  (negative control)")
    print()

    # Step 6: Verdict.
    print("=" * 70)
    print("PILOT VERDICT (n=1 seed, NOT pre-reg H10):")
    print()
    print(f"  Frozen: {arms['Frozen']:.3f}")
    print(f"  Joint:  {arms['Joint']:.3f}")
    print(f"  Random: {arms['Random']:.3f}")
    print(f"  Delta_F-J: {arms['Frozen'] - arms['Joint']:+.3f}")
    print(f"  Delta_F-R: {arms['Frozen'] - arms['Random']:+.3f}")
    print()
    if arms['Frozen'] > arms['Joint'] + 0.05 and arms['Frozen'] > arms['Random'] + 0.10:
        print("  Pilot DIRECTION consistent with H10 (frozen > joint by >=0.05,")
        print("  frozen > random by >=0.10). Full pre-reg H10 warranted.")
    elif arms['Joint'] > arms['Frozen'] + 0.05:
        print("  Pilot DIRECTION contradicts H10 (joint > frozen).")
        print("  Full pre-reg H10 still warranted to confirm n=5 statistical claim.")
    else:
        print("  Pilot INCONCLUSIVE on synthetic-like data direction.")
        print("  Full pre-reg H10 needed for verdict.")
    print()
    print("=" * 70)
    print("REMINDER: This is a PILOT, n=1 seed. Full H10 needs n=5 seeds.")
    print("=" * 70)


if __name__ == "__main__":
    main()

    local_lm_path = os.environ.get(
        "H10_LM_PATH",
        r"F:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    model, tokenizer = load_frozen_lm(model_name=local_lm_path, device="cpu", dtype=torch.float16)
