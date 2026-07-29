"""
Project D -- Small LM as DLR Type Checker (H12 pilot).

Uses a small LM (default: Qwen2.5-1.5B-Instruct) to check whether
a given DLR predicate is TRUE / FALSE / UNCERTAIN for a given state.

This is the FIRST experiment under Project D's revived direction
(after H10 REFUTED). The hypothesis H12 tests whether a small LM
can match the DLR baseline (97.8% mean across 4 envs) at predicate
verification, without per-env training.

NO_SELF_DECEPTION.md compliance:
- This is a SYNTHETIC smoke test using toy state vectors, NOT the
  pre-registered H12 result. The real H12 experiment requires:
  - 50 held-out LunarLander trajectories per seed
  - n=5 seeds
  - Welch t-test comparing LM vs DLR baseline accuracy
- The smoke test here validates the architecture and prompt template.
"""

import os
import torch
import numpy as np


# Default LM (cached locally).
DEFAULT_LM = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_LM_PATH = r"F:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


# Pre-registered predicate definitions (LunarLander).
PREDICATES = {
    "upright": lambda state: abs(state[4]) < 0.1,  # angle small
    "near_ground": lambda state: state[1] < 0.3,  # y_pos small
    "moving_slow": lambda state: abs(state[2]) < 0.1 and abs(state[3]) < 0.1,  # velocity small
    "stable": lambda state: abs(state[4]) < 0.1 and abs(state[2]) < 0.1 and abs(state[3]) < 0.1,
}


def state_to_str(state):
    """Convert 8-dim LunarLander state to a string for the LM prompt."""
    fields = ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "ang_vel", "leg_l", "leg_r"]
    pairs = [f"{name}={val:.3f}" for name, val in zip(fields, state)]
    return "[" + ", ".join(pairs) + "]"


def build_prompt(state, predicate):
    """Build the LM prompt for checking a predicate on a state."""
    return (
        "You are a type checker for a robotics system.\n"
        f"Given the following state vector: {state_to_str(state)}\n"
        f'Is the predicate "{predicate}" TRUE, FALSE, or UNCERTAIN?\n'
        "Answer with exactly one of: TRUE / FALSE / UNCERTAIN."
    )


def load_lm(model_path=LOCAL_LM_PATH, device="cpu", dtype=torch.float16):
    """Load the small LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def lm_check(model, tokenizer, state, predicate, device="cpu", max_new_tokens=10):
    """Check a single (state, predicate) pair using the LM.

    Returns:
        label: "TRUE" / "FALSE" / "UNCERTAIN"
    """
    prompt = build_prompt(state, predicate)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Parse the response.
    text_upper = text.upper()
    if "TRUE" in text_upper and "FALSE" not in text_upper:
        return "TRUE"
    elif "FALSE" in text_upper:
        return "FALSE"
    elif "UNCERTAIN" in text_upper:
        return "UNCERTAIN"
    else:
        return "UNCERTAIN"  # fallback for unparseable output


def ground_truth(state, predicate):
    """Compute the ground truth label using the pre-registered predicate definition."""
    if predicate not in PREDICATES:
        raise ValueError(f"Unknown predicate: {predicate}")
    return "TRUE" if PREDICATES[predicate](state) else "FALSE"


def random_check(state, predicate, rng=None):
    """Random baseline: uniformly random TRUE/FALSE/UNCERTAIN."""
    if rng is None:
        rng = np.random.default_rng()
    return str(rng.choice(["TRUE", "FALSE", "UNCERTAIN"]))


def evaluate_arm(check_fn, states, predicates, name="arm"):
    """Evaluate an arm (LM / DLR / Random) on a set of (state, predicate) pairs."""
    correct = 0
    total = 0
    per_pred = {p: {"correct": 0, "total": 0} for p in predicates}
    for state, pred in zip(states, predicates):
        pred_label = check_fn(state, pred)
        gt_label = ground_truth(state, pred)
        # Map UNCERTAIN to a 50/50 guess (counted as half-correct).
        if pred_label == "UNCERTAIN":
            score = 0.5
        elif pred_label == gt_label:
            score = 1.0
        else:
            score = 0.0
        correct += score
        total += 1
        per_pred[pred]["correct"] += score
        per_pred[pred]["total"] += 1
    accuracy = correct / max(total, 1)
    return {
        "name": name,
        "accuracy": accuracy,
        "per_pred_acc": {p: v["correct"] / max(v["total"], 1) for p, v in per_pred.items()},
        "n": total,
    }


def dlr_baseline_check(state, predicate, threshold=0.7):
    """DLR baseline (mock): use a pre-registered predicate function and a
    learned threshold. This is a STAND-IN for the actual DLR MLP head;
    the real DLR baseline would load trained models from Project E.

    For the smoke test, we use the same predicate function as ground
    truth (which is what DLR learns to approximate). We add a small
    noise to simulate DLR's 95.5% accuracy (LunarLander env).
    """
    rng = np.random.default_rng()
    # DLR is 95.5% accurate on LunarLander; 4.5% noise flip.
    gt = PREDICATES[predicate](state)
    if rng.random() < 0.045:
        # Wrong prediction.
        return "TRUE" if not gt else "FALSE"
    return "TRUE" if gt else "FALSE"


def make_synthetic_states(n_states=20, seed=0):
    """Generate synthetic LunarLander-like 8-dim states for smoke test."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        # Synthetic state with all 8 fields in plausible ranges.
        x_pos = rng.uniform(-0.5, 0.5)
        y_pos = rng.uniform(0.0, 1.5)
        x_vel = rng.uniform(-1.0, 1.0)
        y_vel = rng.uniform(-1.0, 1.0)
        angle = rng.uniform(-0.5, 0.5)
        ang_vel = rng.uniform(-1.0, 1.0)
        leg_l = rng.uniform(0.0, 1.0)
        leg_r = rng.uniform(0.0, 1.0)
        states.append(np.array([x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_l, leg_r]))
    return states


def main():
    print("=" * 70)
    print("Project D H12 smoke test (small LM as DLR type checker)")
    print("=" * 70)
    print("NOTE: synthetic data, NOT the pre-reg H12 result.")
    print()

    n_states = 1  # truly minimal: 1 LM call
    states = make_synthetic_states(n_states=n_states, seed=42)
    predicates = ["upright"]
    # All (state, predicate) pairs.
    all_pairs = [(s, p) for s in states for p in predicates]
    all_states = [s for s, _ in all_pairs]
    all_preds = [p for _, p in all_pairs]
    print(f"Test set: {len(states)} states x {len(predicates)} predicates = {len(all_pairs)} pairs")
    print()

    # Arm 1: Load small LM.
    print("Loading Qwen2.5-1.5B-Instruct...")
    model, tokenizer = load_lm()
    print("Loaded.")
    print()

    # Arm 1: Small LM type checker.
    def lm_check_fn(state, predicate):
        return lm_check(model, tokenizer, state, predicate)
    print("Running LM type checker...")
    lm_result = evaluate_arm(lm_check_fn, all_states, all_preds, name="LM")
    print(f"LM accuracy: {lm_result['accuracy']:.3f}")
    print(f"  per-predicate: {lm_result['per_pred_acc']}")
    print()

    # Arm 2: DLR baseline (mocked, ~95.5% on LunarLander).
    dlr_result = evaluate_arm(dlr_baseline_check, all_states, all_preds, name="DLR")
    print(f"DLR accuracy: {dlr_result['accuracy']:.3f}")
    print(f"  per-predicate: {dlr_result['per_pred_acc']}")
    print()

    # Arm 3: Random baseline.
    rng = np.random.default_rng(0)
    def random_check_fn(state, predicate):
        return random_check(state, predicate, rng)
    random_result = evaluate_arm(random_check_fn, all_states, all_preds, name="Random")
    print(f"Random accuracy: {random_result['accuracy']:.3f}")
    print(f"  per-predicate: {random_result['per_pred_acc']}")
    print()

    print("=" * 70)
    print("Smoke test verdict (synthetic data only):")
    if lm_result["accuracy"] >= 0.85:
        print(f"  LM >= 0.85 threshold ({lm_result['accuracy']:.3f}) - promising for real H12")
    elif lm_result["accuracy"] > random_result["accuracy"] + 0.20:
        print(f"  LM > Random by {lm_result['accuracy'] - random_result['accuracy']:.3f} - LM is learning")
    else:
        print(f"  LM near random ({lm_result['accuracy']:.3f}); H12 may be REFUTED")
    print("=" * 70)


if __name__ == "__main__":
    main()