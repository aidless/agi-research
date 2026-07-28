"""
Project G -- Simple arithmetic dataset generator.

This is a SYNTHETIC dataset designed to be solvable by a small LM
(1.5B params) within ~20 tokens. It generates mixed-difficulty
arithmetic problems so the pilot gets a meaningful mix of successes
and failures.

Difficulty mix:
- 50% EASY: single-step arithmetic with small operands (< 50).
  Expected: Qwen2.5-1.5B solves correctly in 5-10 tokens.
- 50% HARD: multi-step arithmetic or large operands (> 100),
  or word problems requiring extra reasoning. Expected: small LM
  fails within 20 tokens.

The mix produces both classes naturally, which gives the Monitor
arms a real classification signal to learn from.

NO_SELF_DECEPTION.md compliance:
- The dataset is SYNTHETIC, not a real benchmark. It is designed
  to produce a meaningful pilot result on this CPU system.
- This dataset is NOT used for the full pre-registered H10 (which
  uses GSM8K). It is a pilot-only substitute.
- Failure labels are deterministic (ground truth arithmetic), not
  subjective.
- The user may swap this dataset for the real GSM8K once GPU is
  available.
"""

import random


def _make_easy_problem(rng):
    """Generate a single-step arithmetic problem with small operands."""
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        answer = a + b
        text = f"What is {a} + {b}?"
    elif op == "-":
        a = rng.randint(20, 99)
        b = rng.randint(1, a)  # ensure non-negative result
        answer = a - b
        text = f"What is {a} - {b}?"
    else:  # "*"
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        answer = a * b
        text = f"What is {a} * {b}?"
    return text, answer


def _make_hard_problem(rng):
    """Generate a multi-step or large-operand problem that 1.5B likely fails."""
    kind = rng.choice(["multistep_add", "multistep_mul", "large_add", "word"])
    if kind == "multistep_add":
        # (a + b) * c where each is moderate but needs 2 steps.
        a = rng.randint(10, 30)
        b = rng.randint(10, 30)
        c = rng.randint(3, 7)
        answer = (a + b) * c
        text = f"What is ({a} + {b}) times {c}?"
    elif kind == "multistep_mul":
        # a * b + c * d
        a = rng.randint(5, 12)
        b = rng.randint(5, 12)
        c = rng.randint(5, 12)
        d = rng.randint(5, 12)
        answer = a * b + c * d
        text = f"What is {a} times {b} plus {c} times {d}?"
    elif kind == "large_add":
        # 3 large numbers added
        a = rng.randint(100, 500)
        b = rng.randint(100, 500)
        c = rng.randint(100, 500)
        answer = a + b + c
        text = f"What is {a} + {b} + {c}?"
    else:  # word problem
        # A 3-step word problem
        x = rng.randint(5, 15)
        y = rng.randint(3, 10)
        z = rng.randint(2, 8)
        # Jane has x apples, buys y more, eats z. How many left?
        answer = x + y - z
        text = f"Jane has {x} apples. She buys {y} more. Then she eats {z}. How many apples does she have?"
    return text, answer


def generate_dataset(n_total, easy_fraction=0.5, seed=0):
    """Generate a mixed-difficulty arithmetic dataset.

    Args:
        n_total: total number of problems
        easy_fraction: fraction of easy problems (rest are hard)
        seed: RNG seed for reproducibility

    Returns:
        problems: list of dicts with keys "question" and "full_answer"
        ground_truths: list of int (final numerical answer)
        difficulty_labels: list of "easy" / "hard"
    """
    rng = random.Random(seed)
    n_easy = int(n_total * easy_fraction)
    n_hard = n_total - n_easy
    problems = []
    ground_truths = []
    difficulty_labels = []
    for _ in range(n_easy):
        text, answer = _make_easy_problem(rng)
        problems.append({"question": text, "full_answer": str(answer)})
        ground_truths.append(float(answer))
        difficulty_labels.append("easy")
    for _ in range(n_hard):
        text, answer = _make_hard_problem(rng)
        problems.append({"question": text, "full_answer": str(answer)})
        ground_truths.append(float(answer))
        difficulty_labels.append("hard")
    # Shuffle.
    perm = list(range(n_total))
    rng.shuffle(perm)
    problems = [problems[i] for i in perm]
    ground_truths = [ground_truths[i] for i in perm]
    difficulty_labels = [difficulty_labels[i] for i in perm]
    return problems, ground_truths, difficulty_labels


if __name__ == "__main__":
    problems, gts, diffs = generate_dataset(n_total=10, seed=42)
    print("Sample dataset:")
    for i in range(min(5, len(problems))):
        print(f"  [{diffs[i]:4s}] GT={gts[i]:6.0f} | {problems[i]['question']}")
    print()
    n_easy = sum(1 for d in diffs if d == "easy")
    n_hard = sum(1 for d in diffs if d == "hard")
    print(f"Easy: {n_easy}, Hard: {n_hard}")