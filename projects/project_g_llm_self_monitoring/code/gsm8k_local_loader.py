"""Project G -- GSM8K local arrow loader (no `datasets` library needed).

This module reads the GSM8K test set directly from the cached arrow
file at F:\hf_cache\datasets\openai___gsm8k\...\gsm8k-test.arrow using
pyarrow.ipc. It bypasses the `datasets` library (not installed in the
TRAE Python environment) and supports seed-based sampling so different
H10 seeds see different problems.

NO_SELF_DECEPTION.md compliance:
- The loader reads the SAME cached arrow file that the `datasets`
  library would have used; no data transformation.
- Seed-based sampling is deterministic (uses random.Random(seed)).
- The ground truth is extracted from the `#### N` pattern in the
  answer field, exactly matching the existing extract_final_number.

Usage:
    from gsm8k_local_loader import load_gsm8k_local
    problems, gts, labels = load_gsm8k_local(n_samples=8, seed=42)
"""

import os
import re
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_CACHED_ARROW = Path(
    r"F:\hf_cache\datasets\openai___gsm8k"
    r"\main\0.0.0\740312add88f781978c0658806c59bc2815b9866"
    r"\gsm8k-test.arrow"
)


def _read_arrow(path):
    """Read an IPC arrow stream file and return parallel question/answer lists."""
    with open(path, "rb") as f:
        r = ipc.open_stream(f)
        tbl = pa.Table.from_batches(list(r))
    return tbl["question"].to_pylist(), tbl["answer"].to_pylist()


def _extract_gt(answer_text):
    """Extract final numerical answer from GSM8K answer field via #### pattern."""
    m = re.search(r"####\s*(-?\d+\.?\d*)", answer_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def load_gsm8k_local(split="test", n_samples=None, seed=0, cache_path=None):
    """Load GSM8K from local cached arrow file with seed-based sampling.

    Args:
        split: only "test" is supported by the current cache (train not
            needed for the H10 protocol).
        n_samples: number of problems to return (None = all 1319).
        seed: RNG seed for sampling. Different seeds produce different
            8-problem subsets; identical seeds produce identical subsets.
        cache_path: override the cached arrow path (default: train/default cache).

    Returns:
        problems: list of dicts with keys "question" and "full_answer"
        ground_truths: list of float (or None if not parseable)
        difficulty_labels: list of "gsm8k" placeholder strings
    """
    path = Path(cache_path) if cache_path else DEFAULT_CACHED_ARROW
    if not path.exists():
        raise FileNotFoundError(f"GSM8K cached arrow not found at {path}")
    questions, answers = _read_arrow(path)
    n_total = len(questions)
    if n_samples is not None and n_samples < n_total:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(n_total), n_samples))
        questions = [questions[i] for i in idx]
        answers = [answers[i] for i in idx]
    problems = [{"question": q, "full_answer": a} for q, a in zip(questions, answers)]
    ground_truths = [_extract_gt(a) for a in answers]
    return problems, ground_truths, ["gsm8k"] * len(problems)


if __name__ == "__main__":
    p, g, d = load_gsm8k_local(n_samples=8, seed=42)
    for i in range(min(3, len(p))):
        print(f"[{d[i]}] GT={g[i]} | Q: {p[i]['question'][:80]}...")
    print()
    p2, g2, _ = load_gsm8k_local(n_samples=8, seed=43)
    same = (p[0]["question"] == p2[0]["question"])
    print(f"seed 42 vs seed 43 first-question same? {same}")
    p3, g3, _ = load_gsm8k_local(n_samples=8, seed=42)
    same42 = (p[0]["question"] == p3[0]["question"])
    print(f"seed 42 vs seed 42 (re-runnable) first-question same? {same42}")
