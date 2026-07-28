"""
Project G -- Failure-label generator.

Defines what counts as a "failure" trace. The label definition is
separated from the trace generator so it can be reviewed in isolation
and changed without modifying the rollout collector.

CANDIDATE failure-label definitions (pre-registered selection):

1. **GSM8K-style**: trace ends in incorrect final answer (ground
   truth comparison). Clean, deterministic, no ambiguity. PRE-
   REGISTERED DEFAULT.
2. **Logical contradiction**: DLR verifier detects a logical
   contradiction in the trace (e.g., "x = 5" then "x = 7" without
   reconciliation). Requires DLR verifier; adds complexity.
3. **Length overflow**: trace exceeds a length budget (e.g., 512
   tokens). Crude but cheap.

The H10 pre-registration defaults to definition 1 (GSM8K-style).
For the synthetic smoke test, the label is provided externally
(failure_rate parameter in frozen_rollout_collector.py).

NO_SELF_DECEPTION.md compliance:
- The label definition is in this separate file, not embedded in the
  Monitor or rollout collector.
- The user can review and change the label definition without
  changing the architecture.
"""

# Default failure-label definition.
DEFAULT_FAILURE_LABEL = "gsm8k_final_answer_incorrect"


def make_failure_label_function(label_kind=DEFAULT_FAILURE_LABEL):
    """Return a function that maps (trace, ground_truth) -> 0/1.

    The returned function takes:
        trace_tokens: list[int]
        trace_text: str (decoded)
        ground_truth: any

    Returns:
        int: 1 if the trace is a failure, 0 otherwise.
    """
    if label_kind == "gsm8k_final_answer_incorrect":
        def f(trace_tokens, trace_text, ground_truth):
            # Extract final number from trace text.
            # This is a placeholder; the real implementation parses
            # the trace's final answer and compares to ground_truth.
            try:
                # Heuristic: find the last number in trace_text.
                import re
                numbers = re.findall(r"-?\d+\.?\d*", trace_text)
                if not numbers:
                    return 1  # no number = failure
                final = float(numbers[-1])
                return 0 if abs(final - float(ground_truth)) < 1e-3 else 1
            except Exception:
                return 1
        return f
    elif label_kind == "length_overflow":
        max_len = 512
        def f(trace_tokens, trace_text, ground_truth):
            return 1 if len(trace_tokens) > max_len else 0
        return f
    else:
        raise ValueError(f"unknown failure-label kind: {label_kind}")


if __name__ == "__main__":
    f = make_failure_label_function()
    # Smoke test: trace that ends in 42 vs ground truth 42.
    is_fail = f([1, 2, 3], "the answer is 42", 42)
    assert is_fail == 0, f"expected 0, got {is_fail}"
    # Smoke test: trace that ends in 99 vs ground truth 42.
    is_fail = f([1, 2, 3], "the answer is 99", 42)
    assert is_fail == 1, f"expected 1, got {is_fail}"
    print("Failure-label generator smoke test OK")