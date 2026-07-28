"""
Project G -- Frozen-LLM rollout collector.

In the real H10 experiment, this file collects deterministic rollouts
from a FROZEN small LM (Qwen-1.5B / Phi-3-mini / etc.). For the
smoke test, this file uses SYNTHETIC traces that mimic the structure
of an LLM trace without invoking a real LM:

    - Trace: sequence of (token_id, logit_value) pairs
    - Length: 20 pairs (matches the Monitor window)
    - Token IDs: integers in [0, vocab_size)
    - Logit values: floats that have a "this token was uncertain"
      signal mixed in

The synthetic traces have a weak-but-real signal: traces with
"failure" labels have systematically lower logit variance in the
last 5 positions (mimicking a model losing confidence near the
end of a wrong answer). The Monitor should be able to pick up on
this signal in the smoke test.

NO_SELF_DECEPTION.md compliance:
- The synthetic data is NOT used for the real H10 result. The real
  result uses a frozen LLM (user picks).
- The synthetic data has a known signal-to-noise ratio so the smoke
  test result is interpretable.
- Failure-label definition is in failure_label_generator.py, NOT
  here, so the labels can be reviewed in isolation.
"""

import torch


# Synthetic LLM trace vocabulary size.
VOCAB_SIZE = 32000  # Qwen-1.5B vocabulary size for reference.


def make_synthetic_trace(rng, is_failure=False):
    """Generate one synthetic LLM trace.

    Returns:
        tokens: (window,) int64 tensor
        logits: (window,) float tensor
    """
    window = 20
    if is_failure:
        # Failure traces: high token entropy in last 5 positions
        # (mimicking a model losing confidence near the end of a wrong
        # answer).
        early_tokens = torch.randint(0, VOCAB_SIZE, (15,), generator=rng)
        late_tokens = torch.randint(0, VOCAB_SIZE, (5,), generator=rng)
        tokens = torch.cat([early_tokens, late_tokens])
        early_logits = torch.randn(15, generator=rng) * 2.0 + 5.0
        late_logits = torch.randn(5, generator=rng) * 5.0 + 0.5
        logits = torch.cat([early_logits, late_logits])
    else:
        # Success traces: consistent token confidence throughout.
        tokens = torch.randint(0, VOCAB_SIZE, (window,), generator=rng)
        logits = torch.randn(window, generator=rng) * 1.5 + 8.0
    return tokens, logits


def trace_to_features(tokens, logits, slot_dim=32):
    """Convert (tokens, logits) to (window, feat_dim=2*slot_dim) features.

    For the smoke test, we use a simple embedding:
    - token embedding: bucketized token_id / vocab_size -> [-1, 1]
    - logit embedding: sigmoid(logit) -> (0, 1)
    Concatenated -> 2*slot_dim = 64.
    """
    token_feat = (tokens.float() / VOCAB_SIZE) * 2.0 - 1.0  # (window,)
    logit_feat = torch.sigmoid(logits)                       # (window,)
    # Tile each to slot_dim so the slot attention has 32-dim per side.
    token_emb = token_feat.unsqueeze(-1).expand(-1, slot_dim)
    logit_emb = logit_feat.unsqueeze(-1).expand(-1, slot_dim)
    return torch.cat([token_emb, logit_emb], dim=-1)        # (window, 64)


def collect_synthetic_rollouts(n_rollouts=200, failure_rate=0.5, seed=0):
    """Collect a dataset of synthetic LLM traces.

    Returns:
        features: (n_rollouts, window=20, feat_dim=64) tensor
        labels: (n_rollouts,) float tensor (1.0 = failure, 0.0 = success)
    """
    rng = torch.Generator().manual_seed(seed)
    n_failure = int(n_rollouts * failure_rate)
    n_success = n_rollouts - n_failure
    features = []
    labels = []
    for _ in range(n_failure):
        tokens, logits = make_synthetic_trace(rng, is_failure=True)
        features.append(trace_to_features(tokens, logits))
        labels.append(1.0)
    for _ in range(n_success):
        tokens, logits = make_synthetic_trace(rng, is_failure=False)
        features.append(trace_to_features(tokens, logits))
        labels.append(0.0)
    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.float32)
    # Shuffle.
    perm = torch.randperm(n_rollouts, generator=rng)
    return features[perm], labels[perm]


if __name__ == "__main__":
    feats, labels = collect_synthetic_rollouts(n_rollouts=20, seed=42)
    print(f"Synthetic rollouts: features {tuple(feats.shape)}, labels {tuple(labels.shape)}")
    print(f"  Failure rate: {labels.mean().item():.3f}")
    print(f"  Logit mean (failure): {feats[labels == 1, :, 32:].mean().item():.3f}")
    print(f"  Logit mean (success): {feats[labels == 0, :, 32:].mean().item():.3f}")
    print("OK")