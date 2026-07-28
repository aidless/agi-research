"""
Project G -- Slot-Monitor adapted to LLM traces.

Architecture (mirrors Project A Slot-Monitor, adapted to text traces):

    LLM trace (token, logit) pairs
        |
        v
    Slot attention (4 slots, 32 dim, 20-pair window)
        |
        v
    MLP head (32 -> 16 -> 1)
        |
        v
    failure probability (sigmoid)

The Slot-Monitor is trained with BCE loss on (failure_prob,
is_failure) pairs. The "failure" label is a deterministic function
of the trace (e.g., final-answer correctness on GSM8K).

This file implements the architecture only. The H10 pre-registered
experiment is defined in experiments_log/2026-07-28-PRE-REGISTERED-H10.md
and is NOT run by this smoke test. Use h10_smoke.py for the smoke
test.

NO_SELF_DECEPTION.md compliance:
- Architecture mirrors Project A Slot-Monitor (validated AUROC 0.989).
- No synthetic data in this file; the LLM trace generator lives in
  frozen_rollout_collector.py.
- Failure-label generator is a separate file (failure_label_generator.py)
  so the label definition is reviewable in isolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Slot attention over (token, logit) pairs.

    Mirrors Locatello et al. (2020) slot attention, applied to 1-D
    sequence input (last 20 token-logit pairs).
    """

    def __init__(self, n_slots=4, slot_dim=32, n_iter=3):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iter = n_iter
        self.norm_input = nn.LayerNorm(slot_dim * 2)  # token + logit
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim * 2, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim * 2, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, x):
        """x: (batch, seq_len, 2*slot_dim) -> (batch, n_slots, slot_dim)"""
        B, T, _ = x.shape
        slots = torch.randn(B, self.n_slots, self.slot_dim, device=x.device)
        x_norm = self.norm_input(x)
        for _ in range(self.n_iter):
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)
            k = self.to_k(x_norm)
            v = self.to_v(x_norm)
            attn_logits = torch.einsum("bsd,btd->bst", q, k) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=1)  # over slots
            updates = torch.einsum("bst,btd->bsd", attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots.reshape(-1, self.slot_dim),
            ).reshape(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class LLMSlotMonitor(nn.Module):
    """Slot-Monitor architecture for LLM traces.

    Input: (batch, seq_len=20, features=64)  # 20 (token_emb, logit_emb) pairs
    Output: (batch,) failure probability
    """

    def __init__(self, n_slots=4, slot_dim=32, window=20, feat_dim=64):
        super().__init__()
        assert feat_dim == slot_dim * 2, "feat_dim must equal slot_dim * 2"
        self.window = window
        self.slot_dim = slot_dim
        self.slot_attn = SlotAttention(n_slots=n_slots, slot_dim=slot_dim)
        # Aggregate slot_dim * n_slots -> 16 -> 1
        self.head = nn.Sequential(
            nn.Linear(slot_dim * n_slots, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, trace):
        """trace: (batch, window, feat_dim=64) -> (batch,) failure prob"""
        slots = self.slot_attn(trace)             # (B, n_slots, slot_dim)
        flat = slots.reshape(slots.size(0), -1)   # (B, n_slots * slot_dim)
        logit = self.head(flat).squeeze(-1)       # (B,)
        return torch.sigmoid(logit)


if __name__ == "__main__":
    # Smoke test: random input -> output shape check.
    monitor = LLMSlotMonitor()
    x = torch.randn(8, 20, 64)  # batch=8, window=20, feat=64
    y = monitor(x)
    assert y.shape == (8,), f"expected (8,), got {y.shape}"
    assert (y >= 0).all() and (y <= 1).all(), "sigmoid output out of [0,1]"
    print(f"LLMSlotMonitor smoke test: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"  Param count: {sum(p.numel() for p in monitor.parameters())}")
    print(f"  Output sample: {y[:4].tolist()}")
    print("OK")