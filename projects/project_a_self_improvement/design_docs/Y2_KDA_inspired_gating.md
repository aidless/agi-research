# Y2 Project A Design Doc: KDA-Inspired Channel-Wise Forget Gate

> Date: 2026-07-29
> Author: Liu Zewen (with Codex agent)
> Status: Design doc (no code yet). For Y2 implementation.
> Inspiration: Kimi K3 (Moonshot AI, 2026) Kimi Delta Attention (KDA).

---

## 1. Background

### 1.1 KDA channel-wise forget gate (from Kimi K3)

KDA recurrence (per head):
```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

Where:
- `alpha_t in (0,1)^d_k`: **channel-wise one-step retention factor**
  (forget gate per channel)
- `beta_t`: write strength
- `S_t in R^{d_k x d_v}`: recurrent state

The `Diag(alpha_t)` is the key innovation: **a per-channel scalar
forget signal** that decides what to retain from past state. This
is a learned, content-dependent signal (parameterized by a low-rank
projection of the input).

### 1.2 Y1 Project A Monitor (current)

The current Y1.3 Monitor is a uniform signal:
- 4 slots x 32 dim slot attention
- MLP head -> failure probability (single scalar)
- All 32 channels treated equally in the BCE loss

The Monitor's slot attention does **implicit channel weighting** (via
the attention coefficients) but the recurrent state is not explicitly
channel-wise gated.

## 2. Design proposal: Y2 Monitor with channel-wise forget gate

Add a per-channel forget gate to the Monitor's slot state update:

```
slot_state_t = Diag(g_alpha_t) slot_state_{t-1} + W_in new_features_t
g_alpha_t = sigmoid(W_alpha x_t) in (0,1)^{slot_dim}
output = MLP(slot_state_t)
```

Where:
- `slot_state_t in R^{n_slots x slot_dim}`: per-slot latent state.
- `Diag(g_alpha_t)`: per-channel forget gate.
- `W_alpha x_t`: input-conditioned gate parameters.
- `W_in new_features_t`: new evidence to update the state.

The Monitor's failure probability is the output of an MLP over the
final slot state.

## 3. Why this could help

### 3.1 Adaptive history length

A **uniform** forget signal (e.g., exponential decay with fixed
rate) treats all channels the same. A **channel-wise** signal lets
the Monitor:
- Retain long-term memory for "stable" channels (low alpha).
- Forget quickly for "transient" channels (high alpha).
- Adapt the retention per-trajectory (input-conditioned).

This is closer to how a real failure signal works: some failure modes
manifest gradually (need long memory), others are sudden (need
short memory).

### 3.2 Comparison to current Y1.3

Current Y1.3:
- 4 slots x 32 dim, no explicit forgetting.
- BCE loss on failure_prob, all 32 channels equally weighted.

Proposed Y2:
- 4 slots x 32 dim, per-channel forget gate.
- The slot state is updated each timestep with new evidence.
- BCE loss on failure_prob, but the channel-wise gate lets the
  model selectively attend to past evidence.

Expected outcome: better failure prediction on long trajectories
where the failure mode is heterogeneous.

## 4. Pre-registered hypothesis: H13

**H13**: A Monitor with a channel-wise forget gate (KDA-inspired)
achieves higher failure-prediction AUROC than the current Y1.3
Monitor (uniform slot attention) on long LunarLander trajectories
(>= 200 steps) where the failure mode is heterogeneous.

**Decision rule** (pre-registered before any data collection):
- VALIDATED if KDA-Monitor AUROC > Y1.3-Monitor AUROC by delta > 0.05
  AND Welch t > 2.0 on n=5 seeds.
- REFUTED if KDA-Monitor AUROC < Y1.3-Monitor AUROC.
- INCONCLUSIVE if delta < 0.05 or Welch t < 2.0.

**Pre-registered sample size**: n=5 seeds, 200 trajectories per seed,
long trajectories (>= 200 steps).

**Pre-registered compute**: ~30 minutes per seed on CPU. Total: 2.5
hours for n=5. Plus Monitor training: ~10 minutes. Total pilot: ~3
hours on CPU.

## 5. What is novel vs existing work

### 5.1 vs KDA
KDA is a linear-time **self-attention** with channel-wise gating,
designed for language modeling at 2.8T parameter scale. Our Monitor
is a small **failure-prediction** network for RL agents. Different
domain, different scale, different task.

### 5.2 vs RNN gating
Standard RNN gating (LSTM, GRU) is per-cell (scalar) or per-feature
(per-channel vector). Our design is per-channel within a slot, which
is closer to the KDA / Mamba-2 family.

### 5.3 vs Monitor literature
The Monitor literature (Park 2024, Lightman 2023) treats the failure
signal as a single scalar; per-channel gating has not been explored
in this context. If H13 validates, this is a novel contribution.

## 6. Implementation outline

```python
class KDAMonitor(nn.Module):
    def __init__(self, n_slots=4, slot_dim=32, feat_dim=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        # Per-channel gate projection: input -> per-channel forget
        self.gate_proj = nn.Linear(feat_dim, slot_dim)
        # New-evidence projection
        self.in_proj = nn.Linear(feat_dim, slot_dim)
        # Slot state: per-slot latent
        self.register_buffer("slot_state", torch.zeros(n_slots, slot_dim))
        # Failure head
        self.head = nn.Sequential(
            nn.Linear(n_slots * slot_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def step(self, x):
        # x: (feat_dim,)
        g = torch.sigmoid(self.gate_proj(x))  # (slot_dim,)
        new = self.in_proj(x)  # (slot_dim,)
        # Per-channel forget + new evidence
        self.slot_state = self.slot_state * g.unsqueeze(0) + new.unsqueeze(0)
        return self.head(self.slot_state.flatten()).squeeze()

    def forward(self, trajectory):
        # trajectory: (T, feat_dim)
        for t in range(trajectory.shape[0]):
            self.step(trajectory[t])
        return torch.sigmoid(self.head(self.slot_state.flatten()).squeeze())
```

This is a **minimal** design for testing H13. The actual Y2
implementation may include slot attention, BCE loss, etc.

## 7. Relation to existing Archimedes work

### 7.1 Project A Y1.3 baseline
The H13 hypothesis is a **direct extension** of Y1.3: same input
format, same loss, same architecture except for the channel-wise
gate. The diff is minimal, so the comparison is clean.

### 7.2 Project C slot attention
Y2 Monitor's slot state is similar to Project C's slot world model
state, but the gating mechanism is different (per-channel forget
vs. slot attention coefficient). If H13 validates, the two could
be unified: a single slot world model that doubles as a Monitor.

### 7.3 Project G H10 (REFUTED)
H13 is in the same family (decoupled Monitor for failure prediction)
but uses a different mechanism (channel-wise gate) than H10 (joint
vs frozen Monitor training on LLM traces). They are independent
hypotheses; H13 is a Y2 Project A direction, not a Project G revival.

## 8. Risk and honest framing

### 8.1 Risks
- The KDA gate is not a silver bullet; it may not help if the
  failure signal is not channel-structured.
- The Y1.3 baseline is already strong (15 seeds, p<0.001); beating
  it by 0.05+ AUROC is a high bar.
- CPU budget for full H13 is ~3 hours; if we cannot afford that, the
  pilot may be inconclusive.

### 8.2 Honest framing
This is a **design doc**, not a claim. The KDA-inspired gate is a
hypothesis, not a result. The H13 pre-registration is the rigorous
test; if it validates, that's a real contribution; if it refutes, the
Y1.3 baseline remains the canonical result.

## 9. Next step

For Y2 implementation (not in current session):
1. Implement `KDAMonitor` per the outline above.
2. Add H13 pre-registration file with hard decision rule.
3. Run n=5 seeds at 200 trajectories/seed on LunarLander.
4. Compute Welch t-test for KDA-Monitor vs Y1.3-Monitor AUROC.
5. If H13 validates, write a Y2 paper on the contribution.

This design doc is sufficient as a Y2 roadmap; the actual
implementation can begin when Y2 funding / time allows.

---

*Design doc prepared 2026-07-29 by Codex agent. Inspiration: Kimi K3
KDA channel-wise forget gate [46]. For Y2 Project A implementation.*