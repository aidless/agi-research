# STUDY_NOTES: SOTA Open-Source LLMs (Kimi K3 + FlashKDA)

> Date: 2026-07-29
> Sources:
> - C:\Users\Administrator\Downloads\Kimi-K3-main.zip (Moonshot AI Kimi K3)
> - C:\Users\Administrator\Downloads\FlashKDA-master.zip (Moonshot AI FlashKDA)
> Extracted to E:\agi-research\study_materials\
> Reader: Codex agent (with NO_SELF_DECEPTION.md discipline)

---

## 1. Kimi K3: Overview

Kimi K3 is an open-weight, native multimodal agentic model from Moonshot AI:

| Attribute | Value |
|-----------|-------|
| Total parameters | 2.8T |
| Activated parameters | 104B |
| Layers | 93 (1 dense + 92 MoE) |
| Attention layers | 69 KDA + 24 Gated MLA |
| Hidden dim | 7168 |
| Attention heads | 96 |
| Latent MoE dim | 3584 |
| MoE experts | 896 (16 routed + 2 shared per token) |
| Vocabulary | 160K |
| Context length | 1,048,576 (1M) |
| Vision encoder | MoonViT-V2 (401M params) |
| Quantization | MXFP4 weights / MXFP8 activations (QAT from SFT) |
| Modality | Text + Image |
| Released | Open weights under Kimi K3 License |

**Key positioning**: "world's first open 3T-class model" — frontier
intelligence across long-horizon coding, knowledge work, reasoning.

---

## 2. Kimi K3 Architecture

### 2.1 Hybrid Attention (3:1 KDA / Gated MLA)

Each block contains **3 KDA layers** followed by **1 Gated MLA layer**.
A final Gated MLA at the end of the backbone ensures global attention
at the deepest layer.

Rationale:
- KDA: efficient linear-time attention for token mixing over long
  sequences (1M context).
- Gated MLA: full-rank gated multi-head latent attention for
  selective high-capacity attention.

### 2.2 Kimi Delta Attention (KDA)

KDA extends the delta-rule recurrence with a channel-wise forget gate.

**Core recurrence** (per head, single token):

```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

Where:
- `alpha_t in (0,1)^d_k`: channel-wise one-step retention factor (forget gate)
- `beta_t in (0,1)`: write strength for delta-rule
- `q_t, k_t, v_t`: query, key, value (head dim)
- `S_t in R^{d_k x d_v}`: recurrent state

**Per-head projections**:
- `q, k`: L2Norm(Swish(ShortConv(W_qk * x)))
- `v`: Swish(ShortConv(W_v * x))
- `beta`: Sigmoid(W_beta * x) in (0,1)
- `z_alpha = W_up_alpha * x + b_alpha` (low-rank projection)

### 2.3 Kimi K3 innovations vs Kimi Linear

**Lower-bounded log-decay**:
- Kimi Linear: `g = -e^A * Softplus(z)` (unbounded negative)
- Kimi K3: `g = g_min * Sigmoid(e^A * z)` with `g_min = -5`
- Benefit: prevents numerical overflow; allows dense Tensor Core
  MMA in chunkwise computation (vs explicit position-pair).

**Chunkwise parallel form**:
- Recurrent across chunks (size C), parallel within each chunk.
- KDA in K3: chunkwise state update + per-tile dense matmul.

### 2.4 Attention Residuals (AttnRes)

Standard residual connections compress all prior info into one state.
AttnRes applies attention over depth:
- Each layer l has learnable pseudo-query `q_l = w_l in R^d`.
- Keys/values from embedding + all preceding layer outputs.
- `h_l = sum_{i<l} alpha_{i->l} * v_i` where `alpha_{i->l} = softmax(q_l, k_i)`.
- O(L^2 * d) computation, but L < 100 so affordable.

### 2.5 Stable LatentMoE

Three components to stabilize MoE training:
1. **RMSNorm before up-projection** (scale control).
2. **SiTU-GLU** activation: Sigmoid-Tanh-Unit GLU, bounded
   (`|f(x)| <= beta_1 * beta_2`), suppresses activation explosion.
3. **Quantile Balancing (QB)**: load-balancing rule (alternative
   to standard aux-loss), balances expert utilization.

Shared experts (2 per layer) + routed experts (16 of 896 selected).

### 2.6 NoPE (No Positional Encoding)

Kimi K3 uses **no explicit positional embedding**. Position info
comes implicitly from KDA's gating/decay mechanism. Extrapolates
directly to 1M context without RoPE rescaling/interpolation.

---

## 3. Kimi K3 Training

### 3.1 Pre-training
- **Context**: starts at 8k tokens, extends to 64k.
- **Data cleaning**: exact + fuzzy dedup, perceptual hash for video,
  classifier-based quality filter, structural validation.
- **Long-context synthesis**: permute + concatenate multimodal
  documents to create long-range dependency tasks.

### 3.2 Long-context extension (to 1M)
- Progressive context extension.
- NoPE extrapolates directly without RoPE modification.
- Synthetic long-context data so model can't degenerate into
  local patterns.

### 3.3 Post-training: Agentic RL
- **Partial rollout scheme**: pause generation as soon as lambda *
  N * K trajectories complete, allowing stragglers to continue
  in next iteration.
- **Per-token regularization**: constrains policy updates within
  localized neighborhood, tolerating extreme off-policy.
- Multiple domains: coding, general tool use, web development,
  agentic search, professional workflows, agentic chart understanding,
  agentic visual puzzles, kernel optimization.

### 3.4 Thinking mode
- Always enabled by default.
- Returns `reasoning_content` field.
- `reasoning_effort` configurable: "low" / "high" / "max".

---

## 4. FlashKDA: Kernel Implementation

FlashKDA is the CUTLASS-based CUDA kernel for Kimi Delta Attention,
developed by Moonshot AI for their own model deployment. Open-sourced
for the community.

### 4.1 Chunk Size Selection: 16 (not 64)

| Reasoning | Benefit |
|-----------|---------|
| Numerical range fits in bf16 | No intra-chunk rescaling tricks |
| Cheap 16x16 matrix inverse | Neumann series expansion directly |
| SM80-only MMA path | Portable across modern NVIDIA GPUs |

### 4.2 Two-Kernel Split (15%+ speedup)

- **K1** (token-parallel, grid = N x H x num_chunks):
  g activation -> L2 norm -> decay apply -> L/Mqk construction
  -> matrix inversion.
- **K2** (head-parallel only, grid = N x H):
  chunk-by-chunk delta-rule recurrence -> output projection
  -> running state accumulation.

Single-kernel prototypes had SMs idle waiting on the K2 stage's
lower parallelism.

### 4.3 Numerical Precision

- **bf16 on-chip recurrent state** + **fp32 FMA updates**: cuts
  shared-memory footprint in half; no measurable accuracy loss.
- **Sigmoid via PTX tanh.approx.f32**: faster than generic exp.
- **FP16 16x16 matrix inverse**: elements bounded in [-1, 1] so
  fp16 dynamic range sufficient; avoids bf16->fp32 casts.

### 4.4 Other Optimizations

- **Base-2 exponent**: `ex2.approx.ftz.f32` instead of `exp`; higher
  throughput.
- **__launch_bounds__(256, 8)**: trade register spilling for SM
  occupancy.
- **MOVM_T**: register-file transposes eliminate shared-memory
  round trips.

### 4.5 Integration

FlashKDA is dispatched automatically by `flash-linear-attention`
library's `chunk_kda`. Set `FLA_FLASH_KDA=0` to fall back to
Triton path.

---

## 5. Relevance to Archimedes Project

### 5.1 Direct relevance

| Archimedes component | Kimi K3 / FlashKDA relevance |
|----------------------|-------------------------------|
| Project A (decoupled Monitor) | KDA is a linear attention with gating — could be a Monitor target, but we don't use linear attention. |
| Project C (slot world model) | Slot attention is from earlier work; Kimi K3 uses KDA, not slots. Different paradigm. |
| Project D (language-as-type-system) | Kimi K3's "thinking" content suggests LM can do structured reasoning; H12 (LM as DLR type checker) is on the right track. |
| Project E (DLR) | DLR is a verifier; Kimi K3's AttnRes provides pseudo-query attention for selective retrieval. Hybrid: DLR verifies, AttnRes-like mechanism retrieves. |
| Project F (multi-agent) | Kimi K3's partial rollout scheme is exactly what multi-agent async training needs. |

### 5.2 Indirect relevance

1. **KDA's gating is the same idea as our Monitor**: the alpha_t
   channel-wise forget gate decides what to remember from past
   state. Our Monitor's "frozen vs joint" question is about *who*
   decides what to remember — Kimi K3's KDA picks a *fixed* gating
   parameterization but trains everything jointly.

2. **AttnRes is a "depth-wise attention"**: each layer attends
   to all preceding layers. We use slot attention across time, but
   AttnRes is across depth. Conceptually related.

3. **NoPE via KDA gating/decay**: position info encoded implicitly
   in the recurrent state. Our Monitor's frozen-then-evaluated
   approach is similar: temporal info is in the frozen policy's
   rollouts, not in positional embeddings.

4. **Stable LatentMoE's Quantile Balancing**: a smarter load
   balancing rule for expert routing. Relevant to Project F's
   multi-agent (if we add expert-like routing per agent).

### 5.3 What we can NOT reproduce

- 2.8T parameter model training: we have CPU + 1.5B Qwen2.5 only.
- FlashKDA kernels: requires CUDA + SM90+ GPU.
- MXFP4 / MXFP8 quantization: requires compatible hardware.
- Partial rollout RL at scale: requires distributed training infra.

### 5.4 What we CAN learn / borrow

1. **H12 (LM as DLR type checker)**: still on track. Kimi K3's
   structured reasoning capabilities suggest 7B+ LMs could do this
   task. We tested with 1.5B and got LM = Random. With 7B, the
   result might be different. (But 7B may not fit on CPU.)

2. **Channel-wise forget gates** for our Monitor signals: a small
   modification to consider in Y2 (Project A).

3. **NoPE philosophy**: our Monitor signals are also "position-free"
   (we look at failure probability, not absolute trajectory position).
   The KDA gating/decay pattern is a useful conceptual parallel.

4. **Synthetic long-context data**: relevant to Y2 Project F
   multi-agent if we need long interaction histories.

---

## 6. Honest gaps in this study

- Did NOT run FlashKDA kernels (no GPU).
- Did NOT load Kimi K3 weights (too large for CPU).
- Did NOT verify quantitative claims (e.g., 2.5x scaling efficiency
  over Kimi K2) by reproducing.
- Did NOT compare KDA's theoretical complexity vs standard
  softmax attention mathematically.

This is a **study notes** document, not an empirical comparison. For
empirical validation, would need GPU access + much more compute.

---

## 7. Citations to add to Archimedes thesis

If the Archimedes thesis is updated with current SOTA context:

- KDA: Moonshot AI tech report (2026).
- AttnRes: paper citation [58] in Kimi K3 tech report.
- SiTU-GLU: paper citation in tech report.
- FlashKDA: GitHub repo (MoonshotAI/FlashKDA).
- NoPE for linear attention: Mamba paper (Gu & Dao, 2023) cited in
  Kimi K3 tech report as prior work for the gating parameterization.

These are outside the scope of this study but worth tracking for
future thesis updates.

---

*Study notes prepared 2026-07-29 by Codex agent. Source files:
study_materials/FlashKDA/ and study_materials/Kimi-K3/. NO_SELF_
DECEPTION.md compliance: honest about what was studied vs what was
empirically verified.*