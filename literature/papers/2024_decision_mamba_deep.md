# Decision Mamba (various, 2024+)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **MEDIUM** (recent, several variants exist, paper landscape
  moving fast)
> One-line: Replace Transformer self-attention with state-space (Mamba)
> sequence mixing. Linear-time inference. Strong performance on long-context
> decision-making tasks (Atari, robotic control, long-horizon RL).

## Problem
Transformers (used in Decision Transformer, Gato, etc.) scale O(N^2) in context
length. Long-horizon RL demands long context (10K+ state-action pairs).

## Method
Mamba (Gu & Dao 2023) introduced selective state-space models:
- Linear-time inference (vs quadratic for self-attention).
- Input-dependent gating (only some state dimensions updated per step).
- Hardware-efficient on modern GPUs (kernel fusion).

Decision Mamba applies this to offline RL:
- Replace Transformer backbone with Mamba block.
- Train on (state, action, reward, return-to-go) sequences.
- Inference: predict next action autoregressively.

## Empirical result
- Atari 100k: Decision Mamba matches or beats Decision Transformer.
- Long-context: Mamba scales linearly where Transformer OOMs.
- Throughput: 5x faster training than Transformer-baseline.
- Strong on partial observability (long memory).

## Criticisms (specific)
1. **Memory of state dimensions has limited bandwidth** vs attention.
2. **Sequence-to-sequence alignment** still required for multi-modal.
3. **Recurrent hidden state is opaque** - hard to interpret.
4. **Sparse adoption in industry** still; Transformers dominate.

## Connection to our program
For Project A monitor:
- Monitor history vector (concat of (obs, action, reward) over time)
  could be processed by Mamba rather than MLP. Sub-linear latency over
  long trajectories.

For Project C (slot-WM):
- The dynamics model could use Mamba for temporal mixing of slots.
- Faster rollout generation = more reasoning depth in same compute.

For Project D (language types):
- LLM on slots could use Mamba as backbone for longer context windows.

## Confidence
MEDIUM.

## Related
- Mamba (Gu & Dao 2023) - foundational
- RWKV (Peng 2023) - parallel RNN
- Jamba (Lieber 2024) - hybrid Mamba-Transformer
- S4 (Gu 2022) - structured state spaces
- Decision Transformer (Chen 2021)
- Trajectory Transformer (Janner 2021)
- Q-Transformer (Kumar 2023)

## Status
- cited in Project A Related Work (history encoder option)
- cited in Project C (dynamics model option)
- future: benchmark Mamba backbone for world model
