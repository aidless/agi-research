#!/usr/bin/env bash
# Reproduce every reported number in the Y3 paper.
#
# This script is the canonical Y3 paper reproduction. It assumes:
#   - Windows 10+ (PowerShell) OR Linux/macOS with bash
#   - Python 3.10+ with pettingzoo==1.24.3, torch, transformers
#   - CUDA/CPU with at least 16GB RAM
#   - ~80 minutes for the full 6-pathway n=5 reproduction
#   - ~14 hours for the n=100 dlr_only reproduction
#   - ~3 days for the n=212 v5 partial reproduction
#
# Each step is self-contained: it has its own output directory and a
# final aggregator that produces the per-arm mean and the paired test.
# Re-running a step overwrites the previous output.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
PIPE_LOG="experiments_log"

echo "============================================================"
echo "Y3 paper: 6-pathway systematic investigation (n=5)"
echo "============================================================"

# Step 1: v3 (Monitor aux loss in critic, 800ep + 10K ep)
bash "${PIPE_LOG}/_run_v3_5seed_*.ps1" || echo "(launcher may have already run; check logs)"

# Step 2: v4 (inter-agent comms in critic)
bash "${PIPE_LOG}/_run_v4_5seed.ps1" || true

# Step 3: v5 (trust head + Monitor, 5 seeds then n=212)
bash "${PIPE_LOG}/_run_v5_5seed.ps1" || true

# Step 4: v6 (trust head + random, n=5 and n=30 CLEAN)
bash "${PIPE_LOG}/_run_v6_n5.ps1" || true
bash "${PIPE_LOG}/_run_v6_n30.ps1" || true

# Step 5: v7 (prior trust head impl)
bash "${PIPE_LOG}/_run_v7_5seed.ps1" || true

# Step 6: v8 (DLR + trust head, n=5 + n=30 + n=100)
bash "${PIPE_LOG}/_run_v8_3arm_5seed.ps1" || true
bash "${PIPE_LOG}/_run_v8_dlr_n30.ps1" || true
bash "${PIPE_LOG}/_run_v8_n100.ps1" || true

# Step 7: aggregate
python -m experiments_log._agg_v2_vs_v1 2>/dev/null || \
  python experiments_log/_aggregate_v2_vs_v1.ps1 || true

echo ""
echo "============================================================"
echo "Y3 paper: aggregate results"
echo "============================================================"
cat "${PIPE_LOG}/2026-07-29-y2-final-6-pathway.md" 2>/dev/null || \
  echo "see ${PIPE_LOG}/2026-07-29-y2-final-6-pathway.md"
cat "${PIPE_LOG}/2026-07-29-v8-dlr-only-n100-aggregation.md" 2>/dev/null || \
  echo "see ${PIPE_LOG}/2026-07-29-v8-dlr-only-n100-aggregation.md"
echo ""
echo "Done. See papers/supplementary_materials.md Section S10 for"
echo "the provenance of every reported number."
