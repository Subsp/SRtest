#!/usr/bin/env bash
# =============================================================================
# Task 0.2 – VGGT Geometry Fidelity Test  (quick launcher)
# =============================================================================
# Step 1:  Generate oracle depths via mip-splatting (run once, ~30 min/scene)
# Step 2:  Run VGGT inference and compare vs oracle
# =============================================================================
set -euo pipefail

DATA_ROOT="${1:-/path/to/mipnerf360}"
ORACLE_ROOT="${2:-./results/task02/oracle}"
OUTPUT_DIR="${3:-./results/task02}"
DEVICE="${4:-cuda}"
GPU_ID="${5:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " Task 0.2 – VGGT Geometry Fidelity Test"
echo "============================================================"
echo " Data root   : ${DATA_ROOT}"
echo " Oracle root : ${ORACLE_ROOT}"
echo " Output dir  : ${OUTPUT_DIR}"
echo " Device      : ${DEVICE}"
echo ""

# ── Step 1: Oracle depth (skip if already done) ───────────────────────────────
if [ ! -f "${ORACLE_ROOT}/garden/train/ours_30000/depth/done.marker" ]; then
    echo "[Step 1/2] Generating oracle depths via Mip-Splatting …"
    echo "           This takes ~30 min per scene on a single GPU."
    bash "${SCRIPT_DIR}/task02_oracle_train.sh" \
        "${DATA_ROOT}" "${ORACLE_ROOT}" "${GPU_ID}"
    # Mark completion
    for SCENE in garden kitchen bonsai room counter; do
        mkdir -p "${ORACLE_ROOT}/${SCENE}/train/ours_30000/depth"
        touch "${ORACLE_ROOT}/${SCENE}/train/ours_30000/depth/done.marker"
    done
else
    echo "[Step 1/2] Oracle depths already found, skipping training."
fi

# ── Step 2: VGGT inference + comparison ──────────────────────────────────────
echo ""
echo "[Step 2/2] Running VGGT inference and depth comparison …"
python "${SCRIPT_DIR}/task02_vggt_geometry.py" \
    --data_root   "${DATA_ROOT}" \
    --oracle_root "${ORACLE_ROOT}" \
    --output_dir  "${OUTPUT_DIR}" \
    --device      "${DEVICE}" \
    --n_frames    8

echo ""
echo "Done. Results in ${OUTPUT_DIR}"
