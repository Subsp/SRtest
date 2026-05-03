#!/usr/bin/env bash
# =============================================================================
# Task 0.2 – Oracle Depth Generation via vanilla Mip-Splatting
# =============================================================================
# Trains Mip-Splatting on FULL-resolution (images_2) images for each scene
# and renders depth maps to be used as oracle depth in task02_vggt_geometry.py
#
# Usage:
#   bash task02_oracle_train.sh <mipnerf360_root> <oracle_output_root> [gpu_id]
#
# Example:
#   bash task02_oracle_train.sh /data/mipnerf360 ./results/task02/oracle 0
#
# Output layout:
#   oracle_output_root/
#     <scene>/
#       train/
#         ours_30000/
#           depth/
#             <frame>.npy    ← depth maps for task02_vggt_geometry.py
# =============================================================================

set -euo pipefail

MIPNERF360_ROOT="${1:?Usage: $0 <mipnerf360_root> <oracle_output_root> [gpu_id]}"
ORACLE_ROOT="${2:?Usage: $0 <mipnerf360_root> <oracle_output_root> [gpu_id]}"
GPU_ID="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIP_ROOT="$(realpath "${SCRIPT_DIR}/../mip-splatting")"
SCENES=(garden kitchen bonsai room counter)

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================================"
echo " Task 0.2 Oracle Depth – Mip-Splatting Training"
echo "============================================================"
echo " MipNeRF360 root : ${MIPNERF360_ROOT}"
echo " Oracle output   : ${ORACLE_ROOT}"
echo " GPU             : ${GPU_ID}"
echo " Scenes          : ${SCENES[*]}"
echo ""

for SCENE in "${SCENES[@]}"; do
    SCENE_DIR="${MIPNERF360_ROOT}/${SCENE}"
    OUT_DIR="${ORACLE_ROOT}/${SCENE}"

    if [ ! -d "${SCENE_DIR}" ]; then
        echo "[SKIP] Scene not found: ${SCENE_DIR}"
        continue
    fi

    echo "------------------------------------------------------------"
    echo " Training ${SCENE} …"
    echo "------------------------------------------------------------"

    # ── Train Mip-Splatting (HR images, 30k iterations) ──────────────────────
    python "${MIP_ROOT}/train.py" \
        -s "${SCENE_DIR}" \
        -m "${OUT_DIR}" \
        --images images_2 \
        --iterations 30000 \
        --test_iterations 30000 \
        --save_iterations 30000 \
        --checkpoint_iterations 30000 \
        --densify_until_iter 15000 \
        --position_lr_max_steps 30000 \
        --quiet

    echo " Training done → ${OUT_DIR}"

    # ── Render depth maps ─────────────────────────────────────────────────────
    echo " Rendering depth maps …"
    python "${MIP_ROOT}/render.py" \
        -m "${OUT_DIR}" \
        --skip_test \
        --quiet

    # ── Convert rendered depth to .npy ───────────────────────────────────────
    echo " Converting depth renders to .npy …"
    python "${SCRIPT_DIR}/task02_oracle_render.py" \
        --model_dir "${OUT_DIR}" \
        --scene_dir "${SCENE_DIR}" \
        --output_dir "${OUT_DIR}/train/ours_30000/depth"

    echo " Done: ${SCENE}"
done

echo ""
echo "============================================================"
echo " All oracle depths saved to: ${ORACLE_ROOT}"
echo " Next step: python task02_vggt_geometry.py \\"
echo "              --data_root ${MIPNERF360_ROOT} \\"
echo "              --oracle_root ${ORACLE_ROOT}"
echo "============================================================"
