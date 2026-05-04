#!/usr/bin/env bash
# =============================================================================
# Kitchen：HR Head 训练（oracle 蒸馏 + priors）；显存友好（每步 1 视角 forward）
#
#   bash run_train_hr_head_kitchen.sh
#
# 先确保 ORACLE_DIR 下已有对应帧的 oracle .npy（task02 oracle 管线）。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

: "${SCENE_ROOT:=/root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_lr_ablation_v1/mipsplatting_x8to2_baseline_directsrc_v1}"
: "${SPARSE_DIR:=/root/autodl-tmp/kitchen}"
: "${PRIORS_DIR:=/root/autodl-tmp/kitchen/priors}"
: "${ORACLE_DIR:=./results/task02/oracle/kitchen}"
: "${CKPT_DIR:=./checkpoints/hr_head_kitchen_v1}"
: "${DEVICE:=cuda}"
: "${EPOCHS:=400}"
: "${LR:=3e-4}"
: "${N_FRAMES:=32}"
: "${VIEWS_PER_FORWARD:=1}"
: "${DEPTH_SOURCE:=colmap}"

mkdir -p "${CKPT_DIR}"

python train_hr_head.py \
  --scene_root "${SCENE_ROOT}" \
  --sparse_dir "${SPARSE_DIR}" \
  --auto_images \
  --priors_dir "${PRIORS_DIR}" \
  --oracle_dir "${ORACLE_DIR}" \
  --output_dir "${CKPT_DIR}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_frames "${N_FRAMES}" \
  --views_per_forward "${VIEWS_PER_FORWARD}" \
  --depth_source "${DEPTH_SOURCE}" \
  --device "${DEVICE}"

echo "==> last: ${CKPT_DIR}/hr_head_last.pt"
