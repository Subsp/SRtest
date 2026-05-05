#!/usr/bin/env bash
# Task 0.1：StableSR warp 一致性评测，warp 几何深度改用 **HR Head** 输出的 *_depth_hr.npy。
#
# 先在同一批帧上跑 task22（或确保 DEPTH_HR_DIR 下有与 frames stem 对齐的 npy），再：
#
#   export DEPTH_HR_DIR=/root/autodl-tmp/SRtest/experiments/results/task22_kitchen_compare_<...>
#   bash run_task01_kitchen_hr_head_warp.sh
#
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${DEPTH_HR_DIR:-}" ]]; then
  echo "[FATAL] 请先: export DEPTH_HR_DIR=<task22 输出目录（含 DSCF*_depth_hr.npy）>"
  exit 1
fi

# <DATA_ROOT>/kitchen/images_8；SR： <SR_DIR>/kitchen/<SR_SUBDIR>/
: "${DATA_ROOT:=/root/autodl-tmp}"
: "${SR_DIR:=/root/autodl-tmp}"
: "${OUTPUT_DIR:=./results/task01_kitchen_hr_head_warp}"
: "${DEVICE:=cuda}"
: "${SCENES:=kitchen}"
: "${N_FRAMES:=8}"
: "${IMAGE_SUBDIR:=images_8}"
: "${SR_SUBDIR:=priors}"

mkdir -p "${OUTPUT_DIR}"

python task01_2dsr_consistency.py \
  --data_root "${DATA_ROOT}" \
  --sr_dir "${SR_DIR}" \
  --sr_subdir "${SR_SUBDIR}" \
  --scenes ${SCENES} \
  --n_frames "${N_FRAMES}" \
  --image_subdir "${IMAGE_SUBDIR}" \
  --depth_mode hr_head \
  --depth_hr_dir "${DEPTH_HR_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"

echo "==> 结果目录: $(realpath "${OUTPUT_DIR}" 2>/dev/null || echo "${OUTPUT_DIR}")"
