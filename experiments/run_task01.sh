#!/usr/bin/env bash
# =============================================================================
# Task 0.1 – 2DSR View-Inconsistency Test  (quick launcher)
# =============================================================================
#
# 用法（模式 A，传预计算 SR 图）：
#   bash run_task01.sh <data_root> <sr_dir> [output_dir] [device]
#
#   data_root  : MipNeRF360 数据集根目录（提供 COLMAP 相机）
#   sr_dir     : 预计算 800×800 SR 图根目录，结构：<sr_dir>/<scene>/<frame>.png
#   output_dir : 结果输出目录（默认 ./results/task01）
#   device     : cuda 或 cpu（默认 cuda）
#
# 用法（模式 B，实时 SwinIR）：
#   bash run_task01.sh <data_root> "" [output_dir] [device]
# =============================================================================
set -euo pipefail

DATA_ROOT="${1:?请提供 MipNeRF360 根目录作为第一个参数}"
SR_DIR="${2:-}"          # 空字符串 = 实时 SwinIR
OUTPUT_DIR="${3:-./results/task01}"
DEVICE="${4:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Task 0.1: 2DSR View-Inconsistency Test"
echo "  Data root  : ${DATA_ROOT}"
echo "  SR source  : ${SR_DIR:-实时 SwinIR ×4}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Device     : ${DEVICE}"
echo ""

SR_DIR_ARG=""
if [ -n "${SR_DIR}" ]; then
    SR_DIR_ARG="--sr_dir ${SR_DIR}"
fi

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/task01_2dsr_consistency.py" \
    --data_root   "${DATA_ROOT}" \
    ${SR_DIR_ARG} \
    --output_dir  "${OUTPUT_DIR}" \
    --device      "${DEVICE}" \
    --n_frames    8 \
    --save_visuals
