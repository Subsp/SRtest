#!/usr/bin/env bash
# =============================================================================
# Kitchen：HR Head（可带 ckpt）+ oracle 指标 + VGGT 双线性 HR baseline 同协议对比
#
# 使用前请改下面「可改」路径；或 export 覆盖默认值后执行本脚本。
#
#   bash run_task22_kitchen_eval.sh
#
# 依赖：conda 已激活 srtest（或你的环境），且在 experiments/ 下有本仓库代码。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ── 可改 ───────────────────────────────────────────────────────────────────
: "${SCENE_ROOT:=/root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_lr_ablation_v1/mipsplatting_x8to2_baseline_directsrc_v1}"
: "${SPARSE_DIR:=/root/autodl-tmp/kitchen}"
: "${PRIORS_DIR:=/root/autodl-tmp/kitchen/priors}"
: "${ORACLE_DIR:=./results/task02/oracle/kitchen}"
: "${HR_HEAD_CKPT:=./checkpoints/hr_head_kitchen_v0/hr_head_last.pt}"
: "${OUT_DIR:=./results/task22_kitchen_compare_$(date +%Y%m%d_%H%M)}"
: "${DEVICE:=cuda}"
: "${N_FRAMES:=8}"
# depth_source: colmap（默认）或 vggt；开 baseline 时会再跑一遍冻结 VGGT 供对比
: "${DEPTH_SOURCE:=colmap}"

echo "==> OUT_DIR = ${OUT_DIR}"

python task22_hr_head_realdata.py \
  --scene_root "${SCENE_ROOT}" \
  --sparse_dir "${SPARSE_DIR}" \
  --auto_images \
  --priors_dir "${PRIORS_DIR}" \
  --depth_source "${DEPTH_SOURCE}" \
  --ckpt "${HR_HEAD_CKPT}" \
  --oracle_dir "${ORACLE_DIR}" \
  --eval_vggt_upsampled_baseline \
  --n_frames "${N_FRAMES}" \
  --output_dir "${OUT_DIR}" \
  --device "${DEVICE}"

echo ""
echo "==> 完成。查看："
echo "    ${OUT_DIR}/depth_metrics_vs_oracle.csv              (HR Head)"
echo "    ${OUT_DIR}/depth_metrics_vggt_upsampled_vs_oracle.csv  (VGGT→HR 双线性)"
echo "    ${OUT_DIR}/*_depth_hr.npy 等导出"
