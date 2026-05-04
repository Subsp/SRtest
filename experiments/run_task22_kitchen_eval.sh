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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── 可改：OUT_DIR 可写绝对路径或相对 experiments 的目录名 ─────────────────
: "${SCENE_ROOT:=/root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_lr_ablation_v1/mipsplatting_x8to2_baseline_directsrc_v1}"
: "${SPARSE_DIR:=/root/autodl-tmp/kitchen}"
: "${PRIORS_DIR:=/root/autodl-tmp/kitchen/priors}"
: "${ORACLE_DIR:=./results/task02/oracle/kitchen}"
: "${HR_HEAD_CKPT:=./checkpoints/hr_head_kitchen_v0/hr_head_last.pt}"
: "${DEVICE:=cuda}"
: "${N_FRAMES:=8}"
: "${DEPTH_SOURCE:=colmap}"

TS="$(date +%Y%m%d_%H%M)"
if [[ -z "${OUT_DIR:-}" ]]; then
  REL_OUT="results/task22_kitchen_compare_${TS}"
else
  REL_OUT="${OUT_DIR#./}"
fi
if [[ "${REL_OUT}" = /* ]]; then
  OUT_ABS="${REL_OUT}"
else
  OUT_ABS="${SCRIPT_DIR}/${REL_OUT}"
fi

mkdir -p "${OUT_ABS}"

echo "==> OUT_DIR (绝对路径) = ${OUT_ABS}"

# -u：避免缓冲「瞬间完成但实际没跑」的假象；若在跑请稍等 VGGT/H R Head 推理日志
python -u task22_hr_head_realdata.py \
  --scene_root "${SCENE_ROOT}" \
  --sparse_dir "${SPARSE_DIR}" \
  --auto_images \
  --priors_dir "${PRIORS_DIR}" \
  --depth_source "${DEPTH_SOURCE}" \
  --ckpt "${HR_HEAD_CKPT}" \
  --oracle_dir "${ORACLE_DIR}" \
  --eval_vggt_upsampled_baseline \
  --n_frames "${N_FRAMES}" \
  --output_dir "${OUT_ABS}" \
  --device "${DEVICE}"

N_SAVE=$(find "${OUT_ABS}" -maxdepth 1 -name '*_depth_hr.npy' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${N_SAVE}" -lt 1 ]]; then
  echo ""
  echo "[FATAL] 输出目录里没有 *_depth_hr.npy — Python 侧等于没跑进 main 或半途崩了但没退出码。"
  echo "       很常见原因：服务端 task22_hr_head_realdata.py 仍是旧版（缺少 if __name__=='__main__'）。请先："
  echo "         cd ${SCRIPT_DIR}/.. && git pull --ff-only origin main"
  echo "       再在 experiments 目录重跑本脚本。"
  exit 1
fi

echo ""
echo "==> 完成。查看（请直接复制下面路径）："
echo "    ${OUT_ABS}/compare_hrhead_vs_vggt_lr_bilinear.txt   (并排：HR Head vs VGGT LR→HR 双线性)"
echo "    ${OUT_ABS}/depth_metrics_vs_oracle.csv"
echo "    ${OUT_ABS}/depth_metrics_vggt_upsampled_vs_oracle.csv"
echo "    ${OUT_ABS}/*_depth_hr.npy"
if [[ ! -f "${OUT_ABS}/depth_metrics_vs_oracle.csv" ]]; then
  echo "[WARN] CSV 未找到，可搜索最近一次输出："
  echo "    find ${SCRIPT_DIR}/results -name 'depth_metrics_vs_oracle.csv' -mmin -30 2>/dev/null | tail -5"
fi
