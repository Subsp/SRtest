#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
SCENE_NAME="${SCENE_NAME:-kitchen}"
SCENE_ROOT="${SCENE_ROOT:-${WORK_ROOT}/${SCENE_NAME}}"
SCENE_ASSET_ROOT="${SCENE_ASSET_ROOT:-${SCENE_ROOT}/_hrgsrefiner_assets}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HR_IMAGES_SUBDIR="${HR_IMAGES_SUBDIR:-images_2}"
LR_ANCHOR_DIR="${LR_ANCHOR_DIR:-${SCENE_ROOT}/images_8}"

SR_PRIOR_NAME="${SR_PRIOR_NAME:-qwen_steps1_seed42_rcgm_aligned_images2_train244_v0}"
PREPARED_SR_PRIOR_ROOT="${PREPARED_SR_PRIOR_ROOT:-${SCENE_ASSET_ROOT}/prepared_sr_priors/${SR_PRIOR_NAME}}"
SR_PRIOR_SUBDIR="${SR_PRIOR_SUBDIR:-fused_priors}"
SR_PRIOR_MASK_SUBDIR="${SR_PRIOR_MASK_SUBDIR:-usable_masks}"
REQUIRE_PRIOR_MASK="${REQUIRE_PRIOR_MASK:-0}"

RUN_NAME="${RUN_NAME:-${SR_PRIOR_NAME}_sp_ie_srgs_routed_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/sp_ie_srgs_v0/${SCENE_NAME}}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

ITERATIONS="${ITERATIONS:-30000}"
TEST_ITERATIONS="${TEST_ITERATIONS:-7000 30000}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000 30000}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-1}"
TRAIN_PORT="${TRAIN_PORT:-6009}"
LLFFHOLD="${LLFFHOLD:-8}"

SP_GEO_LAMBDA_DSSIM="${SP_GEO_LAMBDA_DSSIM:-0.2}"
SP_APP_L1_WEIGHT="${SP_APP_L1_WEIGHT:-0.05}"
SP_APP_LAMBDA_DSSIM="${SP_APP_LAMBDA_DSSIM:-0.0}"
SP_APP_LR_WEIGHT="${SP_APP_LR_WEIGHT:-0.01}"
SP_AUDIT_INTERVAL="${SP_AUDIT_INTERVAL:-100}"
SP_DISABLE_DENSIFICATION_ROUTE="${SP_DISABLE_DENSIFICATION_ROUTE:-0}"
SP_MAX_POINTS="${SP_MAX_POINTS:-700000}"

SP_SURFACE_ENABLE="${SP_SURFACE_ENABLE:-0}"
SP_LAMBDA_SURFACE="${SP_LAMBDA_SURFACE:-1.0}"
SP_LAMBDA_DISTORTION="${SP_LAMBDA_DISTORTION:-1000.0}"
SP_LAMBDA_DEPTH_NORMAL="${SP_LAMBDA_DEPTH_NORMAL:-0.05}"
SP_LAMBDA_NORMAL_SMOOTH="${SP_LAMBDA_NORMAL_SMOOTH:-0.01}"
SP_SURFACE_RAMP_START_ITER="${SP_SURFACE_RAMP_START_ITER:-1000}"
SP_SURFACE_RAMP_END_ITER="${SP_SURFACE_RAMP_END_ITER:-5000}"

POSITION_LR_INIT="${POSITION_LR_INIT:-0.00016}"
POSITION_LR_FINAL="${POSITION_LR_FINAL:-0.0000016}"
FEATURE_LR="${FEATURE_LR:-0.0025}"
OPACITY_LR="${OPACITY_LR:-0.05}"
SCALING_LR="${SCALING_LR:-0.005}"
ROTATION_LR="${ROTATION_LR:-0.001}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-500}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-3000}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"

MIN_MATCH_RATIO="${MIN_MATCH_RATIO:-0.95}"
NPSE_CACHE_ROOT="${NPSE_CACHE_ROOT:-}"

read -r -a TEST_ITER_ARRAY <<< "${TEST_ITERATIONS}"
read -r -a SAVE_ITER_ARRAY <<< "${SAVE_ITERATIONS}"
read -r -a CKPT_ITER_ARRAY <<< "${CHECKPOINT_ITERATIONS}"

PREFLIGHT_ARGS=(
  --scene_root "${SCENE_ROOT}"
  --hr_images_subdir "${HR_IMAGES_SUBDIR}"
  --lr_anchor_dir "${LR_ANCHOR_DIR}"
  --prepared_sr_prior_root "${PREPARED_SR_PRIOR_ROOT}"
  --sr_prior_subdir "${SR_PRIOR_SUBDIR}"
  --sr_prior_mask_subdir "${SR_PRIOR_MASK_SUBDIR}"
  --llffhold "${LLFFHOLD}"
  --min_match_ratio "${MIN_MATCH_RATIO}"
  --json_out "${MODEL_DIR}/sp_preflight.json"
)
if [[ "${REQUIRE_PRIOR_MASK}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require_prior_mask)
fi
if [[ -n "${NPSE_CACHE_ROOT}" ]]; then
  PREFLIGHT_ARGS+=(--npse_cache_root "${NPSE_CACHE_ROOT}")
fi

echo "[sp-ie-srgs-v0] repo      : ${REPO_ROOT}"
echo "[sp-ie-srgs-v0] scene     : ${SCENE_ROOT}"
echo "[sp-ie-srgs-v0] HR images : ${HR_IMAGES_SUBDIR}"
echo "[sp-ie-srgs-v0] LR anchor : ${LR_ANCHOR_DIR}"
echo "[sp-ie-srgs-v0] prior     : ${PREPARED_SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}"
echo "[sp-ie-srgs-v0] masks     : ${PREPARED_SR_PRIOR_ROOT}/${SR_PRIOR_MASK_SUBDIR}"
echo "[sp-ie-srgs-v0] output    : ${MODEL_DIR}"
echo "[sp-ie-srgs-v0] surface   : enable=${SP_SURFACE_ENABLE} lambda=${SP_LAMBDA_SURFACE} ramp=${SP_SURFACE_RAMP_START_ITER}-${SP_SURFACE_RAMP_END_ITER}"
echo "[sp-ie-srgs-v0] densify   : disable_route=${SP_DISABLE_DENSIFICATION_ROUTE} max_points=${SP_MAX_POINTS}"

mkdir -p "${MODEL_DIR}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/sp_ie_srgs_preflight.py" "${PREFLIGHT_ARGS[@]}"

TRAIN_ARGS=(
  -s "${SCENE_ROOT}"
  -m "${MODEL_DIR}"
  -i "${HR_IMAGES_SUBDIR}"
  -r "${TRAIN_RESOLUTION}"
  --eval
  --iterations "${ITERATIONS}"
  --test_iterations "${TEST_ITER_ARRAY[@]}"
  --save_iterations "${SAVE_ITER_ARRAY[@]}"
  --prepared_sr_prior_root "${PREPARED_SR_PRIOR_ROOT}"
  --sr_prior_subdir "${SR_PRIOR_SUBDIR}"
  --sr_prior_mask_subdir "${SR_PRIOR_MASK_SUBDIR}"
  --sp_lr_anchor_dir "${LR_ANCHOR_DIR}"
  --sp_geo_lambda_dssim "${SP_GEO_LAMBDA_DSSIM}"
  --sp_app_l1_weight "${SP_APP_L1_WEIGHT}"
  --sp_app_lambda_dssim "${SP_APP_LAMBDA_DSSIM}"
  --sp_app_lr_weight "${SP_APP_LR_WEIGHT}"
  --sp_audit_interval "${SP_AUDIT_INTERVAL}"
  --sp_max_points "${SP_MAX_POINTS}"
  --sp_lambda_surface "${SP_LAMBDA_SURFACE}"
  --sp_lambda_distortion "${SP_LAMBDA_DISTORTION}"
  --sp_lambda_depth_normal "${SP_LAMBDA_DEPTH_NORMAL}"
  --sp_lambda_normal_smooth "${SP_LAMBDA_NORMAL_SMOOTH}"
  --sp_surface_ramp_start_iter "${SP_SURFACE_RAMP_START_ITER}"
  --sp_surface_ramp_end_iter "${SP_SURFACE_RAMP_END_ITER}"
  --position_lr_init "${POSITION_LR_INIT}"
  --position_lr_final "${POSITION_LR_FINAL}"
  --feature_lr "${FEATURE_LR}"
  --opacity_lr "${OPACITY_LR}"
  --scaling_lr "${SCALING_LR}"
  --rotation_lr "${ROTATION_LR}"
  --densify_from_iter "${DENSIFY_FROM_ITER}"
  --densify_until_iter "${DENSIFY_UNTIL_ITER}"
  --densification_interval "${DENSIFICATION_INTERVAL}"
  --opacity_reset_interval "${OPACITY_RESET_INTERVAL}"
  --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD}"
  --port "${TRAIN_PORT}"
)
if [[ "${#CKPT_ITER_ARRAY[@]}" -gt 0 ]]; then
  TRAIN_ARGS+=(--checkpoint_iterations "${CKPT_ITER_ARRAY[@]}")
fi
if [[ -n "${START_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--start_checkpoint "${START_CHECKPOINT}")
fi
if [[ "${REQUIRE_PRIOR_MASK}" == "1" ]]; then
  TRAIN_ARGS+=(--sp_require_prior_mask)
fi
if [[ "${SP_SURFACE_ENABLE}" == "1" ]]; then
  TRAIN_ARGS+=(--sp_surface_enable)
fi
if [[ "${SP_DISABLE_DENSIFICATION_ROUTE}" == "1" ]]; then
  TRAIN_ARGS+=(--sp_disable_densification_route)
fi

cd "${REPO_ROOT}"
echo "[sp-ie-srgs-v0] launch train_sp_ie_srgs.py"
"${PYTHON_BIN}" "${REPO_ROOT}/train_sp_ie_srgs.py" "${TRAIN_ARGS[@]}"
