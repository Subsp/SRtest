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
LLFFHOLD="${LLFFHOLD:-8}"
MIN_MATCH_RATIO="${MIN_MATCH_RATIO:-0.95}"
NPSE_CACHE_ROOT="${NPSE_CACHE_ROOT:-}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-${WORK_ROOT}/sp_ie_srgs_preflight.json}"

echo "[sp-ie-srgs-env] repo      : ${REPO_ROOT}"
echo "[sp-ie-srgs-env] python    : $(${PYTHON_BIN} -V 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "[sp-ie-srgs-env] nvidia-smi not found"
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

print("[sp-ie-srgs-env] executable:", sys.executable)
print("[sp-ie-srgs-env] version   :", sys.version.replace("\n", " "))

required = [
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "diff_gaussian_rasterization",
    "simple_knn._C",
]

for name in required:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "")
    print(f"[sp-ie-srgs-env] import ok: {name} {version}".rstrip())

import torch

print("[sp-ie-srgs-env] torch cuda:", torch.version.cuda)
print("[sp-ie-srgs-env] cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[sp-ie-srgs-env] cuda device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available to torch")
PY

"${PYTHON_BIN}" -m py_compile \
  train_sp_ie_srgs.py \
  routing/grad_audit.py \
  routing/param_groups.py \
  routing/train_step.py \
  surface/losses.py \
  surface/metrics.py \
  surface/render_utils.py \
  scripts/sp_ie_srgs_preflight.py

bash -n scripts/run_sp_ie_srgs_v0_kitchen.sh

PREFLIGHT_ARGS=(
  --scene_root "${SCENE_ROOT}"
  --hr_images_subdir "${HR_IMAGES_SUBDIR}"
  --lr_anchor_dir "${LR_ANCHOR_DIR}"
  --prepared_sr_prior_root "${PREPARED_SR_PRIOR_ROOT}"
  --sr_prior_subdir "${SR_PRIOR_SUBDIR}"
  --sr_prior_mask_subdir "${SR_PRIOR_MASK_SUBDIR}"
  --llffhold "${LLFFHOLD}"
  --min_match_ratio "${MIN_MATCH_RATIO}"
  --json_out "${PREFLIGHT_JSON}"
)
if [[ "${REQUIRE_PRIOR_MASK}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require_prior_mask)
fi
if [[ -n "${NPSE_CACHE_ROOT}" ]]; then
  PREFLIGHT_ARGS+=(--npse_cache_root "${NPSE_CACHE_ROOT}")
fi

"${PYTHON_BIN}" scripts/sp_ie_srgs_preflight.py "${PREFLIGHT_ARGS[@]}"
echo "[sp-ie-srgs-env] ok"
