#!/usr/bin/env bash
set -euo pipefail

DTU_ROOT="${DTU_ROOT:-/data/dtu_3dgs}"
DTU_OFFICIAL_ROOT="${DTU_OFFICIAL_ROOT:-/data/DTU}"
ASSET_OUT="${1:-${PWD}/dtu_scan24_asset.tar.gz}"

SCAN_DIR="${DTU_ROOT}/scan24"
STL_PATH="${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
OBSMASK_PATH="${DTU_OFFICIAL_ROOT}/ObsMask/ObsMask24_10.mat"
PLANE_PATH="${DTU_OFFICIAL_ROOT}/ObsMask/Plane24.mat"

count_files() {
  local root="$1"
  local pattern="$2"
  find "${root}" -type f -name "${pattern}" 2>/dev/null | wc -l | tr -d ' '
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "missing required asset path: ${path}" >&2
    exit 2
  fi
}

if [[ ! -d "${SCAN_DIR}" ]]; then
  echo "missing scan24 directory: ${SCAN_DIR}" >&2
  exit 2
fi

if [[ ! -f "${STL_PATH}" ]]; then
  echo "missing DTU STL: ${STL_PATH}" >&2
  exit 2
fi

if [[ ! -f "${OBSMASK_PATH}" ]]; then
  echo "missing DTU ObsMask: ${OBSMASK_PATH}" >&2
  exit 2
fi

if [[ ! -f "${PLANE_PATH}" ]]; then
  echo "missing DTU Plane: ${PLANE_PATH}" >&2
  exit 2
fi

require_path "${SCAN_DIR}/images"
require_path "${SCAN_DIR}/sparse/0"
require_path "${SCAN_DIR}/depths"
require_path "${SCAN_DIR}/points.ply"

IMAGE_COUNT="$(count_files "${SCAN_DIR}/images" "*.png")"
DEPTH_COUNT="$(count_files "${SCAN_DIR}/depths" "*.pt")"
STL_BYTES="$(wc -c < "${STL_PATH}" | tr -d ' ')"

if [[ "${IMAGE_COUNT}" -lt 40 ]]; then
  echo "scan24 looks incomplete: expected >=40 images, found ${IMAGE_COUNT} in ${SCAN_DIR}/images" >&2
  exit 2
fi

if [[ "${DEPTH_COUNT}" -lt 40 ]]; then
  echo "scan24 looks incomplete: expected >=40 depth maps, found ${DEPTH_COUNT} in ${SCAN_DIR}/depths" >&2
  exit 2
fi

if [[ "${STL_BYTES}" -lt 10000000 ]]; then
  echo "DTU STL looks too small: ${STL_PATH} has ${STL_BYTES} bytes" >&2
  exit 2
fi

ASSET_OUT="$(python - <<'PY' "${ASSET_OUT}"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
mkdir -p "$(dirname "${ASSET_OUT}")"

STAGE_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${STAGE_DIR}"
}
trap cleanup EXIT

mkdir -p "${STAGE_DIR}/dtu_3dgs" "${STAGE_DIR}/DTU/Points/stl" "${STAGE_DIR}/DTU/ObsMask"
ln -s "${SCAN_DIR}" "${STAGE_DIR}/dtu_3dgs/scan24"
ln -s "${STL_PATH}" "${STAGE_DIR}/DTU/Points/stl/stl024_total.ply"
ln -s "${OBSMASK_PATH}" "${STAGE_DIR}/DTU/ObsMask/ObsMask24_10.mat"
ln -s "${PLANE_PATH}" "${STAGE_DIR}/DTU/ObsMask/Plane24.mat"

cat > "${STAGE_DIR}/manifest.json" <<EOF
{
  "asset": "dtu_scan24_asset",
  "dataset": "dtu_core",
  "scene": "scan24",
  "layout": {
    "scan24": "dtu_3dgs/scan24",
    "stl": "DTU/Points/stl/stl024_total.ply",
    "obs_mask": "DTU/ObsMask/ObsMask24_10.mat",
    "plane": "DTU/ObsMask/Plane24.mat"
  }
}
EOF

rm -f "${ASSET_OUT}" "${ASSET_OUT}.sha256"
LC_ALL=C tar -czhf "${ASSET_OUT}" -C "${STAGE_DIR}" manifest.json dtu_3dgs DTU

if command -v sha256sum >/dev/null 2>&1; then
  LC_ALL=C sha256sum "${ASSET_OUT}" > "${ASSET_OUT}.sha256"
else
  LC_ALL=C shasum -a 256 "${ASSET_OUT}" > "${ASSET_OUT}.sha256"
fi

echo "[dtu-asset] wrote ${ASSET_OUT}"
cat "${ASSET_OUT}.sha256"
du -sh "${ASSET_OUT}" "${ASSET_OUT}.sha256" 2>/dev/null || true
