#!/usr/bin/env bash
set -euo pipefail

DTU_ROOT="${DTU_ROOT:-/data/dtu_3dgs}"
DTU_OFFICIAL_ROOT="${DTU_OFFICIAL_ROOT:-/data/DTU}"
CACHE_DIR="${CACHE_DIR:-/data/_downloads/2dgs_dtu}"
HF_DATASET="${HF_DATASET:-dylanebert/2DGS}"
DTU_TAR_URL="${DTU_TAR_URL:-https://huggingface.co/datasets/dylanebert/2DGS/resolve/main/dtu.tar.gz}"

mkdir -p "${DTU_ROOT}" "${DTU_OFFICIAL_ROOT}/Points/stl" "${CACHE_DIR}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2
    exit 2
  fi
}

need_cmd python
need_cmd curl
need_cmd tar

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
  python -m pip install -U huggingface_hub
fi

echo "[dtu-scan24] DTU_ROOT=${DTU_ROOT}"
echo "[dtu-scan24] DTU_OFFICIAL_ROOT=${DTU_OFFICIAL_ROOT}"
echo "[dtu-scan24] CACHE_DIR=${CACHE_DIR}"

if [[ ! -f "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" ]]; then
  echo "[dtu-scan24] downloading eval_dtu metadata/ground truth from ${HF_DATASET}"
  python - <<'PY' "${HF_DATASET}" "${CACHE_DIR}"
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["eval_dtu/**"],
)
PY

  STL_PATH="$(find "${CACHE_DIR}" -type f -name 'stl024_total.ply' -print -quit)"
  if [[ -z "${STL_PATH}" ]]; then
    echo "could not find stl024_total.ply under ${CACHE_DIR}" >&2
    echo "fallback: download official Points.zip and extract only stl024_total.ply" >&2
    echo "  curl -L -o /data/_downloads/Points.zip https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip" >&2
    echo "  unzip -j /data/_downloads/Points.zip '*stl024_total.ply' -d ${DTU_OFFICIAL_ROOT}/Points/stl" >&2
    exit 3
  fi
  cp -f "${STL_PATH}" "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
fi

if [[ ! -d "${DTU_ROOT}/scan24" ]]; then
  echo "[dtu-scan24] streaming scan24 from ${DTU_TAR_URL}"
  TMP_EXTRACT="$(mktemp -d "${CACHE_DIR}/scan24_extract.XXXXXX")"
  set +e
  curl -L --retry 5 --retry-delay 5 "${DTU_TAR_URL}" \
    | tar -xzf - -C "${TMP_EXTRACT}" --wildcards 'scan24/*' '*/scan24/*'
  TAR_STATUS=$?
  set -e
  if [[ ${TAR_STATUS} -ne 0 ]]; then
    echo "streaming extraction failed; remove partial dir: ${TMP_EXTRACT}" >&2
    exit ${TAR_STATUS}
  fi

  SCAN_DIR="$(find "${TMP_EXTRACT}" -type d -name 'scan24' -print -quit)"
  if [[ -z "${SCAN_DIR}" ]]; then
    echo "could not locate scan24 after extraction under ${TMP_EXTRACT}" >&2
    exit 4
  fi

  rm -rf "${DTU_ROOT}/scan24.tmp"
  mv "${SCAN_DIR}" "${DTU_ROOT}/scan24.tmp"
  rm -rf "${DTU_ROOT}/scan24"
  mv "${DTU_ROOT}/scan24.tmp" "${DTU_ROOT}/scan24"
  rm -rf "${TMP_EXTRACT}"
fi

echo "[dtu-scan24] done"
test -d "${DTU_ROOT}/scan24"
test -f "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
du -sh "${DTU_ROOT}/scan24" "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" 2>/dev/null || true
