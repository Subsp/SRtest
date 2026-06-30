#!/usr/bin/env bash
set -euo pipefail

DTU_ROOT="${DTU_ROOT:-/data/dtu_3dgs}"
DTU_OFFICIAL_ROOT="${DTU_OFFICIAL_ROOT:-/data/DTU}"
CACHE_DIR="${CACHE_DIR:-/data/_downloads/2dgs_dtu}"
HF_DATASET="${HF_DATASET:-dylanebert/2DGS}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
DTU_TAR_URL="${DTU_TAR_URL:-${HF_ENDPOINT}/datasets/dylanebert/2DGS/resolve/main/dtu.tar.gz}"
POINTS_ZIP_URL="${POINTS_ZIP_URL:-https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip}"

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

download_official_stl_only() {
  local out_path="${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
  echo "[dtu-scan24] trying official Points.zip range extraction"
  if ! python -c "import remotezip" >/dev/null 2>&1; then
    python -m pip install -U remotezip
  fi
  python - <<'PY' "${POINTS_ZIP_URL}" "${out_path}" "${CACHE_DIR}"
import shutil
import sys
from pathlib import Path

from remotezip import RemoteZip

zip_url, out_path, cache_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
tmp_dir = cache_dir / "official_points_stl024"
tmp_dir.mkdir(parents=True, exist_ok=True)

with RemoteZip(zip_url) as zf:
    names = zf.namelist()
    matches = [
        name for name in names
        if name.endswith("/stl/stl024_total.ply") or name.endswith("stl024_total.ply")
    ]
    if not matches:
        raise SystemExit("stl024_total.ply not found in official Points.zip")
    member = matches[0]
    zf.extract(member, tmp_dir)
    extracted = tmp_dir / member
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(extracted, out_path)
    print(f"[dtu-scan24] extracted {member} -> {out_path}")
PY
}

echo "[dtu-scan24] DTU_ROOT=${DTU_ROOT}"
echo "[dtu-scan24] DTU_OFFICIAL_ROOT=${DTU_OFFICIAL_ROOT}"
echo "[dtu-scan24] CACHE_DIR=${CACHE_DIR}"
echo "[dtu-scan24] HF_ENDPOINT=${HF_ENDPOINT}"

if [[ ! -f "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" ]]; then
  echo "[dtu-scan24] downloading eval_dtu metadata/ground truth from ${HF_DATASET}"
  set +e
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
  HF_STATUS=$?
  set -e
  if [[ ${HF_STATUS} -ne 0 ]]; then
    echo "[dtu-scan24] Hugging Face snapshot failed; trying direct URLs/fallbacks" >&2
  fi

  STL_PATH="$(find "${CACHE_DIR}" -type f -name 'stl024_total.ply' -print -quit)"
  if [[ -z "${STL_PATH}" ]]; then
    for url in \
      "${HF_ENDPOINT}/datasets/${HF_DATASET}/resolve/main/eval_dtu/Points/stl/stl024_total.ply" \
      "${HF_ENDPOINT}/datasets/${HF_DATASET}/resolve/main/eval_dtu/Points/stl/stl024_total.ply?download=true"; do
      echo "[dtu-scan24] trying ${url}"
      if curl -fL --retry 5 --retry-delay 5 -o "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" "${url}"; then
        break
      fi
    done
  fi
  if [[ -n "${STL_PATH}" ]]; then
    cp -f "${STL_PATH}" "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
  elif [[ ! -f "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" ]]; then
    download_official_stl_only
  fi
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
