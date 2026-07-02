#!/usr/bin/env bash
set -euo pipefail

DTU_ROOT="${DTU_ROOT:-/data/dtu_3dgs}"
DTU_OFFICIAL_ROOT="${DTU_OFFICIAL_ROOT:-/data/DTU}"
CACHE_DIR="${CACHE_DIR:-/data/_downloads/2dgs_dtu}"
DTU_SCAN24_ASSET_URL="${DTU_SCAN24_ASSET_URL:-https://github.com/Subsp/SRtest/releases/download/dtu-scan24-v1/dtu_scan24_asset.tar.gz}"
ALLOW_EXTERNAL_DTU_DOWNLOAD="${ALLOW_EXTERNAL_DTU_DOWNLOAD:-0}"
POINTS_ZIP_URL="${POINTS_ZIP_URL:-https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip}"
GDRIVE_DTU_FOLDER_URL="${GDRIVE_DTU_FOLDER_URL:-https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9}"

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

assets_ready() {
  validate_scan24_dir "${DTU_ROOT}/scan24" >/dev/null 2>&1 \
    && validate_stl_path "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" >/dev/null 2>&1 \
    && validate_dtu_eval_assets "${DTU_OFFICIAL_ROOT}" >/dev/null 2>&1
}

count_files() {
  local root="$1"
  local pattern="$2"
  find "${root}" -type f -name "${pattern}" 2>/dev/null | wc -l | tr -d ' '
}

validate_scan24_dir() {
  local scan_dir="$1"
  local image_count
  local depth_count

  [[ -d "${scan_dir}" ]] || {
    echo "missing scan24 directory: ${scan_dir}" >&2
    return 2
  }
  [[ -d "${scan_dir}/images" ]] || {
    echo "scan24 missing images directory: ${scan_dir}/images" >&2
    return 2
  }
  [[ -d "${scan_dir}/sparse/0" ]] || {
    echo "scan24 missing sparse/0 directory: ${scan_dir}/sparse/0" >&2
    return 2
  }
  [[ -d "${scan_dir}/depths" ]] || {
    echo "scan24 missing depths directory: ${scan_dir}/depths" >&2
    return 2
  }
  [[ -f "${scan_dir}/points.ply" ]] || {
    echo "scan24 missing points.ply: ${scan_dir}/points.ply" >&2
    return 2
  }

  image_count="$(count_files "${scan_dir}/images" "*.png")"
  depth_count="$(count_files "${scan_dir}/depths" "*.pt")"
  [[ "${image_count}" -ge 40 ]] || {
    echo "scan24 has too few images: ${image_count}" >&2
    return 2
  }
  [[ "${depth_count}" -ge 40 ]] || {
    echo "scan24 has too few depth maps: ${depth_count}" >&2
    return 2
  }
}

validate_stl_path() {
  local stl_path="$1"
  local stl_bytes

  [[ -f "${stl_path}" ]] || {
    echo "missing DTU STL: ${stl_path}" >&2
    return 2
  }
  stl_bytes="$(wc -c < "${stl_path}" | tr -d ' ')"
  [[ "${stl_bytes}" -ge 10000000 ]] || {
    echo "DTU STL looks too small: ${stl_path} has ${stl_bytes} bytes" >&2
    return 2
  }
}

validate_dtu_eval_assets() {
  local dtu_official_root="$1"
  [[ -f "${dtu_official_root}/ObsMask/ObsMask24_10.mat" ]] || {
    echo "missing DTU ObsMask: ${dtu_official_root}/ObsMask/ObsMask24_10.mat" >&2
    return 2
  }
  [[ -f "${dtu_official_root}/ObsMask/Plane24.mat" ]] || {
    echo "missing DTU Plane: ${dtu_official_root}/ObsMask/Plane24.mat" >&2
    return 2
  }
}

install_release_asset() {
  local asset_path="${CACHE_DIR}/dtu_scan24_asset.tar.gz"
  local extract_dir="${CACHE_DIR}/dtu_scan24_asset_extract"

  if [[ -z "${DTU_SCAN24_ASSET_URL}" ]]; then
    echo "DTU_SCAN24_ASSET_URL is empty; skipping GitHub asset download"
    return 0
  fi

  echo "[dtu-scan24] downloading GitHub asset"
  echo "[dtu-scan24] source=${DTU_SCAN24_ASSET_URL}"
  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"
  if ! curl -fL --retry 5 --retry-delay 5 -o "${asset_path}" "${DTU_SCAN24_ASSET_URL}"; then
    echo "[dtu-scan24] GitHub asset download failed: ${DTU_SCAN24_ASSET_URL}" >&2
    return 8
  fi
  if ! LC_ALL=C tar -xzf "${asset_path}" -C "${extract_dir}"; then
    echo "[dtu-scan24] could not extract asset: ${asset_path}" >&2
    return 8
  fi

  if ! validate_scan24_dir "${extract_dir}/dtu_3dgs/scan24"; then
    echo "asset has invalid dtu_3dgs/scan24: ${asset_path}" >&2
    return 6
  fi
  if ! validate_stl_path "${extract_dir}/DTU/Points/stl/stl024_total.ply"; then
    echo "asset has invalid DTU/Points/stl/stl024_total.ply: ${asset_path}" >&2
    return 6
  fi
  if ! validate_dtu_eval_assets "${extract_dir}/DTU"; then
    echo "asset has invalid DTU/ObsMask assets: ${asset_path}" >&2
    return 6
  fi

  rm -rf "${DTU_ROOT}/scan24.tmp"
  mv "${extract_dir}/dtu_3dgs/scan24" "${DTU_ROOT}/scan24.tmp"
  rm -rf "${DTU_ROOT}/scan24"
  mv "${DTU_ROOT}/scan24.tmp" "${DTU_ROOT}/scan24"

  mkdir -p "${DTU_OFFICIAL_ROOT}/Points/stl"
  cp -f \
    "${extract_dir}/DTU/Points/stl/stl024_total.ply" \
    "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
  mkdir -p "${DTU_OFFICIAL_ROOT}/ObsMask"
  cp -f \
    "${extract_dir}/DTU/ObsMask/ObsMask24_10.mat" \
    "${DTU_OFFICIAL_ROOT}/ObsMask/ObsMask24_10.mat"
  cp -f \
    "${extract_dir}/DTU/ObsMask/Plane24.mat" \
    "${DTU_OFFICIAL_ROOT}/ObsMask/Plane24.mat"

  rm -rf "${extract_dir}"
  rm -f "${asset_path}"
}

download_official_stl_only() {
  local out_path="${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
  echo "[dtu-scan24] extracting official DTU STL from ${POINTS_ZIP_URL}"
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

install_gdown_if_needed() {
  if ! python -c "import gdown" >/dev/null 2>&1; then
    python -m pip install -U "gdown>=5,<6"
  fi
}

download_scan24_from_gdrive() {
  local gdrive_dir="${CACHE_DIR}/gdrive_dtu"
  echo "[dtu-scan24] downloading scan24 from Google Drive"
  echo "[dtu-scan24] source=${GDRIVE_DTU_FOLDER_URL}"
  install_gdown_if_needed
  rm -rf "${gdrive_dir}.tmp"
  mkdir -p "${gdrive_dir}.tmp"
  python -m gdown --folder --remaining-ok "${GDRIVE_DTU_FOLDER_URL}" -O "${gdrive_dir}.tmp"
  python - <<'PY' "${gdrive_dir}.tmp" "${DTU_ROOT}" "${CACHE_DIR}"
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

download_root = Path(sys.argv[1])
dtu_root = Path(sys.argv[2])
cache_dir = Path(sys.argv[3])
dest = dtu_root / "scan24"


def find_scan24(root: Path):
    for path in root.rglob("scan24"):
        if path.is_dir():
            return path
    return None


def install_scan24(src: Path) -> None:
    tmp = dtu_root / "scan24.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    dtu_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(tmp))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)
    print(f"[dtu-scan24] installed {dest}")


def safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    base_resolved = base.resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise RuntimeError(f"unsafe archive member path: {member_name}")
    return target


scan_dir = find_scan24(download_root)
if scan_dir is not None:
    install_scan24(scan_dir)
    raise SystemExit(0)

archives = sorted(
    path for path in download_root.rglob("*")
    if path.is_file()
    and (
        path.name.endswith(".tar")
        or path.name.endswith(".tar.gz")
        or path.name.endswith(".tgz")
        or path.name.endswith(".zip")
    )
)

for archive in archives:
    extract_root = cache_dir / f"scan24_extract_{archive.stem.replace('.', '_')}"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"[dtu-scan24] extracting scan24 from {archive}")

    if archive.name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:*") as tf:
            members = [
                member for member in tf.getmembers()
                if "scan24" in Path(member.name).parts
            ]
            if not members:
                shutil.rmtree(extract_root)
                continue
            for member in members:
                safe_target(extract_root, member.name)
            tf.extractall(extract_root, members=members)
    else:
        with zipfile.ZipFile(archive) as zf:
            names = [
                name for name in zf.namelist()
                if "scan24" in Path(name).parts
            ]
            if not names:
                shutil.rmtree(extract_root)
                continue
            for name in names:
                safe_target(extract_root, name)
            zf.extractall(extract_root, members=names)

    scan_dir = find_scan24(extract_root)
    if scan_dir is not None:
        install_scan24(scan_dir)
        shutil.rmtree(extract_root)
        raise SystemExit(0)
    shutil.rmtree(extract_root)

raise SystemExit(f"could not locate scan24 in Google Drive download: {download_root}")
PY
  rm -rf "${gdrive_dir}" "${gdrive_dir}.tmp"
}

echo "[dtu-scan24] DTU_ROOT=${DTU_ROOT}"
echo "[dtu-scan24] DTU_OFFICIAL_ROOT=${DTU_OFFICIAL_ROOT}"
echo "[dtu-scan24] CACHE_DIR=${CACHE_DIR}"
echo "[dtu-scan24] DTU_SCAN24_ASSET_URL=${DTU_SCAN24_ASSET_URL}"

if ! assets_ready && [[ -n "${DTU_SCAN24_ASSET_URL}" ]]; then
  if ! install_release_asset; then
    if [[ "${ALLOW_EXTERNAL_DTU_DOWNLOAD}" == "1" ]]; then
      echo "[dtu-scan24] release asset unavailable; continuing with external source because ALLOW_EXTERNAL_DTU_DOWNLOAD=1" >&2
    else
      echo "DTU release asset is unavailable. Upload dtu_scan24_asset.tar.gz first, or set ALLOW_EXTERNAL_DTU_DOWNLOAD=1 for local asset preparation." >&2
      exit 7
    fi
  fi
fi

if [[ ! -f "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" && "${ALLOW_EXTERNAL_DTU_DOWNLOAD}" == "1" ]]; then
  download_official_stl_only
fi

if [[ ! -d "${DTU_ROOT}/scan24" && "${ALLOW_EXTERNAL_DTU_DOWNLOAD}" == "1" ]]; then
  download_scan24_from_gdrive
fi

if ! assets_ready; then
  echo "DTU scan24 assets are still missing." >&2
  echo "Upload dtu_scan24_asset.tar.gz to ${DTU_SCAN24_ASSET_URL}, or set ALLOW_EXTERNAL_DTU_DOWNLOAD=1." >&2
  exit 7
fi

echo "[dtu-scan24] done"
test -d "${DTU_ROOT}/scan24"
test -f "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply"
test -f "${DTU_OFFICIAL_ROOT}/ObsMask/ObsMask24_10.mat"
test -f "${DTU_OFFICIAL_ROOT}/ObsMask/Plane24.mat"
du -sh \
  "${DTU_ROOT}/scan24" \
  "${DTU_OFFICIAL_ROOT}/Points/stl/stl024_total.ply" \
  "${DTU_OFFICIAL_ROOT}/ObsMask/ObsMask24_10.mat" \
  "${DTU_OFFICIAL_ROOT}/ObsMask/Plane24.mat" 2>/dev/null || true
