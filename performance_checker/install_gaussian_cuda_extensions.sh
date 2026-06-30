#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export CUDA_EXT_LOG_DIR="${CUDA_EXT_LOG_DIR:-${REPO_ROOT}/benchmark_runs/_logs/cuda_ext}"
export REBUILD_GAUSSIAN_EXTENSIONS="${REBUILD_GAUSSIAN_EXTENSIONS:-0}"

if [[ $# -gt 0 ]]; then
  METHOD_ROOTS=("$@")
else
  METHOD_ROOTS=(
    "${REPO_ROOT}/SP-IE-SRGS"
    "${REPO_ROOT}/mip-splatting"
  )
fi

echo "[cuda-ext] REPO_ROOT=${REPO_ROOT}"
echo "[cuda-ext] CUDA_HOME=${CUDA_HOME}"
echo "[cuda-ext] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[cuda-ext] MAX_JOBS=${MAX_JOBS}"
echo "[cuda-ext] CUDA_EXT_LOG_DIR=${CUDA_EXT_LOG_DIR}"

python - <<'PY'
import shutil
import torch

print(f"[cuda-ext] torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"[cuda-ext] cuda_available={torch.cuda.is_available()}")
print(f"[cuda-ext] nvcc={shutil.which('nvcc')}")
PY

python -m pip install --no-cache-dir -U ninja setuptools wheel
mkdir -p "${CUDA_EXT_LOG_DIR}"

build_extension() {
  local ext_dir="$1"
  local import_name="$2"
  local log_name
  local log_path
  if [[ ! -d "${ext_dir}" ]]; then
    echo "[cuda-ext] skip missing ${ext_dir}"
    return 0
  fi

  if [[ "${REBUILD_GAUSSIAN_EXTENSIONS}" != "1" ]] && python -c "import ${import_name}" >/dev/null 2>&1; then
    echo "[cuda-ext] skip installed ${import_name}"
    return 0
  fi

  echo "[cuda-ext] building ${ext_dir}"
  log_name="$(echo "${ext_dir}" | sed 's#[^A-Za-z0-9_.-]#_#g')"
  log_path="${CUDA_EXT_LOG_DIR}/${log_name}.log"
  rm -rf "${ext_dir}/build" "${ext_dir}"/*.egg-info
  set +e
  python -m pip install --no-build-isolation --no-cache-dir -v "${ext_dir}" 2>&1 | tee "${log_path}"
  status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[cuda-ext] build failed: ${ext_dir}" >&2
    echo "[cuda-ext] full log: ${log_path}" >&2
    tail -120 "${log_path}" >&2 || true
    exit "${status}"
  fi
}

for method_root in "${METHOD_ROOTS[@]}"; do
  if [[ ! -d "${method_root}" ]]; then
    echo "[cuda-ext] skip missing method root ${method_root}"
    continue
  fi

  build_extension "${method_root}/submodules/diff-gaussian-rasterization" "diff_gaussian_rasterization"
  build_extension "${method_root}/submodules/simple-knn" "simple_knn._C"
done

python - <<'PY'
import importlib

for name in ["diff_gaussian_rasterization", "simple_knn._C"]:
    importlib.import_module(name)
    print(f"[cuda-ext] import ok: {name}")
PY

echo "[cuda-ext] done"
