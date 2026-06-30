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

python - <<'PY'
import shutil
import torch

print(f"[cuda-ext] torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"[cuda-ext] cuda_available={torch.cuda.is_available()}")
print(f"[cuda-ext] nvcc={shutil.which('nvcc')}")
PY

python -m pip install --no-cache-dir -U ninja setuptools wheel

build_extension() {
  local ext_dir="$1"
  if [[ ! -d "${ext_dir}" ]]; then
    echo "[cuda-ext] skip missing ${ext_dir}"
    return 0
  fi

  echo "[cuda-ext] building ${ext_dir}"
  rm -rf "${ext_dir}/build" "${ext_dir}"/*.egg-info
  python -m pip install --no-build-isolation --no-cache-dir -v "${ext_dir}"
}

for method_root in "${METHOD_ROOTS[@]}"; do
  if [[ ! -d "${method_root}" ]]; then
    echo "[cuda-ext] skip missing method root ${method_root}"
    continue
  fi

  build_extension "${method_root}/submodules/diff-gaussian-rasterization"
  build_extension "${method_root}/submodules/simple-knn"
done

python - <<'PY'
import importlib

for name in ["diff_gaussian_rasterization", "simple_knn._C"]:
    importlib.import_module(name)
    print(f"[cuda-ext] import ok: {name}")
PY

echo "[cuda-ext] done"
