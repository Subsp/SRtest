#!/usr/bin/env bash
set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-Subsp/SRtest}"
RELEASE_TAG="${RELEASE_TAG:-dtu-scan24-v1}"
ASSET_PATH="${1:-${PWD}/dtu_scan24_asset.tar.gz}"
SHA_PATH="${ASSET_PATH}.sha256"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2
    exit 2
  fi
}

need_cmd gh

if [[ ! -f "${ASSET_PATH}" ]]; then
  echo "missing asset: ${ASSET_PATH}" >&2
  exit 2
fi

if [[ ! -f "${SHA_PATH}" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    LC_ALL=C sha256sum "${ASSET_PATH}" > "${SHA_PATH}"
  else
    LC_ALL=C shasum -a 256 "${ASSET_PATH}" > "${SHA_PATH}"
  fi
fi

gh auth status

if gh release view "${RELEASE_TAG}" --repo "${GITHUB_REPO}" >/dev/null 2>&1; then
  echo "[dtu-asset] uploading to existing release ${GITHUB_REPO}/${RELEASE_TAG}"
  gh release upload "${RELEASE_TAG}" "${ASSET_PATH}" "${SHA_PATH}" \
    --repo "${GITHUB_REPO}" \
    --clobber
else
  echo "[dtu-asset] creating release ${GITHUB_REPO}/${RELEASE_TAG}"
  gh release create "${RELEASE_TAG}" "${ASSET_PATH}" "${SHA_PATH}" \
    --repo "${GITHUB_REPO}" \
    --title "DTU scan24 benchmark asset" \
    --notes "Reusable DTU scan24 asset for the SRtest single-scene performance checker."
fi

echo "[dtu-asset] done"
echo "https://github.com/${GITHUB_REPO}/releases/download/${RELEASE_TAG}/$(basename "${ASSET_PATH}")"
