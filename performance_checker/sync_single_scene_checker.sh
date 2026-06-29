#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:?Set REMOTE to user@host or host alias}"
REMOTE_DIR="${REMOTE_DIR:?Set REMOTE_DIR to the remote SRtest checkout path}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

RSYNC_FLAGS=(-av --delete)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

"${RSYNC_BIN}" "${RSYNC_FLAGS[@]}" \
  "${ROOT_DIR}/performance_checker/" \
  "${REMOTE}:${REMOTE_DIR%/}/performance_checker/"

echo "Synced performance_checker/ to ${REMOTE}:${REMOTE_DIR%/}/performance_checker/"
