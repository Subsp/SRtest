#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:Subsp/SRtest.git}"
BRANCH="${BRANCH:-main}"
CHECKOUT_DIR="${CHECKOUT_DIR:-$PWD/SRtest}"

if [[ -d "${CHECKOUT_DIR}/.git" ]]; then
  cd "${CHECKOUT_DIR}"
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
else
  mkdir -p "$(dirname "${CHECKOUT_DIR}")"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${CHECKOUT_DIR}"
  cd "${CHECKOUT_DIR}"
fi

echo "Synced ${REPO_URL} ${BRANCH} into ${CHECKOUT_DIR}"
python3 performance_checker/checker.py plan --commands
