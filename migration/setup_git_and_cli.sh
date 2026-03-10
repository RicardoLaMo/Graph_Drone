#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! git -C "$ROOT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Not inside a git repository: $ROOT_DIR" >&2
  exit 1
fi

git -C "$ROOT_DIR" config --local fetch.prune true
git -C "$ROOT_DIR" config --local pull.rebase false
git -C "$ROOT_DIR" config --local push.autoSetupRemote true
git -C "$ROOT_DIR" config --local rerere.enabled true
git config --global --add safe.directory "$ROOT_DIR"

if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git config --global user.name "$GIT_USER_NAME"
fi
if [[ -n "${GIT_USER_EMAIL:-}" ]]; then
  git config --global user.email "$GIT_USER_EMAIL"
fi

echo "Git remote:"
git -C "$ROOT_DIR" remote -v | sed -n '1,4p'
echo
echo "Git identity:"
echo "  user.name=$(git config --global --get user.name || echo '<unset>')"
echo "  user.email=$(git config --global --get user.email || echo '<unset>')"

echo
if command -v gh >/dev/null 2>&1; then
  if gh auth status >/dev/null 2>&1; then
    echo "GitHub CLI auth: ready"
  else
    echo "GitHub CLI auth: missing"
    echo "Run: gh auth login"
  fi
else
  echo "GitHub CLI not found."
  echo "Install 'gh' with your system package manager, then run: gh auth login"
fi

echo
if command -v hf >/dev/null 2>&1; then
  if hf auth whoami >/dev/null 2>&1; then
    echo "Hugging Face CLI auth: ready"
  else
    echo "Hugging Face CLI auth: missing"
    echo "Run: hf auth login"
  fi
elif command -v huggingface-cli >/dev/null 2>&1; then
  if huggingface-cli whoami >/dev/null 2>&1; then
    echo "Hugging Face CLI auth: ready"
  else
    echo "Hugging Face CLI auth: missing"
    echo "Run: huggingface-cli login"
  fi
else
  echo "Hugging Face CLI not found."
  echo "Run migration/bootstrap_h200.sh or install: python -m pip install 'huggingface_hub[cli]'"
fi
