#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-}"

if [[ -z "${UV_BIN}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    echo "error: 'uv' 명령을 찾을 수 없습니다. https://docs.astral.sh/uv/ 안내에 따라 설치하세요." >&2
    exit 1
  fi
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "error: 'tesseract' 명령을 찾을 수 없습니다. 다음 명령으로 설치하세요:" >&2
  echo "  sudo apt install tesseract-ocr" >&2
  exit 1
fi

(
  cd "${SCRIPT_DIR}"
  if [[ ! -d ".venv" ]]; then
    echo "[bootstrap] creating virtualenv via 'uv venv'"
    "${UV_BIN}" venv
  fi
  echo "[bootstrap] syncing dependencies via 'uv sync'"
  "${UV_BIN}" sync
)

VENV_PY="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "error: .venv 초기화에 실패했습니다." >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VENV_PY}" -m mypkg.cli.main "$@"