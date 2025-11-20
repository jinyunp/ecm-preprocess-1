#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo "   Qwen2-VL Local Environment + Model Setup"
echo "=============================================="

# -----------------------------
# Config (필요시 수정/환경변수로 override)
# -----------------------------
REPO_ID="${REPO_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"   # HF repo id
TARGET_DIR="${TARGET_DIR:-/workspace/qwen}"         # 모델 저장 경로
HF_TOKEN="${HF_TOKEN:-}"                            # export HF_TOKEN=... 로도 사용 가능

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip3}"
VENV_DIR="${VENV_DIR:-qwen_env}"

echo "[Config] REPO_ID    = ${REPO_ID}"
echo "[Config] TARGET_DIR = ${TARGET_DIR}"
echo "[Config] VENV_DIR   = ${VENV_DIR}"
echo "[Config] PYTHON_BIN = ${PYTHON_BIN}"
echo "[Config] PIP_BIN    = ${PIP_BIN}"
echo "=============================================="

# -----------------------------
# 1) System update (optional)
# -----------------------------
echo "[1/7] Updating system..."
sudo apt-get update -y

# -----------------------------
# 2) Install base deps
# -----------------------------
echo "[2/7] Installing Python, pip, venv, git, git-lfs..."
sudo apt-get install -y python3 python3-pip python3-venv git git-lfs
git lfs install || true

# -----------------------------
# 3) Create & activate venv
# -----------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[3/7] Creating Python virtual environment at ${VENV_DIR} ..."
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
else
  echo "[3/7] Virtual environment already exists at ${VENV_DIR}, reusing..."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# venv 안에서는 python/pip 별칭 고정
PYTHON_BIN="python"
PIP_BIN="pip"

# -----------------------------
# 4) Install PyTorch
# -----------------------------
echo "[4/7] Installing PyTorch..."

GPU_CHECK=$(lspci | grep -i nvidia || true)

if [[ -n "$GPU_CHECK" ]]; then
    echo "NVIDIA GPU detected → installing CUDA-enabled PyTorch (cu121)."
    ${PIP_BIN} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected → installing CPU PyTorch."
    ${PIP_BIN} install torch torchvision torchaudio
fi

# -----------------------------
# 5) Install Transformers + etc
# -----------------------------
echo "[5/7] Installing Transformers + dependencies..."
${PIP_BIN} install -U pip
${PIP_BIN} install "transformers>=4.40" pillow accelerate safetensors "huggingface_hub[cli]"

# -----------------------------
# 6) (Optional) Hugging Face login
# -----------------------------
if [[ -n "${HF_TOKEN}" ]]; then
  echo "[6/7] Logging in to Hugging Face with HF_TOKEN (non-interactive)..."
  ${PYTHON_BIN} - <<PY
import os, subprocess
token = os.environ.get("HF_TOKEN","").strip()
if token:
    cmd = ["huggingface-cli","login","--token", token, "--add-to-git-credential"]
    print("[CLI]", " ".join(cmd))
    subprocess.run(cmd, check=False)
else:
    print("[INFO] HF_TOKEN empty, skipping login.")
PY
else
  echo "[6/7] HF_TOKEN not set, skipping HF login."
fi

# -----------------------------
# 7) Download Qwen model to /workspace/qwen
# -----------------------------
echo "[7/7] Downloading Qwen model snapshot via huggingface_hub ..."
${PYTHON_BIN} - <<PY
from huggingface_hub import snapshot_download
import os

repo_id = "${REPO_ID}"
local_dir = "${TARGET_DIR}"

os.makedirs(local_dir, exist_ok=True)
print(f"[INFO] Downloading snapshot of {repo_id} to {local_dir} ...")
snapshot_download(
    repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print("[INFO] Snapshot download complete.")
PY

echo
echo "=============================================="
echo " ✔ Qwen2-VL Environment & Model Ready"
echo "=============================================="
echo ""
echo "Virtualenv:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Model local path:"
echo "  ${TARGET_DIR}"
echo ""
echo "Python 예시 코드:"
echo "  from transformers import AutoProcessor, AutoModelForVision2Seq"
echo "  model = AutoModelForVision2Seq.from_pretrained('${TARGET_DIR}')"
echo "  processor = AutoProcessor.from_pretrained('${TARGET_DIR}')"
echo ""
echo "이미지 요약 스크립트 실행 예시:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python summarize_images_with_qwen.py /root/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json"
echo ""
