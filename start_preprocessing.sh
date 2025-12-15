#!/usr/bin/env bash
set -e

echo "=============================================="
echo "   ECMiner Preprocessing Environment Setup"
echo "   (GPU + Qwen2.5-VL-3B-Instruct + Qwen2.5-7B-Instruct)"
echo "=============================================="

# -----------------------------------------------------------------------------------------
# 0) 프로젝트 루트 이동
# -----------------------------------------------------------------------------------------
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# Qwen 모델 디렉터리 설정
BASE_QWEN_DIR="/workspace/qwen"
VL_MODEL_DIR="$BASE_QWEN_DIR/VL"
TXT_MODEL_DIR="$BASE_QWEN_DIR/txt"

VL_MODEL_REPO_HF="Qwen/Qwen2.5-VL-3B-Instruct"
TXT_MODEL_REPO_HF="Qwen/Qwen2.5-7B-Instruct"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Qwen base dir: $BASE_QWEN_DIR"
echo "[INFO] VL model dir : $VL_MODEL_DIR"
echo "[INFO] TXT model dir: $TXT_MODEL_DIR"

# -----------------------------------------------------------------------------------------
# 1) 필수 시스템 패키지 설치
# -----------------------------------------------------------------------------------------
echo "[1/7] Installing system packages (python, git, curl, OCR etc.)..."

if command -v apt-get &> /dev/null; then
    apt-get update -y || true
    apt-get install -y \
        python3 python3-venv python3-pip \
        git curl build-essential \
        tesseract-ocr tesseract-ocr-kor \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
        || true
else
    echo "[WARN] apt-get not supported on this system."
fi

# -----------------------------------------------------------------------------------------
# 2) Git-Xet 설치 (HF 공식 설치 스크립트 + --system)
# -----------------------------------------------------------------------------------------
echo "[2/7] Installing Git-Xet..."

if ! git xet --version &> /dev/null; then
    set +e
    curl --proto '=https' --tlsv1.2 -sSf \
      https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh \
      | sh
    XET_SCRIPT_EXIT=$?
    set -e

    if [ $XET_SCRIPT_EXIT -ne 0 ]; then
        echo "[WARN] Git-Xet installation failed. (exit=$XET_SCRIPT_EXIT)"
    else
        git xet install --system || echo "[WARN] git xet install --system failed."
    fi
else
    echo "[INFO] Git-Xet already installed."
fi

# -----------------------------------------------------------------------------------------
# 3) Python 가상환경 생성
# -----------------------------------------------------------------------------------------
echo "[3/7] Creating Python virtual environment (.venv)..."

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip

# -----------------------------------------------------------------------------------------
# 4) GPU용 PyTorch + torchvision 설치
# -----------------------------------------------------------------------------------------
echo "[4/7] Installing GPU torch + torchvision..."

pip uninstall -y torch torchvision || true

# CUDA 12.1 권장 (RunPod 대부분)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# GPU 확인
python3 - <<EOF
import torch
print("[CHECK] torch version:", torch.__version__)
print("[CHECK] CUDA available:", torch.cuda.is_available())
EOF

# -----------------------------------------------------------------------------------------
# 5) Transformers, HF, OCR libs, utils 설치
# -----------------------------------------------------------------------------------------
echo "[5/7] Installing transformers + supporting libraries..."

pip install \
  "transformers>=4.40.0" \
  huggingface_hub \
  accelerate \
  pillow \
  pytesseract \
  safetensors \
  einops \
  sentencepiece \
  httpx \
  hf_transfer

# -----------------------------------------------------------------------------------------
# 6) Qwen 모델 다운로드 (VL + TEXT 분리)
# -----------------------------------------------------------------------------------------
echo "[6/7] Downloading Qwen models ..."

mkdir -p "$VL_MODEL_DIR"
mkdir -p "$TXT_MODEL_DIR"

# fast download 강제 off (hf_transfer 문제 방지)
echo "[6.1] Downloading VL model: $VL_MODEL_REPO_HF -> $VL_MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER=0 python3 - <<EOF
from huggingface_hub import snapshot_download

vl_repo = "$VL_MODEL_REPO_HF"
vl_dir = "$VL_MODEL_DIR"

print(f"[INFO] Downloading VL model: {vl_repo}")
snapshot_download(repo_id=vl_repo, local_dir=vl_dir)
print("[INFO] VL model downloaded successfully to", vl_dir)
EOF

echo "[6.2] Downloading TEXT model: $TXT_MODEL_REPO_HF -> $TXT_MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER=0 python3 - <<EOF
from huggingface_hub import snapshot_download

txt_repo = "$TXT_MODEL_REPO_HF"
txt_dir = "$TXT_MODEL_DIR"

print(f"[INFO] Downloading TEXT model: {txt_repo}")
snapshot_download(repo_id=txt_repo, local_dir=txt_dir)
print("[INFO] TEXT model downloaded successfully to", txt_dir)
EOF

# -----------------------------------------------------------------------------------------
# 7) 환경 변수 설정
# -----------------------------------------------------------------------------------------
echo "[7/7] Setting env variables (.venv persist)..."

{
  echo ""
  echo "# ECMiner preprocessing environment"
  echo "export QWEN_MODEL_PATH=\"$VL_MODEL_DIR\""
  echo "export QWEN_TEXT_MODEL_PATH=\"$TXT_MODEL_DIR\""
  echo "export PYTHONPATH=\"$PROJECT_ROOT/src:\$PYTHONPATH\""
} >> .venv/bin/activate

echo "[INFO] QWEN_MODEL_PATH      will be set to: $VL_MODEL_DIR"
echo "[INFO] QWEN_TEXT_MODEL_PATH will be set to: $TXT_MODEL_DIR"
echo "[INFO] PYTHONPATH will include: $PROJECT_ROOT/src"

echo "=============================================="
echo "Setup Complete!"
echo ""
echo "To start using the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "VL model path  (Qwen-VL):     \$QWEN_MODEL_PATH"
echo "TEXT model path (Qwen-Text):  \$QWEN_TEXT_MODEL_PATH"
echo ""
echo "To run preprocessing:"
echo "    python run_full_preprocess.py /path/to/docx_or_folder"
echo "=============================================="
