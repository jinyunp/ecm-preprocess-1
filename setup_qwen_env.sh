#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo "   ECMiner RAG + Qwen2-VL Setup Script"
echo "=============================================="

PROJECT_PATH="$(pwd)"
BASHRC="$HOME/.bashrc"

QWEN_VENV="qwen_env"
APP_VENV=".venv"
QDRANT_CONTAINER="ecminer-qdrant"
QDRANT_PORT=6333
QDRANT_VOLUME="$PROJECT_PATH/data/index"

echo "[INFO] Project path: $PROJECT_PATH"


### ─────────────────────────────────────────────
### 1) System update & base deps (Ubuntu/WSL)
### ─────────────────────────────────────────────
if command -v apt >/dev/null 2>&1; then
  echo "[1/6] Updating system & installing base packages..."
  sudo apt-get update -y
  sudo apt-get install -y \
    python3 python3-pip python3-venv \
    git curl ca-certificates \
    lsb-release
else
  echo "[WARN] 'apt' not found. Skipping system package install."
fi


### ─────────────────────────────────────────────
### 2) Qwen2-VL 전용 venv(qwen_env) 생성 및 모델 다운로드
### ─────────────────────────────────────────────
echo "[2/6] Setting up Qwen2-VL environment (${QWEN_VENV})..."

if [ ! -d "$QWEN_VENV" ]; then
  echo "[INFO] Creating Python virtual environment for Qwen: $QWEN_VENV"
  python3 -m venv "$QWEN_VENV"
else
  echo "[INFO] Qwen venv already exists. Reusing: $QWEN_VENV"
fi

# shellcheck disable=SC1091
source "$QWEN_VENV/bin/activate"

echo "[INFO] Installing PyTorch for Qwen..."
GPU_CHECK=$( (lspci | grep -i nvidia) || true )

if [ -n "$GPU_CHECK" ]; then
  echo "[INFO] NVIDIA GPU detected → installing CUDA-enabled PyTorch (cu121)."
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "[INFO] No NVIDIA GPU detected → installing CPU PyTorch."
  pip install --upgrade pip
  pip install torch torchvision torchaudio
fi

echo "[INFO] Installing Transformers + PIL + other libs (Qwen side)..."
pip install "transformers>=4.40" pillow accelerate safetensors

echo "[INFO] Preloading Qwen2-VL model (this may take some time)..."
python - << 'EOF'
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
print(f"[INFO] Preloading model: {MODEL_NAME} (first run may take time)")
AutoProcessor.from_pretrained(MODEL_NAME)
AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
print("[INFO] Qwen2-VL model download complete.")
EOF

deactivate
echo "[INFO] Qwen2-VL environment ready: source ${QWEN_VENV}/bin/activate"


### ─────────────────────────────────────────────
### 3) 앱용 venv(.venv) 생성 및 requirements 설치
### ─────────────────────────────────────────────
echo "[3/6] Setting up app environment (${APP_VENV})..."

if [ ! -d "$APP_VENV" ]; then
  echo "[INFO] Creating app virtual environment: $APP_VENV"
  python3 -m venv "$APP_VENV"
else
  echo "[INFO] App venv already exists. Reusing: $APP_VENV"
fi

# shellcheck disable=SC1091
source "$APP_VENV/bin/activate"

echo "[INFO] Installing app dependencies from requirements.txt..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "[WARN] requirements.txt not found. Skipping app dependency install."
fi

deactivate
echo "[INFO] App environment ready: source ${APP_VENV}/bin/activate"


### ─────────────────────────────────────────────
### 4) Docker & Qdrant 컨테이너 구성
### ─────────────────────────────────────────────
echo "[4/6] Checking Docker & launching Qdrant..."

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] Docker is not installed or not in PATH."
  echo "        Please install Docker (or Docker Desktop + WSL2 integration) first."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon is not running."
  echo "        Start Docker Desktop (on Windows) or 'sudo systemctl start docker' (Linux)."
  exit 1
fi

mkdir -p "$QDRANT_VOLUME"

if docker ps -q -f "name=^/${QDRANT_CONTAINER}$" >/dev/null; then
  echo "[INFO] Qdrant container already running: ${QDRANT_CONTAINER}"
elif docker ps -aq -f "name=^/${QDRANT_CONTAINER}$" >/dev/null; then
  echo "[INFO] Found stopped Qdrant container. Starting: ${QDRANT_CONTAINER}"
  docker start "$QDRANT_CONTAINER"
else
  echo "[INFO] Creating new Qdrant container: ${QDRANT_CONTAINER}"
  docker run -d \
    --name "$QDRANT_CONTAINER" \
    --restart unless-stopped \
    -p ${QDRANT_PORT}:6333 \
    -v "$QDRANT_VOLUME:/qdrant/storage" \
    qdrant/qdrant:latest
fi

sleep 2
if command -v curl >/dev/null 2>&1; then
  if curl -fsS "http://localhost:${QDRANT_PORT}/readyz" >/dev/null 2>&1; then
    echo "[INFO] Qdrant is ready at http://localhost:${QDRANT_PORT}"
  else
    echo "[WARN] Qdrant readyz not responding yet, but container is running."
  fi
else
  echo "[WARN] curl not available. Skipping Qdrant readyz check."
fi


### ─────────────────────────────────────────────
### 5) 전처리 + BM25 훈련 + Qdrant 인덱싱
### ─────────────────────────────────────────────
echo "[5/6] Running preprocess + BM25 training + Qdrant indexing..."

# shellcheck disable=SC1091
source "$APP_VENV/bin/activate"

# 필요에 따라 사용/해제 가능
if [ -f "scripts/preprocess.py" ]; then
  echo "[INFO] Running scripts/preprocess.py ..."
  python scripts/preprocess.py
else
  echo "[WARN] scripts/preprocess.py not found. Skipping."
fi

if [ -f "scripts/train_sparse_vectorizer.py" ]; then
  echo "[INFO] Running scripts/train_sparse_vectorizer.py ..."
  python scripts/train_sparse_vectorizer.py
else
  echo "[WARN] scripts/train_sparse_vectorizer.py not found. Skipping."
fi

if [ -f "scripts/init_index.py" ]; then
  echo "[INFO] Running scripts/init_index.py (Qdrant indexing) ..."
  python scripts/init_index.py
else
  echo "[WARN] scripts/init_index.py not found. Skipping indexing."
fi

deactivate
echo "[INFO] Preprocess + indexing steps completed."


### ─────────────────────────────────────────────
### 6) 완료 안내 (UI 실행 방법)
### ─────────────────────────────────────────────
echo
echo "=============================================="
echo " ✔ All setup complete!"
echo "=============================================="
echo ""
echo "[Qwen2-VL 이미지 요약 환경]"
echo "  source ${QWEN_VENV}/bin/activate"
echo "  python summarize_images_with_qwen.py <sanitized_json_path>"
echo ""
echo "[RAG 앱 + Qdrant]"
echo "  1) App venv 활성화:"
echo "       source ${APP_VENV}/bin/activate"
echo ""
echo "  2) FastAPI UI 실행:"
echo "       uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  3) 브라우저에서 접속:"
echo "       http://localhost:8000"
echo ""
echo "※ Qdrant URL은 설정에서 QDRANT_URL=http://localhost:${QDRANT_PORT} 으로 맞춰주세요."
echo "=============================================="
