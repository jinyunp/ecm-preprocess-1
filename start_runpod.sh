#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo "   ECMiner RAG + Qwen2-VL Setup (Runpod)"
echo "=============================================="

PROJECT_PATH="$(pwd)"
QWEN_VENV="qwen_env"
APP_VENV=".venv"
QDRANT_CONTAINER="ecminer-qdrant"
QDRANT_PORT=6333
QDRANT_VOLUME="$PROJECT_PATH/data/index"

echo "[INFO] Project path: $PROJECT_PATH"

### 1) Qwen2-VL venv (qwen_env)
echo "[1/5] Setting up Qwen2-VL environment (${QWEN_VENV})..."

if [ ! -d "$QWEN_VENV" ]; then
  echo "[INFO] Creating Qwen venv: $QWEN_VENV"
  python3 -m venv "$QWEN_VENV"
else
  echo "[INFO] Qwen venv already exists. Reusing."
fi

# shellcheck disable=SC1091
source "$QWEN_VENV/bin/activate"

echo "[INFO] Installing PyTorch for Qwen..."
GPU_CHECK=$( (command -v nvidia-smi && nvidia-smi >/dev/null 2>&1 && echo "GPU") || true )

pip install --upgrade pip
if [ -n "$GPU_CHECK" ]; then
  echo "[INFO] GPU detected → installing CUDA PyTorch (cu121)."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "[INFO] No GPU detected → installing CPU PyTorch."
  pip install torch torchvision torchaudio
fi

echo "[INFO] Installing Transformers + PIL + etc..."
pip install "transformers>=4.40" pillow accelerate safetensors

echo "[INFO] Preloading Qwen2-VL model..."
python - << 'EOF'
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
print(f"[INFO] Preloading model: {MODEL_NAME}")
AutoProcessor.from_pretrained(MODEL_NAME)
AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
print("[INFO] Qwen2-VL model download complete.")
EOF

deactivate
echo "[INFO] Qwen2-VL env ready: source ${QWEN_VENV}/bin/activate"


### 2) App venv (.venv) + requirements
echo "[2/5] Setting up app environment (${APP_VENV})..."

if [ ! -d "$APP_VENV" ]; then
  echo "[INFO] Creating app venv: $APP_VENV"
  python3 -m venv "$APP_VENV"
else
  echo "[INFO] App venv already exists. Reusing."
fi

# shellcheck disable=SC1091
source "$APP_VENV/bin/activate"

pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing app dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "[WARN] requirements.txt not found. Skipping app deps install."
fi

deactivate
echo "[INFO] App env ready: source ${APP_VENV}/bin/activate"


### 3) Qdrant Docker (있으면 실행, 없으면 경고만)
echo "[3/5] Checking Docker & Qdrant..."

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "[INFO] Docker daemon is running."

    mkdir -p "$QDRANT_VOLUME"

    if docker ps -q -f "name=^/${QDRANT_CONTAINER}$" >/dev/null; then
      echo "[INFO] Qdrant container already running: ${QDRANT_CONTAINER}"
    elif docker ps -aq -f "name=^/${QDRANT_CONTAINER}$" >/dev/null; then
      echo "[INFO] Starting existing Qdrant container: ${QDRANT_CONTAINER}"
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
        echo "[WARN] Qdrant readyz not responding yet."
      fi
    fi
  else
    echo "[WARN] Docker found but daemon not running. Qdrant 부분은 수동으로 띄워야 합니다."
  fi
else
  echo "[WARN] Docker not found in Runpod image. Qdrant는 별도 Pod/서비스로 띄우거나, Docker 지원 템플릿을 사용해야 합니다."
fi


### 4) preprocess + BM25 + init_index
echo "[4/5] Running preprocess / BM25 training / init_index..."

# shellcheck disable=SC1091
source "$APP_VENV/bin/activate"

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
  echo "[WARN] scripts/train_sparse_vectorizer.py not found. Skipping BM25 training."
fi

if [ -f "scripts/init_index.py" ]; then
  echo "[INFO] Running scripts/init_index.py (Qdrant indexing) ..."
  python scripts/init_index.py
else
  echo "[WARN] scripts/init_index.py not found. Skipping indexing."
fi

deactivate
echo "[INFO] Preprocess + indexing step finished."


### 5) 안내
echo
echo "=============================================="
echo " ✔ Setup finished (Runpod)"
echo "=============================================="
echo ""
echo "[Qwen2-VL 이미지 요약용]"
echo "  source ${QWEN_VENV}/bin/activate"
echo "  python summarize_images_with_qwen.py <sanitized_json_path>"
echo ""
echo "[RAG + Qdrant + UI]"
echo "  source ${APP_VENV}/bin/activate"
echo "  # Qdrant URL 은 settings/.env 에서 QDRANT_URL=http://localhost:${QDRANT_PORT}"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "=============================================="
