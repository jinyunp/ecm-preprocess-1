#!/usr/bin/env bash
set -e

echo "=============================================="
echo "   Qwen2-VL Local Environment Setup Script"
echo "=============================================="

# -----------------------------
# 1) System update (optional)
# -----------------------------
echo "[1/7] Updating system..."
sudo apt-get update -y

# -----------------------------
# 2) Install Python deps
# -----------------------------
echo "[2/7] Installing Python & pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# -----------------------------
# 3) Create virtual environment
# -----------------------------
echo "[3/7] Creating Python virtual environment..."
python3 -m venv qwen_env
source qwen_env/bin/activate

# -----------------------------
# 4) Install PyTorch
# -----------------------------
echo "[4/7] Installing PyTorch..."

# GPU available?
GPU_CHECK=$(lspci | grep -i nvidia || true)

if [ -n "$GPU_CHECK" ]; then
    echo "NVIDIA GPU detected → installing CUDA-enabled PyTorch."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected → installing CPU PyTorch."
    pip install torch torchvision torchaudio
fi


# -----------------------------
# 5) Install Transformers + PIL + other libs
# -----------------------------
echo "[5/7] Installing Transformers + dependencies..."
pip install "transformers>=4.40" pillow accelerate safetensors

# -----------------------------
# 6) Download Qwen2-VL model (auto cache)
#     - This step may take time depending on model size
# -----------------------------
echo "[6/7] Preparing Qwen2-VL model cache..."
python3 - << 'EOF'
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
print(f"[INFO] Preloading model: {MODEL_NAME} (first run may take time)")
AutoProcessor.from_pretrained(MODEL_NAME)
AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
print("[INFO] Model download complete.")
EOF


# -----------------------------
# 7) User instructions
# -----------------------------
echo "[7/7] Setup completed successfully!"
echo ""
echo "To activate environment:"
echo "    source qwen_env/bin/activate"
echo ""
echo "To run the image LLM summarization script:"
echo "    python summarize_images_with_qwen.py <sanitized_json_path>"
echo ""
echo "Example:"
echo "    python summarize_images_with_qwen.py \\"
echo "      /root/ecm-preprocess-1/output/processed/1장_v3.1/v0/_sanitized/1장_v3.1_sanitized.json"
echo ""
echo "=============================================="
echo " ✔ Qwen2-VL Local Environment Ready"
echo "=============================================="
