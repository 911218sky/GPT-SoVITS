#!/bin/bash
#
# GPT-SoVITS Docker 安裝腳本
# 呼叫 install.sh 並下載 Docker 專用的額外模型
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 啟動 uv 環境
source "$HOME/uv/etc/profile.d/uv.sh"

# 轉換 CUDA 版本格式：12.6.0 -> CU126, 13.0.0 -> CU130
CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1,2 | tr -d '.')
DEVICE_ARG="CU${CUDA_MAJOR_MINOR}"

# 呼叫主安裝腳本（PyTorch、依賴、預訓練模型都在這裡處理）
echo "[INFO] Running main install script with device: $DEVICE_ARG"
bash install.sh --device "$DEVICE_ARG" --source HF --download-uvr5

# === Docker 專用：下載額外模型 ===

# 安裝模型下載工具
echo "[INFO] Installing model download tools..."
uv pip install huggingface_hub modelscope -q

# 下載 FunASR 模型
echo "[INFO] Downloading FunASR models..."
python3 "$SCRIPT_DIR/download_models.py"

# 下載 fast-langdetect 模型
echo "[INFO] Downloading fast-langdetect model..."
mkdir -p GPT_SoVITS/pretrained_models/fast_langdetect
curl -L -o GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin \
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# === 清理快取（Docker image 優化）===
echo "[INFO] Cleaning up caches..."
uv cache clean
rm -rf /tmp/* /var/tmp/* 2>/dev/null || true
rm -rf "$HOME/.cache"

echo "[INFO] Installation completed!"
