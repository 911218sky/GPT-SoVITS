#!/bin/bash
#
# GPT-SoVITS Docker 安裝腳本
# 呼叫 install.sh 並下載 Docker 專用的額外模型
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 啟動 micromamba 環境
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 呼叫主安裝腳本（PyTorch、依賴、預訓練模型都在這裡處理）
echo "[INFO] Running main install script..."
bash install.sh --device "CU${CUDA_VERSION//./}" --source HF --download-uvr5

# === Docker 專用：下載額外模型 ===

# 安裝模型下載工具
echo "[INFO] Installing model download tools..."
python -m pip install huggingface_hub modelscope -q

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
python -m pip cache purge
rm -rf /tmp/* /var/tmp/* 2>/dev/null || true
rm -rf "$HOME/miniconda3/pkgs" "$HOME/.conda" "$HOME/.cache"
mkdir -p "$HOME/miniconda3/pkgs"

echo "[INFO] Installation completed!"
