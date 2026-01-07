#!/bin/bash
#
# GPT-SoVITS Docker 安裝腳本 (使用 uv)
# 安裝 PyTorch、依賴、預訓練模型
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 啟動 uv 環境
source "$HOME/uv/etc/profile.d/uv.sh"

UV="$HOME/uv/uv"

# === 安裝 PyTorch ===
echo "[INFO] Installing PyTorch for CUDA ${CUDA_VERSION}..."
case "${CUDA_VERSION}" in
    "12.6")
        $UV pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
        ;;
    "12.8")
        $UV pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
        ;;
    *)
        echo "[ERROR] Unsupported CUDA version: ${CUDA_VERSION}"
        exit 1
        ;;
esac

# === 安裝依賴 ===
echo "[INFO] Installing dependencies..."
# 移除 --no-binary 限制
grep -v "^--no-binary" requirements.txt > requirements_optimized.txt || cp requirements.txt requirements_optimized.txt
$UV pip install -r requirements_optimized.txt
$UV pip install -r extra-req.txt
rm -f requirements_optimized.txt

# === 下載預訓練模型 ===
echo "[INFO] Installing model download tools..."
$UV pip install huggingface_hub modelscope -q

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
$UV cache clean
rm -rf /tmp/* /var/tmp/* 2>/dev/null || true
rm -rf "$HOME/.cache"

echo "[INFO] Installation completed!"
