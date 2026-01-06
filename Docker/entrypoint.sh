#!/bin/bash

set -e

cd /workspace/GPT-SoVITS

echo "=========================================="
echo "  GPT-SoVITS Docker Container"
echo "=========================================="
echo ""

# Download URLs (HuggingFace)
PRETRAINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"

# Check and download pretrained models
if [ ! -d "GPT_SoVITS/pretrained_models/chinese-hubert-base" ]; then
    echo "[INFO] Downloading Pretrained Models..."
    rm -rf pretrained_models.zip
    wget -q --show-progress "$PRETRAINED_URL" -O pretrained_models.zip
    unzip -q -o pretrained_models.zip -d GPT_SoVITS
    rm -rf pretrained_models.zip
    echo "[DONE] Pretrained Models Downloaded"
else
    echo "[INFO] Pretrained Models already exist, skipping download"
fi

# Check and download G2PWModel
if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "[INFO] Downloading G2PWModel..."
    rm -rf G2PWModel.zip
    wget -q --show-progress "$G2PW_URL" -O G2PWModel.zip
    unzip -q -o G2PWModel.zip -d GPT_SoVITS/text
    rm -rf G2PWModel.zip
    echo "[DONE] G2PWModel Downloaded"
else
    echo "[INFO] G2PWModel already exists, skipping download"
fi

echo ""
echo "[INFO] Starting WebUI..."
echo "[INFO] Access the WebUI at: http://localhost:9874"
echo ""

# Start the WebUI
exec python webui.py
