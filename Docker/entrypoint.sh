#!/bin/bash

set -e

cd /workspace/GPT-SoVITS

echo "=========================================="
echo "  GPT-SoVITS Docker Container"
echo "=========================================="
echo ""

echo "[INFO] Starting WebUI..."
echo "[INFO] Access the WebUI at: http://localhost:9874"
echo ""

# Start the WebUI
exec python webui.py
