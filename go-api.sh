#!/bin/bash

echo "=========================================="
echo "  GPT-SoVITS API Server"
echo "  Default Port: 9880"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/runtime/python/bin:$PATH"

echo "[INFO] Starting API Server..."
echo "[INFO] API will be available at: http://127.0.0.1:9880"
echo "[INFO] Press Ctrl+C to stop the server"
echo ""

./runtime/python/bin/python api_v2.py -a 127.0.0.1 -p 9880
