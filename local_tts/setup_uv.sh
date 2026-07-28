#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"

command -v uv >/dev/null 2>&1 || {
    echo "找不到 uv，請先安裝 uv。" >&2
    exit 1
}

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "警告：找不到 ffmpeg；API/WebUI 可啟動，但音訊清理與合併需要先安裝 ffmpeg。" >&2
fi

uv venv --allow-existing --python 3.10 "$ENV_DIR"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    uv pip install --python "$ENV_DIR/bin/python" fastapi==0.115.6 requests starlette==0.41.3 tqdm
    echo "偵測到專案既有 .venv，啟動器會共用其 GPT-SoVITS 依賴。"
else
    uv pip install --python "$ENV_DIR/bin/python" -r "$PROJECT_ROOT/requirements.txt"
    uv pip install --python "$ENV_DIR/bin/python" fastapi==0.115.6 starlette==0.41.3
fi

echo "uv 環境已建立：$ENV_DIR"
echo "啟動 API：$SCRIPT_DIR/start_api.sh"
echo "啟動 Web：$SCRIPT_DIR/start_web.sh"
