#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
RUNTIME_ENV_DIR="$PROJECT_ROOT/.venv"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "找不到 local_tts/.venv，請先執行：$SCRIPT_DIR/setup_uv.sh" >&2
    exit 1
fi

if [[ ! -x "$RUNTIME_ENV_DIR/bin/python" ]]; then
    echo "找不到 GPT-SoVITS 主環境：$RUNTIME_ENV_DIR/bin/python" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$SCRIPT_DIR"
export NLTK_DATA="$PROJECT_ROOT/.nltk_data:$HOME/nltk_data${NLTK_DATA:+:$NLTK_DATA}"
cd "$PROJECT_ROOT"
exec uv run --no-project --python "$RUNTIME_ENV_DIR/bin/python" -m local_tts.start_web "$@"
