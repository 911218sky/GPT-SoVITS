#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "找不到 local_tts/.venv，請先執行：$SCRIPT_DIR/setup_uv.sh" >&2
    exit 1
fi

export PYTHONPATH="$ENV_DIR/lib/python3.10/site-packages:$PROJECT_ROOT:$SCRIPT_DIR:$PROJECT_ROOT/.venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec uv run --no-project --python "$ENV_DIR/bin/python" -m local_tts.merge_audio "$@"
