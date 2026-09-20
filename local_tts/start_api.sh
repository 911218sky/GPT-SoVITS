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
export NLTK_DATA="$PROJECT_ROOT/.nltk_data:$HOME/nltk_data${NLTK_DATA:+:$NLTK_DATA}"
# 長篇批次：每 N 次請求 empty_cache（預設 64，偏速度），壓碎片、少吃共享顯存；
# 0=從不，1=每請求（最穩最慢）。仍漲顯存可改 16 或 1。T2S tqdm 預設關。
export GPT_SOVITS_EMPTY_CACHE="${GPT_SOVITS_EMPTY_CACHE:-64}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
cd "$PROJECT_ROOT"
exec uv run --no-project --python "$ENV_DIR/bin/python" -m local_tts.start_api "$@"
