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
# 長篇批次：每 N 次請求 empty_cache（預設 1=每請求，最穩）；
# 0=從不，N=每 N 次。想加速可改 16。T2S tqdm 預設關。
export GPT_SOVITS_EMPTY_CACHE="${GPT_SOVITS_EMPTY_CACHE:-1}"
# 8GB GPU: CUDA Graph static buffers starve VITS. default off; force with CUDAGraph=1.
export CUDAGraph="${CUDAGraph:-0}"
# Per-token T2S CUDA graph. Portable across CUDA GPUs. Set 0 to force eager T2S.
export GPT_SOVITS_T2S_STEP_GRAPH="${GPT_SOVITS_T2S_STEP_GRAPH:-1}"
# Winning 8GB novel TTS: micro=16 + empty_cache=1 => ~230 chars/s (CUDAGraph must stay 0)
export GPT_SOVITS_VITS_MICROBATCH="${GPT_SOVITS_VITS_MICROBATCH:-16}"
export GPT_SOVITS_T2S_MAX_BATCH="${GPT_SOVITS_T2S_MAX_BATCH:-8}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
cd "$PROJECT_ROOT"
exec "$ENV_DIR/bin/python" -m local_tts.start_api "$@"
