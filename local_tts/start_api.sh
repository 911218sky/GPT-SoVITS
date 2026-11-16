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
# 推理途中不要每請求 empty_cache（CUDA sync，會把熱跑從約 320 字/秒拉低）。
# 0=批次途中不收；跑完由 batch_tts 打 /control?command=empty_cache。
# 長篇若顯存一路爬升，再改 16 或 1。T2S tqdm 預設關。
export GPT_SOVITS_EMPTY_CACHE="${GPT_SOVITS_EMPTY_CACHE:-0}"
# 8GB GPU: CUDA Graph static buffers starve VITS. default off; force with CUDAGraph=1.
export CUDAGraph="${CUDAGraph:-0}"
# Per-token T2S CUDA graph. Portable across CUDA GPUs. Set 0 to force eager T2S.
export GPT_SOVITS_T2S_STEP_GRAPH="${GPT_SOVITS_T2S_STEP_GRAPH:-1}"
# Winning 8GB novel TTS: step graph + micro=16 + empty_cache=0, release once at end.
export GPT_SOVITS_VITS_MICROBATCH="${GPT_SOVITS_VITS_MICROBATCH:-16}"
export GPT_SOVITS_T2S_MAX_BATCH="${GPT_SOVITS_T2S_MAX_BATCH:-8}"
# v5turbo on 8GB. CFM groups of 16 and T2S slices of 32; keep the last 1.6s of the reference mel.
export GPT_SOVITS_V5_CFM_BATCH="${GPT_SOVITS_V5_CFM_BATCH:-16}"
export GPT_SOVITS_V5_T2S_BATCH="${GPT_SOVITS_V5_T2S_BATCH:-32}"
export GPT_SOVITS_V5_REF_FRAMES="${GPT_SOVITS_V5_REF_FRAMES:-160}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
# Run-ons longer than this get one extra comma. 20-char split measured ~297 chars/s on 吃神者 4827.
export GPT_SOVITS_FAST_SPLIT="${GPT_SOVITS_FAST_SPLIT:-20}"
cd "$PROJECT_ROOT"
exec "$ENV_DIR/bin/python" -m local_tts.start_api "$@"
