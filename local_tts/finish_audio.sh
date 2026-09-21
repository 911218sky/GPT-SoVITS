#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "找不到 local_tts/.venv，請先執行：$SCRIPT_DIR/setup_uv.sh" >&2
    exit 1
fi

usage() {
    cat <<'EOF'
用法：
  ./local_tts/finish_audio.sh <指令> [參數…]

指令：
  all     推薦：每段只編碼一次（清音±調速）後合併（較快，效果同舊三步）
  clean   只清靜音
  tempo   只調語速
  merge   只合併

範例：
  ./local_tts/finish_audio.sh all \
    --input "local_tts/output/班级公交求生_658_真人男" \
    --tempo 0.9

  ./local_tts/finish_audio.sh clean --input ... --output ...
  ./local_tts/finish_audio.sh tempo --input ... --output ... --tempo 0.9
  ./local_tts/finish_audio.sh merge --input-folder ... --output-dir ...
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    -h|--help|help)
        usage
        exit 0
        ;;
    all|clean|tempo|merge)
        ;;
    *)
        echo "未知指令：$1" >&2
        usage >&2
        exit 2
        ;;
esac

export PYTHONPATH="$ENV_DIR/lib/python3.10/site-packages:$PROJECT_ROOT:$SCRIPT_DIR:$PROJECT_ROOT/.venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec uv run --no-project --python "$ENV_DIR/bin/python" -m local_tts.finish_audio "$@"
