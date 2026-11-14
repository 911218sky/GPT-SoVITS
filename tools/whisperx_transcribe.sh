#!/usr/bin/env bash
# WhisperX transcribe (+ optional diarization) for GPT-SoVITS dataset prep.
# Usage:
#   ./tools/whisperx_transcribe.sh <audio> <output_dir> [--diarize] [--model PATH] [--language zh]
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${WHISPERX_PYTHON:-$ROOT/.venv/bin/python}"
BIN="${WHISPERX_BIN:-}"
if [[ -z "$BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/whisperx" ]]; then
    BIN="$ROOT/.venv/bin/whisperx"
  else
    BIN="$PY -m whisperx"
  fi
fi

AUDIO="${1:-}"
OUT_DIR="${2:-}"
shift 2 || true

if [[ -z "$AUDIO" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 <audio_file> <output_dir> [--diarize] [extra whisperx args...]" >&2
  exit 1
fi

MODEL_DEFAULT="${WHISPERX_MODEL:-}"
# common local layout outside repo (override with WHISPERX_MODEL)
if [[ -z "$MODEL_DEFAULT" && -f "$HOME/code/whisperx/models/faster-whisper-medium/model.bin" ]]; then
  MODEL_DEFAULT="$HOME/code/whisperx/models/faster-whisper-medium"
elif [[ -z "$MODEL_DEFAULT" && -f "/home/sky/code/whisperx/models/faster-whisper-medium/model.bin" ]]; then
  MODEL_DEFAULT="/home/sky/code/whisperx/models/faster-whisper-medium"
fi

DIARIZE=0
EXTRA=()
LANG="zh"
MODEL="$MODEL_DEFAULT"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --diarize) DIARIZE=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --language) LANG="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Set WHISPERX_MODEL to faster-whisper model dir (containing model.bin)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CMD=($BIN "$AUDIO" --model "$MODEL" --language "$LANG" --device cuda --compute_type float16 \
  --output_dir "$OUT_DIR" --output_format json --no_align)
if [[ "$DIARIZE" -eq 1 ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: --diarize usually needs HF_TOKEN (and accepting gated model terms)." >&2
  fi
  CMD+=(--diarize)
fi
CMD+=("${EXTRA[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
