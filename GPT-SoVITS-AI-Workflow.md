# GPT-SoVITS AI Workflow

Playbook for assistants and humans. Ask before long or destructive steps.

| Input | Workflow |
|-------|----------|
| Audio with BGM / recording | **A** clean → **WhisperX** (text + who) → filter speaker → train |
| Novel TXT | **B** role TTS |
| Video URL / file (e.g. Bilibili) | **C** yt-dlp → extract → then **A** |

Also: [Operations Guide](./GPT-SoVITS-Operations-Guide.md) · [Environment Setup](./GPT-SoVITS-Environment-Setup.md) · [local_tts README](./local_tts/README.md)

Host-specific paths/tokens belong in a **local** note outside this repo.

---

## 0. Preflight

```bash
cd /path/to/GPT-SoVITS
test -x .venv/bin/python && test -x local_tts/.venv/bin/python
command -v ffmpeg
command -v yt-dlp
command -v whisperx   # wrapper recommended
# Local Whisper model (medium, 8GB-friendly):
test -f /path/to/faster-whisper-medium/model.bin
```

**Default ASR + speaker labels: WhisperX** (not project FunASR).  
FunASR remains optional fallback for Chinese-only when Whisper quality is poor.

---

## 1. WhisperX (primary: text + who is speaking)

### 1.1 Why WhisperX

- One tool: transcription **and** speaker diarization (`--diarize`)
- Replaces the old FunASR-only path for dataset labeling when speaker ID matters
- On ~8GB GPU use **`medium`** (`float16`); drop to `small` on OOM

### 1.2 ASR-only (works without Hugging Face token)

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"  # if Hub is slow

whisperx "<audio.wav>" \
  --model "/path/to/faster-whisper-medium" \
  --language zh \
  --device cuda \
  --compute_type float16 \
  --output_dir "<out_dir>" \
  --output_format json \
  --no_align
```

Output JSON has `segments[].text` / `start` / `end`.

### 1.3 With speaker IDs (`--diarize`)

Requires:

1. Hugging Face account + token (`HF_TOKEN` or `--hf_token`)
2. Accept model terms on the diarization repo (WhisperX prints the URL, e.g. `pyannote/speaker-diarization-community-1`)

```bash
export HF_TOKEN="hf_..."   # user-provided; do not commit

whisperx "<audio.wav>" \
  --model "/path/to/faster-whisper-medium" \
  --language zh \
  --device cuda \
  --compute_type float16 \
  --diarize \
  --output_dir "<out_dir>" \
  --output_format json
```

Then **ask the user** which `SPEAKER_00` / `SPEAKER_01` / … to keep (play short clips or show durations).  
Export only that speaker’s segments → training clips / `.list`.

### 1.4 Convert WhisperX JSON → GPT-SoVITS `.list` (after speaker filter)

```text
/abs/path/clip.wav|SpeakerName|ZH|transcript text
```

Cut audio with `ffmpeg -ss -to` per segment if needed, or keep one merged track of the chosen speaker only.

### 1.5 FunASR fallback (optional)

Only if user wants project FunASR for Chinese:

```bash
.venv/bin/python -s tools/asr/funasr_asr.py -i "<clips>" -o "<out>" -l zh --backend fun-asr-nano
```

Do **not** use FunASR as default when the goal is “who spoke”.

---

## 2. Workflow A — Audio → dataset → train

Ask: path, `exp_name`, language, UVR?, slice params?, **WhisperX diarize + which speaker?**, train version (default **v2Pro**), go?

Steps:

1. UVR if BGM (`HP3` → `onnx_dereverb` → `DeEchoAggressive`) — does **not** separate people  
2. Slice if needed (`tools/slice_audio.py`) **or** cut by WhisperX segment times  
3. **WhisperX** (§1) → pick speaker → build `.list`  
4. Optional QA deletes (ask first)  
5. Train via `./local_tts/start_web.sh` after user says go → notify  

---

## 3. Workflow B — Novel TTS

Ask role → `start_api` → `batch_tts` → `finish_audio`. See [local_tts README](./local_tts/README.md).

---

## 4. Workflow C — Video → then A

```bash
yt-dlp -o "<dir>/%(title)s [%(id)s].%(ext)s" --no-playlist "<URL>"
# or audio-only:
yt-dlp -x --audio-format wav -o "<dir>/%(title)s.%(ext)s" --no-playlist "<URL>"

ffmpeg -y -i "<video>" -vn -ac 1 -ar 16000 "<dir>/raw.wav"
```

Then Workflow A + WhisperX. Multi-speaker: always run `--diarize` when HF token available; otherwise ASR-only and ask user for time ranges / reference clip.

---

## 5. Tool map

| Step | Tool |
|------|------|
| Download | `yt-dlp` |
| Extract | `ffmpeg` |
| BGM remove | UVR5 WebUI |
| **ASR + who spoke** | **WhisperX** |
| Chinese ASR fallback | `tools/asr/funasr_asr.py` |
| Train / novel TTS | `local_tts/start_web.sh`, `batch_tts.sh`, … |

---

## 6. Limits

1. UVR ≠ speaker ID  
2. `--diarize` needs HF token + accepted gated models  
3. Overlapped speech is hard  
4. First model download is large (~1.5GB for medium); use `aria2c -x16` for speed  
5. One heavy GPU job at a time on 8GB  

---

## 7. Quick path: Bilibili → one speaker → train

1. Ask URL, who to keep, `exp_name`  
2. yt-dlp → wav  
3. UVR if needed  
4. WhisperX `--diarize` → user picks speaker  
5. Build dataset → user says go → train → notify  
