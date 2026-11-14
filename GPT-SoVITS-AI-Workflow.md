# GPT-SoVITS AI Workflow

Playbook for assistants and humans. **Ask before converting or training.** Do not start WhisperX / FunASR / UVR / train until the user answers the gates below (or says “use your recommendation / go”).

| Input | Flow |
|-------|------|
| Audio / recording | **A** |
| Novel TXT | **B** |
| Video URL / file | **C** → then often **A** |

Also: [Operations Guide](./GPT-SoVITS-Operations-Guide.md) · [Environment Setup](./GPT-SoVITS-Environment-Setup.md) · [local_tts README](./local_tts/README.md)

Scripts in this repo:

| Script | Purpose |
|--------|---------|
| [`tools/whisperx_transcribe.sh`](./tools/whisperx_transcribe.sh) | Run WhisperX (optional `--diarize`) |
| [`tools/whisperx_json_to_list.py`](./tools/whisperx_json_to_list.py) | WhisperX JSON → cut clips + GPT-SoVITS `.list` |

Host-specific model paths / tokens: keep in a **local** note outside the repo (`WHISPERX_MODEL`, `HF_TOKEN`).

---

## 0. Preflight

```bash
cd /path/to/GPT-SoVITS
test -x .venv/bin/python && test -x local_tts/.venv/bin/python
command -v ffmpeg && command -v yt-dlp
./tools/whisperx_transcribe.sh 2>/dev/null || true   # prints usage
# Model dir must contain model.bin (set WHISPERX_MODEL if not default)
```

---

## 0.1 Mandatory interview — BEFORE any convert / ASR

Present options, give a **recommendation**, wait for answers.

### Q1 — What is the input?

- [ ] Local audio  
- [ ] Video file  
- [ ] URL (Bilibili / YouTube / …)  
- [ ] Novel TXT only → jump to Workflow **B**

### Q2 — Transcription / labeling method (ASR)

| Option | When | Recommend? |
|--------|------|------------|
| **A. WhisperX** | Default: text + can label who spoke | **Yes (default)** |
| **B. FunASR** (`tools/asr/funasr_asr.py`) | Chinese-only, no need for “who spoke” | Fallback |
| **C. Skip ASR** | User already has `.list` / labels | Rare |

**Recommendation to user:** use **WhisperX (`medium`, CUDA float16)** on ~8GB GPUs.

### Q3 — Extract a specific person? (multi-speaker)

| Option | Meaning | Needs |
|--------|---------|--------|
| **1. Yes — diarize, then I pick** | WhisperX `--diarize` → show SPEAKER_00/01… → user chooses | `HF_TOKEN` + accept gated model |
| **2. Yes — I already know time range / give reference clip** | Cut with ffmpeg / match later | No HF token |
| **3. No — keep all speech** | ASR only, one “speaker” name in `.list` | No HF token |
| **4. Unsure** | Assistant recommends **1** if conversation/podcast/multi-host; else **3** | — |

**Recommendation:**

- One clear host / single voice source → **3**  
- Interview / multi-person video → **1** (if token available), else **2**

Ask explicitly: “要不要只留某一個人的聲音？若要，用自動分人（需 HF token）還是你指定時間／參考音？”

### Q4 — Remove BGM / UVR?

Recommend **yes** if music/noise is obvious: `HP3` → `onnx_dereverb` → `DeEchoAggressive`.  
UVR does **not** separate person A vs B.

### Q5 — Slice settings

WebUI defaults vs shorter/longer presets (see §2.3). Ask or offer “use defaults”.

### Q6 — After labels: train now?

Version (recommend **v2Pro** if matching `local_tts`), `exp_name`, register role later?  
**Start train only after user says go.**

### Q7 — Anything else custom?

If user says “沒問題 / 用推薦 / 開跑” → proceed with the chosen options.

---

## 1. WhisperX how-to

### 1.1 ASR only

```bash
export WHISPERX_MODEL="/path/to/faster-whisper-medium"
./tools/whisperx_transcribe.sh "<audio.wav>" "<out_dir>"
```

### 1.2 ASR + who spoke

```bash
export HF_TOKEN="hf_..."   # user-provided; never commit
./tools/whisperx_transcribe.sh "<audio.wav>" "<out_dir>" --diarize
```

Then ask which `SPEAKER_xx` to keep (list durations / sample clips).

### 1.3 Build GPT-SoVITS `.list`

```bash
# keep one speaker:
.venv/bin/python tools/whisperx_json_to_list.py \
  "<out_dir>/<stem>.json" \
  --audio "<audio.wav>" \
  --out-dir "<data>/<exp>/slicer" \
  --speaker SPEAKER_00 \
  --speaker-name "<exp_name>" \
  --lang ZH

# keep all (no --speaker); only works well if JSON has no conflicting speakers
```

### 1.4 FunASR fallback

```bash
.venv/bin/python -s tools/asr/funasr_asr.py -i "<clips>" -o "<out>" -l zh --backend fun-asr-nano
```

---

## 2. Workflow A — Audio → dataset → train

1. Run **§0.1 interview**  
2. UVR if chosen  
3. WhisperX / FunASR as chosen  
4. If diarize: user picks speaker → `whisperx_json_to_list.py`  
5. Optional QA deletes (ask)  
6. Confirm train version → user says go → `./local_tts/start_web.sh` → notify  

Slice CLI (if used): `tools/slice_audio.py` — defaults threshold `-34`, min_length `4000`, …

---

## 3. Workflow B — Novel TTS

Interview: TXT path, **which role** (list `ROLE_PROFILES` + missing assets), tempo, output dir, OOM policy → then API + `batch_tts` + `finish_audio`.

---

## 4. Workflow C — Video → A

Interview adds: URL/path, cookies if login wall, keep video file?

```bash
yt-dlp -o "<dir>/%(title)s [%(id)s].%(ext)s" --no-playlist "<URL>"
ffmpeg -y -i "<video>" -vn -ac 1 -ar 16000 "<dir>/raw.wav"
```

Then continue **§0.1** (ASR method + specific person?) → A.

---

## 5. Recommendation cheat-sheet (say this to the user)

> 建議用 **WhisperX medium**：可轉文字，也能分人。  
> 若影片裡很多人、你只要其中一個：建議開 **diarize**，分完後你選 SPEAKER_xx（需要 HF token）。  
> 若只有一個人或不要分人：WhisperX 只轉寫即可。  
> 有 BGM：先 UVR 再 WhisperX。  
> 訓練版本預設 **v2Pro**（與現有 local_tts 角色一致時）。

---

## 6. Limits

1. UVR ≠ speaker ID  
2. `--diarize` needs HF token + gated model accept  
3. Overlap speech is hard  
4. Medium model ~1.5GB download; use `aria2c -x16`  
5. One heavy GPU job at a time on 8GB  

---

## 7. Quick path

1. **Interview §0.1** (method + specific person?)  
2. Download / extract if video  
3. UVR if needed  
4. WhisperX (± diarize) → user picks speaker if needed → `.list`  
5. User says go → train → notify  
