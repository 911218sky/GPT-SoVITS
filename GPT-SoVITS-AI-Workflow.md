# GPT-SoVITS AI Workflow

Playbook for assistants and humans. Ask before long or destructive steps, then run the pipeline.

**Inputs this playbook covers:**

| Input | Workflow |
|-------|----------|
| Audio with BGM / raw recording | **A** → clean → slice → ASR → train |
| Novel TXT | **B** → role TTS |
| Video URL or local video (e.g. Bilibili) | **C** → download → extract audio → (optional target speaker) → then **A** |

Also see:

- [Operations Guide](./GPT-SoVITS-Operations-Guide.md)
- [Environment Setup](./GPT-SoVITS-Environment-Setup.md)
- [local_tts README](./local_tts/README.md)
- [Performance Guide](./GPT-SoVITS-Performance-Guide.md)

Host-specific inventory (what is already installed on one PC) belongs in a **local** note outside this repo.

---

## 0. Preflight (every task)

```bash
cd /path/to/GPT-SoVITS
test -x .venv/bin/python && test -x local_tts/.venv/bin/python
command -v ffmpeg
# Video pipeline:
command -v yt-dlp || echo "MISS yt-dlp — see Environment Setup"
ls tools/uvr5/uvr5_weights/HP3_all_vocals.pth \
   tools/uvr5/uvr5_weights/VR-DeEchoAggressive.pth \
   tools/uvr5/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx 2>/dev/null || true
```

### Roles

Read `local_tts/common.py` → `ROLE_PROFILES`. Verify each role’s `.ckpt` / `.pth` / ref `.wav`.  
Default recommendation when present: **真人男** (v2Pro).

---

## 1. Workflow A — Audio → dataset → train

### 1.1 Ask the user

1. Input audio path  
2. Speaker / `exp_name`  
3. Language (`zh` default)  
4. UVR order: recommended three-step or custom?  
5. Slice preset A/B/C/D?  
6. Optional denoise?  
7. Strong AI quality review + delete bad clips?  
8. Train version (recommend **v2Pro** if matching `local_tts`)  
9. Register into `local_tts` after train?  
10. User says **go** only when ready  

### 1.2 Folders

```text
<data_root>/<exp_name>/
  raw/
  uvr_vocal/
  slicer/
  slicer_good/
  slicer.list
```

### 1.3 UVR5 (BGM / reverb — not different people)

UVR separates **vocals vs accompaniment**, **not** speaker A vs speaker B.

Recommended: `HP3_all_vocals` → `onnx_dereverb_By_FoxJoy` → `VR-DeEchoAggressive`.  
UI: Gradio only (`tools/uvr5/webui.py`, port ~9873). No headless CLI.

### 1.4 Slice / ASR / QA / train

Same as before:

- Slice: `tools/slice_audio.py` (WebUI defaults: threshold `-34`, min_length `4000`, …)  
- ASR: `tools/asr/funasr_asr.py` → `.list` = `path|speaker|LANG|text`  
- QA: ask before deletes  
- Train: `./local_tts/start_web.sh` after user says go; notify when done  

---

## 2. Workflow B — Novel TXT → audio

Ask: TXT path, **role**, tempo / output / OOM policy → `start_api` → `batch_tts` → `finish_audio`.  
Details: [local_tts README](./local_tts/README.md).

---

## 3. Workflow C — Video → audio dataset (then A)

Use when the user gives a **Bilibili / YouTube / other URL** or a local video file, and wants training data (or a clean voice track).

### 3.1 Ask first

1. **Video source:** URL and/or local path  
2. **Which voice to keep?** (critical if multi-speaker — see §3.4)  
3. Keep video file after extract, or audio-only?  
4. Need remove BGM (UVR) after extract?  
5. Then continue into Workflow **A** (slice / ASR / train)? Which `exp_name`?  
6. Output directory under data root  

### 3.2 Download (URL)

Prefer **yt-dlp** + system **ffmpeg**:

```bash
# Best audio-friendly: download best video+audio, or audio-only
yt-dlp -o "<out_dir>/%(title)s [%(id)s].%(ext)s" \
  --no-playlist \
  "<URL>"

# Audio-only extract while downloading (m4a/webm often):
yt-dlp -o "<out_dir>/%(title)s [%(id)s].%(ext)s" \
  -x --audio-format wav --audio-quality 0 \
  --no-playlist \
  "<URL>"
```

Notes:

- Bilibili usually works; login-gated / 大会员 content may need cookies (`--cookies-from-browser` or cookies file) — **ask the user** before using browser cookies.  
- Prefer `--no-playlist` unless the user wants a whole series.  
- If download fails, report the error; try another format `-f bv+ba/b` or update yt-dlp.

### 3.3 Extract audio from local video

```bash
ffmpeg -y -i "<video>" -vn -ac 1 -ar 32000 "<out_dir>/raw_audio.wav"
```

(32 kHz mono matches GPT-SoVITS slicer expectations; stereo OK if downstream converts.)

Save under: `<data_root>/<exp_name>/raw/` then proceed to UVR / slice as in **A**.

### 3.4 Multiple people in one video — can we pick one person?

**Yes, but UVR alone cannot do it.** UVR ≠ speaker ID.

| Goal | Approach |
|------|----------|
| Remove music / noise, keep all speech | UVR (Workflow A) |
| Keep **only one person’s** speech | **Speaker diarization** (+ user picks which ID), or **reference-voice match** |

#### Option 1 — Diarization then user picks (recommended default)

1. Run speaker diarization on the (preferably UVR-cleaned) vocal track.  
2. Export one folder per speaker: `spk00/`, `spk01/`, … with short sample clips.  
3. **Ask the user:** “Which speaker is the target? (spk00 / spk01 / …)” — play or list durations.  
4. Keep only that speaker’s segments → merge or keep as clips → Workflow A slice/ASR if needed.  

Typical stack: **pyannote.audio** (may need Hugging Face token + model download on first use). If not installed, tell the user, install with consent, then run.

#### Option 2 — User provides a short reference of the target person

1. User gives ~5–15 s of **only** the target voice (or points to a time range in the video).  
2. Match segments by speaker embedding / verification; discard others.  
3. Confirm a few clips with the user before deleting the rest.

#### Option 3 — Time ranges only

User says “only 03:20–12:00” or “skip the guest at the end” → `ffmpeg -ss -to` cut first, then A.  
Works when speakers don’t overlap much in time; fails for frequent turn-taking.

#### Overlapping speech

If two people talk at once, automatic separation is **hard**. Say so; offer: keep as-is and rely on later QA deletes, or ask for a cleaner source.

### 3.5 Suggested folder layout for video jobs

```text
<data_root>/<exp_name>/
  video/                 # yt-dlp output (optional keep)
  raw/                   # extracted wav
  uvr_vocal/             # after UVR
  diarize/spk00|spk01/   # after diarization
  target/                # chosen speaker only
  slicer/
  slicer.list
```

### 3.6 End-to-end gate order (video)

```text
[C0] URL/path + exp_name + which voice strategy
[C1] yt-dlp download (ask cookies if needed)
[C2] ffmpeg extract wav
[C3] UVR if BGM (ask)
[C4] If multi-speaker: diarize → user picks ID (or reference match)
[C5] Continue Workflow A from slice → ASR → QA → train version → go
```

---

## 4. State machine

```text
Video URL / file?     → Workflow C (then often A)
Audio / recording?    → Workflow A
Novel TXT?            → Workflow B
Else                  → ask which of A / B / C
```

Always offer a **recommended default + one-line reason**.

---

## 5. Code / tool map

| Step | Tool / file |
|------|-------------|
| Download video | `yt-dlp` |
| Extract / cut audio | `ffmpeg` |
| Remove BGM | `tools/uvr5/webui.py` |
| Slice | `tools/slice_audio.py` |
| ASR | `tools/asr/funasr_asr.py` |
| Target speaker | diarization (e.g. pyannote) — install on demand |
| Train | `webui.py` / `./local_tts/start_web.sh` |
| Novel TTS | `local_tts/start_api.sh`, `batch_tts.sh`, `finish_audio.sh` |

---

## 6. Limitations (say upfront)

1. UVR does not separate different people.  
2. Diarization needs extra install + often a HF token; first run downloads models.  
3. Login-walled Bilibili may need user-provided cookies.  
4. Overlapped speech cannot be cleanly split with this stack alone.  
5. Single GPU: do not stack download+UVR+train+long TTS.

---

## 7. Quick path: “Bilibili video → train one person’s voice”

1. Ask URL, `exp_name`, language, **who to keep** (name/description or “I’ll pick after diarize”).  
2. `yt-dlp` → `ffmpeg` wav → `raw/`.  
3. UVR if needed.  
4. If multi-speaker: diarize → **user picks speaker ID** → `target/`.  
5. Slice → ASR → optional QA.  
6. Ask train version (default v2Pro) → user says go → train → notify.  
