# GPT-SoVITS AI Workflow

Playbook for assistants and humans. **Ask before converting or training.** Do not start WhisperX / UVR / train until the user answers the gates below (or says “use your recommendation / go”).

| Input | Flow |
|-------|------|
| Audio / recording | **A** |
| Novel TXT | **B** |
| Video URL / file | **C** → then often **A** |

Also: [Operations Guide](./GPT-SoVITS-Operations-Guide.md) · [`GPT-SoVITS-Audio-Quality-Guide.md`](./GPT-SoVITS-Audio-Quality-Guide.md)（好／壞音質標準與優化） · [Environment Setup](./GPT-SoVITS-Environment-Setup.md) · [local_tts README](./local_tts/README.md)

Scripts in this repo:

| Script | Purpose |
|--------|---------|
| [`tools/whisperx_transcribe.sh`](./tools/whisperx_transcribe.sh) | WhisperX ASR (± `--diarize`) |
| [`tools/whisperx_json_to_list.py`](./tools/whisperx_json_to_list.py) | WhisperX JSON → clips + `.list` |
| [`tools/slice_for_training.py`](./tools/slice_for_training.py) | slicer2 + max-length quietest split (3–10s) |
| [`local_tts/batch_tts.sh`](./local_tts/batch_tts.sh) | **Novel TTS** (use this, not ad-hoc curls) |
| [`local_tts/finish_audio.sh`](./local_tts/finish_audio.sh) | denoise-ish cleanup + tempo + merge |
| [`local_tts/start_api.sh`](./local_tts/start_api.sh) | API for TTS |

Host-specific paths / tokens: local note outside git (`WHISPERX_MODEL`, `HF_TOKEN` in `~/.config/gpt-sovits-secrets.env`).

**Lesson from `cintia` (2026-11):** quality + VRAM control matter more than “run everything in one Python process”. Follow **§8** and **§9**. **Before any train / after cleaning audio:** read and apply [`GPT-SoVITS-Audio-Quality-Guide.md`](./GPT-SoVITS-Audio-Quality-Guide.md) (spectrogram + `rel_qh` / `bleed` gates). Do **not** train on dirty clips.

---

## 0. Preflight

```bash
cd /path/to/GPT-SoVITS
test -x .venv/bin/python && test -x local_tts/.venv/bin/python
command -v ffmpeg && command -v yt-dlp
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"   # uv + nvidia-smi on WSL
# Check VRAM before heavy work:
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

If API / leftover train is holding VRAM: stop `api_v2.py` / `local_tts.start_api` / train PIDs first (see §8).

---



## 0.2 Mandatory audio quality gate（訓練前必做）

**凡走 Workflow A／C（做資料集或訓練）**，AI **必須**先讀：

[`GPT-SoVITS-Audio-Quality-Guide.md`](./GPT-SoVITS-Audio-Quality-Guide.md)

然後在開訓前完成（對使用者簡要報告數字／結論）：

1. 畫頻譜：長人聲 ≥1 張 + 抽樣 clip 3–5 張（句間應接近黑）  
2. 對保留 clip 算 **`rel_qh` / `bleed`**（門檻見 Quality Guide §1.2）  
3. 丟掉英文歌、低 SNR、高殘留、過短／過長段  
4. 選最乾淨 clip 當 `ref.wav`  
5. 訓後用 TTS 抽測再畫頻譜對照（可對 `真人男` ref）  

**未過閘門 → 禁止開訓**；改回 UVR／spectral_clean／篩選（Quality Guide §2–§4）。  
純小說轉語音（Workflow B、角色已有）不需重跑此閘門，但若使用者抱怨「髒／不平」→ 立刻改走 Quality Guide 優化資料並重訓。

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
| **A. WhisperX** | Default: text + can label who spoke | **Yes (always)** |
| **B. FunASR** | Legacy only | Avoid for new work |
| **C. Skip ASR** | User already has `.list` | Rare |

**Recommendation:** always **WhisperX `faster-whisper-medium`, CUDA float16**. Do not start `tools/asr/fasterwhisper_asr.py` (it may download large FunASR / large-v3).

### Q3 — Extract a specific person?

| Option | Meaning | Needs |
|--------|---------|--------|
| **1. Yes — diarize, then I pick** | `--diarize` → SPEAKER_00/01… | `HF_TOKEN` + gated accept |
| **2. Time range / reference** | ffmpeg cut | — |
| **3. Keep all** | single speaker name in `.list` | — |

**Recommendation:** multi-person → **1**; clear solo vlog/narration → **3** (still diarize once to confirm one dominant speaker).

### Q4 — Remove BGM / UVR?

Recommend **yes** if BGM / room reverb is audible.

**Proven order on 8GB (cintia):**

1. `HP3_all_vocals` (GPU, fast)  
2. **`VR-DeEchoAggressive`** (GPU, fast)  
3. **Skip `onnx_dereverb_By_FoxJoy` by default** — runs mostly on CPU, very slow, can thrash RAM/VRAM when chained in one process  

Optional later: denoise (`tools/cmd-denoise.py`) only if still hissy.

UVR ≠ speaker ID.

### Q5 — Slice / quality settings

Target **~3–10 s** per clip (community / GPT-SoVITS #1070). Prefer `tools/slice_for_training.py` (slicer2 + max-length quietest split) **or** energy split on diarized ranges.

**Quality gates (do these before train)** — details & thresholds in [`GPT-SoVITS-Audio-Quality-Guide.md`](./GPT-SoVITS-Audio-Quality-Guide.md):

- Drop English-song / non-target language clips (CJK-dominant check)  
- Drop low SNR / high high-band residual (music bleed); rank by **`rel_qh` / `bleed`**  
- Drop near-silent / clipped / <~2.5 s or >~10 s  
- Re-ASR each kept clip with WhisperX medium; drop empty / garbage text  
- **Must pass §0.2** (spectrogram + metrics) before Q6 train  

Dataset length: **5–15 min** clean speech is the practical sweet spot; <5 min works if very clean (prefer quality over count).

### Q6 — Train now?

Recommend **v2Pro** when using `local_tts` (same family as Hitomi / 真人男).  
Confirm `exp_name`, then **start only after user says go**.

After train: register role in `local_tts/common.py` + `assets/` (weights + `ref.wav` 3–10 s). Keep **only final** weights (e.g. SoVITS `e8`/`e10` + GPT `e15`); delete mid checkpoints.

### Q7 — Novel TTS role?

If training a new voice for novels: ask whether to add to `ROLE_PROFILES` (default **yes**).

---

## ASR policy (host default)

**Always WhisperX.** Secrets: `~/.config/gpt-sovits-secrets.env` + `~/.cache/huggingface/token`.

For sliced folders: batch medium model → `.list` lines `wav_basename|SpeakerName|ZH|text`.

---

## 1. WhisperX how-to

### 1.1 ASR only

```bash
export WHISPERX_MODEL="$HOME/code/whisperx/models/faster-whisper-medium"
./tools/whisperx_transcribe.sh "<audio.wav>" "<out_dir>" --language zh
```

### 1.2 ASR + diarize

```bash
. "$HOME/.config/gpt-sovits-secrets.env"
./tools/whisperx_transcribe.sh "<audio.wav>" "<out_dir>" --diarize --language zh
```

Show speaker durations; user picks `SPEAKER_xx` (or auto-pick dominant if solo).

### 1.3 Build `.list`

```bash
.venv/bin/python tools/whisperx_json_to_list.py \
  "<out>/<stem>.json" --audio "<audio.wav>" --out-dir "<data>/<exp>/slicer" \
  --speaker SPEAKER_00 --speaker-name "<exp>" --lang ZH
```

Prefer **re-slice + re-ASR** after UVR (long WhisperX segments are too long for training).

---

## 2. Workflow A — Audio → clean dataset → train

1. §0.1 interview  
2. UVR: **HP3 → DeEchoAggressive** (separate processes; §8)  
3. WhisperX ± diarize → keep target speaker  
4. Slice to 3–10 s + quality gates (§0.1 Q5 + **§0.2 Audio Quality Guide**)  
5. WhisperX ASR on clips → `slicer.list`  
6. User says go → feature prep → SoVITS → GPT (free VRAM between stages)  
7. Keep final weights only → register `local_tts` role → notify  

**Train defaults that worked on RTX 3060 Ti 8GB (`cintia`):**

| Stage | Version | batch_size | epochs | Notes |
|-------|---------|------------|--------|-------|
| SoVITS | v2Pro | **4** (safe) / 8 if VRAM free | 8–10 | `mkdir -p logs/<exp>/logs_s2_v2Pro` before train |
| GPT | v2Pro (`s1v3` init) | **4** | 15 | save every last epoch only if disk tight |

Feature prep order: `1-get-text` → `2-get-hubert-wav32k` → `2-get-sv` (Pro) → `3-get-semantic`. Clear GPU between stages.

---

## 3. Workflow B — Novel TTS (use existing scripts)

**Do not invent a new TTS path.** Use:

```bash
export PATH="$HOME/.local/bin:$PATH"
./local_tts/start_api.sh --role <RoleName>
./local_tts/batch_tts.sh \
  --file-path /path/to/novel.txt \
  --role <RoleName> \
  --output-dir /path/to/out \
  --no-set-model          # if API already loaded this role
./local_tts/finish_audio.sh all --input /path/to/out --tempo 0.9 --max-size-mb 1024
```

Interview: TXT path, **role** (from `ROLE_PROFILES`), tempo, output dir, OOM (`--batch-size 48/40` if needed).

After TTS demo: **stop API** to free VRAM (`pkill -f api_v2.py`).

Register a new role (example fields): GPT `*-e15.ckpt`, SoVITS `*_e*_s*.pth`, `assets/Data/<role>/ref.wav`, `prompt_text` matching ref, `speed_factor`.

Roles already include: Lele, Lele_Pro, 阿甘, Sesame, Hitomi, 真人男, **cintia**.

---

## 4. Workflow C — Video → A

```bash
# Audio-only is enough for voice datasets; avoid aria2c on some Bilibili CDNs (HTTP 567)
yt-dlp --no-playlist -f "ba/b" -x --audio-format wav -o "raw.%(ext)s" --no-keep-video "<URL>"
ffmpeg -y -i raw.wav -ac 2 -ar 44100 uvr_in.wav   # UVR likes 44.1k stereo
```

Then §0.1 → A.

---

## 5. Recommendation cheat-sheet

> 建議 **WhisperX medium**；多人就 **diarize** 後選 SPEAKER。  
> 有 BGM：先 **HP3 → DeEchoAggressive**（預設跳過慢速 MDX onnx）。  
> 訓練切片目標 **3–10 秒**，丟掉英文歌／低 SNR／殘響髒段。
> 音質標準與優化決策樹見 **[`GPT-SoVITS-Audio-Quality-Guide.md`](./GPT-SoVITS-Audio-Quality-Guide.md)**（先量 `rel_qh`／頻譜，再訓）。  
> 訓練版本預設 **v2Pro**；8GB 用 **batch_size=4** 較穩。  
> 轉小說一律：`start_api.sh` → **`batch_tts.sh`** → `finish_audio.sh`。

---

## 6. Limits

1. UVR ≠ speaker ID  
2. Diarize needs HF token + gated model  
3. Overlap speech is hard  
4. Medium model ~1.5GB; prefer `aria2c` for HF weights, not always for Bilibili  
5. **One heavy GPU job at a time on 8GB**  
6. Chaining HP3+MDX+DeEcho in **one** long Python process can pin ~7–8GB and feel “stuck”

---

## 7. Quick path

1. Interview §0.1  
2. Download / extract if video  
3. UVR HP3 → DeEcho (separate procs)  
4. WhisperX (± diarize) → quality slice → ASR → `.list`  
5. User says go → train (low batch) → final weights only → register role  
6. Novel test via **batch_tts.sh** → stop API → notify  

---

## 7.1 Clean vocal preprocess (BGM-heavy / Disney vlog)

Preferred stack on 8GB (one stage = one process):

1. **bs_roformer** (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) — best BGM separation
2. **VR-DeEchoDeReverb** — mono reverb/echo
3. **HP5_only_main_vocal** — kill residual accompaniment
4. **`tools/spectral_clean.py`** — soft gate + highband hush (no heavy model)
5. Slice 3–10s → WhisperX medium → rank by **relative quiet-high / bleed**, keep cleanest ~40 clips

Helpers: `tools/uvr_one_stage.py`, `tools/spectral_clean.py`, `tools/filter_training_clips.py`.

Skip FRCRN unless deps installed. Skip MDX onnx by default (slow/CPU thrash).

Measure before train: `rel_qh` of kept set should be ≪ old HP3-only set (cintia_v3 ≈6× cleaner than cintia_v2).

## 8. VRAM / speed control (mandatory on 8GB)

**Rules learned from cintia retrain:**

1. **One stage = one process.** When the process exits, VRAM returns. Do not keep HP3 + MDX + WhisperX loaded together.  
2. After each GPU stage: `torch.cuda.empty_cache()`, then verify with `nvidia-smi` (WSL: `/usr/lib/wsl/lib/nvidia-smi`). Kill leftover compute PIDs if memory stays high.  
3. **Stop TTS API** before UVR / train / WhisperX (`pkill -f api_v2.py`). API alone can hold ~2GB+.  
4. Prefer **HP3 + DeEcho**; skip MDX onnx unless user insists (CPU-bound, ~10s+/chunk).  
5. Train with **batch_size=4** first; raise to 8 only if `memory.used` stays comfortable.  
6. Before `s2_train.py`: `mkdir -p logs/<exp>/logs_s2_v2Pro` (save can fail if dir missing).  
7. After train: delete mid `e5`/`e10` GPT ckpts and huge `logs/*/G_*.pth` if only finals needed.  
8. Never start a second heavy job “while waiting” on the same GPU.

Quick free:

```bash
pkill -f 'api_v2.py' || true
pkill -f 'local_tts.start_api' || true
# then kill any nvidia-smi compute PIDs if still listed
```

---

## 9. Worked example — `cintia` (video → role)

Source: Bilibili vlog → voice clone role for novels.

| Step | What we did |
|------|-------------|
| Download | `yt-dlp` audio-only WAV (no aria2c) |
| UVR | HP3 → DeEchoAggressive; **skipped MDX** |
| Diarize | WhisperX medium → almost all `SPEAKER_00` |
| Slice + QA | ~3–8.5 s, SNR / high-band filter, drop EN song → **65 clips (~5.5 min)** |
| Train | v2Pro, **bs=4**, SoVITS e10 + GPT e15; keep finals only |
| Role | `local_tts/common.py` + `assets/.../cintia*` + `Data/cintia/ref.wav` |
| Novel test | `batch_tts.sh --role cintia` on `local_tts/examples/test_novel.txt` excerpt + `finish_audio.sh` |
| Cleanup | Stop API; VRAM back to ~300 MiB idle |

If output still flat/noisy: need **more minutes of clean speech**, not higher epochs alone.

