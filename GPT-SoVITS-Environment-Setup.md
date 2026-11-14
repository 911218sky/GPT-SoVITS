# GPT-SoVITS Environment Setup

Generic install and verification for this repository.  
For machine-specific status (GPU snapshot, what is already downloaded on one host), keep a **local** note outside the repo (do not commit host inventory here).

Also see:

- [AI Workflow](./GPT-SoVITS-AI-Workflow.md)
- [Operations Guide](./GPT-SoVITS-Operations-Guide.md)
- [local_tts README](./local_tts/README.md)
- Upstream: [docs/cn/README.md](./docs/cn/README.md)

---

## 1. Two Python environments

| Env | Path | Use |
|-----|------|-----|
| Project root | `.venv` | WebUI, training, UVR, ASR, features |
| local_tts | `local_tts/.venv` | `batch_tts` / `finish_audio` / launch wrappers |

Do not manually mix packages across the two envs.

---

## 2. System packages (Linux / WSL)

```bash
sudo apt update
sudo apt install -y ffmpeg wget git
sudo apt install -y libsox-dev sox
sudo apt install -y unzip || true
```

Confirm GPU is visible in WSL/Linux (`nvidia-smi`) before installing CUDA builds of PyTorch.

---

## 3. Recommended: uv + Python 3.10

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /path/to/GPT-SoVITS
uv venv --allow-existing --python 3.10 .venv

# Pick a torch CUDA index that matches your driver (example: cu128)
uv pip install --python .venv/bin/python torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv/bin/python -r extra-req.txt --no-deps
uv pip install --python .venv/bin/python -r requirements.txt

./local_tts/setup_uv.sh
```

`local_tts/setup_uv.sh` creates `local_tts/.venv`. If root `.venv` already exists, it only adds launcher deps and reuses project packages.

If `.venv` already works, **do not** reinstall blindly — run §6 checks first.

---

## 4. Upstream: conda + `install.sh`

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
cd /path/to/GPT-SoVITS
bash install.sh --device CU128 --source HF --download-uvr5
```

- `--device`: `CU126|CU128|ROCM|MPS|CPU`
- `--source`: `HF|HF-Mirror|ModelScope`
- `--download-uvr5`: fetch UVR weights zip

Still run `./local_tts/setup_uv.sh` afterward if you use novel batch TTS.

---

## 5. Weights

### 5.1 Pretrained (training / base)

After a successful `install.sh`, `GPT_SoVITS/pretrained_models/` should exist (including `sv/`).  
Manual zip: `XXXXRT/GPT-SoVITS-Pretrained` → extract into `GPT_SoVITS/pretrained_models/`.

Versions used by `config.py` include: `v1`, `v2`, `v2Pro`, `v2ProPlus`, `v3`, `v4`, `v5dev`, `v5turbo`.

### 5.2 UVR5 (BGM / reverb)

Directory: `tools/uvr5/uvr5_weights/`

Recommended minimum for speech + BGM:

| File | Role |
|------|------|
| `HP3_all_vocals.pth` | Remove accompaniment |
| `onnx_dereverb_By_FoxJoy/vocals.onnx` | Dereverb |
| `VR-DeEchoAggressive.pth` | De-echo |

```bash
# Official zip (same as install.sh --download-uvr5)
wget -O /tmp/uvr5_weights.zip \
  "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
python3 -c "import zipfile; zipfile.ZipFile('/tmp/uvr5_weights.zip').extractall('tools/uvr5')"

# If HP3 is missing from the zip:
wget -O tools/uvr5/uvr5_weights/HP3_all_vocals.pth \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth"
```

### 5.3 ASR / denoise / voice roles

- FunASR / Faster-Whisper: under `tools/asr/models/` or auto-download on first use
- Denoise: optional; empty `tools/denoise-model/` falls back to ModelScope
- Inference roles: `local_tts/assets/` + `local_tts/common.py` (`ROLE_PROFILES`)

### 5.4 Data layout (suggested)

```text
GPT-SoVITS-DATA/   # sibling folder, not required name
  01_raw_audio/
  02_datasets/
  03_novels/
  04_models/
  05_outputs/
```

Keep large audio, novels, and weights out of git.

---



## 5.5 Optional: yt-dlp (video / Bilibili download)

Used by [AI Workflow](./GPT-SoVITS-AI-Workflow.md) **Workflow C**.

```bash
# Recommended with uv:
uv tool install yt-dlp
export PATH="$HOME/.local/bin:$PATH"
yt-dlp --version

# Also need ffmpeg (system package)
sudo apt install -y ffmpeg   # if missing
```

Upgrade later: `uv tool upgrade yt-dlp`.

Speaker diarization (pick one person from multi-speaker audio) is **not** bundled; install on demand (e.g. pyannote) when Workflow C needs it, with user consent (HF token may be required).



## 5.6 WhisperX (primary ASR + diarization)

Install into the **same** project `.venv` that already has CUDA torch (avoid reinstalling torch):

```bash
uv pip install --python .venv/bin/python whisperx --no-deps
uv pip install --python .venv/bin/python \
  "faster-whisper>=1.1.1" ctranslate2 "transformers>=4.48.0" av \
  "pyannote.audio>=3.3.2" pandas nltk onnxruntime
```

Download **faster-whisper-medium** weights (~1.5GB). Prefer multi-connection download:

```bash
sudo apt install -y aria2
mkdir -p /path/to/faster-whisper-medium && cd /path/to/faster-whisper-medium
aria2c -c -x 16 -s 16 -o model.bin \
  "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin"
# also: config.json tokenizer.json vocabulary.txt from the same repo
```

CLI wrapper example: call `.venv/bin/whisperx` with `--model /path/to/faster-whisper-medium`.

Diarization needs `HF_TOKEN` and accepting gated model terms on Hugging Face.

**Speed tip:** On some hosts official `pypi.org` is faster than regional mirrors—benchmark before forcing a mirror. Large Hub files: use `aria2c -x16`, not single-thread `wget`.

## 6. Verification checklist

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/GPT-SoVITS

command -v uv && uv --version
command -v ffmpeg
command -v yt-dlp || echo MISS yt-dlp
command -v whisperx || echo MISS whisperx
test -x .venv/bin/python && .venv/bin/python -V
test -x local_tts/.venv/bin/python && local_tts/.venv/bin/python -V

.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"

for f in \
  tools/uvr5/uvr5_weights/HP3_all_vocals.pth \
  tools/uvr5/uvr5_weights/VR-DeEchoAggressive.pth \
  tools/uvr5/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx
 do
  test -f "$f" && echo OK "$f" || echo MISS "$f"
done

test -d GPT_SoVITS/pretrained_models/v2Pro && echo OK v2Pro
test -f GPT_SoVITS/pretrained_models/s1v3.ckpt && echo OK s1v3
```

Optional smoke: `./local_tts/start_web.sh` (9874), `./local_tts/start_api.sh --role <role>` (9880).

---

## 7. Troubleshooting

| Issue | Action |
|-------|--------|
| `uv` not found | §3 install + PATH |
| Missing `local_tts/.venv` | `./local_tts/setup_uv.sh` |
| CUDA OOM on 8GB | Lower batch size; do not run UVR + train + long TTS together |
| Empty UVR dropdown | Populate `tools/uvr5/uvr5_weights/` |
| `finish_audio` fails | Install ffmpeg |
| No `unzip` | `apt install unzip` or Python `zipfile` |

---

## 8. Reinstall policy (for AI / ops)

1. §6 passes → do not reinstall; follow AI Workflow  
2. Only missing weights → §5 only  
3. Broken `.venv` / no CUDA → rebuild §3 (or upstream §4 if user insists)  
4. New machine → §2 → §3 → §5 → §6 → [AI Workflow](./GPT-SoVITS-AI-Workflow.md)
