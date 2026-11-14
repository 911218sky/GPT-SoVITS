# GPT-SoVITS AI Workflow

Playbook for assistants and humans: when the user gives **audio with BGM** or a **novel TXT**, ask before acting, then run the pipeline.

Also see:

- [Operations Guide](./GPT-SoVITS-Operations-Guide.md)
- [Environment Setup](./GPT-SoVITS-Environment-Setup.md)
- [local_tts README](./local_tts/README.md)
- [Performance Guide](./GPT-SoVITS-Performance-Guide.md) / [local_tts/Performance-Guide.md](./local_tts/Performance-Guide.md)

Host-specific inventory (what is already installed on one PC) belongs in a **local** note outside this repo, not here.

---

## 0. Preflight (every task)

```bash
cd /path/to/GPT-SoVITS
test -x .venv/bin/python && test -x local_tts/.venv/bin/python
ls tools/uvr5/uvr5_weights/HP3_all_vocals.pth \
   tools/uvr5/uvr5_weights/VR-DeEchoAggressive.pth \
   tools/uvr5/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx
command -v ffmpeg
```

If anything is missing, follow [Environment Setup](./GPT-SoVITS-Environment-Setup.md).

### Roles (`local_tts/common.py` → `ROLE_PROFILES`)

List available roles from that file and verify each role’s `.ckpt` / `.pth` / ref `.wav` under `local_tts/assets/`.  
Default recommendation when present: **真人男** (v2Pro). Skip roles with missing assets.

---

## 1. Workflow A — Audio with BGM → dataset → train

### 1.1 Ask the user (do not start training until answered)

1. Input audio path  
2. Speaker / `exp_name`  
3. Language for ASR/train (`zh` default, or `yue`/`ja`/`en`/`ko`/`auto`)  
4. UVR model order: accept recommended three-step or custom?  
5. Slice preset A/B/C/D or custom numbers?  
6. Optional denoise? (may download models)  
7. Strong AI quality review + delete bad clips? (yes/no)  
8. Train version: recommend **v2Pro** if matching existing `local_tts` roles; else `v2ProPlus`/`v2`/`v3`/`v4`/`v5*`/custom  
9. After train: register into `local_tts`? which ref wav?  
10. Any other custom settings? User says **go** only when ready  

### 1.2 Suggested folders

```text
<data_root>/<exp_name>/
  raw/
  uvr_vocal/
  slicer/
  slicer_good/          # optional after QA
  slicer.list
```

### 1.3 UVR5 (remove BGM / reverb)

- Code: `tools/uvr5/webui.py` (Gradio; **no headless CLI**)  
- Weights: `tools/uvr5/uvr5_weights/`  
- Start via `./local_tts/start_web.sh` then UVR UI (port from `config.py`, default **9873**)

Recommended order for speech + BGM:

1. `HP3_all_vocals` — drop BGM  
2. `onnx_dereverb_By_FoxJoy` — dereverb  
3. `VR-DeEchoAggressive` — de-echo  

Alternatives: `HP5_only_main_vocal` (with harmony), `HP2_all_vocals`, other DeEcho models.

If the assistant cannot drive a browser, pause and ask the user to finish UVR (or use browser automation).

### 1.4 Slice (keep clearer speech)

```bash
.venv/bin/python -s tools/slice_audio.py \
  "<input file or dir>" "<out dir>" \
  <threshold> <min_length> <min_interval> <hop_size> <max_sil_kept> \
  <_max> <alpha> <i_part> <n_parts>
```

WebUI defaults (`webui.py`): threshold `-34`, min_length `4000`, min_interval `300`, hop_size `10`, max_sil_kept `500`, `_max` `0.9`, `alpha` `0.25`. Single file forces `n_parts=1`.

Ask the user:

- **A** defaults  
- **B** shorter/clearer: stricter threshold / smaller min_length  
- **C** longer segments  
- **D** fully custom  

Report clip count and duration; ask whether to re-slice.

### 1.5 Denoise (optional)

```bash
.venv/bin/python -s tools/cmd-denoise.py -i "<in>" -o "<out>" -p float16
```

Skip unless the user agrees (may need network if local denoise weights are absent).

### 1.6 ASR

Prefer Chinese FunASR:

```bash
.venv/bin/python -s tools/asr/funasr_asr.py \
  -i "<clips dir>" -o "<out dir>" -l zh --backend fun-asr-nano
```

`.list` format:

```text
/abs/path/clip.wav|SpeakerName|ZH|transcript text
```

### 1.7 Quality review (ask: stronger check?)

If yes: scan `.list` for empty/junk text, bad duration, broken paths; propose deletes; **wait for confirm** before deleting; copy keepers to `slicer_good/` and rewrite list.  
If no: only light checks (path exists, non-empty text).

### 1.8 Train version (ask)

Recommend **v2Pro** when aligning with current `local_tts` v2Pro assets. Confirm epochs/batch or WebUI defaults. On a single mid-range GPU, do not run UVR + train + long TTS in parallel.

**Start training only after the user says go.**

### 1.9 Train

```bash
./local_tts/start_web.sh
# http://127.0.0.1:9874
```

Order: clean clips → (slice) → ASR `.list` → check labels → BERT/SSL/semantic → SoVITS → GPT.  
Outputs under `logs/<exp>/` and versioned `SoVITS_weights_*` / `GPT_weights_*` (see `config.py`).

Notify the user when done; ask whether to copy weights into `local_tts/assets/` and register `ROLE_PROFILES`.

---

## 2. Workflow B — Novel TXT → audio

### 2.1 Ask

1. TXT path (UTF-8)  
2. Which role from `ROLE_PROFILES` (warn on missing assets)  
3. API already up + `batch_tts --no-set-model`?  
4. Output dir name  
5. `finish_audio.sh all --tempo 0.9` OK? custom tempo / max MB?  
6. On OOM, auto lower `--batch-size` (64→48→40)?  

### 2.2 Commands

```bash
./local_tts/start_api.sh --role <role>

./local_tts/batch_tts.sh \
  --file-path "/path/to/novel.txt" \
  --role <role> \
  --no-set-model \
  --output-dir "local_tts/output/<name>_<role>"

./local_tts/finish_audio.sh all \
  --input "local_tts/output/<name>_<role>" \
  --tempo 0.9 \
  --max-size-mb 1024
```

Defaults tuned for ~8GB GPU are documented in Performance Guide (`max_text_length=1200`, `batch_size=64`, etc.). One `/tts` at a time; resume by re-running the same command.

---

## 3. State machine

```text
Audio? → Workflow A
Novel TXT? → Workflow B
Else ask: train voice or novel TTS?
```

Gates A0–A8 / B0–B3: always offer a **recommended default + one-line reason**; user may reply “use defaults / go”.

---

## 4. Code map

| Step | Files |
|------|--------|
| UVR | `tools/uvr5/webui.py`, `vr.py`, `mdxnet.py`, `bsroformer.py` |
| Slice | `tools/slice_audio.py`, `tools/slicer2.py` |
| Denoise | `tools/cmd-denoise.py` |
| ASR | `tools/asr/funasr_asr.py`, `fasterwhisper_asr.py` |
| Train UI | `webui.py`, `config.py` |
| Roles | `local_tts/common.py` |
| Novel TTS | `local_tts/start_api.sh`, `batch_tts.sh`, `finish_audio.sh` |

`local_tts` does not patch upstream `api_v2.py` / `webui.py`.

---

## 5. Known limitations (generic)

1. UVR has no headless CLI (Gradio)  
2. Local denoise weights may be absent (network fallback)  
3. Some roles may lack ref wav — verify before use  
4. Single GPU: do not stack heavy jobs  

---

## 6. Quick path: “30-min audio with BGM”

1. Ask §1.1  
2. Store under data `raw/`  
3. UVR: HP3 → onnx_dereverb → DeEchoAggressive  
4. Slice with agreed params  
5. FunASR → `.list`  
6. Optional QA deletes  
7. Confirm version (default v2Pro) → user says go → train  
8. Notify + optional `local_tts` registration  
9. Later novels → Workflow B (ask role first)
