# GPT-SoVITS 環境安裝與驗證（給後續 AI／人類）

> **主機範例：** `sky-3060ti-linux`（WSL2 + RTX 3060 Ti 8GB）  
> **專案根目錄：** `/home/sky/code/GPT-SoVITS`  
> **資料根目錄：** `/home/sky/code/GPT-SoVITS-DATA`  
> **遠端操作：** 本機 PowerShell 只當遙控器；腳本寫入 `D:\cursor\sshfs\tmp\_ai_task_*.sh` 後  
> `cmd /c "ssh sky-3060ti-linux bash -s < script.sh"`。細節見 `D:\cursor\sshfs\docs\AI-讀寫與搜尋.md`。

相關文件：

- 互動工作流（音檔訓練／轉小說）：[`GPT-SoVITS-AI-Workflow.md`](./GPT-SoVITS-AI-Workflow.md)
- 專案總覽：[`GPT-SoVITS-Operations-Guide.md`](./GPT-SoVITS-Operations-Guide.md)
- 批次 TTS：[`local_tts/README.md`](./local_tts/README.md)
- 上游中文安裝說明：[`docs/cn/README.md`](./docs/cn/README.md)

---

## 0. 本機實測現況（2026-11-14，`sky-3060ti-linux`）

| 項目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 3060 Ti 8GB |
| 根環境 | `/home/sky/code/GPT-SoVITS/.venv`（Python **3.10.20**，uv） |
| 批次環境 | `/home/sky/code/GPT-SoVITS/local_tts/.venv`（Python 3.10.20） |
| PyTorch | `2.12.1+cu132`，`cuda.is_available()=True` |
| uv | `/home/sky/.local/bin/uv`（例：0.11.x） |
| ffmpeg | 系統已裝（例：8.x） |
| 預訓練模型 | `GPT_SoVITS/pretrained_models/` 已有 v1/v2/v2Pro/v3/v4/v5 等 |
| UVR5 | `tools/uvr5/uvr5_weights/` 已含 HP2/HP3/HP5、DeEcho、onnx_dereverb、bs_roformer |
| FunASR 本地 | `tools/asr/models/` 已有部分 |
| denoise 本地 | **空**（首次降噪會連 ModelScope） |
| unzip | 系統可能未裝；解壓可用 Python `zipfile` |

**兩個 venv 不要手動混裝：**

| 環境 | 用途 |
|------|------|
| 專案根 `.venv` | WebUI、訓練、UVR、ASR、特徵提取 |
| `local_tts/.venv` | `batch_tts`／`finish_audio`／啟動包裝；可共用根環境的 GPT-SoVITS 依賴 |

---

## 1. 系統依賴（WSL／Ubuntu）

```bash
sudo apt update
sudo apt install -y ffmpeg wget git
sudo apt install -y libsox-dev sox
sudo apt install -y unzip || true
```

確認：

```bash
ffmpeg -version | head -n1
nvidia-smi -L
```

NVIDIA 驅動需讓 WSL 看得到 GPU（`nvidia-smi` 有卡再繼續）。

---

## 2. 安裝 uv（本機實際路線，推薦）

上游 `install.sh` 預設走 **conda**。本宿舍機是用 **uv + Python 3.10**；換機或重裝時優先對齊這條。

### 2.1 安裝 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv --version
```

### 2.2 建立專案根 `.venv` 並裝依賴

```bash
cd /home/sky/code/GPT-SoVITS
uv venv --allow-existing --python 3.10 .venv

# 依 GPU 選擇 torch 索引（本機現況為 cu132；新機常見 cu128）
uv pip install --python .venv/bin/python torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv/bin/python -r extra-req.txt --no-deps
uv pip install --python .venv/bin/python -r requirements.txt
```

> 若已有可用的 `.venv`（本機即是），**不要無故砍掉重裝**；先跑 §6 驗證。

### 2.3 建立 `local_tts` 環境

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/setup_uv.sh
```

腳本會：要求 `uv`；用 Python 3.10 建 `local_tts/.venv`；若根目錄已有 `.venv` 則只補 fastapi／requests／starlette／tqdm 並共用根依賴；否則安裝整份 `requirements.txt`。缺 ffmpeg 時會警告（`finish_audio.sh` 需要它）。

---

## 3. 上游官方路線（conda + `install.sh`）

適合全新 clone、願意用 conda 的機器。腳本：`GPT-SoVITS/install.sh`。

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
cd /home/sky/code/GPT-SoVITS
bash install.sh --device CU128 --source HF --download-uvr5
```

`--device`：`CU126|CU128|ROCM|MPS|CPU`  
`--source`：`HF|HF-Mirror|ModelScope`  
加 `--download-uvr5` 會下載 UVR 權重 zip。

腳本大致會：裝 torch、裝 `extra-req.txt`／`requirements.txt`、下預訓練與 G2PW／nltk 等、可選 UVR zip。

**注意：** 需要 `conda`、`wget`、最好有 `unzip`；官方列到 CU128，本機已是 **cu132** 自建環境，環境正常就不必為對齊文件而降版。裝完仍建議跑 `./local_tts/setup_uv.sh`。

Windows 原生見上游 `install.ps1`／`docs/cn/README.md`；本 playbook 以 WSL Linux 為準。

---

## 4. 模型權重安裝

### 4.1 預訓練

若 `install.sh` 成功且存在 `GPT_SoVITS/pretrained_models/sv/`，通常已齊。

手動來源：`XXXXRT/GPT-SoVITS-Pretrained` 的 `pretrained_models.zip`，或 [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS)。解壓到 `GPT_SoVITS/pretrained_models/`。

`config.py` 版本對照（訓練切版前確認檔案存在）：

| 版本 | SoVITS 底模 | GPT 底模 |
|------|-------------|----------|
| v1 | `pretrained_models/s2G488k.pth` | `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt` |
| v2 | `gsv-v2final-pretrained/s2G2333k.pth` | `gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` |
| v2Pro | `v2Pro/s2Gv2Pro.pth` | `s1v3.ckpt` |
| v2ProPlus | `v2Pro/s2Gv2ProPlus.pth` | `s1v3.ckpt` |
| v3 | `s2Gv3.pth` | `s1v3.ckpt` |
| v4 | `gsv-v4-pretrained/s2Gv4.pth` | `s1v3.ckpt` |
| v5dev/v5turbo | `gsv-v5-pretrained/s2Gv5*.pth` | `s1v3.ckpt` |

另需：`chinese-hubert-base`、`chinese-roberta-wwm-ext-large`、`sv/` 等。

### 4.2 UVR5

目錄：`tools/uvr5/uvr5_weights/`

推薦最少集合：`HP3_all_vocals.pth`、`onnx_dereverb_By_FoxJoy/vocals.onnx`、`VR-DeEchoAggressive.pth`。

```bash
cd /tmp
wget -O uvr5_weights.zip \
  "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
python3 -c "import zipfile; zipfile.ZipFile('/tmp/uvr5_weights.zip').extractall('/home/sky/code/GPT-SoVITS/tools/uvr5')"

# zip 若缺 HP3，單補：
cd /home/sky/code/GPT-SoVITS/tools/uvr5/uvr5_weights
wget -O HP3_all_vocals.pth \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth"
```

### 4.3 ASR／降噪／角色資產

- FunASR：`tools/asr/models/`（可首次自動下）
- Faster-Whisper：首次用時自動下
- 降噪：本地 `tools/denoise-model/` 可空，會走 ModelScope
- 角色：`local_tts/assets/` + `common.py`；本機 **Sesame 參考 WAV 缺**

---

## 5. 建議資料目錄

```text
/home/sky/code/GPT-SoVITS-DATA/
  01_raw_audio/  02_datasets/  03_novels/  04_models/  05_outputs/
```

大型音檔、小說、權重不要提交 Git。

---

## 6. 安裝後驗證清單

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/sky/code/GPT-SoVITS

command -v uv && uv --version
command -v ffmpeg
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

可選煙霧測試：`./local_tts/start_web.sh`（9874）、`./local_tts/start_api.sh --role 真人男`（9880）。通過後走 [`GPT-SoVITS-AI-Workflow.md`](./GPT-SoVITS-AI-Workflow.md)。

---

## 7. 常見問題

| 現象 | 處理 |
|------|------|
| 找不到 uv | §2.1，並把 `~/.local/bin` 加 PATH |
| 找不到 local_tts/.venv | `./local_tts/setup_uv.sh` |
| 8GB OOM | 降 batch；勿並行 UVR＋訓練＋長篇 TTS |
| UVR 列表空 | 補 `tools/uvr5/uvr5_weights/` |
| finish_audio 失敗 | 裝 ffmpeg |
| 無 unzip | `apt install unzip` 或 Python zipfile |
| Sesame 失敗 | 補參考 WAV |
| install.sh 與 cu132 不一致 | 環境正常以現有 `.venv` 為準 |

---

## 8. AI 重裝決策

1. §6 全過 → 不要重裝，直接做工作流  
2. 只缺權重 → 只補 §4  
3. `.venv` 壞／無 CUDA → §2 重建；使用者堅持再用 §3 conda  
4. 新機器 → §1 → §2 → §4 → §2.3 → §6 → AI 工作流  

*更新環境或新下載權重後，請同步改本檔 §0 與 `GPT-SoVITS-AI-Workflow.md` §0。*
