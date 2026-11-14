# GPT-SoVITS AI 工作流（給後續 AI／人類）

> **主機：** `sky-3060ti-linux`（WSL）  
> **專案根目錄：** `/home/sky/code/GPT-SoVITS`  
> **資料根目錄：** `/home/sky/code/GPT-SoVITS-DATA`  
> **遠端操作規範：** 本機 PowerShell 只當遙控器；指令寫入 `D:\cursor\sshfs\tmp\_ai_task_*.sh` 後  
> `cmd /c "ssh sky-3060ti-linux bash -s < script.sh"`。禁止用 PS 直接改 `X:` 掛載。  
> **對照：** Windows 觀察用 `X:\` ＝ 遠端 `/home/sky/`。

本文件是**可執行 playbook**：使用者給你帶 BGM 的音檔或 TXT 小說時，依本流程**逐步詢問、確認後再動手**。  
細節指令也可對照：

- [`GPT-SoVITS-Operations-Guide.md`](./GPT-SoVITS-Operations-Guide.md)（總覽）
- 環境安裝／驗證：[`GPT-SoVITS-Environment-Setup.md`](./GPT-SoVITS-Environment-Setup.md)
- [`local_tts/README.md`](./local_tts/README.md)（批次 TTS／finish_audio）
- [`local_tts/Performance-Guide.md`](./local_tts/Performance-Guide.md)（3060 Ti 掃速）

---

## 0. 開工前硬性檢查（每次任務先跑）

在遠端執行（可打包進 `_ai_task_*.sh`）：

```bash
cd /home/sky/code/GPT-SoVITS
test -x .venv/bin/python && test -x local_tts/.venv/bin/python
ls tools/uvr5/uvr5_weights/HP3_all_vocals.pth \
   tools/uvr5/uvr5_weights/VR-DeEchoAggressive.pth \
   tools/uvr5/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx
command -v ffmpeg
```

### 0.1 權重／環境就緒表（2026-11-14 實測）

| 項目 | 狀態 | 路徑／說明 |
|------|------|------------|
| 根 `.venv` | OK | `/home/sky/code/GPT-SoVITS/.venv` |
| `local_tts/.venv` | OK | 批次 TTS／finish_audio |
| UVR HP2 / **HP3** / HP5 | OK | `tools/uvr5/uvr5_weights/*.pth` |
| UVR MDX `onnx_dereverb_By_FoxJoy` | OK | `.../onnx_dereverb_By_FoxJoy/vocals.onnx` |
| UVR DeEcho Aggressive/Normal/DeReverb | OK | 同目錄 `VR-DeEcho*.pth` |
| bs_roformer ckpt | OK | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| bs_roformer yaml | 缺（可跑） | 程式有內建預設；非必需 |
| 本地 denoise-model | **空** | `tools/denoise-model/` 僅 `.gitignore`；首次降噪會連 ModelScope 下 `damo/speech_frcrn_ans_cirm_16k` |
| FunASR 相關模型 | OK | `tools/asr/models/`（vad／標點／paraformer 等） |
| Faster-Whisper 快取 | 無 | 首次用會自動下載 |
| 預訓練 v1/v2/v2Pro/v2ProPlus/v3/v4/v5 | OK | `GPT_SoVITS/pretrained_models/` |
| ffmpeg | OK | 系統已裝 |

### 0.2 `local_tts` 角色資產（推理用）

定義於 `local_tts/common.py` → `ROLE_PROFILES`。實測：

| 角色 | GPT ckpt | SoVITS pth | 參考 WAV |
|------|----------|------------|----------|
| 真人男 | OK | OK | OK |
| Hitomi | OK | OK | OK |
| Lele / Lele_Pro | OK | OK | OK |
| 阿甘 | OK | OK | OK |
| Sesame | OK | OK | **MISSING**（無 `local_tts/assets/Data/Sesame/`） |

**預設推薦角色：`真人男`（v2Pro）。** 選 Sesame 前必須先補參考音。

---

## 1. 工作流 A：帶背景音樂的音檔 → 訓練素材 → 訓練

### 1.1 必問使用者（開始前一次問清，或逐步確認）

用條列請使用者回答，**未回答的項目不要擅自用預設開跑訓練**：

1. **輸入音檔路徑**（本機或遠端絕對路徑；可先 scp／放到 `GPT-SoVITS-DATA/01_raw_audio/<專案名>/`）
2. **說話者／實驗名稱** `exp_name`（例如 `Hitomi`、`新角色A`）— 只能英文／數字／底線較穩妥；中文目錄可，但 list 內 speaker 建議簡單
3. **語言**（ASR／訓練）：`zh`（預設）／`yue`／`ja`／`en`／`ko`／`auto`
4. **去 BGM 流程**（見 §1.3）：接受「推薦三步」或自訂模型順序？
5. **切分參數**（見 §1.4）：用預設或自訂？要偏「只留明顯人聲、短句」還是「較長段落」？
6. **是否做降噪**（可選；本地模型空，需網路）？
7. **是否加強 AI 素材檢查**（聽感／ASR 文字品質，刪差段）？預設問「要／不要」
8. **訓練版本**：`v2Pro`（本機現有角色皆此；**預設推薦**）／`v2ProPlus`／`v2`／`v3`／`v4`／`v5dev`／`v5turbo`／自訂
9. **訓練完成後**：是否自動登記進 `local_tts/common.py`？參考句要用哪一段 WAV？
10. **有沒有還沒說的自訂**？使用者說「沒問題／開跑」後才開始長時間訓練

### 1.2 建議目錄配置

```text
/home/sky/code/GPT-SoVITS-DATA/
└── <exp_name>/
    ├── raw/                 # 原始上傳（可含 BGM）
    ├── uvr_vocal/           # 去 BGM／去混響後人聲
    ├── slicer/              # 切段 WAV
    ├── slicer_good/         # （可選）AI／人工篩過的保留段
    └── slicer.list          # ASR：音檔|說話者|語言|文字
```

亦可沿用文件慣例：`01_raw_audio`／`02_datasets`／`04_models`／`05_outputs`。

### 1.3 去背景音樂／混響（UVR5）

**代碼：** `tools/uvr5/webui.py`（Gradio）、`vr.py`／`mdxnet.py`／`bsroformer.py`  
**權重目錄：** `tools/uvr5/uvr5_weights/`  
**啟動：** 主 WebUI 內開「人声分离」，或：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_web.sh
# 主 WebUI 預設 http://127.0.0.1:9874
# UVR5 子頁埠：9873（config.py webui_port_uvr5 / 環境變數 GPT_SOVITS_WEBUI_PORT_UVR5）
```

**重要限制（已核對）：沒有無介面 CLI。** 批次路徑靠 UVR Gradio 的「輸入資料夾」。AI 若無法操作瀏覽器，應：

- 請使用者在 UVR WebUI 跑完；或  
- 用瀏覽器 MCP／遠端桌面代操；或  
- 明確告知此步需 WebUI，並暫停等輸出目錄出現人聲檔。

#### 推薦模型順序（說話＋BGM → 訓練素材）

官方 UI 文案「最乾淨」＝ **先 MDX-Net 再 DeEcho-Aggressive**；去伴奏另用 HP 系列。

| 步驟 | 模型下拉名稱（檔名去副檔名） | 檔案 | 用途 |
|------|------------------------------|------|------|
| 1 | `HP3_all_vocals` | `HP3_all_vocals.pth` | **去 BGM、保留人聲**（無和聲說話首選） |
| 2 | `onnx_dereverb_By_FoxJoy` | `onnx_dereverb_By_FoxJoy/vocals.onnx` | 去雙聲道混響 |
| 3 | `VR-DeEchoAggressive` | `VR-DeEchoAggressive.pth` | 去殘響（官方推薦第二段） |

備選：

- 有和聲：`HP5_only_main_vocal`
- 人聲略弱可試：`HP2_all_vocals`
- 更重殘響：`VR-DeEchoDeReverb`
- bs_roformer：有 ckpt、無 yaml，可用內建預設（非首選）

輸出：人聲放到 `.../uvr_vocal/`，非人聲可丟棄或另存。

### 1.4 自動切分（只留較明顯人聲段）

**代碼：** `tools/slice_audio.py` ← `tools/slicer2.py`  
**WebUI 呼叫格式（`webui.py` `open_slice`）：**

```bash
cd /home/sky/code/GPT-SoVITS
.venv/bin/python -s tools/slice_audio.py \
  "<輸入檔或資料夾>" "<輸出目錄>" \
  <threshold> <min_length> <min_interval> <hop_size> <max_sil_kept> \
  <_max> <alpha> <i_part> <n_parts>
```

**WebUI 預設（請當「標準」；改參數前先問使用者）：**

| 參數 | 預設 | 意義 |
|------|------|------|
| threshold | `-34` | 低於此音量當靜音切割候選 |
| min_length | `4000` | 每段最短（採樣點尺度；與 slicer 內部一致） |
| min_interval | `300` | 最短切割間隔 |
| hop_size | `10` | 音量曲線步進 |
| max_sil_kept | `500` | 切完最多保留的靜音 |
| _max | `0.9` | 正規化峰值 |
| alpha | `0.25` | 正規化混合比 |
| n_parts | 檔案強制 `1`；資料夾可並行 | 分片並行 |

**輸出檔名：** `<原檔名>_<start10位>_<end10位>.wav`，取樣率 32k。

**問使用者怎麼切（給選項）：**

- **A 預設（推薦入門）：** 上表原樣  
- **B 偏短句／只留清楚聲：** `threshold=-38`～`-40`，`min_length` 降到 `2000`～`3000`（段更碎、靜音更嚴）  
- **C 偏長段：** `threshold=-30`～`-32`，`min_length=6000`～`8000`  
- **D 完全自訂：** 使用者給每個數字  

切完後回報：段數、總時長、最短／最長段，再問是否重切。

### 1.5 降噪（可選）

**代碼：** `tools/cmd-denoise.py`  
**現況：** 本地 `tools/denoise-model/` **無權重**，會走 ModelScope 線上下載。

```bash
cd /home/sky/code/GPT-SoVITS
.venv/bin/python -s tools/cmd-denoise.py \
  -i "<slicer目錄>" \
  -o "<denoise輸出目錄>" \
  -p float16
```

預設**不問就不做**；要做需使用者同意（首次下載）。

### 1.6 ASR（每段轉文字）

**中文首選 FunASR：** `tools/asr/funasr_asr.py`（`tools/asr/models` 已有本地模型）  
**多語／備選：** `tools/asr/fasterwhisper_asr.py`（首次下載）

```bash
cd /home/sky/code/GPT-SoVITS
.venv/bin/python -s tools/asr/funasr_asr.py \
  -i "<切段或篩選後目錄>" \
  -o "<輸出目錄，與 list 同層或指定>" \
  -l zh \
  --backend fun-asr-nano
```

**`.list` 格式（訓練必備）：**

```text
/絕對路徑/xxx.wav|SpeakerName|ZH|這裡是辨識出的文字
```

範例見：`GPT-SoVITS-DATA/Hitomi/slicer.list`。

### 1.7 AI 素材品質檢查（必須問「是否加強」）

若使用者同意加強檢查，AI 應：

1. 讀取 `.list`，統計：空文字、極短文字、重複段、路徑失效  
2. 依規則建議刪除（**先列出清單請使用者確認再刪**）：  
   - 無字／亂碼／明顯非目標語言  
   - 時長過短（例如 &lt; 0.8s）或過長異常  
   - 文字與檔名／上下文明顯不相關  
3. 可抽聽若干段（若環境允許）標「保留／可疑／刪」  
4. 通過後把保留檔複製到 `slicer_good/`，重寫 `slicer_good.list`  
5. 若使用者拒絕加強檢查：僅做「路徑存在＋非空文字」的輕量檢查，然後繼續

**禁止**在未確認前大量刪檔。

### 1.8 選訓練版本與權重（必須問）

向使用者說明並請選：

| 版本 | 本機預訓練 | 說明／推薦 |
|------|------------|------------|
| **v2Pro** | OK | **預設推薦**；現有 `local_tts` 角色皆 v2Pro |
| v2ProPlus | OK | 同系列加強；需對應權重目錄 `*_v2ProPlus` |
| v2 | OK | 經典 v2 |
| v3 / v4 | OK | 另一路線；推理／訓練流程與 v2Pro 不同（v3+ 走 lora 腳本） |
| v5dev / v5turbo | OK | 較新；確認使用者要實驗再用 |
| 自訂 | — | 使用者自備底模路徑 |

同時問：

- SoVITS／GPT **epoch／batch** 要用 WebUI 預設還是自訂？  
- GPU：本機單卡 3060 Ti 8GB → **一次只跑一個重任務**（別並行 UVR＋訓練＋TTS）

**使用者明確說「沒問題／開始訓練」後才啟動訓練。**

### 1.9 訓練怎麼開

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_web.sh
# http://127.0.0.1:9874
```

WebUI 順序（操作指南 §3）：

1. 準備／放入乾淨人聲切段  
2. 切分（若尚未 CLI 做完）  
3. ASR／取得 `.list`  
4. 檢查 `音檔|說話者|語言|文字`  
5. 文本／BERT → SSL → 語意 Token  
6. SoVITS 訓練 → GPT 訓練  

產物通常在：

- 實驗過程：`GPT-SoVITS/logs/<exp_name>/`  
- 匯出權重根目錄（依版本）：`SoVITS_weights_v2Pro/`、`GPT_weights_v2Pro/` 等（見 `config.py` 的 `*_weight_version2root`）

訓練結束後 **主動通知使用者**，並問是否：

1. 挑選哪個 epoch 的 `.ckpt`／`.pth`  
2. 複製到 `local_tts/assets/GPT_weights_v2Pro/` 與 `SoVITS_weights_v2Pro/`  
3. 選一段參考 WAV → `local_tts/assets/Data/<角色>/`  
4. 在 `local_tts/common.py` 登記 `ROLE_PROFILES`  
5. 用 `./local_tts/start_api.sh --role <名>` 試講一句

---

## 2. 工作流 B：TXT 小說 → 有聲（批次 TTS）

### 2.1 必問使用者

1. **小說 TXT 路徑**（UTF-8；可放 `GPT-SoVITS-DATA/03_novels/`）  
2. **角色／模型**：下列哪一個？（列出 `ROLE_PROFILES` 現況；標出 Sesame 缺參考音）  
3. 是否先 `start_api` 再 `batch_tts --no-set-model`？  
4. 輸出目錄名？預設 `local_tts/output/<書名>_<角色>`  
5. 收尾：`finish_audio.sh all --tempo 0.9` 是否接受？tempo／分片 MB 要自訂嗎？  
6. OOM 時是否允許自動降 `--batch-size`（64→48→40）？

### 2.2 標準指令

```bash
# 終端 A
cd /home/sky/code/GPT-SoVITS
./local_tts/start_api.sh --role 真人男

# 終端 B
./local_tts/batch_tts.sh \
  --file-path "/home/sky/code/GPT-SoVITS-DATA/03_novels/某小說.txt" \
  --role 真人男 \
  --no-set-model \
  --output-dir "local_tts/output/某小說_真人男"

# 轉完
./local_tts/finish_audio.sh all \
  --input "local_tts/output/某小說_真人男" \
  --tempo 0.9 \
  --max-size-mb 1024
```

**3060 Ti 預設（已掃速）：** `max_text_length=1200`，`batch_size=64`，`cut5`，`split_bucket=True`，`parallel_infer=True`。  
規則：單卡**不要並行多個 `/tts`**。中斷可重跑同指令續接。

成品：`local_tts/output/<名>_merged/`。

---

## 3. AI 對話狀態機（照著問、照著做）

```text
收到音檔？ ──是──► 工作流 A
   │
   └否─ 收到小說 TXT？ ──是──► 工作流 B
              │
              └否─ 問使用者要訓練還是轉小說，或兩者
```

### 工作流 A 閘門

```text
[A0] 路徑／exp_name／語言
[A1] 去 BGM：推薦三步 or 自訂？ → 跑 UVR（WebUI）→ 確認輸出
[A2] 切分參數 A/B/C/D？ → slice → 回報段數
[A3] 要降噪嗎？
[A4] ASR → 得到 .list
[A5] 要加強 AI 檢查刪差段嗎？→ 清單確認 → 刪／保留
[A6] 訓練版本＋epoch 等 → 推薦 v2Pro
[A7] 「還有要改的嗎？」→ 使用者說開跑 → 訓練
[A8] 完成通知 → 是否登記 local_tts 角色
```

### 工作流 B 閘門

```text
[B0] TXT 路徑
[B1] 角色列表＋缺件警告 → 使用者選角
[B2] tempo／輸出／OOM 策略
[B3] 開 API → batch_tts → finish_audio → 通知成品路徑
```

**原則：** 每個閘門預設給「推薦選項＋一行理由」；使用者可不改直接回「用推薦／沒問題」。

---

## 4. 關鍵代碼對照（勿改上游除非使用者要求）

| 用途 | 檔案 |
|------|------|
| UVR UI | `tools/uvr5/webui.py` |
| 切段 | `tools/slice_audio.py`, `tools/slicer2.py` |
| 降噪 | `tools/cmd-denoise.py` |
| ASR | `tools/asr/funasr_asr.py`, `tools/asr/fasterwhisper_asr.py` |
| 主 WebUI／訓練 | `webui.py`, `config.py` |
| 版本底模表 | `config.py` → `pretrained_*_name`, `*_weight_version2root` |
| 角色／路徑 | `local_tts/common.py` |
| API／批次／收尾 | `local_tts/start_api.sh`, `batch_tts.sh`, `finish_audio.sh` |
| 對應實作 | `local_tts/start_api.py`, `batch_tts.py`, `finish_audio.py` |

`local_tts` **不修改**上游 `api_v2.py`／`webui.py`；只包啟動與路徑。

---

## 5. 已知缺口（AI 要主動說）

1. **UVR 無 headless CLI** — 去 BGM 依賴 WebUI／瀏覽器操作。  
2. **denoise 本地權重空** — 可選；要做需網路。  
3. **Sesame 參考 WAV 缺失** — 勿選該角直到補齊。  
4. **bs_roformer 無 yaml** — 可跑內建預設，非推薦主路徑。  
5. 單卡 8GB — 訓練／UVR／長篇 TTS 不要疊著跑。

---

## 6. 一頁速查：使用者只丢「半小時有 BGM 的音檔」

1. 問 §1.1 清單（至少：存放路徑、角色名、切法、是否加強檢查、訓練版）。  
2. 音檔進 `GPT-SoVITS-DATA/<名>/raw/`。  
3. UVR：`HP3` → `onnx_dereverb` → `DeEchoAggressive` → `uvr_vocal/`。  
4. 依約定參數 `slice_audio.py` → `slicer/`。  
5. FunASR → `slicer.list`。  
6. 若要檢查：列刪除建議 → 確認 → `slicer_good/`。  
7. 確認版本（預設 v2Pro）→ 使用者說開跑 → WebUI 訓練。  
8. 結束通知＋是否掛進 `local_tts`。  
9. 之後轉小說走工作流 B，先問角色。

---

*本文件應與實機權重同步；若下載新模型或新增角色，請更新 §0 狀態表與 `ROLE_PROFILES`。*
