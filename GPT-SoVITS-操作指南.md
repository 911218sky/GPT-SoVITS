# GPT-SoVITS 專案快速操作指南（給人類與 AI）

這是 `/home/sky/code/GPT-SoVITS` 的總覽文件。目標是讓新的 AI 或使用者快速知道：資料放哪裡、如何訓練聲音、如何載入模型，以及如何把 TXT 小說轉成真人男聲音檔。

詳細的本機批次 TTS 指令請直接閱讀：

[`GPT-SoVITS/local_tts/README.md`](./GPT-SoVITS/local_tts/README.md)

效能參數、掃速換機、最優化想法請看：

[`GPT-SoVITS-效能優化指南.md`](./GPT-SoVITS-效能優化指南.md)

## 1. 專案位置與主要資料夾

```text
/home/sky/code/
├── GPT-SoVITS/                 # 程式碼、WebUI、API
│   ├── local_tts/              # 本機啟動器、批次轉小說、音檔處理
│   ├── GPT_SoVITS/             # 核心模型程式與 pretrained_models
│   ├── logs/                   # WebUI 訓練產物（私人資料，不提交 Git）
│   └── .venv/                 # GPT-SoVITS 主 Python 環境
└── GPT-SoVITS-DATA/            # 建議放使用者資料與大型模型
    ├── 01_raw_audio/           # 原始錄音
    ├── 02_datasets/            # 切音、標註後的訓練資料
    ├── 03_novels/              # 待轉換 TXT 小說
    ├── 04_models/              # 訓練完成的 GPT/SoVITS 權重
    └── 05_outputs/             # 轉換、清理、合併後的音檔
```

檔案類型建議：原始/訓練語音使用 WAV（單人、乾淨、無背景音）；小說使用 UTF-8 `.txt`；GPT 模型是 `.ckpt`；SoVITS 模型是 `.pth`。大型模型、錄音、小說與輸出音檔不應提交到 Git。

GPT-SoVITS 內建或推理需要的基礎模型放在：

```text
GPT-SoVITS/GPT_SoVITS/pretrained_models/
```

訓練完成後給 `local_tts` 使用的角色模型則放在：

```text
GPT-SoVITS/local_tts/assets/GPT_weights_v2Pro/
GPT-SoVITS/local_tts/assets/SoVITS_weights_v2Pro/
GPT-SoVITS/local_tts/assets/Data/<角色>/
```

## 2. 安裝與啟動環境（WSL/Linux）

```bash
sudo apt update
sudo apt install -y ffmpeg
cd /home/sky/code/GPT-SoVITS
uv venv --allow-existing --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./local_tts/setup_uv.sh
```

`local_tts/.venv` 是批次工具環境；主 WebUI/訓練使用專案根目錄的 `.venv`。不要把兩個環境手動混裝。

## 3. 啟動 WebUI 與訓練模型

訓練不需要先啟動 API。在一個終端機執行：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_web.sh
```

瀏覽器開啟 `http://127.0.0.1:9874`。WebUI 訓練順序為：準備錄音 → 切分音檔 → ASR/文字標註 → 檢查 `音檔|說話者|語言|文字` → 文本/BERT → SSL 特徵 → 語意 Token → SoVITS 訓練 → GPT 訓練。

訓練輸出通常在 `GPT-SoVITS/logs/<實驗名稱>/`。訓練 `v2Pro` 時，預訓練 GPT、SoVITS-G/SoVITS-D 與訓練版本必須一致，不要混用 v2、v3 或 v4 權重。

## 4. 將訓練模型登記為角色

把訓練完成的 `.ckpt`、`.pth` 與一段對應的參考 WAV 放入 `local_tts/assets/`。真人男角色目前的設定在 `local_tts/common.py`，預期檔案為：

```text
local_tts/assets/GPT_weights_v2Pro/真人男-e15.ckpt
local_tts/assets/SoVITS_weights_v2Pro/真人男_e8_s112.pth
local_tts/assets/Data/真人男/<參考音訊>.wav
```

若要新增角色，需同步在 `common.py` 登記 GPT 權重、SoVITS 權重、參考音訊、參考文字與語言。

## 5. 啟動 API 並把小說轉成真人男聲

第一個終端機保持 API 執行：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_api.sh --role 真人男
```

Windows 的 `D:\APP\...` 在 WSL 通常寫成 `/mnt/d/...`。例如把小說放在 `D:\APP\TomatoNovelDownloader\罪名不朽_310.txt` 時，第二個終端機執行：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/batch_tts.sh \
  --file-path "/mnt/d/APP/TomatoNovelDownloader/罪名不朽_310.txt" \
  --role 真人男 \
  --no-set-model \
  --output-dir "local_tts/output/罪名不朽_310_真人男"
```

輸出會是一組編號 WAV。中斷後重新執行相同指令即可續跑，已存在的編號檔會跳過。單張 GPU 不要同時執行多個批次工作。預設已用 3060 Ti 掃速最快組（約 1200 字/段、`batch_size=64`）。CUDA OOM 時依序嘗試 `--batch-size 56`、`48`、`40`。

## 6. 清理、調語速與合併音檔

統一用 **`local_tts/finish_audio.sh`**（需 `ffmpeg`）。詳細參數表見 [`local_tts/README.md`](./GPT-SoVITS/local_tts/README.md)「清理、調速與合併」。

### 6.1 推薦：一條龍（最快，效果同舊三步）

TTS 轉完編號 WAV 後：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/finish_audio.sh all \
  --input "local_tts/output/罪名不朽_310_真人男" \
  --tempo 0.9 \
  --max-size-mb 1024
```

| 會產生 | 路徑 |
|--------|------|
| 中繼 MP3 | `.../罪名不朽_310_真人男_prepared/` |
| 合併成品 | `.../罪名不朽_310_真人男_merged/` |

說明：

- **不必寫 `--workers`**：預設用滿全部 CPU。
- **不必寫 `--silence-steps`**：預設 `0.5:-30,2.0:-20`（等同以前兩輪清音）。
- `--tempo 0.9`＝變慢；不調速就拿掉 `--tempo`。
- `--max-size-mb 1024`＝每個合併檔來源大約最多 1GB，超過會切多檔。
- 已存在的檔會跳過，可中斷續跑。
- `真人男` 推理保持 `speed_factor=1.0`；語速在這裡用 `--tempo` 調。

為什麼更快：多輪靜音 + 調速串成**一次 FFmpeg 編碼**，再 copy 合併（舊流程大約編碼 3 次）。

自訂靜音輪次範例：

```bash
./local_tts/finish_audio.sh all \
  --input "local_tts/output/罪名不朽_310_真人男" \
  --tempo 0.9 \
  --silence-steps "0.3:-35,0.5:-30,2.0:-20" \
  --max-size-mb 1024
```

### 6.2 分步指令（等同舊三支腳本）

```bash
./local_tts/finish_audio.sh clean \
  --input "local_tts/output/罪名不朽_310_真人男" \
  --output "local_tts/output/罪名不朽_310_真人男_clean"

./local_tts/finish_audio.sh tempo \
  --input "local_tts/output/罪名不朽_310_真人男_clean" \
  --output "local_tts/output/罪名不朽_310_真人男_tempo" \
  --tempo 0.9 \
  --suffix mp3

./local_tts/finish_audio.sh merge \
  --input-folder "local_tts/output/罪名不朽_310_真人男_tempo" \
  --output-dir "local_tts/output/罪名不朽_310_真人男_merged" \
  --suffix mp3 \
  --max-size-mb 1024
```

指令一覽：`all` / `clean` / `tempo` / `merge`。查說明：`./local_tts/finish_audio.sh help`。

## 7. AI 快速定位規則

- 要訓練或開 WebUI：看本文件第 3 節，執行 `local_tts/start_web.sh`。
- 要把 TXT 轉語音：看 [`local_tts/README.md`](./GPT-SoVITS/local_tts/README.md) 的「批次產生音訊」章節。
- 要找角色模型：看 `local_tts/common.py` 與 `local_tts/assets/`。
- 要找訓練產物：看 `logs/<實驗名稱>/`。
- 要處理長篇小說：使用批次工具，不要對單張 GPU 開多個 API worker。
- 要清音／調速／合併：`local_tts/finish_audio.sh all --tempo 0.9`。

停止 WebUI 或 API：在對應終端機按 `Ctrl+C`。
