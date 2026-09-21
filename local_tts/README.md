# GPT-SoVITS 本機啟動、訓練與小說轉語音

這份文件專門說明 `local_tts` 的實際操作。專案整體資料夾、WebUI 訓練概念與模型管理總覽請先看：

[`../../GPT-SoVITS-操作指南.md`](../../GPT-SoVITS-操作指南.md)

換機掃速、最優化想法與本機改動清單：

[`效能優化指南.md`](./效能優化指南.md)

以下指令都以 WSL/Linux 的專案根目錄 `/home/sky/code/GPT-SoVITS` 為例。Windows 的 `D:\` 磁碟在 WSL 通常對應 `/mnt/d/`。

`local_tts` 不修改 GPT-SoVITS 上游的 `api_v2.py`、`webui.py`，只提供固定路徑、角色模型與 uv 啟動入口。

## 一次建立環境

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/setup_uv.sh
```

首次使用主 WebUI/訓練時，根目錄也必須有 `.venv`；完整建立方式請看總指南。`setup_uv.sh` 只建立 `local_tts/.venv`，不是訓練 WebUI 的完整環境。

## 資料與模型放置規則

建議把個人資料集中放在專案旁的 `/home/sky/code/GPT-SoVITS-DATA/`：

```text
01_raw_audio/   原始真人錄音（WAV 優先）
02_datasets/    切分與標註後的訓練資料
03_novels/      UTF-8 TXT 小說
04_models/      GPT .ckpt、SoVITS .pth、參考 WAV
05_outputs/     轉換、清理、合併輸出
```

`local_tts` 實際載入的角色資產仍必須位於 `local_tts/assets/`，路徑由 `local_tts/common.py` 的 `ROLE_PROFILES` 決定。真人男目前需要：

```text
local_tts/assets/GPT_weights_v2Pro/真人男-e15.ckpt
local_tts/assets/SoVITS_weights_v2Pro/真人男_e8_s112.pth
local_tts/assets/Data/真人男/还是你来吧，我突然间觉得好像也没有那么迫切的想要脱单了。.wav
```

模型資產預期位於 `local_tts/assets/`：

- `GPT_weights_v2Pro/*.ckpt`
- `SoVITS_weights_v2Pro/*.pth`
- `Data/<角色>/*.wav`

已設定的角色包括 `Lele`、`Lele_Pro`、`阿甘`、`Sesame`、`真人男` 與 `Hitomi`。`Hitomi` 需要以下本機私有資產，不會提交到 Git：

```text
local_tts/assets/GPT_weights_v2Pro/Hitomi-e15.ckpt
local_tts/assets/SoVITS_weights_v2Pro/Hitomi_e8_s224.pth
local_tts/assets/Data/Hitomi/今天晚上有那个哎公司厨艺争霸战，感觉会很有趣。在阿基的台。.wav
```

音訊清理與合併需要 FFmpeg；WSL/Ubuntu 可執行：

```bash
sudo apt update && sudo apt install -y ffmpeg
```

## 啟動 API

```bash
./local_tts/start_api.sh --role 真人男
```

API 預設監聽 `http://127.0.0.1:9880`，啟動完成後會自動切換到角色模型。也可以只啟動服務、不切模型：

```bash
./local_tts/start_api.sh --no-set-model
```

啟動 Hitomi：

```bash
./local_tts/start_api.sh --role Hitomi
```

## 呼叫 API

```bash
curl -X POST http://127.0.0.1:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "你好，這是一段 GPT-SoVITS API 測試。",
    "text_lang": "zh",
    "ref_audio_path": "/home/sky/code/GPT-SoVITS/local_tts/assets/Data/真人男/还是你来吧，我突然间觉得好像也没有那么迫切的想要脱单了。.wav",
    "prompt_text": "还是你来吧，我突然间觉得好像也没有那么迫切的想要脱单了。",
    "prompt_lang": "zh",
    "text_split_method": "cut2",
    "media_type": "wav"
  }' --output local_tts/output/test.wav
```

## 批次產生音訊

```bash
./local_tts/batch_tts.sh \
  --file-path /path/to/input.txt \
  --role 真人男
```

使用 Hitomi 批次輸出：

```bash
./local_tts/batch_tts.sh \
  --file-path /path/to/input.txt \
  --role Hitomi \
  --output-dir local_tts/output/Hitomi
```

若 API 啟動時已指定同一個角色，可加上 `--no-set-model` 避免每批次重新載入權重。

### RTX 3060 Ti 長篇小說建議

`batch_tts.sh` 預設（本機掃速最快組）：每次約 `1200` 字、`cut5`、`batch_size=64`、`split_bucket=True`、`parallel_infer=True`、`top_k=15`、`fragment_interval=0.01`。相對舊預設（2400 字 / batch 56）約快 **30%**。

**怎麼丟 API 最快：**
1. **只開一個 API、一次只打一個 `/tts` 請求**（做完立刻打下一段）。不要並行多請求，8GB 會搶顯存變慢或 OOM。
2. **每段約 1200 字 + `batch_size=64`**：讓單次請求內塞滿句批次，又不要把整段拉太長拖垮顯存。
3. 保持 `split_bucket=True`、`parallel_infer=True`、`speed_factor=1.0`（非 1.0 會關 bucket）。
4. 清音／調速可邊轉邊做（CPU）；不要為此再開第二條 TTS。

`真人男` 預設 `speed_factor=1.0`；語速若要變慢，用 `finish_audio.sh all --tempo 0.9`（或單獨 `tempo`）。文本 BERT 特徵批次抽取（`GPT_SOVITS_BERT_BATCH_SIZE`，預設 64）。OOM 時依序降 `--batch-size 56`、`48`、`40`。

```bash
./local_tts/start_api.sh --role 真人男

./local_tts/batch_tts.sh \
  --file-path /mnt/d/novels/my_novel.txt \
  --role 真人男 \
  --no-set-model
```

如果遇到 CUDA out of memory，先降到 `--batch-size 48`，仍不足再試 `40` 或 `24`。想更容易中斷續跑，可以把 `--max-text-length` 降到 `1800`。

本機附一份原創測試小說，可用來確認 API 與批次流程：

```bash
./local_tts/batch_tts.sh \
  --file-path local_tts/examples/test_novel.txt \
  --role 真人男 \
  --no-set-model \
  --output-dir local_tts/output/test_novel_3060ti
```

## 清理、調速與合併（`finish_audio.sh`）

需要系統已安裝 `ffmpeg`。舊的 `process_audio` / `tempo_audio` / `merge_audio` 已合併成一支腳本與一支 Python：

- 腳本：`local_tts/finish_audio.sh`
- 實作：`local_tts/finish_audio.py`

```bash
./local_tts/finish_audio.sh <指令> [參數…]
./local_tts/finish_audio.sh help
./local_tts/finish_audio.sh all --help   # 或其他指令
```

| 指令 | 做什麼 | 何時用 |
|------|--------|--------|
| `all` | **推薦**：清靜音 ± 調速（一次編碼）→ 合併 | 長篇轉完後收尾 |
| `clean` | 只清靜音（每輪各編碼一次，較慢） | 只要清音、或邊 TTS 邊清 |
| `tempo` | 只調語速 | 已有 clean MP3，只改語速 |
| `merge` | 只合併編號音檔 | 已有 prepared/tempo MP3 |

### 為什麼 `all` 比較快、效果卻一樣？

以前三步大約是：**清音編碼 2 次 + 調速編碼 1 次 + 合併 copy**。  
現在 `all`（不要加 `--legacy`）把多輪靜音與調速串成**一條 FFmpeg filter**，每段只 **MP3 編碼 1 次**，再 `-c copy` 合併。預設靜音輪次仍是舊的兩輪，聽感參數對齊。

| 項目 | 舊三步 | `all`（預設） |
|------|--------|----------------|
| 每段編碼次數 | ≈ 3 | **1** |
| 靜音預設 | `0.5s/-30dB` 再 `2.0s/-20dB` | 同左（`--silence-steps`） |
| 調速 | 另一步 | 同一條 filter 的 `--tempo` |
| 合併 | copy | copy |
| 並行 | 約 CPU−2 | **預設用滿全部 CPU**（可不寫 `--workers`） |

### 最常見指令（等同以前：清音兩輪 + 語速 0.9 + 合併 1GB）

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/finish_audio.sh all \
  --input "local_tts/output/你的小說_真人男" \
  --tempo 0.9 \
  --max-size-mb 1024
```

- 中繼：`<input>_prepared/`（編號 MP3）
- 成品：`<input>_merged/`（如 `1_tts_powerful_output.mp3`…）
- 已存在的檔會跳過，可中斷續跑。
- 不調速就省略 `--tempo`（預設 `1.0`）。

### `all` 參數一覽

| 參數 | 預設 | 說明 |
|------|------|------|
| `--input` | 必填 | `batch_tts` 編號 WAV 目錄 |
| `--output-dir` | `<input>_merged` | 合併成品目錄 |
| `--work-dir` | `<input>_prepared` | 中繼 MP3 目錄 |
| `--tempo` | `1.0` | 語速；`0.9`=變慢、`1.1`=變快（音高不變） |
| `--silence-steps` | `0.5:-30,2.0:-20` | 多輪靜音 list（見下） |
| `--max-size-mb` | `1024` | 每個合併分片的來源總大小上限（MB） |
| `--output-name` | `tts_powerful_output.mp3` | 合併檔名 |
| `--workers` | **全部 CPU** | 並行 FFmpeg 數；通常不用手動指定 |
| `--quality` | `4` | MP3 品質 0–9（數字越大越快、檔越小） |
| `--volume-boost` | `1.0` | 音量倍率 |
| `--legacy` | 關 | 走舊三步（每輪各編碼，較慢，僅除錯） |

### `--silence-steps`（多輪靜音 list）

格式：`秒數:閾值dB`，多輪用逗號（或分號）分隔。

```bash
# 預設＝舊兩輪（可省略不寫）
--silence-steps "0.5:-30,2.0:-20"

# 只清一輪
--silence-steps "0.5:-30"

# 三輪
--silence-steps "0.3:-35,0.5:-30,2.0:-20"
```

`all` 會把這些輪次串在同一條 FFmpeg 裡；`clean` 則每輪各編碼一次。

### `clean`（只清靜音）

```bash
./local_tts/finish_audio.sh clean \
  --input local_tts/output/GPT_真人男_小說 \
  --output local_tts/output/GPT_真人男_小說_clean \
  --silence-steps "0.5:-30,2.0:-20"
```

支援與 `all` 相同的 `--silence-steps` / `--workers` / `--quality` / `--volume-boost`。輸出為編號 MP3。

### `tempo`（只調語速）

```bash
./local_tts/finish_audio.sh tempo \
  --input local_tts/output/GPT_真人男_小說_clean \
  --output local_tts/output/GPT_真人男_小說_tempo \
  --tempo 0.9 \
  --suffix mp3
```

| 參數 | 說明 |
|------|------|
| `--input` / `--output` | 必填 |
| `--tempo` | 必填，語速倍數 |
| `--suffix` | 只處理該副檔名（如 `mp3`） |
| `--all-files` | 不限編號檔名 |
| `--workers` | 預設用滿 CPU |
| `--quality` | MP3 品質 0–9 |

預設只處理 `0.mp3`、`1.mp3`…。

### `merge`（只合併）

只合併檔名為數字的音檔，避免把已合併輸出再吃進去：

```bash
./local_tts/finish_audio.sh merge \
  --input-folder local_tts/output/GPT_真人男_小說_prepared \
  --output-dir local_tts/output/GPT_真人男_小說_merged \
  --suffix mp3 \
  --max-size-mb 1024
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--input-folder` / `--input` | 必填 | 編號音檔目錄 |
| `--output-dir` / `--output` | 必填 | 合併輸出目錄 |
| `--suffix` | `mp3` | 也可 `wav`（直接併 WAV） |
| `--max-size-mb` | `1024` | 分片大小上限 MB |
| `--output-name` | `tts_powerful_output.mp3` | 檔名 |
| `--workers` | 用滿 CPU（且 ≤ 分片數） | 多分片並行合併 |

### 分步範例（等同舊三支腳本）

```bash
./local_tts/finish_audio.sh clean  --input ".../小說_真人男" --output ".../小說_真人男_clean"
./local_tts/finish_audio.sh tempo  --input ".../小說_真人男_clean" --output ".../小說_真人男_tempo" --tempo 0.9 --suffix mp3
./local_tts/finish_audio.sh merge  --input-folder ".../小說_真人男_tempo" --output-dir ".../小說_真人男_merged" --suffix mp3 --max-size-mb 1024
```

一般情況直接用上面的 `all` 即可，更快且效果相同。

## 啟動 WebUI

```bash
./local_tts/start_web.sh --language Auto
```

主 WebUI 預設使用 GPT-SoVITS 的 `9874` 埠號。可用 `--port` 覆蓋，`--cpu` 強制 CPU，`--share` 啟用 Gradio 公開分享連結。

WebUI 用於資料標註、特徵處理與 GPT/SoVITS 訓練；訓練時不需要先啟動 API。訓練完成後，從 `logs/<實驗名稱>/` 取出同版本的 `.ckpt` 與 `.pth`，放入上述 `assets` 目錄，再用 API 或 WebUI 推理測試。

## 完整範例：罪名不朽_310.txt → 真人男聲

第一個終端機：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_api.sh --role 真人男
```

第二個終端機：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/batch_tts.sh \
  --file-path "/mnt/d/APP/TomatoNovelDownloader/罪名不朽_310.txt" \
  --role 真人男 \
  --no-set-model \
  --output-dir "local_tts/output/罪名不朽_310_真人男"
```

清理、調語速、合併（推薦一次做完）：

```bash
./local_tts/finish_audio.sh all \
  --input "local_tts/output/罪名不朽_310_真人男" \
  --tempo 0.9 \
  --max-size-mb 1024
```

若 API 沒有先以相同角色啟動，移除批次指令中的 `--no-set-model`，讓批次工具自行載入模型。若顯示 CUDA out of memory，加入 `--batch-size 40 --max-text-length 1800`。批次中斷後可重跑相同指令續接，已存在的編號檔不會重做。

## 常見問題

- `找不到 local_tts/.venv`：先執行 `./local_tts/setup_uv.sh`。
- `模型權重不存在`：檢查 `local_tts/assets/` 與 `common.py` 的檔名是否完全一致。
- WebUI 啟動失敗：使用 `./local_tts/start_web.sh`，不要用 `local_tts/.venv/bin/python webui.py`。
- 沒有聲音或音質很差：確認 GPT/SoVITS 是同一次訓練、同一版本，且參考 WAV 與 `prompt_text` 對應。
- 清理或合併失敗：安裝 `ffmpeg`，再重新執行對應指令。
