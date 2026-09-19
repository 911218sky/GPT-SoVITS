# GPT-SoVITS 本機啟動、訓練與小說轉語音

這份文件專門說明 `local_tts` 的實際操作。專案整體資料夾、WebUI 訓練概念與模型管理總覽請先看：

[`../../GPT-SoVITS-操作指南.md`](../../GPT-SoVITS-操作指南.md)

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

`batch_tts.sh` 預設使用每次約 `2400` 字、`cut5` 句子切分、`batch_size=56`、`split_bucket=True`、`parallel_infer=True`、`top_k=15`。`真人男` 預設 `speed_factor=1.0`（非 1.0 會關閉 bucket）。文本 BERT 特徵會批次抽取（預設 64 句/批，環境變數 `GPT_SOVITS_BERT_BATCH_SIZE` 可改）。不要同時開多個 `batch_tts.sh` 搶同一張 GPU；速度主要靠 API 內部批次，而不是多個 HTTP 請求併發。

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

## 清理靜音

需要系統已安裝 `ffmpeg`：

```bash
./local_tts/process_audio.sh \
  --input local_tts/output/GPT_真人男_小說 \
  --output local_tts/output/GPT_真人男_小說_clean
```

預設會先以 `0.5` 秒、`-30 dB` 去除短靜音，再以 `2.0` 秒、`-20 dB` 去除長靜音。只執行一個步驟時加上 `--single-step`。

## 合併前調整語速

`真人男` 預設 `speed_factor=1.0` 以保留 bucket 加速。若要接近以前的 `0.9` 語速（或任意倍數），在清理後、合併前對每個編號片段批次調整：

```bash
./local_tts/tempo_audio.sh \
  --input local_tts/output/GPT_真人男_小說_clean \
  --output local_tts/output/GPT_真人男_小說_tempo \
  --tempo 0.9 \
  --suffix mp3
```

- `--tempo 0.9`：變慢；`1.1`：變快。使用 FFmpeg `atempo`，音高不變。
- 預設只處理 `0.mp3`、`1.mp3` 這類編號檔；已存在的輸出會跳過，可中斷續跑。
- 合併時改吃 `--output` 那個資料夾。

## 合併音檔

`merge_audio.sh` 只會合併檔名為 `0.mp3`、`1.mp3`、`2.mp3` 這類編號音檔，避免把已合併的輸出檔再次納入：

```bash
./local_tts/merge_audio.sh \
  --input-folder local_tts/output/GPT_真人男_小說_tempo \
  --output-dir local_tts/output/merged
```

`process_audio.sh` 清理後會輸出 MP3，因此清理後合併時使用 `--suffix mp3`。如果直接合併批次工具產生的 WAV，才改用 `--suffix wav`。用 `--max-size-mb` 限制每個輸出分片的來源總大小，單位為 MB，預設是 `1024`：

```bash
./local_tts/merge_audio.sh \
  --input-folder local_tts/output/GPT_真人男_小說_tempo \
  --output-dir local_tts/output/merged \
  --max-size-mb 1024
```

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

清理、調語速、合併：

```bash
./local_tts/process_audio.sh \
  --input "local_tts/output/罪名不朽_310_真人男" \
  --output "local_tts/output/罪名不朽_310_真人男_clean"

# 合併前把每個編號片段調成 0.9 倍語速（可改成其他倍數）
./local_tts/tempo_audio.sh \
  --input "local_tts/output/罪名不朽_310_真人男_clean" \
  --output "local_tts/output/罪名不朽_310_真人男_tempo" \
  --tempo 0.9 \
  --suffix mp3

./local_tts/merge_audio.sh \
  --input-folder "local_tts/output/罪名不朽_310_真人男_tempo" \
  --output-dir "local_tts/output/罪名不朽_310_真人男_merged" \
  --suffix mp3 \
  --max-size-mb 1024
```

若 API 沒有先以相同角色啟動，移除批次指令中的 `--no-set-model`，讓批次工具自行載入模型。若顯示 CUDA out of memory，加入 `--batch-size 40 --max-text-length 1800`。批次中斷後可重跑相同指令續接，已存在的編號檔不會重做。

## 常見問題

- `找不到 local_tts/.venv`：先執行 `./local_tts/setup_uv.sh`。
- `模型權重不存在`：檢查 `local_tts/assets/` 與 `common.py` 的檔名是否完全一致。
- WebUI 啟動失敗：使用 `./local_tts/start_web.sh`，不要用 `local_tts/.venv/bin/python webui.py`。
- 沒有聲音或音質很差：確認 GPT/SoVITS 是同一次訓練、同一版本，且參考 WAV 與 `prompt_text` 對應。
- 清理或合併失敗：安裝 `ffmpeg`，再重新執行對應指令。
