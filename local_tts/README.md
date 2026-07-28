# GPT-SoVITS 本機啟動與調用

`local_tts` 不修改 GPT-SoVITS 上游的 `api_v2.py`、`webui.py`，只提供固定路徑、角色模型與 uv 啟動入口。

## 一次建立環境

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/setup_uv.sh
```

模型資產預期位於 `local_tts/assets/`：

- `GPT_weights_v2Pro/*.ckpt`
- `SoVITS_weights_v2Pro/*.pth`
- `Data/<角色>/*.wav`

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

若 API 啟動時已指定同一個角色，可加上 `--no-set-model` 避免每批次重新載入權重。

## 清理靜音

需要系統已安裝 `ffmpeg`：

```bash
./local_tts/process_audio.sh \
  --input local_tts/output/GPT_真人男_小說 \
  --output local_tts/output/GPT_真人男_小說_clean
```

預設會先以 `0.5` 秒、`-30 dB` 去除短靜音，再以 `2.0` 秒、`-20 dB` 去除長靜音。只執行一個步驟時加上 `--single-step`。

## 合併音檔

`merge_audio.sh` 只會合併檔名為 `0.mp3`、`1.mp3`、`2.mp3` 這類編號音檔，避免把已合併的輸出檔再次納入：

```bash
./local_tts/merge_audio.sh \
  --input-folder local_tts/output/GPT_真人男_小說_clean \
  --output-dir local_tts/output/merged
```

可用 `--suffix wav` 合併 WAV，或用 `--max-size` 限制每個輸出分片的來源總大小。

## 啟動 WebUI

```bash
./local_tts/start_web.sh --language Auto
```

主 WebUI 預設使用 GPT-SoVITS 的 `9874` 埠號。可用 `--port` 覆蓋，`--cpu` 強制 CPU，`--share` 啟用 Gradio 公開分享連結。
