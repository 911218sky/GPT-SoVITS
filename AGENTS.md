# GPT-SoVITS 專案操作指南

這份文件提供給協作 AI 與開發者使用。修改本專案前，先閱讀本文件與更深層目錄中的 `AGENTS.md`。使用者的明確要求優先於本文件。

## 專案結構

- `webui.py`：GPT-SoVITS 主 WebUI 入口。
- `api_v2.py`：GPT-SoVITS HTTP API 入口。
- `config.py`：推理、WebUI、API 埠號與模型設定。
- `tools/uvr5/webui.py`：UVR5 人聲分離子 WebUI。
- `local_tts/`：本機 API/WebUI 啟動器、批次 TXT 轉語音、音訊清理與合併工具。
- `GPT_SoVITS/pretrained_models/`：訓練與推理需要的基礎預訓練模型。
- `logs/`：GPT-SoVITS 訓練實驗輸出。

## 不可提交的資產

以下內容可能非常大、包含私人資料，或不適合放進 Git。不要使用 `git add -f` 強制加入：

- `.venv/`、`local_tts/.venv/`
- `local_tts/assets/` 內的 GPT/SoVITS 權重與參考音訊
- `local_tts/output/`、`output/`、`logs/`
- 個人 TXT、WAV、MP3 與訓練資料集

GitHub 普通 Git 儲存庫不適合存放數百 MB 到數 GB 的模型。模型資產必須由使用者另外下載或從本機複製。

## 系統需求

建議環境：

- WSL2 Ubuntu 或 Linux
- Python 3.10
- `uv`
- GPU 推理/訓練時使用相容的 NVIDIA 驅動與 CUDA
- `ffmpeg`（音訊清理、轉檔與合併需要）

安裝基本工具：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 建立環境

在專案根目錄執行：

```bash
cd /home/sky/code/GPT-SoVITS
uv venv --allow-existing --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m nltk.downloader -d "$HOME/nltk_data" averaged_perceptron_tagger_eng
./local_tts/setup_uv.sh
```

`local_tts/.venv` 是約數十 MB 的輕量工具環境，主要提供本機工具與 API 啟動器依賴；訓練 WebUI 必須由 `local_tts/start_web.sh` 使用根目錄 `.venv` 啟動。不要手動把兩個環境的 `site-packages` 混合放進 `PYTHONPATH`，否則可能造成 Gradio、FastAPI、Starlette 版本不相容。

WebUI 目前固定使用 `fastapi==0.115.6` 與 `starlette==0.41.3`，搭配已安裝的 Gradio 4.x。若更新依賴後出現 `TypeError: unhashable type: 'dict'` 或 `TypeError: argument of type 'bool' is not iterable`，先重新同步根目錄 `.venv`，並用 `./local_tts/start_web.sh` 啟動，不要直接使用 `local_tts/.venv/bin/python webui.py`。

目前 GPU 環境使用 `torch==2.12.1+cu132` 與 `torchvision==0.27.1+cu132`。PyTorch 2.12.1 沒有相同版本的 `torchaudio` wheel，因此核心讀音檔與重採樣改由 `audio_compat.py` 使用 `soundfile` 與 `scipy` 完成；不要重新安裝舊版 `torchaudio`。

`requirements.txt` 固定使用 `onnxruntime==1.23.2` 的 CPU 版本。`onnxruntime-gpu==1.23.2` 需要 CUDA 12 的 `libcublas.so.12` 等動態庫，與本環境的 CUDA 13.2 不相容；GPT-SoVITS 的 Torch 推理仍使用 GPU，G2PW 與 UVR5 的 ONNX 推理改用 CPU。

## 模型資產配置

`local_tts/common.py` 會從以下位置尋找角色模型：

```text
local_tts/assets/GPT_weights_v2Pro/<角色>-e15.ckpt
local_tts/assets/SoVITS_weights_v2Pro/<角色>_e8_s*.pth
local_tts/assets/Data/<角色>/*.wav
```

目前啟用的角色是：`Lele`、`Lele_Pro`、`Sesame`、`真人男`、`阿甘`。新增角色時，必須同步修改 `local_tts/common.py` 的角色設定，並確認 GPT 權重、SoVITS 權重、參考音訊與 prompt 文字都存在。

Windows D 槽在 WSL 中通常使用 `/mnt/d/`，例如：

```bash
cp -a /mnt/d/你的模型資料夾/. /home/sky/code/GPT-SoVITS/local_tts/assets/
```

訓練用的基礎模型放在 `GPT_SoVITS/pretrained_models/`，不要把它們複製到 `local_tts/assets/`。

## 啟動訓練 WebUI

訓練模型只需要啟動 WebUI，不需要先啟動 `start_api.sh`：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_web.sh
```

瀏覽器開啟：

```text
http://127.0.0.1:9874
```

如果需要讓其他介面連線：

```bash
./local_tts/start_web.sh --host 0.0.0.0
```

訓練流程在 WebUI 內依序完成：資料標註/檢查、SSL 特徵提取、語音切分與 SoVITS/GPT 訓練。訓練輸出通常在 `logs/`。不要把訓練輸出直接當成 `local_tts/assets/`，完成訓練後再依角色設定複製或指定對應權重。

本專案目前預設使用 `v2` 訓練版本，因為本機已配置：

```text
GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
```

在 WebUI 的「預訓練模型路徑」中，`預訓練 GPT 模型`、`預訓練 SoVITS-G 模型` 與左側的訓練版本必須相互對應。不要選 `v1`、`v2Pro` 或 `v2ProPlus` 後仍保留 v2 的模型路徑。

訓練集格式化一鍵流程的建議順序：

1. 準備 `音檔|說話者|語言|文字` 格式的 `.list` 檔。
2. 確認音檔已完成切分，且 `GPT_SoVITS/pretrained_models/chinese-hubert-base` 存在。
3. 使用 `v2`，依序執行文本/BERT、SSL、語意 Token 三個步驟。
4. 確認 `logs/<實驗名>/2-name2text.txt`、`4-cnhubert/`、`6-name2semantic.tsv` 都已產生，再開始訓練。

純英文標註即使被 ASR 誤標為 `ZH`，格式化腳本也會自動改走英文 G2P；混合中文句子仍應標記為 `ZH`。

停止服務使用 `Ctrl+C`。

## 啟動 API 與批次轉語音

啟動 API 並載入角色：

```bash
./local_tts/start_api.sh --role 真人男
```

預設 API 位址是 `http://127.0.0.1:9880`。如果模型已由 API 載入，批次處理可以避免重新切換模型：

```bash
./local_tts/batch_tts.sh \
  --file-path "/mnt/d/文字/new_text.txt" \
  --role 真人男 \
  --no-set-model
```

如果 API 尚未載入相同角色，省略 `--no-set-model`：

```bash
./local_tts/batch_tts.sh \
  --file-path "/mnt/d/文字/new_text.txt" \
  --role 真人男
```

預設輸出會放在：

```text
local_tts/output/GPT_<角色>_<TXT檔名>/
```

每段語音先輸出為 WAV；可以用 `--output-dir` 指定其他位置。

## 音訊清理與合併

去除短靜音、長靜音並轉成 MP3：

```bash
./local_tts/process_audio.sh \
  --input local_tts/output/GPT_真人男_new_text \
  --output local_tts/output/GPT_真人男_new_text_clean
```

合併編號音檔：

```bash
./local_tts/merge_audio.sh \
  --input-folder local_tts/output/GPT_真人男_new_text_clean \
  --output-dir local_tts/output/merged
```

合併工具只處理 `0.mp3`、`1.mp3`、`2.mp3` 這類數字檔名，避免把已產生的合併檔再次納入。可用 `--suffix wav` 處理 WAV。

## HTTP API 範例

API 啟動後，可以直接呼叫：

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

## 常見問題

- `找不到 uv`：先安裝 `uv`，再重新執行 `local_tts/setup_uv.sh`。
- `找不到 ffmpeg`：安裝 `sudo apt install -y ffmpeg`。
- 模型找不到：檢查 `local_tts/assets/` 的檔名、角色名稱與路徑大小寫。
- API 無法連線：確認 `9880` 沒有被其他程序占用，並先啟動 `start_api.sh`。
- WebUI 無法連線：確認 `9874` 沒有被占用；WSL 通常使用 `http://localhost:9874` 或 `http://127.0.0.1:9874`。
- `TypeError: unhashable type: 'dict'` 或 `argument of type 'bool' is not iterable`：通常是 Gradio 與 FastAPI/Starlette 版本不相容，或兩個虛擬環境被混用；使用 `./local_tts/start_web.sh`，不要自行設定跨環境 `PYTHONPATH`。
- UVR5 人聲分離與音頻標註是主 WebUI 的選用子服務；任一子服務失敗不代表 GPT/SoVITS 訓練或 TTS API 失敗。
- Fun-ASR-Nano 使用 `qwen3` 架構，請使用 `requirements.txt` 指定的 Transformers 版本；若看到 `KeyError: 'qwen3'`，先重新同步根目錄 `.venv`。
- `torch.cat(): expected a non-empty list of Tensors`：通常是英文或空白文字被標成 `ZH`；重新執行格式化，並檢查 `.list` 的語言欄位與文字是否正確。
- `找不到 6-name2semantic-0.tsv`：先檢查前一個 1A/1B 步驟是否非零退出，再確認 `預訓練 SoVITS-G 模型` 路徑存在；不要只重跑最後一個步驟。
- `Failed to load libcublasLt.so.12`、`libcudart.so.12`：不要另外安裝 CUDA 12 或舊版 Torch；重新執行 `uv pip install --python .venv/bin/python -r requirements.txt`，確認使用 CPU 版 `onnxruntime==1.23.2`。
- `Resource averaged_perceptron_tagger_eng not found`：執行 `.venv/bin/python -m nltk.downloader -d "$HOME/nltk_data" averaged_perceptron_tagger_eng`；中文參考文字與目標文字請在介面選 `中文` 或 `auto`，不要誤選 `English`。
- `Input type Half and bias type float`：代表以 CPU/非半精度模式讀取了 GPU 產生的半精度 HuBERT 特徵；目前語意抽取會自動轉成 `float32`。

## AI 修改規範

1. 修改前先檢查相關程式、`git status` 與更深層 `AGENTS.md`。
2. 優先沿用 `local_tts/` 現有啟動器，不要另建第二套環境或重複安裝 PyTorch/CUDA。
3. 不要提交模型、參考音訊、輸出檔、個人資料或虛擬環境。
4. 改動 WebUI、API 或音訊流程後，至少執行對應的編譯檢查與實際端點/命令 smoke test。
5. 修改埠號時使用 `GPT_SOVITS_WEBUI_PORT`、`GPT_SOVITS_API_PORT` 等既有環境變數，不要硬編碼新的埠號。
6. 不要為了讓測試變綠而刪除或弱化既有功能；遇到上游相容性問題，優先做最小、可回復的修正。
7. 提交前確認 `git diff --check`，並用 `git check-ignore` 確認模型和輸出沒有被加入暫存區。

## 最小驗證

```bash
.venv/bin/python -m py_compile webui.py GPT_SoVITS/prepare_datasets/1-get-text.py GPT_SoVITS/prepare_datasets/3-get-semantic.py tools/subfix_webui.py
local_tts/.venv/bin/python -m compileall -q local_tts tools/uvr5/webui.py
git diff --check
```

API/WebUI 有條件時，還要實際使用 `start_api.sh` 或 `start_web.sh`，並以 `curl` 或瀏覽器確認服務可以連線。
