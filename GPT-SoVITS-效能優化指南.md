# GPT-SoVITS 效能優化指南（每台電腦可重跑）

這份文件說明：

1. 本機為加速小說轉語音**一共改了什麼**
2. **最優化想法**（為什麼這樣比較快）
3. 換電腦 / 換顯卡時，如何**自己掃速找出最快參數**並寫回預設

詳細日常操作仍看：[`GPT-SoVITS-操作指南.md`](./GPT-SoVITS-操作指南.md) 與 [`GPT-SoVITS/local_tts/README.md`](./GPT-SoVITS/local_tts/README.md)。

---

## 一、一共改了什麼

### A. 真正影響轉小說速度的預設（已套用）

檔案：`GPT-SoVITS/local_tts/batch_tts.py`

| 常數 | 舊值 | 新值（RTX 3060 Ti 8GB 掃速最快組） |
|------|------|-------------------------------------|
| `DEFAULT_MAX_TEXT_LENGTH` | 2400 | **1200** |
| `DEFAULT_BATCH_SIZE` | 56 | **64** |
| `DEFAULT_FRAGMENT_INTERVAL` | 0.05 | **0.01** |

維持不變（經驗上必要）：

- `split_method=cut5`
- `top_k=15`
- `split_bucket=True`
- `parallel_infer=True`
- `batch_threshold=0.75`
- 角色 `speed_factor=1.0`（非 1.0 會關掉 bucket，變慢）

本機實測相對舊預設約 **+30%** 字/秒；再加上微優化（略過每請求 `empty_cache`、關閉 T2S tqdm、快取 v2Pro SV embedding）約再 **+8%**，合計最快組附近約 **328 字/秒**。之後直接跑 `batch_tts.sh` 即使用新預設；開跑時會依此吞吐顯示整本預估時間。

### B. 文件同步

- `GPT-SoVITS/local_tts/README.md`：3060 Ti 建議與「怎麼丟 API」
- `GPT-SoVITS/AGENTS.md`：同上
- `GPT-SoVITS-操作指南.md`：OOM 降級順序更新

### C. 量測 / 實驗工具（新增，不影響預設行為）

| 檔案 | 用途 |
|------|------|
| `local_tts/bench_sweep.py` + `bench_sweep.sh` | 掃 `batch_size` × 每段字數等，輸出最快組 |
| `local_tts/bench_overlap.py` + `bench_overlap.sh` | 對照 text 預取、TTS∥清音 |
| `local_tts/output/_bench_sweep/winner.json` | 本機上次掃速的最快參數 |

### D. 實驗功能（預設關閉，一般轉小說不用開）

| 改動 | 說明 | 結論 |
|------|------|------|
| `TTS_infer_pack/TTS.py` 的 `pipeline_prefetch` | T2S 時預取下一段 text/BERT | 幾乎沒贏（BERT 也佔 GPU） |
| `api_v2.py` 增加 `pipeline_prefetch: bool = False` | 給上面用的請求欄位 | 預設關 |
| `batch_tts.py` 每段 `client_wall` log | 方便看每段耗時 | 不影響速度 |

### E. 沒有當成預設的做法（刻意不做）

- **不要**同時開兩個 API / 兩個 `batch_tts` 搶同一張 GPU
- **不要**為了「餵滿 GPU」同時跑兩路 T2S，或硬做 T2S∥VITS（8GB 上收益小、易 OOM）
- text 預取、雙流推理：**實驗過，不划算**

---

## 二、最優化想法（為什麼這樣快）

### 1. 瓶頸在哪

單次 `/tts` 大致分成：

```text
text/BERT（約 15–20%）→ T2S（約 50–55%）→ VITS（約 10%）→ 後處理
```

- **T2S** 是自回歸，本來就很難把 GPU 利用率拉到訓練那種滿載，這是模型特性。
- **BERT 也在 GPU**，所以「一邊 T2S、一邊預取 BERT」會搶卡，牆鐘幾乎不贏。
- **VITS 只佔約一成**，T2S∥VITS 流水線工程大、8GB 還可能要砍 batch，總吞吐常沒比較好。

### 2. 正確的「餵滿」方式

在 **同一個 HTTP 請求裡** 用：

- `parallel_infer=True`：多句一起走 T2S
- `batch_size` 盡量大（顯存允許）
- `split_bucket=True`：長度接近的句子分桶，減少 padding 浪費

這才是官方設計的併發，不是多開幾個請求。

### 3. 每段字數（`max_text_length`）要適中

| 太長（例如 2400） | 太短（例如 800） |
|------------------|------------------|
| 單次活化值大，8GB 上容易變慢或抖動 | HTTP / 固定開銷變多，也不利大 batch |
| 本機舊預設偏慢 | 掃速裡偏慢 |

**3060 Ti 甜蜜點：約 1200 字/段 + batch_size=64。**

換卡後甜蜜點會變，所以要重跑掃速（見下一節）。

### 4. 請求節奏：做完立刻下一段

```text
請求 N ──────────────────► 寫完 N.wav
請求 N+1 ────────────────► 寫完 N+1.wav
（中間不要並行第二個 /tts）
```

- 一次只服務一個推理請求（`api_v2` 也是 `workers=1`）
- `batch_tts` 本來就是「一段接一段」；保持這樣即可
- 清音 / 調速是 **CPU + ffmpeg**，可以邊轉邊清（藏掉後處理時間），但不要為此再開第二條 TTS

### 5. 絕對別關的開關

| 關掉會怎樣（本機實測） |
|------------------------|
| `parallel_infer=False` → 速度崩到約 1/10 |
| `split_method=cut2` → 明顯變慢 |
| `split_bucket=False` → 略慢 |
| `speed_factor≠1.0` → 關 bucket，推理變慢（語速改留給 `finish_audio.sh all --tempo`） |

---

## 三、每台電腦怎麼找出「自己的最快組」

顯卡 VRAM、驅動、WSL 記憶體不同，**最快參數不能直接抄死**。流程如下。

### 步驟 0：準備

```bash
cd /home/sky/code/GPT-SoVITS   # 或你的專案路徑

# 需要：已安裝環境、ffmpeg、角色模型在 local_tts/assets/
./local_tts/start_api.sh --role 真人男
```

另開一個終端。準備一段約 4000–6000 字的 UTF-8 測試文本（可重複短文拼長）：

```bash
# 若已有範例：
# local_tts/output/_bench_timing_input.txt
# 或自己放一個 test.txt
```

### 步驟 1：跑掃速

```bash
./local_tts/bench_sweep.sh \
  --file-path local_tts/output/_bench_timing_input.txt \
  --role 真人男 \
  --output-dir local_tts/output/_bench_sweep \
  --repeats 2
```

會自動：

1. **Phase1**：掃 `batch_size ∈ {32,40,48,56,64,72}` × `max_text_length ∈ {1200,1600,2000,2400}`
2. **Phase2**：在最快組附近微調（bucket / top_k / threshold / cut / fragment 等）
3. **Confirm**：最快組再跑 2 次取平均
4. 對照舊預設，印加速比

輸出：

```text
local_tts/output/_bench_sweep/winner.json   # 最快參數
local_tts/output/_bench_sweep/summary.json  # 摘要
local_tts/output/_bench_sweep/results.json  # 全部結果
```

終端會印 `SWEEP TOP 10` 與 `WINNER`。

### 步驟 2：把最快組寫進預設

打開 `local_tts/batch_tts.py`，改這幾個常數為 `winner.json` 的值：

```python
DEFAULT_MAX_TEXT_LENGTH = 1200   # winner.max_text_length
DEFAULT_BATCH_SIZE = 64          # winner.batch_size
DEFAULT_FRAGMENT_INTERVAL = 0.01 # winner.fragment_interval
# 若 winner 的 top_k / split_method / batch_threshold 不同，一併改 DEFAULT_*
```

也可暫時不改檔，跑書時手動帶參數：

```bash
./local_tts/batch_tts.sh \
  --file-path /path/to/novel.txt \
  --role 真人男 \
  --no-set-model \
  --max-text-length 1200 \
  --batch-size 64 \
  --fragment-interval 0.01 \
  --output-dir local_tts/output/my_novel
```

### 步驟 3：OOM 時怎麼退

掃速若某組失敗（CUDA OOM），腳本會記 error 並繼續。手動降級順序建議：

1. `--batch-size`：64 → 56 → 48 → 40
2. 仍 OOM 再略降 `--max-text-length`（例如 1200 → 1000）
3. **不要**用關 `parallel_infer` 來省顯存換速度（會極慢）

### 步驟 4：（可選）對照後處理重疊

```bash
./local_tts/bench_overlap.sh \
  --file-path local_tts/output/_bench_timing_input.txt \
  --role 真人男
```

本機結論參考：

- text/BERT 預取：約 +5%，不值得當預設
- TTS∥清音：約 +12% 總牆鐘（清音佔比小時收益有限，長篇清音重時更明顯）

---

## 四、換機時的經驗規則（快速猜起始點）

| 顯存 | 建議起始 `batch_size` | 建議起始每段字數 |
|------|----------------------|------------------|
| 8GB（如 3060 Ti） | 56–64 | 1000–1400 |
| 12GB | 64–80 | 1200–2000 |
| 16GB+ | 80–96 | 1600–2400 |
| 6GB 或更小 | 24–40 | 800–1200 |

猜完仍以 `bench_sweep.sh` 為準。

其他環境因素：

- **WSL 記憶體**太小（例如只給 8GB）會拖垮大 batch；主機 RAM 夠時可在 `.wslconfig` 調高 `memory=`
- 掃速時關閉其他佔 GPU 的程式（WebUI 訓練、遊戲、第二個 API）

---

## 五、日常轉完整本小說（優化後）

```bash
cd /home/sky/code/GPT-SoVITS

# 終端 1
./local_tts/start_api.sh --role 真人男

# 終端 2（使用已寫入的最快預設）
./local_tts/batch_tts.sh \
  --file-path "/mnt/d/novels/我的小說.txt" \
  --role 真人男 \
  --no-set-model \
  --output-dir "local_tts/output/我的小說_真人男"
```

然後用 `finish_audio.sh all` 清音（可選調速）並合併，見操作指南第 6 節 / `local_tts/README.md`。  
`all` 預設一次編碼（多輪靜音 list + tempo 串在同一條 FFmpeg），比舊三步快；`--workers` 不寫＝用滿 CPU。  
可選：TTS 產出 WAV 的同時，另一終端先跑 `finish_audio.sh clean`（CPU∥GPU）；全部轉完再用 `all`（或 `tempo`+`merge`）收尾。

---

## 六、本機（RTX 3060 Ti）掃速摘要備查

| 項目 | 數值 |
|------|------|
| 最快參數組 | `batch_size=64`, `max_text_length=1200`, `fragment_interval=0.01`, bucket/parallel 開, cut5, top_k=15 |
| 相對舊預設（僅參數） | 約 **+29.5%** 字/秒（~227 → ~295） |
| 微優化後（empty_cache 略過 + TQDM 關 + SV cache） | 約 **328 字/秒**（相對舊行為同參數再 **+7.9%**） |
| 絕對峰值單次 | `bs=64/len=1200` 偶發 ~305–330 字/秒（屬抖動） |
| 大敗筆 | `parallel_infer=False` ≈ 31 字/秒；`cut2` ≈ 116 字/秒 |

微優化由 `start_api.sh` 預設啟用（`GPT_SOVITS_EMPTY_CACHE=0`、`TQDM_DISABLE=1`）。長跑若 OOM：

```bash
GPT_SOVITS_EMPTY_CACHE=1 ./local_tts/start_api.sh --role 真人男
```

`batch_tts.sh` 開跑會印「依 N 字/秒預估約需 …」，並在每段後用實測吞吐更新剩餘時間。可用 `--chars-per-sec 328` 覆寫初始預估值。

完整掃速表：`GPT-SoVITS/local_tts/output/_bench_sweep/results.json`。
微優化對照：`GPT-SoVITS/local_tts/output/_bench_micro/`。

---

## 七、檢查清單

- [ ] 單卡只跑一個 API
- [ ] `batch_tts` 用掃速最快組（或本機預設已更新）
- [ ] `parallel_infer` / `split_bucket` 保持開
- [ ] `speed_factor=1.0`，語速交給後處理
- [ ] OOM 只降 `batch_size`，不關 parallel
- [ ] 換顯卡後重跑 `bench_sweep.sh` 並更新 `batch_tts.py` 預設
