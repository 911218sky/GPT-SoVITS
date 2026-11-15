# GPT-SoVITS TTS 速度調校（RTX 3060 Ti 8GB / sky-3060ti-linux）

> **給後續 AI／人類：** 本機小說轉語音變慢、或要重新找「這台電腦」最快設定時，先讀這份再動手。  
> **主機：** `sky-3060ti-linux`（WSL2）  
> **GPU：** NVIDIA GeForce RTX 3060 Ti **8GB**  
> **專案：** `/home/sky/code/GPT-SoVITS`  
> **資料：** `/home/sky/code/GPT-SoVITS-DATA`  
> **遠端操作：** PowerShell 只當遙控器；腳本寫入 `D:\cursor\sshfs\tmp\_ai_task_*.sh` 後  
> `cmd /c "ssh sky-3060ti-linux bash -s < script.sh"`。細節見 `D:\cursor\sshfs\docs\AI-讀寫與搜尋.md`。

相關文件：

- 小說轉語音流程：[`local_tts/README.md`](./local_tts/README.md)
- AI 工作流：[`GPT-SoVITS-AI-Workflow.md`](./GPT-SoVITS-AI-Workflow.md)
- 本機狀態（勿推遠端）：`/home/sky/code/GPT-SoVITS-LOCAL-STATUS.md`

---

## 接手先看

本機是 RTX 3060 Ti 8GB，WSL。一次只開一個 API：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_api.sh --role 真人男
```

這些環境變數已經寫在 `local_tts/start_api.sh`，不要改：

| 變數 | 值 | 原因 |
|---|---|---|
| `CUDAGraph` | `0` | 舊的整段圖會吃滿 8GB，速度掉到每秒幾十個字 |
| `GPT_SOVITS_T2S_STEP_GRAPH` | `1` | 語意 token 用逐步圖，算完就放掉 |
| `GPT_SOVITS_VITS_MICROBATCH` | `16` | v2Pro 聲學一次最多合 16 句。v5 不走這條 |
| `GPT_SOVITS_EMPTY_CACHE` | `0` | 推理途中不收記憶體。整批結束再清一次 |
| `GPT_SOVITS_V5_CFM_BATCH` | `24` | v5turbo 一次送進 CFM 的句數 |
| `GPT_SOVITS_V5_REF_FRAMES` | `160` | v5 參考梅爾只留最後 160 幀（1.6 秒） |

批次預設在 `local_tts/batch_tts.py`：每次約 1200 字、`split_method=auto`、`batch_size=64`、`fragment_interval=0`、分桶開、平行推理開、`speed_factor=1.0`。語速要變慢就在轉完後用 `finish_audio.sh`。推理時改語速會關掉分桶。

`auto` 看目前載入的模型。v2Pro 用 `cut5`（遇標點就切）。v5turbo 用 `cut20`（標點子句收到大約 20 字）。同一段《吃神者》前 3000 字、句間不補靜音：v2Pro 約 **258 字/秒**，v5turbo 約 **98 字/秒**。v5 再切短會變慢；收到 30 字幾乎沒差；收到 40 字會在 batch 48 爆顯存。細節在第 12 節。

`fragment_interval=0` 表示每一段合成完不再在尾巴接一段全 0 的取樣。這個參數的作用是換氣，上游預設 0.3 秒。小說不要這段空白。程式只在值大於 0 時才接。

v2Pro 乾淨稿大約 300–360 字/秒。章節標題很多的吃神者 4827 字，熱跑大約 304 字/秒。比這慢很多時，先看 log 裡的 `[TTS_TIMING]`。

平常不用再改的程式：

| 做什麼 | 在哪 |
|---|---|
| v2 用 cut5、v5 用 cut20 | `default_split_method`、`cut20` |
| 沒有標點而且長過 120 字才硬切 | `split_for_fast_t2s` |
| v5 多句一起跑 CFM，聲碼器最多 48 句 | `synthesize_v5_mels`、`using_vocoder_synthesis_batched_infer` |
| 長句不要讓整桶重跑 | `t2s_model.py` 的 `_batch_decode_step_graph` |
| 下一桶注音跟聲學一起做 | `TTS.py` 的 `_PhoneBucketPrefetcher` |

轉完小說要關掉 API，把顯存讓出來。下面從第 0 節起是較早的測量。和第 12 節衝突時，以這一節和第 12 節為準。

---


## 0. 一句話結論（2026-11-15 實測）

慢的主因仍是舊的 **Accel 靜態 CUDA Graph** 在 8GB 上把顯存吃滿。那個開關繼續關。

現在另有一條 **逐步 CUDA Graph**（`GPT_SOVITS_T2S_STEP_GRAPH`，預設開）：只把 T2S 每一步的 Transformer 收成圖，算完就放掉 KV，再跑 VITS。這條用的是 PyTorch 標準 CUDA Graph，不綁 RTX 3060 Ti，也不綁 UEFI 張量解鎖。

**本機目前實測（cintia，fp16，WSL2）：**

| 文章 | 牆鐘 | 字/秒 | 伺服器內部分段 |
|---|---|---|---|
| 1200 字，熱身 | 4.11 秒 | **292** | t2s 2.11 秒、vits 0.70 秒、文字 0.66 秒 |
| 5000 字，一筆 | 13.92 秒 | **359** | t2s 6.15 秒、vits 4.21 秒、文字 2.97 秒 |

顯存峰值約 6.8GB，閒置約 3.5GB。

上表是比較乾淨的 cintia 稿。有章節標題和大量數字時不會到 359，吃神者那種稿大約 300 字/秒，見第 11 節。

```bash
export CUDAGraph=0
export GPT_SOVITS_T2S_STEP_GRAPH=1
export GPT_SOVITS_VITS_MICROBATCH=16
export GPT_SOVITS_EMPTY_CACHE=0
# 這組註解是當天早上的 v2Pro 測量。現在的請求預設見「接手先看」：auto、fragment_interval=0。
# cut5, max_text_length=1200, batch_size=64, split_bucket=True, parallel_infer=True
# top_k=15, speed_factor=1.0
```

已寫進 `local_tts/start_api.sh` 預設。啟動：

```bash
cd /home/sky/code/GPT-SoVITS
./local_tts/start_api.sh --role cintia   # 或 真人男 等
```

**每次小說轉完一定要關 API、放顯存**（使用者硬性要求）：

```bash
pkill -f api_v2.py; pkill -f local_tts.start_api
# 必要時再清殘留
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

---

## 1. 歷史對照：為什麼「以前快、現在慢」

| Commit / 時期 | 特徵 | 8GB 上的體感 |
|---|---|---|
| `db790e44…`（約 2026-09） | **沒有** `tools/acceleration.py` / Accel CUDA Graph | 小說轉換快（使用者記憶 ~200+ 字/秒） |
| `bf8c5a3`（2026-11-14 merge） | 併入 CUDA Graph、`GPT_SoVITS/Accel/**`，`TTS.py` 大改 | 預設開 Graph → 顯存 ~7.5/8GB → **20–40 字/秒** |
| 2026-11-14 | 還沒有逐步圖。Graph 關、micro=16、每請求清記憶體 | 熱身約 230–240 字/秒 |

查某個 commit 有沒有 Accel：

```bash
cd /home/sky/code/GPT-SoVITS
git cat-file -e db790e4:tools/acceleration.py && echo YES || echo NO_ACCEL
git log --oneline -- GPT_SoVITS/Accel tools/acceleration.py | head
```

**重點心智模型：**  
在 8GB 上，**T2S 的 CUDA Graph「看起來加速」會偷走 VITS 的顯存**；小說路徑 VITS 佔牆鐘時間大頭，所以整體反而更慢。

---

## 2. 怎麼找重點（診斷流程，照順序）

### 2.1 先看顯存，不要先猜模型

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
ps -ef | grep -E 'api_v2|start_api' | grep -v grep
```

| 啟動後 idle 顯存 | 含義 |
|---|---|
| ~2.0–3.5 GB | 正常（無 CUDA Graph） |
| ~7.0–7.9 GB | **幾乎肯定 Graph/Accel 佔滿** → 後面 VITS 會暴慢 |

### 2.2 打開分段計時 `TTS_TIMING`

```bash
export GPT_SOVITS_TIMING_LOG=/tmp/tts_bench/timing.log
export PYTHONUNBUFFERED=1
# 再啟動 API，跑一筆 /tts 後看 log
```

輸出示例：

```text
[TTS_TIMING] ref=… text=… bucket_prep=… t2s=… vits=… post=… total=… prefetch=…
```

**怎麼讀：**

| 欄位 | 正常熱身（1200 字、快設定） | 異常訊號 |
|---|---|---|
| `vits` | ~1.2–1.5s | **>5–20s** → 顯存 thrash 或整段超大 concat |
| `t2s` | ~2.5–3.5s | 若 Graph 開著但 total 仍慢，多半不是 T2S 的錯 |
| `text` | ~0.7–1.0s | 含分句 + BERT，通常不是主兇 |
| `post` | ~0.05s（`fragment_interval=0.01`） | 若很大，檢查 interval / 後處理 |
| `total` | ~5s → **~240 字/秒** | `1200/total` 即字/秒 |

字/秒快速算法：`chars / total_seconds`（只算非空白字元即可）。

### 2.3 確認行程真的吃到你設的環境變數

`start_api.sh` → `start_api.py` → `subprocess` 拉起 `api_v2.py`。  
**一定要以 `api_v2.py` 的 PID 查 environ**，不要只看 shell：

```bash
PID=$(pgrep -f 'api_v2.py' | head -1)
tr '\0' '\n' < /proc/$PID/environ | grep -E 'CUDAGraph|VITS_MICRO|EMPTY_CACHE|TIMING'
```

若你設了 `CUDAGraph=0` 但這裡仍是 `auto/1`，代表啟動路徑被蓋掉或起錯行程——先 `pkill` 乾淨再啟。

### 2.4 用「同一段 1200 字」冷跑 + 熱跑

- **冷跑**：含第一次 ref 編碼，會慢（`ref` 常 2–3s）。  
- **熱跑**：才是跟「以前轉小說」可比的數字。  
- 至少熱跑 2–3 次；若熱跑一次比一次慢且顯存爬升，多半是碎片／沒 `empty_cache`。

---

## 3. 根因清單（依發生頻率）

### A. CUDA Graph 吃滿 8GB（最大兇）

- **現象：** idle ~7.5GB；`vits` 從 ~1s 變成數十秒；總速 20–40 字/秒。  
- **修復：** `CUDAGraph=0`（本機預設）。  
- **代碼：** `tools/acceleration.py` 的 `cuda_graph_available()`：≤10GB 預設視為 off（除非 `CUDAGraph=1` 強制）。  
- **推論入口：** `GPT_SoVITS/TTS_infer_pack/TTS.py` 讀 `use_cuda_graph` 時也有 auto-off。

### B. VITS「整段超大 concat」在顯存不足時 thrash

- **現象：** `parallel_infer` + `cut5` 很多短句，全部 `torch.cat` 進一次 `decode`；顯存頂滿後 `vits` 爆炸。  
- **對照：** `db790e4` 也是 method-2 整段 concat，但**當時沒有 Graph**，有足夠顯存所以快。  
- **修復：** `GPT_SOVITS_VITS_MICROBATCH=N`  
  - `1`：一句一解（最省顯存，但開銷大，~110–130 字/秒）  
  - `16`：**本機甜蜜點**（~230+）  
  - `0`：等同整段 concat（顯存鬆時可快，但熱身後易頂到 ~7.8GB，下一次 `vits` 變慢）

### C. 每句重算 speaker embedding `ref_enc`

- **現象：** microbatch/逐句時，每句重跑 `ref_enc`，VITS 額外變慢。  
- **修復：** `GPT_SoVITS/module/models.py` `SynthesizerTrn.decode` 的 `_tts_ge_cache`。

### D. `empty_cache` 的取捨

- **推理途中不做（`EMPTY_CACHE=0`，目前預設）：** 少一次 CUDA sync，熱跑才能停在每秒 300 字以上。  
- **每請求做（`EMPTY_CACHE=1`）：** 最穩，但同步成本會把整段平均從約 320 拉回 270 附近。  
- **跑完一定清一次：** `GET /control?command=empty_cache`。`batch_tts` 結束（含中途失敗）會自動打。權重留在顯存，下一輪不用重載。  
- 整本小說若顯存一路爬、`vits` 變長，把 `EMPTY_CACHE` 改回 `16`，不要改回每請求。

### E. 其他（次要）

- `load_audio_gpu` 曾誤用未匯入的 `torchaudio` → 改走 `load_audio()`（`TTS.py`）。  
- T2S Accel 因 `batch_size` 變化反覆 rebuild CUDA Graph → 已改成容量制（`GPT_SOVITS_T2S_MAX_BATCH`）。  
- **不要**用 `start_api.sh` 的前景 `wait` 卡死 SSH；測速請 `setsid`／另開 SSH，API ready 就立刻打 `/tts`。

---

## 4. 本機最優配置（鎖定值）

寫在 `local_tts/start_api.sh`：

| 變數 | 預設 | 說明 |
|---|---|---|
| `CUDAGraph` | `0` | 舊的 Accel **靜態**圖。8GB 必須關 |
| `GPT_SOVITS_T2S_STEP_GRAPH` | `1` | 逐步圖。程式預設開；設 `0` 退回普通 T2S |
| `GPT_SOVITS_VITS_MICROBATCH` | `16` | VITS 每次最多合 16 句 |
| `GPT_SOVITS_EMPTY_CACHE` | `0` | 推理途中不收。`batch_tts` 結束打一次 `/control?command=empty_cache` |
| `GPT_SOVITS_T2S_MAX_BATCH` | `8` | 只在強制開舊靜態圖時使用 |
| `TQDM_DISABLE` | `1` | 少印進度 |

批次小說參數（`local_tts/batch_tts.py` 預設，勿隨便改大）：

| 參數 | 值 |
|---|---|
| `max_text_length` | 1200 |
| `text_split_method` | auto（v2Pro 為 cut5，v5turbo 為 cut20） |
| `batch_size` | 64（v5 聲碼器內部再以 48 句一組） |
| `split_bucket` | True |
| `parallel_infer` | True |
| `fragment_interval` | 0 |
| `top_k` | 15 |

下面的速度列是 v2Pro、`cut5`、句間仍為 0.01 秒時量到的。0.01 秒不進模型，所以那些字/秒仍然描述 v2。現在小說不再補這段靜音。

**實測（cintia，1200 字，2026-11-14）：**

| 狀態 | 字/秒 | 備註 |
|---|---|---|
| Graph 開滿顯存 | ~20–40 | 修前 |
| Graph 關 + micro=1 | ~110–130 | 太碎 |
| Graph 關 + micro=0（整段） | ~150–165 但不穩 | 熱跑顯存頂高 |
| Graph 關 + micro=16 + empty=1 | ~230–240 | 2026-11-14，尚無逐步圖 |
| **逐步圖開 + 同上** | **292（1200 字）／359（5000 字）** | **2026-11-15 採用** |
| 逐步圖 + 途中不清記憶體 | 真人男 3 萬字約 310 字/秒；吃神者 4827 字約 304 字/秒 | 目前就是這組設定。文章不同，數字不同 |

---

## 5. 標準修復／重測劇本（給 AI 直接跑）

### 5.1 啟動（一定要 detach，不要卡 SSH）

```bash
cd /home/sky/code/GPT-SoVITS
mkdir -p /tmp/tts_bench
pkill -f api_v2.py; pkill -f local_tts.start_api; sleep 1

setsid env CUDAGraph=0 GPT_SOVITS_VITS_MICROBATCH=16 GPT_SOVITS_EMPTY_CACHE=0 \
  GPT_SOVITS_TIMING_LOG=/tmp/tts_bench/timing.log PYTHONUNBUFFERED=1 \
  ./local_tts/start_api.sh --role cintia --host 127.0.0.1 --port 9880 \
  >/tmp/tts_bench/api.log 2>&1 < /dev/null &
echo $!
```

另開連線：curl `/docs` = 200 就立刻測，不要傻等 log。

### 5.2 最小測速（熱跑為準）

對 `POST http://127.0.0.1:9880/tts` 送與 `batch_tts` 相同欄位的 JSON（`text_split_method=auto`、`batch_size=64`、`parallel_infer=true`、`fragment_interval=0`、`use_cuda_graph=false`）。  
同一 `1200` 字跑 **cold + warm×2**，看 `/tmp/tts_bench/timing.log`。

目標：**warm `total` ≤ 6s（1200 字）⇒ ≥ 200 字/秒**。

### 5.3 若又掉速：掃 micro 甜蜜點（8GB 專用）

固定 `CUDAGraph=0`，只掃：

```text
MICROBATCH ∈ {4, 8, 16, 32, 0}
EMPTY_CACHE ∈ {1, 16}
```

每次重啟 API → cold + warm×2 → 記錄 `cps` 與 `nvidia-smi`。  
選 **warm 平均最高且顯存峰值 < ~6.5GB** 的組合，寫回 `start_api.sh`。

### 5.4 結束必做

```bash
pkill -f api_v2.py; pkill -f local_tts.start_api
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 應回到 ~200MiB 左右
```

---

## 6. 相關代碼位置（改之前先定位）

| 檔案 | 職責 |
|---|---|
| `tools/acceleration.py` | CUDA Graph / Accel 是否啟用；≤10GB auto-off |
| `GPT_SoVITS/Accel/**` | AR CUDA Graph 實作（吃顯存） |
| `GPT_SoVITS/TTS_infer_pack/TTS.py` | 推論主路徑、`TTS_TIMING`、VITS microbatch、`use_cuda_graph` auto-off |
| `GPT_SoVITS/module/models.py` | `decode()` 的 `_tts_ge_cache` |
| `local_tts/start_api.sh` | **本機預設環境變數（最優配置入口）** |
| `local_tts/batch_tts.py` | 小說批次參數 |
| `api_v2.py` | HTTP API |

本地未提交的調校可能存在；用 `git diff -- GPT_SoVITS/TTS_infer_pack/TTS.py GPT_SoVITS/module/models.py tools/acceleration.py local_tts/start_api.sh` 查看。

---

## 7. 給後續 AI 的決策樹

```text
速度 < 100 字/秒？
  ├─ idle 顯存 > 6GB → 關 CUDAGraph，重測
  ├─ vits >> t2s 且顯存頂滿 → 降 MICROBATCH（試 16→8→4），EMPTY_CACHE=1
  ├─ vits 慢但顯存很空 → 升 MICROBATCH（試 8→16→32），檢查 ge_cache 是否在
  └─ environ 與預期不符 → 殺乾淨行程再啟，勿信任舊 API
速度 100–180？
  └─ 掃 MICROBATCH + EMPTY_CACHE，鎖定 warm 最高者
速度 ≥ 200？
  └─ 確認 log 有 `[T2S] step_graph` 而沒有 `step graph fallback`
     乾淨稿 1200 字約 290、5000 字約 360。有章節標題的小說約 300 也算正常
```

---

## 8. 禁止事項

1. **不要**在 8GB 上為了「加速」強開 `CUDAGraph=1` 還跑長篇小說（除非重測證明 warm ≥ 200 且顯存仍健康）。  
2. **不要**只改程式不查 `/proc/<api_v2_pid>/environ`。  
3. **不要**用單次冷跑宣稱「變慢／變快」。  
4. **不要**轉完小說還把 API 掛著佔顯存。  
5. **不要**在 Windows 掛載碟（`X:`）直接改檔；用 SSH + 遠端腳本。
6. **不要**因為 `CUDAGraph=0` 就把 `GPT_SOVITS_T2S_STEP_GRAPH` 也關掉。前者是會吃滿 8GB 的靜態圖，後者是算完就釋放的逐步圖。

---

## 9. 變更紀錄

| 日期 | 內容 |
|---|---|
| 2026-11-14 | 診斷 Accel/CUDA Graph 為主因；ge_cache + microbatch；鎖定 `CUDAGraph=0`、`MICROBATCH=16`、`EMPTY_CACHE=1`；熱身實測 ~230–240 字/秒 |
| 2026-11-15 | 加上逐步 CUDA Graph（`GPT_SOVITS_T2S_STEP_GRAPH`）。1200 字 292 字/秒，5000 字 359 字/秒。舊靜態圖維持關閉 |
| 2026-11-15 | 長句不再拖垮整桶：超過 20 字再切、未完成的句子單獨續跑。注音依長度分桶，並在 VITS 期間準備下一桶。吃神者 4827 字從約 106–192 字/秒回到 300 字/秒以上 |
| 2026-11-15 | 小說不再補句間靜音（`fragment_interval=0`）。v2Pro 維持 cut5，v5turbo 用 cut20，CFM 一次 24 句。吃神者 3000 字：v2 約 258 字/秒，v5 約 98 字/秒 |

---

## 10. 其他 GPU，以及怎麼調到最快

這份程式可以在其他 NVIDIA GPU 上跑。逐步圖不讀顯卡型號，只要求目前這份 PyTorch 能在該卡上跑 CUDA，並且 `scaled_dot_product_attention` 可用。已在 RTX 3060 Ti 8GB、WSL2、PyTorch 2.12.1+cu132、fp16 上量過。

UEFI 張量解鎖（`nv-uefi-unlock-playbook`）是另一件事。它只影響 RTX 20／30／40 上的 FP16、BF16、TF32 大矩陣，大約把張量算力加倍；FP32 幾乎不變。沒有解鎖時，同一套程式一樣能跑，只是那些矩陣較慢。解鎖不會自動讓自回歸迴圈變兩倍，因為 T2S 每一步的矩陣很小，時間多半在 kernel 發射。

### 10.1 兩個 Graph 開關

| 變數 | 做什麼 | 建議 |
|---|---|---|
| `CUDAGraph` | Accel 靜態圖，事先佔住整段 KV | **保持 0**。8GB 上開了會讓 VITS 掉到每秒幾十個字 |
| `GPT_SOVITS_T2S_STEP_GRAPH` | 每個 token 重放一次 Transformer，返回前釋放 KV | **保持 1** |

Log 出現 `[T2S] step_graph bsz=… steps=…` 表示這桶走逐步圖。出現 `step graph fallback` 表示這桶已改回普通 T2S，結果仍正確，只是這桶較慢。

兩種自動退回：

- `kv NNNNMB`：這桶的 KV 緩衝超過 2000MB，不分配。8GB 上不要放寬。16GB 以上若想讓超大 bucket 也走圖，改 `GPT_SoVITS/AR/models/t2s_model.py` 裡 `_batch_decode_step_graph` 的 `2000 * 1024 * 1024`。
- 有句子在 320 步內沒結束：已完成的句子留下，只剩一句時把上限放到 1024 繼續走逐步圖。多句同時沒結束才整桶退回普通 T2S。

### 10.2 換顯卡時的調整順序

先固定請求參數，再只動一個環境變數。每次用同一段文字，冷跑一次、熱跑兩次，以熱跑為準。看 log 的 `[TTS_TIMING]`。

請求側（`batch_tts` 已是這組，直接打 API 時要自己帶）：

| 參數 | 起步值 | 原因 |
|---|---|---|
| `text_split_method` | `auto` | v2Pro 落到 `cut5`。v2 用 `cut2` 會明顯變慢。v5turbo 落到 `cut20` |
| `batch_size` | `64` | 8 這種小批次會把 T2S 拆碎。v5 聲碼器內部以 48 句一組 |
| `split_bucket` | 開 | 長度相近的句子才放同一桶 |
| `parallel_infer` | 開 | 關掉大約只剩每秒幾十個字 |
| `fragment_interval` | `0` | 大於 0 會在每段後面補靜音。小說維持 0 |
| `top_k` | `15` | 與目前模型訓練設定一致 |
| `speed_factor` | `1.0` | 不是 1 會關掉分桶 |

環境側，依顯存改：

| 顯存 | `VITS_MICROBATCH` | `EMPTY_CACHE` | `T2S_STEP_GRAPH` | `CUDAGraph` |
|---|---|---|---|---|
| 6GB 或更小 | `4` 或 `8` | `1` | 先 `1`；若 capture 時 OOM 改 `0` | `0` |
| 8GB（本機） | `16` | `0`，跑完再清一次 | `1` | `0` |
| 12–16GB | 先 `16`，再試 `32` | `1`，長跑穩定可試 `0` | `1` | `0`，確認閒置顯存低於一半再考慮 `1` |
| 24GB 以上 | `32` 或 `0`（整段一次解） | `0` 或 `16` | `1` | 可以試 `1`，但先看 VITS 是否變慢 |

判斷方式：

- `vits` 變長、顯存頂到 90% 以上：把 `VITS_MICROBATCH` 減半。
- `vits` 很短、顯存還很空：把 `VITS_MICROBATCH` 加倍，或設 `0` 試整段。
- `t2s` 佔一半以上、而且沒有 `step_graph` 那行：逐步圖沒生效，檢查環境變數是不是 `0`，或是不是每桶都 fallback。
- 閒置顯存超過總量的 75%：多半是舊靜態圖還開著，設 `CUDAGraph=0` 後重啟 API。

本機 8GB 不要為了追數字把 `CUDAGraph` 設成 `1`。先前這樣做，熱身掉到約 39 字/秒。



## 11. 吃神者長文：為什麼會掉到 100 多字/秒，以及當時的做法

角色困芽、v2Pro、fp16、參考音已快取。同一份《吃神者》前 4827 字。

| 跑法 | 牆鐘 | 字/秒 | 原因 |
|---|---:|---:|---|
| 冷跑，參考音重算 | 約 8–9 秒 / 1000 字量級 | 約 106 | `ref` 本身就 2–4 秒，不能拿來跟熱跑比 |
| 熱跑，長句不切 | 25–21 秒 | 192–225 | 沒標點的章節標題讓整桶 T2S 超過 320 步，已完成的句子被丟掉重跑 |
| 熱跑，只修續跑 | 18.3 秒 | 264 | 整桶不再重跑，但長句仍把同桶短句拖到 200 步以後 |
| 熱跑，超過 20 字再切 | 16.0 秒 | 302 | T2S 6.96 秒，VITS 5.85 秒。這本書的峰值在這裡附近 |
| 熱跑，注音跟 VITS 重疊 | 15.8 秒 | 306 | 注音等待 0.28 秒；VITS 因為跟 G2PW 同一張卡，從 5.85 秒變成 7.52 秒 |
| 整理成正式路徑後再測 | 15.89 秒 | 304 | 音檔 1242 秒，與上一列同一段稿。檔案 `GPT-SoVITS-DATA/困芽/bench_v2_5000.wav` |

乾淨文本（cintia 5000 字）沒有這種章節標題，所以可以到 359 字/秒。那不是同一份稿。

做法，由外到內：

1. **當時是超過 20 字就再切。** 後來改成只切沒有標點、長過 120 字的段落（`split_for_fast_t2s`）。每個子句都再切一次會讓聲音重起。v2 的標點切分仍是 `cut5`。v5 的大約 20 字是另一個切法 `cut20`，見第 12 節。
2. **一桶裡只有長句還沒結束時，不要整桶重來。** `_batch_decode_step_graph` 回傳已完成的列。只剩一句就把 token 上限從 320 放到 1024，繼續走逐步圖。
3. **注音跟聲學重疊，不跟語意 token 重疊。** 分桶只用正規化後的字數，這一步很短。G2PW 也在 GPU 上，若在 T2S 擷取 CUDA graph 時同時跑，graph 會失效。所以下一桶注音是在這一桶 T2S 結束、VITS 開始時才啟動。
4. **環境維持鎖定值：** `CUDAGraph=0`、`GPT_SOVITS_T2S_STEP_GRAPH=1`、`GPT_SOVITS_VITS_MICROBATCH=16`、`GPT_SOVITS_EMPTY_CACHE=0`、`batch_size=64`、`split_bucket=True`。`pipeline_prefetch` 預設開；分段或串流模式會關掉。

程式入口：`text_segmentation_method.split_for_fast_t2s`、`Text2SemanticDecoder._batch_decode_step_graph`、`TTS.plan_length_buckets`、`_PhoneBucketPrefetcher`。

## 12. 句間不補靜音，v5 用大約 20 字

同一段《吃神者》前 3000 字，`fragment_interval=0`，參考音已熱身。2026-11-15 晚。

| 模型 | 切段 | batch | 生成 | 速度 | 音檔長度 |
|---|---|---:|---:|---:|---:|
| v2Pro | cut5 | 64 | 11.64 秒 | 257.7 字/秒 | 768 秒 |
| v5turbo | cut20 | 48 | 30.72 秒 | 97.6 字/秒 | 714 秒 |

v5turbo 在 batch 48 下改每段字數：10 字 72.1 字/秒，15 字 92.4，20 字 97.6，30 字 96.7。40 字在 batch 48 聲碼器爆顯存，降到 batch 32 是 89.0 字/秒。v5 因此維持 cut20。

`audio_postprocess` 只在 `fragment_interval > 0` 時，才在每一段後面接 `sampling_rate * fragment_interval` 個 0。上游預設 0.3 秒，用來讓分開合成的句子中間有一段換氣。API、`batch_tts` 和 `audio_postprocess` 的小說預設都是 0。`_set_prompt_semantic` 裡參考音後面的 0.3 秒只給 HuBERT 抽音色，不會寫進輸出。
