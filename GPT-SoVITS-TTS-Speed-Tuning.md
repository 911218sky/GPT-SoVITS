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

## 0. 一句話結論（2026-11-14 實測）

**不是環境壞掉。** 慢的主因是 `bf8c5a3` 併入 **Accel / CUDA Graph** 後，在 **8GB** 卡上把顯存吃滿，VITS 被擠到 thrash。

**本機目前最快預設（熱身約 230–240 字/秒，1200 字約 5 秒）：**

```bash
export CUDAGraph=0
export GPT_SOVITS_VITS_MICROBATCH=16
export GPT_SOVITS_EMPTY_CACHE=1
# 其餘跟小說批次一樣：
# cut5, max_text_length=1200, batch_size=64, split_bucket=True, parallel_infer=True
# fragment_interval=0.01, top_k=15
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
| 本次修復後 | Graph 關 + VITS micro=16 + 每請求 empty_cache | 熱身 **~230–240 字/秒** |

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

### D. 每請求 `empty_cache` 的取捨

- **不做：** 熱身後顯存碎片累積，下一次 `vits` 變慢（曾見整段 concat 後 7.8GB，熱跑變差）。  
- **每請求做（`EMPTY_CACHE=1`）：** 略有同步成本，但本機可穩定 ~230+。  
- **`16`：** 較猛，長跑可能又慢回來。本機預設改回 `1`。

### E. 其他（次要）

- `load_audio_gpu` 曾誤用未匯入的 `torchaudio` → 改走 `load_audio()`（`TTS.py`）。  
- T2S Accel 因 `batch_size` 變化反覆 rebuild CUDA Graph → 已改成容量制（`GPT_SOVITS_T2S_MAX_BATCH`）。  
- **不要**用 `start_api.sh` 的前景 `wait` 卡死 SSH；測速請 `setsid`／另開 SSH，API ready 就立刻打 `/tts`。

---

## 4. 本機最優配置（鎖定值）

寫在 `local_tts/start_api.sh`：

| 變數 | 預設 | 說明 |
|---|---|---|
| `CUDAGraph` | `0` | 8GB 必須關 |
| `GPT_SOVITS_VITS_MICROBATCH` | `16` | VITS 每次最多合 16 句 |
| `GPT_SOVITS_EMPTY_CACHE` | `1` | 每請求清 CUDA 快取 |
| `GPT_SOVITS_T2S_MAX_BATCH` | `8` | 若強制開 Graph 時的 batch 上限 |
| `TQDM_DISABLE` | `1` | 少印進度 |

批次小說參數（`local_tts/batch_tts.py` 預設，勿隨便改大）：

| 參數 | 值 |
|---|---|
| `max_text_length` | 1200 |
| `text_split_method` | cut5 |
| `batch_size` | 64 |
| `split_bucket` | True |
| `parallel_infer` | True |
| `fragment_interval` | 0.01 |
| `top_k` | 15 |

**實測（cintia，1200 字，2026-11-14）：**

| 狀態 | 字/秒 | 備註 |
|---|---|---|
| Graph 開滿顯存 | ~20–40 | 修前 |
| Graph 關 + micro=1 | ~110–130 | 太碎 |
| Graph 關 + micro=0（整段） | ~150–165 但不穩 | 熱跑顯存頂高 |
| **Graph 關 + micro=16 + empty=1** | **~230–240** | **採用** |

---

## 5. 標準修復／重測劇本（給 AI 直接跑）

### 5.1 啟動（一定要 detach，不要卡 SSH）

```bash
cd /home/sky/code/GPT-SoVITS
mkdir -p /tmp/tts_bench
pkill -f api_v2.py; pkill -f local_tts.start_api; sleep 1

setsid env CUDAGraph=0 GPT_SOVITS_VITS_MICROBATCH=16 GPT_SOVITS_EMPTY_CACHE=1 \
  GPT_SOVITS_TIMING_LOG=/tmp/tts_bench/timing.log PYTHONUNBUFFERED=1 \
  ./local_tts/start_api.sh --role cintia --host 127.0.0.1 --port 9880 \
  >/tmp/tts_bench/api.log 2>&1 < /dev/null &
echo $!
```

另開連線：curl `/docs` = 200 就立刻測，不要傻等 log。

### 5.2 最小測速（熱跑為準）

對 `POST http://127.0.0.1:9880/tts` 送與 `batch_tts` 相同欄位的 JSON（`cut5`、`batch_size=64`、`parallel_infer=true`、`fragment_interval=0.01`、`use_cuda_graph=false`）。  
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
  └─ 寫進 start_api.sh 預設；轉完小說關 API 放顯存
```

---

## 8. 禁止事項

1. **不要**在 8GB 上為了「加速」強開 `CUDAGraph=1` 還跑長篇小說（除非重測證明 warm ≥ 200 且顯存仍健康）。  
2. **不要**只改程式不查 `/proc/<api_v2_pid>/environ`。  
3. **不要**用單次冷跑宣稱「變慢／變快」。  
4. **不要**轉完小說還把 API 掛著佔顯存。  
5. **不要**在 Windows 掛載碟（`X:`）直接改檔；用 SSH + 遠端腳本。

---

## 9. 變更紀錄

| 日期 | 內容 |
|---|---|
| 2026-11-14 | 診斷 Accel/CUDA Graph 為主因；ge_cache + microbatch；鎖定 `CUDAGraph=0`、`MICROBATCH=16`、`EMPTY_CACHE=1`；熱身實測 ~230–240 字/秒 |

