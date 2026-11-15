# GPT-SoVITS 各版本訓練方法

> 給之後要重訓的人。這裡寫的是這份程式實際走的路徑，來源是 `config.py`、`webui.py`、`GPT_SoVITS/s1_train.py`、`GPT_SoVITS/s2_train.py`、`GPT_SoVITS/s2_train_v3_lora.py`。
> 主機：`sky-3060ti-linux`。專案：`/home/sky/code/GPT-SoVITS`。Python：專案裡的 `.venv/bin/python`（訓練用這個，不要用 `local_tts/.venv`）。

WebUI 的版本選項目前只列出 `v2`、`v2Pro`、`v2ProPlus`、`v4`。`v1`、`v3`、`v5dev`、`v5turbo` 的訓練函式同一套，把版本字串設對即可。下面的命令就是 WebUI 按下按鈕時會跑的事。

實驗目錄固定是 `logs/<實驗名>/`。

---

## 1. 先準備同一份資料

標註檔每行一筆，用 `|` 分開：

```text
音檔絕對路徑或檔名|說話者|語種|文字
```

語種用 `ZH`、`EN`、`JA`、`KO`、`YUE`。音檔目錄留空時，程式用標註裡的絕對路徑；有填目錄時，只拿標註裡的檔名，再接到這個目錄。

cintia 現成的標註在：

`/home/sky/code/GPT-SoVITS-DATA/cintia/10_dataset/slicer.list`

40 段，約 4.5 分鐘。

然後在專案根目錄做三次前處理。三次都讀環境變數，GPU 用 `_CUDA_VISIBLE_DEVICES`。

### 1A 文字，得到 `2-name2text.txt`

```bash
export inp_text="標註.list"
export inp_wav_dir=""
export exp_name="實驗名"
export opt_dir="logs/實驗名"
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export i_part=0
export all_parts=1
export is_half=True
export _CUDA_VISIBLE_DEVICES=0
.venv/bin/python -s GPT_SoVITS/prepare_datasets/1-get-text.py
mv logs/實驗名/2-name2text-0.txt logs/實驗名/2-name2text.txt
```

這步跑中文 BERT，抽出音素。

### 1B 聲音特徵，得到 `4-cnhubert/` 和 `5-wav32k/`

```bash
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
.venv/bin/python -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
```

`v2Pro` 和 `v2ProPlus` 還要多跑一次說話人嵌入：

```bash
.venv/bin/python -s GPT_SoVITS/prepare_datasets/2-get-sv.py
```

其他版本不要跑這支。

### 1C 語意 token，得到 `6-name2semantic.tsv`

```bash
export pretrained_s2G="該版本的 SoVITS-G 底模"
export s2config_path="GPT_SoVITS/configs/s2.json"   # v2Pro 系改用下面表裡的 json
.venv/bin/python -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
```

腳本會寫出 `6-name2semantic-0.tsv`。在第一行加上 `item_name\tsemantic_audio`，存成 `6-name2semantic.tsv`。

`3-get-semantic.py` 不看版本名字，只看 SoVITS-G 檔案大小：大於 700MB 就當成 `v3` 那類（v3、v4、v5dev、v5turbo 都落在這裡），只用裡面的量化器抽出語意 token。SoVITS 真正的版本字串是在下一步訓練設定裡寫的。

---

## 2. 兩個模型分開訓

每個版本都是兩個權重：

| 模型 | 在做什麼 | 腳本 |
|---|---|---|
| GPT（s1） | 文字一路自回歸出語意 token | `GPT_SoVITS/s1_train.py` |
| SoVITS（s2） | 語意 token 變成聲音 | v1／v2 系：`s2_train.py`。v3／v4／v5：`s2_train_v3_lora.py` |

GPT 的設定檔：`v1` 用 `GPT_SoVITS/configs/s1longer.yaml`，其餘版本用 `GPT_SoVITS/configs/s1longer-v2.yaml`。WebUI 預設 **15** epoch、每 **5** epoch 存一次、DPO 關。batch 用顯存 GB 數除以 2，8GB 就是 4。

SoVITS 的設定檔：`v2Pro` 用 `configs/s2v2Pro.json`，`v2ProPlus` 用 `configs/s2v2ProPlus.json`，其餘用 `configs/s2.json`。訓練前把 `model.version` 改成該版本。

8GB 上一次只跑一個訓練。GPT 和 SoVITS 不要同時開。

---

## 3. 每個版本差在哪

底模路徑都相對專案根目錄。

| 版本 | GPT 底模 | SoVITS-G 底模 | SoVITS-D | SoVITS 怎麼訓 | 8GB 上 SoVITS 預設 |
|---|---|---|---|---|---|
| v1 | `pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt` | `pretrained_models/s2G488k.pth` | 同名把 `s2G` 換成 `s2D` | 全模型，`s2_train.py` | batch 4，8 epoch，每 4 epoch 存 |
| v2 | `pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` | `pretrained_models/gsv-v2final-pretrained/s2G2333k.pth` | `s2D2333k.pth` | 全模型，`s2_train.py` | 同上 |
| v2Pro | `pretrained_models/s1v3.ckpt` | `pretrained_models/v2Pro/s2Gv2Pro.pth` | `s2Dv2Pro.pth` | 全模型，外加 1B 的 SV | 同上。設定檔用 `s2v2Pro.json` |
| v2ProPlus | `pretrained_models/s1v3.ckpt` | `pretrained_models/v2Pro/s2Gv2ProPlus.pth` | `s2Dv2ProPlus.pth` | 全模型，外加 SV | 同上。設定檔用 `s2v2ProPlus.json` |
| v3 | `pretrained_models/s1v3.ckpt` | `pretrained_models/s2Gv3.pth` | 沒有 | 只對 CFM 做 LoRA | batch 1，2 epoch，每 1 epoch 存，LoRA 秩 32 |
| v4 | `pretrained_models/s1v3.ckpt` | `pretrained_models/gsv-v4-pretrained/s2Gv4.pth` | 沒有 | 只對 CFM 做 LoRA | 同上 |
| v5dev | `pretrained_models/s1v3.ckpt` | `pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth` | 沒有 | 只對 CFM 做 LoRA | 同上 |
| v5turbo | `pretrained_models/s1v3.ckpt` | `pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth` | 沒有 | 只對 CFM 做 LoRA | 同上 |

v3、v4、v5dev、v5turbo 的 GPT 是同一顆 `s1v3.ckpt`，所以換的是 SoVITS 底模和 `model.version`。v5dev 和 v5turbo 的差別就是底模：`s2Gv5dev.pth` 對 32 步，`s2Gv5turbo.pth` 對 4 步。要 turbo 就不要拿 dev 的底模來訓。

v3 之後的 LoRA 只打在 CFM 的 `to_q`、`to_k`、`to_v`、`to_out.0`。秩的選項是 16、32、64、128，預設 32。程式自己的註解寫著這類模型 epoch 不要拉高，上限 16，預設 2。

已經訓好的 v2Pro 權重（例如現在的 `cintia-e15.ckpt`、`cintia_e10_s210.pth`）不能拿來當 v5 的起點。v5 必須從上表的底模開始。

### 權重會放哪

| 版本 | GPT | SoVITS |
|---|---|---|
| v1 | `GPT_weights/` | `SoVITS_weights/` |
| v2 | `GPT_weights_v2/` | `SoVITS_weights_v2/` |
| v3 | `GPT_weights_v3/` | `SoVITS_weights_v3/` |
| v4 | `GPT_weights_v4/` | `SoVITS_weights_v4/` |
| v5dev | `GPT_weights_v5dev/` | `SoVITS_weights_v5dev/` |
| v5turbo | `GPT_weights_v5turbo/` | `SoVITS_weights_v5turbo/` |
| v2Pro | `GPT_weights_v2Pro/` | `SoVITS_weights_v2Pro/` |
| v2ProPlus | `GPT_weights_v2ProPlus/` | `SoVITS_weights_v2ProPlus/` |

SoVITS 訓練過程的 checkpoint 在 `logs/<實驗名>/logs_s2_<版本>/`。v3 之後是 `logs_s2_<版本>_lora_<秩>/`。GPT 過程在 `logs/<實驗名>/logs_s1_<版本>/`。

### SoVITS 訓練命令

v1、v2、v2Pro、v2ProPlus：

```bash
.venv/bin/python -s GPT_SoVITS/s2_train.py --config logs/實驗名/tmp_s2.json
```

v3、v4、v5dev、v5turbo：

```bash
.venv/bin/python -s GPT_SoVITS/s2_train_v3_lora.py --config logs/實驗名/tmp_s2.json
```

`tmp_s2.json` 從對應的 `s2*.json` 抄出來，再寫入這些欄位：

```text
train.batch_size
train.epochs
train.text_low_lr_rate          # v1/v2 系用，預設 0.4。v3 之後介面會藏掉
train.pretrained_s2G
train.pretrained_s2D            # v3 之後留空字串
train.if_save_latest            # 預設 true
train.if_save_every_weights     # 預設 true
train.save_every_epoch
train.gpu_numbers               # "0"
train.grad_ckpt                 # 只有 v3 之後有用。介面預設 false
train.lora_rank                 # 只有 v3 之後有用，預設 "32"
model.version                   # 必須是 v5turbo 這種版本字串
data.exp_dir / s2_ckpt_dir      # logs/實驗名
save_weight_dir                 # 上表的 SoVITS 目錄
name                            # 實驗名
version                         # 同上
```

8GB 訓 v3 之後，batch 用 1。這次 cintia 另外把 `grad_ckpt` 打開，用時間換顯存。顯存更大、想照介面原設定時，保持 `false`。

### GPT 訓練命令

```bash
export hz=25hz
.venv/bin/python -s GPT_SoVITS/s1_train.py --config_file logs/實驗名/tmp_s1.yaml
```

`tmp_s1.yaml` 從 `s1longer.yaml` 或 `s1longer-v2.yaml` 抄出，寫入：

```text
train.batch_size
train.epochs                         # 預設 15
train.precision                      # 半精度用 16-mixed
train.save_every_n_epoch             # 預設 5
train.if_save_every_weights          # true
train.if_save_latest                 # true
train.if_dpo                         # false
train.half_weights_save_dir          # 上表的 GPT 目錄
train.exp_name
pretrained_s1                        # 該版本的 GPT 底模
train_semantic_path                  # logs/實驗名/6-name2semantic.tsv
train_phoneme_path                   # logs/實驗名/2-name2text.txt
output_dir                           # logs/實驗名/logs_s1_<版本>
```

---

## 4. 這次 cintia 的 v5turbo

實驗名 `cintia-v5turbo`。資料就是上面那份 40 段標註。

- GPT：從 `s1v3.ckpt` 訓 **15** epoch，每 5 epoch 存一次，batch **2**（8GB 上比介面預設的 4 再小一檔）
- SoVITS：從 `s2Gv5turbo.pth` 做 LoRA，秩 **32**，**2** epoch，batch **1**，`grad_ckpt` 開
- 前處理的語意 token 用 `s2Gv5turbo.pth` 抽

完成後權重在：

- `GPT_weights_v5turbo/cintia-v5turbo-e15.ckpt`
- `SoVITS_weights_v5turbo/cintia-v5turbo_e2_s*_l32.pth`

SoVITS 那個檔是 LoRA，推論時要跟 `s2Gv5turbo.pth` 一起載。推理程式會從權重頭認出版本是 `v5turbo`。

日誌：`logs/cintia-v5turbo/train.log`。背景啟動，SSH 斷線不會停：

```bash
setsid /home/sky/code/GPT-SoVITS/TEMP/run_cintia_v5turbo.sh \
  </dev/null >/home/sky/code/GPT-SoVITS/logs/cintia-v5turbo/wrapper.log 2>&1 &
```

實測（2026-11-15，同一段 5000 字，cintia v5turbo）：

| 聲學怎麼跑 | 牆鐘 | 字/秒 | 穩態一段的 vits |
|---|---|---|---|
| 一句一句 | 112 秒 | 43 | 約 20 秒 |
| 沿時間接句，一次 CFM（v2 的 microbatch 16） | 70 秒 | 69 | 約 9 秒 |

接句上限是 v2 同一個 `GPT_SOVITS_VITS_MICROBATCH`（預設 16），另外受 v5 一個 chunk 500 幀限制。超過一個 chunk 的單句仍逐句算。
