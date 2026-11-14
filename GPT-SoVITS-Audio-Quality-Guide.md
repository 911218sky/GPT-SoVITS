# GPT-SoVITS Audio Quality Guide（給 AI／人類）

> **目的：** 讓後續 AI 能判斷「這份語音適不適合訓練／推論」，以及**怎麼把髒音清到可訓**。  
> **主流程：** [`GPT-SoVITS-AI-Workflow.md`](./GPT-SoVITS-AI-Workflow.md)  
> **工作流強制：** Workflow **§0.2** — 做資料集／訓練前必讀本檔並過閘門；未過禁止開訓。  
> **顯存：** 一律遵守 Workflow **§8**（一階段一進程；8GB 上 UVR／ASR／train 之間必須釋放 VRAM）。

本文件不重複「怎麼開 API 轉小說」；只談**聲音品質的好壞標準與優化手段**。

---

## 0. 一句話標準

**好資料：** 靜音段接近黑、諧波清楚、沒有持續高頻霧／BGM 殘留，模型才學得到乾淨聲線。  
**壞資料：** 頻譜全程紫霧、句間不黑、中高頻一直有粒狀雜訊——再怎麼多訓，輸出也會「平、髒、有底噪」。

對標經驗：`真人男`（乾淨口播）vs `cintia` 初代（迪士尼 Vlog + BGM，僅 HP3→DeEcho）。

| 對照 | 好（接近真人男 ref） | 壞（cintia 初代／未清乾淨） |
|------|----------------------|-----------------------------|
| 句間靜音 | 接近純黑（~−80 dB） | 紫霧／粒狀，整段抬高噪底 |
| 低頻諧波 0–2 kHz | 水平條紋清晰、條間暗 | 糊、垂直拉絲、和伴奏黏在一起 |
| 6–12 kHz | 只有擦音／齒音短暫亮 | **持續**霧狀／音樂殘留 |
| 聽感 | 乾、近麥、無底噪 | 空、金屬感、背景嗡嗡或歌 |

**重要天花板：** 來源若是「有 BGM 的手機 Vlog」，再強前處理也難 100% 等於棚內乾聲；目標是**逼近乾淨口播的噪底與殘留**，不是幻想變成另一個錄音棚。

---

## 1. AI 必做：先量再訓（不要只靠耳朵猜）

在切完訓練片段後、開訓前，對 **整段人聲** 與 **每段 clip** 算指標（可用 `librosa`）。

### 1.1 建議指標

對 32 kHz mono：

1. **`quiet_high`**：靜音幀（RMS ≤ p25）上，6–12 kHz 功率均值  
2. **`bleed`**：靜音幀 mid(200–4k) 功率 ÷ 語音幀 mid 功率  
3. **`rel_qh`**：`quiet_high ÷ 語音幀 mid 功率`（**比絕對 quiet_high 穩**，不受音量影響）  
4. 輔佐：時長 2.8–10 s、CJK 字數 ≥ 4、SNR(p75/p10) 粗估

### 1.2 經驗門檻（3060 Ti 實戰，cintia 管線）

| 指標 | 可接受（可進 10_dataset） | 優秀 | 應丟棄／重清 |
|------|---------------------------|------|--------------|
| `rel_qh`（clip） | ≤ **0.003** | ≤ **0.001** | > 0.008 或排名最差一批 |
| `bleed` | ≤ **0.04–0.05** | ≤ **0.02** | > 0.06 |
| 時長 | 2.8–10 s（均 ~4.5–7 s） | 同左 | <2.5 或 >10（對齊會炸） |
| 語言 | 目標語為主（中文角色要有足夠 CJK） | — | 英文歌詞／BGM ASR 垃圾段 |

**集合健康度：** 保留集 `rel_qh` 均值應明顯低於「只做 HP3」的舊集。  
cintia 實測大致：`HP3 舊集 ~0.010` → `bs_roformer+HP5+spectral ~0.0016` → `再 deepclean ~0.0002`。

整段長音檔的絕對 `quiet_high` 易被 soft-gate 靜音「美化」，**篩選請以 clip 的 `rel_qh`／`bleed` 為準**。

### 1.3 頻譜圖怎麼讀（必會）

畫 STFT（建議 `n_fft=1024–2048`，y 軸 0–12 kHz，magma）：

- **好：** 句間接近黑；0–3 kHz 諧波像清晰橫條；高頻只在 s/sh 短暫亮。  
- **壞：** 全程淡紫霧；高頻垂直粒；句間不黑；諧波被垂直「拉絲」糊掉（常見 UVR 相位／殘響）。  
- **TTS 輸出：** 允許比 ref 略髒（合成本來有假頻），但若比**同角色 ref** 明顯更霧，多半是訓練資料仍髒或權重沒載對。

推論後也應抽 20 s 對比：舊模型 TTS vs 新模型 TTS vs `真人男` ref。

---

## 2. 壞音常見類型 → 對策

| 症狀（聽／看） | 常見原因 | 優先對策 |
|----------------|----------|----------|
| 唱歌／伴奏殘留、中頻 peakiness 高 | BGM 分離不夠 | **bs_roformer** → 必要時 **HP5** |
| 空曠、拖尾、金屬感 | 殘響／echo | **VR-DeEchoDeReverb** 或 DeEchoAggressive；立體混響可再 MDX（慢，8GB 預設跳過） |
| 高頻嘶嘶、紫霧 | 手機底噪／UVR 殘渣 | `tools/spectral_clean.py`（soft gate + highband hush）；可再 9 kHz 低通 |
| 句間不黑但說話時還行 | 持續底噪 | 加強 gate；丟 `rel_qh` 差的 clip |
| ASR 出英文歌詞 | 開頭／過場 BGM | CJK 門檻過濾；不要硬留 |
| TTS 平、髒但 ref 已乾淨 | 權重錯／仍用舊 dirty 權重 | 確認 `/set_sovits_weights` 為角色 `.pth`；重訓後只留最終權重 |
| 顯存爆、極慢 | 多模型同進程 | **一階段一進程**；停 API；跳過 MDX onnx |

---

## 3. 推薦前處理管線（有 BGM／Vlog）

現成專案內工具即可，不必另裝巨型套件（DeepFilterNet／FRCRN 可選，缺依賴就跳過）。

### 3.1 預設「乾淨優先」順序（8GB 驗證過）

每一步結束進程，確認 `nvidia-smi` 回到 ~300 MiB 再下一步：

1. **bs_roformer**（`model_bs_roformer_ep_317_sdr_12.9755.ckpt`）— 人聲分離最強常用項  
2. **VR-DeEchoDeReverb** — 去殘響／delay（比只 DeEchoAggressive 更適合髒場）  
3. **HP5_only_main_vocal** — 再壓殘留伴奏（可能稍削弱人聲，BGM 重時值得）  
4. **`tools/spectral_clean.py`** — soft gate + 高頻壓制；必要時再加 **~9 kHz 低通** + 更強 gate（deepclean）  
5. **`tools/slice_for_training.py`** — 靜音切 + 最安靜點遞迴切到 ≤10 s  
6. **WhisperX medium**（本地 `faster-whisper-medium`）— 禁止預設跑會下載 large-v3／FunASR 的腳本  
7. **排序篩選** — 依 `rel_qh`、`bleed`、CJK、時長；寧少勿髒（~40 段乾淨 > 65 段帶霧）  
8. 選 **最乾淨 3.8–6.5 s** 當 `ref.wav`，`prompt_text` 必須與 ref 文案一致  

CLI 輔助：

- `tools/uvr_one_stage.py` — 單模型單檔，跑完即退出（釋放 VRAM）  
- `tools/spectral_clean.py`  
- `tools/filter_training_clips.py` — 門檻過濾（絕對 quiet_high 偏嚴時改用相對 `rel_qh` 排序）

### 3.2 何時用較短管線

| 來源 | 建議 |
|------|------|
| 乾淨口播／幾乎無 BGM（如真人男類） | 可跳分離；最多輕 spectral_clean；重點在切片與 ASR |
| 輕 BGM 說話 | HP3 → DeEchoAggressive → spectral_clean |
| 重 BGM／樂園／MV 感 Vlog | **必須** §3.1 全套；勿只 HP3 |

### 3.3 官方口頭「最乾淨」與本機取捨

上游常寫：**MDX-Net Dereverb → DeEcho-Aggressive**。  
本機 8GB：**MDX onnx 極慢且易拖垮系統** → **預設跳過**；用 **bs_roformer + DeEchoDeReverb + HP5 + spectral** 取代「乾淨度／速度」平衡。

---

## 4. 優化決策樹（AI 照著走）

```
拿到音檔／影片
  ├─ 有明顯 BGM？ ─是─→ §3.1 全套分離+清理
  │                    └─ 量整段 quiet_high／bleed；仍髒 → 加 HP5／加強 spectral／丟段
  └─ 否 ─→ 輕清理或跳過 UVR
切 3–10s → WhisperX → 算每段 rel_qh／bleed
  ├─ 保留集 rel_qh 均值仍高？ → 回到分離／清理，不要硬訓
  └─ OK → 選乾淨 ref → feature → v2Pro train (batch_size=4 on 8GB)
訓完 → batch_tts 抽測 → 畫 TTS 頻譜 vs ref vs 真人男
  ├─ TTS 仍明顯霧且 ref 也霧 → 資料問題，繼續清
  ├─ ref 乾、TTS 髒 → 查權重是否載入角色 SoVITS；查 finish_audio
  └─ 可接受 → 註冊角色、更新文件、只留最終 .pth/.ckpt
```

**原則：資料品質 > 多幾個 epoch。** 髒資料多訓只會把霧學死。

---

## 5. 訓練／推論側注意（品質相關）

- 版本預設 **v2Pro**；8GB **`batch_size=4`**。  
- `mkdir -p logs/<exp>/logs_s2_v2Pro` 再 `s2_train`。  
- 只保留最終權重；中間 G_/D_/lightning ckpt 可刪。  
- 小說：`start_api.sh --role` → `batch_tts.sh` → `finish_audio.sh`（後者有清靜音／調速，但**救不了**訓練時學進去的底噪）。  
- API 日誌若出現 `fall back to default vits_weights_path`，確認後續有成功 `/set_sovits_weights` 到角色檔。

---

## 6. 好／壞速查清單（開訓前勾選）

- [ ] 已畫至少 1 張長人聲 + 3–5 張 clip 頻譜，句間接近黑  
- [ ] 保留集 `rel_qh` 均值達 §1.2；最差段已丟掉  
- [ ] 無大段英文歌／空轉寫  
- [ ] 時長落在 3–10 s；總時長通常 ≥3–5 min 較穩（極乾淨可略少，但別低於 ~2 min 還硬上高 epoch）  
- [ ] `ref.wav` 是保留集裡最乾淨之一，文案一致  
- [ ] UVR／ASR／train 分進程，VRAM 有釋放  
- [ ] 訓後 TTS 頻譜與舊版／真人男對照過，並把路徑告訴使用者  

---

## 7. 實戰數字備忘（cintia，2026-11）

| 階段 | 重點 | 相對 `rel_qh`（集均值，約） |
|------|------|------------------------------|
| 舊 HP3→DeEcho 集 | 高頻靜音噪大 | ~0.010 |
| bs_roformer→DeReverb→HP5→spectral | 對齊乾淨口播量級 | ~0.0016 |
| + 更強 gate／9 kHz 低通 deepclean | 再降 | ~0.0002 |
| 舊 TTS `quiet_high` | — | ~0.15 |
| deepclean 後 TTS `quiet_high` | — | ~0.026（優於該次 male_ref ~0.029） |

工具與案例細節亦見 Workflow **§7.1 / §8 / §9**。

---

## 8. 給 AI 的硬規則

1. **先量頻譜／`rel_qh`，再決定要不要訓。**  
2. **寧可 40 段乾淨，不要 65 段帶霧。**  
3. **有重 BGM 禁止只跑 HP3 就宣佈完成。**  
4. **不要用「再多訓幾個 epoch」掩蓋髒資料。**  
5. **優化順序：分離 → 去殘響 → 去殘伴奏 → 頻譜閘門 → 篩選；最後才訓練。**  
6. **改完資料必須重跑 feature + 重訓，並用 TTS+頻譜驗收。**  
7. **8GB：一階段一進程；卡住先釋放顯存。**
