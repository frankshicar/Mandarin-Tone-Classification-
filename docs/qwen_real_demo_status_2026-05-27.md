# 2026-05-27 狀態報告：Local Qwen3-TTS + 華語聽能 real demo

## 1. 這次工作的目標

目標是把現有 MVP pipeline 補成可實跑的 real-demo 版本，並保持既有 Stage 1 / 2 / 3 架構不變：

1. 用本地 `Qwen3-TTS` 取代人工錄製刺激音。
2. 對 TTS 產生的刺激音跑自動 QC：
   - 本地 ASR 驗證內容
   - tone model 驗證聲調
   - duration / silence / speech rate / loudness 規則檢查
3. 對患者複誦音檔跑：
   - 本地 Whisper ASR
   - tone inference
   - Stage 2 結構化評分
   - Stage 3 病人報告與復健建議

## 2. 目前已完成的部分

### 2.1 Stage 2 / Stage 3：患者回應分析已能跑通

這一側已經有完整產物，且結果可重現。重點 artifact：

- 題庫：[data/hearing_real_demo/item_bank_real.csv](../data/hearing_real_demo/item_bank_real.csv)
- 患者 seed manifest：[data/hearing_real_demo/patient_responses_seed.csv](../data/hearing_real_demo/patient_responses_seed.csv)
- 患者 ASR 輸出：[output/hearing_real_demo/patient_responses_asr.csv](../output/hearing_real_demo/patient_responses_asr.csv)
- 患者 tone 輸出：[output/hearing_real_demo/patient_responses_scored_inputs.csv](../output/hearing_real_demo/patient_responses_scored_inputs.csv)
- Stage 2 結果：[output/hearing_real_demo/structured_score.json](../output/hearing_real_demo/structured_score.json)
- 題目層級結果：[output/hearing_real_demo/item_level_results.csv](../output/hearing_real_demo/item_level_results.csv)
- 複核清單：[output/hearing_real_demo/human_review_items.csv](../output/hearing_real_demo/human_review_items.csv)
- 混淆摘要：[output/hearing_real_demo/confusion_summary.json](../output/hearing_real_demo/confusion_summary.json)
- 病人報告：[output/hearing_real_demo/patient_report.md](../output/hearing_real_demo/patient_report.md)
- 復健建議：[output/hearing_real_demo/rehab_plan.json](../output/hearing_real_demo/rehab_plan.json)

截至 `2026-05-27T06:02:56+00:00` 的 Stage 2 指標：

- `item_accuracy = 0.50`
- `keyword_accuracy = 0.50`
- `syllable_accuracy = 0.75`
- `initial_accuracy = 0.9375`
- `final_accuracy = 0.75`
- `tone_accuracy = 0.75`
- `review_required_items = 2`
- `asr_readable_rate = 1.0`

tone inference 摘要：

- 檔案：[output/hearing_real_demo/patient_tone_summary.json](../output/hearing_real_demo/patient_tone_summary.json)
- `device = cpu`
- `rows_total = 8`
- `predicted_rows = 8`
- `mean_tone_confidence = 0.8626`

### 2.2 Stage 1：local Qwen3-TTS 的技術串接已完成

以下腳本已可串成「生成 -> ASR -> tone -> QC」流程：

- [scripts/demo/generate_qwen_tts_candidates.py](../scripts/demo/generate_qwen_tts_candidates.py)
- [scripts/demo/run_local_asr.py](../scripts/demo/run_local_asr.py)
- [scripts/demo/run_tone_inference.py](../scripts/demo/run_tone_inference.py)
- [scripts/demo/run_tts_qc.py](../scripts/demo/run_tts_qc.py)

目前已跑過的 Qwen real-demo artifacts：

- `probe`：
  - [output/hearing_real_demo/tts_probe_qc_summary.json](../output/hearing_real_demo/tts_probe_qc_summary.json)
  - [output/hearing_real_demo/tts_probe_aiden_qc_summary.json](../output/hearing_real_demo/tts_probe_aiden_qc_summary.json)
  - [output/hearing_real_demo/tts_probe_v2_qc_summary.json](../output/hearing_real_demo/tts_probe_v2_qc_summary.json)
- `aiden` 8 題 batch：
  - [output/hearing_real_demo/tts_qwen_aiden_qc_summary.json](../output/hearing_real_demo/tts_qwen_aiden_qc_summary.json)
  - [output/hearing_real_demo/tts_qwen_aiden_qc_report.csv](../output/hearing_real_demo/tts_qwen_aiden_qc_report.csv)
  - [output/hearing_real_demo/tts_qwen_aiden_tone_summary.json](../output/hearing_real_demo/tts_qwen_aiden_tone_summary.json)
- `serena` 8 題 batch：
  - [output/hearing_real_demo/tts_qwen_serena_qc_summary.json](../output/hearing_real_demo/tts_qwen_serena_qc_summary.json)
  - [output/hearing_real_demo/tts_qwen_serena_qc_report.csv](../output/hearing_real_demo/tts_qwen_serena_qc_report.csv)
  - [output/hearing_real_demo/tts_qwen_serena_tone_summary.json](../output/hearing_real_demo/tts_qwen_serena_tone_summary.json)

### 2.3 已確認的 Qwen 行為

這些是這次工作中已驗證、之後接手時可直接沿用的結論：

- `qwen_tts` 套件可正常 import。
- 目前安裝版本暴露的是 `Qwen3TTSModel` / `Qwen3TTSTokenizer`，不是先前以為的標準 HF 介面。
- 本地合成已成功產生非空 WAV，不是「模型載不動」的問題。
- `generate_qwen_tts_candidates.py` 已支援直接指定 HF cache snapshot path。
- CPU fallback 已修過，`mem_get_info()` 失敗時不會又跳回 CUDA。
- `max_new_tokens=48` 比先前大 token 上限穩定很多。
- `0.6B custom voice` 對 instruction 幾乎沒有實際控制力，主要有效控制桿是 speaker。
- `dylan` 很不穩定，不應作為預設 speaker。

已知本地模型位置：

- `/home/sbplab/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5`

## 3. 目前真正卡住的地方

### 3.1 主 blocker：Qwen 生成的刺激音尚未穩定通過 Stage 1 QC

這是目前唯一沒有關掉的關鍵問題。

#### `aiden` 結果

- 摘要：[output/hearing_real_demo/tts_qwen_aiden_qc_summary.json](../output/hearing_real_demo/tts_qwen_aiden_qc_summary.json)
- 8 / 8 `FAIL`
- `approved_count = 0`

主要失敗原因：

- `duration_too_long = 8`
- `tone_sequence_mismatch = 7`
- `low_asr_confidence = 6`
- `speech_rate_too_slow = 5`

解讀：

- `aiden` 常把短雙音節詞唸得太長、太慢。
- 某些項目即使 ASR 內容正確，tone model 仍判成錯誤聲調。
- 有些檔案接近拖成長段語音，這會直接把語速和 duration 一起打爆。

#### `serena` 結果

- 摘要：[output/hearing_real_demo/tts_qwen_serena_qc_summary.json](../output/hearing_real_demo/tts_qwen_serena_qc_summary.json)
- 7 `FAIL`, 1 `PASS`
- `approved_count = 1`
- 通過項目只有：
  - [output/hearing_real_demo/tts_qwen_serena_approved.csv](../output/hearing_real_demo/tts_qwen_serena_approved.csv) 內的 `SSB04070450 / 波浪`

主要失敗原因：

- `tone_sequence_mismatch = 7`
- `silence_padding_needs_review = 6`
- `low_asr_confidence = 4`
- `duration_too_long = 2`
- `rms_near_lower_bound = 2`

解讀：

- `serena` 比 `aiden` 短、整體更接近可用。
- 但目前仍主要敗在 tone model 聽到的聲調序列不對。
- silence padding 問題也偏多，不一定是致命問題，但表示音檔尾端/前端仍不夠俐落。

### 3.2 Stage 2 / 3 的 proof 目前不是「真正用 Qwen 刺激音完成」

這點非常重要，之後接手不能誤判：

- `data/hearing_real_demo/patient_responses_seed.csv` 和
  [output/hearing_real_demo/patient_responses_asr.csv](../output/hearing_real_demo/patient_responses_asr.csv)
  裡面的 `audio_path` 是 `data/aishell3/raw/test/...` 的真實 AISHELL-3 音檔。
- 也就是說，患者回應分析這條線目前是用「真實 corpus 音檔」驗證 ASR + tone + scoring + report 會工作。
- 這證明 Stage 2 / 3 pipeline 本身是通的。
- 但它還不能證明「Stage 1 通過 QC 的 Qwen 刺激音」已經真正接到同一套 real-demo 測試中。

### 3.3 session metadata 有一個語意上的不精確點

[data/hearing_real_demo/patient_session_real.json](../data/hearing_real_demo/patient_session_real.json) 目前寫的是：

- `"stimulus_source": "Qwen3-TTS"`

但這次 Stage 2 / 3 proof 實際使用的音檔來源是 AISHELL-3 真實音檔，不是已通過 QC 的 Qwen 刺激音。

所以目前的 [output/hearing_real_demo/patient_report.md](../output/hearing_real_demo/patient_report.md) 應該視為：

- 「病人分析與報告流程可用」的 proof artifact
- 不是「Qwen 刺激音端到端已完成」的 final artifact

### 3.4 文件中的 CLI 旗標有舊版本殘留

[docs/mandarin_hearing_mvp_pipeline.md](../docs/mandarin_hearing_mvp_pipeline.md) 裡的示例命令有一部分仍是舊參數名，例如：

- `--items`：現在腳本實際使用的是 `--item-bank`
- `--approved-out`：現在腳本實際使用的是 `--out-approved`

之後重跑請以腳本本身的 `argparse` 為準，不要直接照舊 doc 貼上執行。

## 4. 目前建議的接續方向

優先順序建議如下：

1. 先把 Stage 1 做到「可交付」。
2. 不要先放寬 QC 門檻。
3. 先換 speaker / 模型，再考慮非常窄的後處理。

具體建議：

1. 先做 `eric` 的完整 8 題 batch，比對 `aiden` / `serena`。
2. 若 `0.6B` 仍過不了，優先評估本機是否已有可直接用的 `1.7B` Qwen3-TTS cache。
3. 若只有前後靜音偏長，可考慮「可重現且固定」的 trim 後處理，但不要先動 duration / speech-rate / tone 規則本身。
4. 若 tone mismatch 仍大量存在，優先懷疑 speaker / model，不建議先把 tone QC 降格。
5. 找到可接受 speaker / model 後，再重跑完整 Stage 1，確認有足夠的 approved stimuli，之後才算完成真正的 end-to-end 整合。

## 5. 之後可直接重跑的命令

所有 shell 命令都要加 `rtk`。

建議先設一個 model path：

```bash
MODEL=/home/sbplab/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

### 5.1 跑某個 speaker 的 Stage 1 batch

```bash
rtk .venv/bin/python -m scripts.demo.generate_qwen_tts_candidates \
  --item-bank data/hearing_real_demo/item_bank_real.csv \
  --output-csv output/hearing_real_demo/tts_candidates_qwen_eric.csv \
  --audio-dir output/hearing_real_demo/audio/tts_qwen_eric \
  --model-id "$MODEL" \
  --speaker eric \
  --device cpu \
  --local-files-only \
  --max-new-tokens 48 \
  --no-do-sample
```

```bash
rtk .venv/bin/python -m scripts.demo.run_local_asr \
  --input-csv output/hearing_real_demo/tts_candidates_qwen_eric.csv \
  --output-csv output/hearing_real_demo/tts_candidates_qwen_eric_asr.csv \
  --model openai/whisper-small \
  --language zh \
  --device cpu
```

```bash
rtk .venv/bin/python -m scripts.demo.run_tone_inference \
  --input-csv output/hearing_real_demo/tts_candidates_qwen_eric_asr.csv \
  --output-csv output/hearing_real_demo/tts_candidates_qwen_eric_scored.csv \
  --summary-json output/hearing_real_demo/tts_qwen_eric_tone_summary.json \
  --item-bank data/hearing_real_demo/item_bank_real.csv \
  --device cpu
```

```bash
rtk .venv/bin/python -m scripts.demo.run_tts_qc \
  --item-bank data/hearing_real_demo/item_bank_real.csv \
  --candidates output/hearing_real_demo/tts_candidates_qwen_eric_scored.csv \
  --out-csv output/hearing_real_demo/tts_qwen_eric_qc_report.csv \
  --out-summary output/hearing_real_demo/tts_qwen_eric_qc_summary.json \
  --out-approved output/hearing_real_demo/tts_qwen_eric_approved.csv
```

### 5.2 重跑患者回應分析這一側

```bash
rtk .venv/bin/python -m scripts.demo.run_local_asr \
  --input-csv data/hearing_real_demo/patient_responses_seed.csv \
  --output-csv output/hearing_real_demo/patient_responses_asr.csv \
  --model openai/whisper-small \
  --language zh \
  --device cpu
```

```bash
rtk .venv/bin/python -m scripts.demo.run_tone_inference \
  --input-csv output/hearing_real_demo/patient_responses_asr.csv \
  --output-csv output/hearing_real_demo/patient_responses_scored_inputs.csv \
  --summary-json output/hearing_real_demo/patient_tone_summary.json \
  --item-bank data/hearing_real_demo/item_bank_real.csv \
  --device cpu
```

```bash
rtk .venv/bin/python -m scripts.demo.score_patient_repetition \
  --item-bank data/hearing_real_demo/item_bank_real.csv \
  --responses output/hearing_real_demo/patient_responses_scored_inputs.csv \
  --out-score output/hearing_real_demo/structured_score.json \
  --out-items output/hearing_real_demo/item_level_results.csv \
  --out-confusion output/hearing_real_demo/confusion_summary.json \
  --out-review output/hearing_real_demo/human_review_items.csv
```

```bash
rtk .venv/bin/python -m scripts.demo.generate_patient_report \
  --score output/hearing_real_demo/structured_score.json \
  --confusion output/hearing_real_demo/confusion_summary.json \
  --items output/hearing_real_demo/item_level_results.csv \
  --session data/hearing_real_demo/patient_session_real.json \
  --template '聽能分析報告/AI華語聽能複誦分析參考報告模板.md' \
  --out-report output/hearing_real_demo/patient_report.md \
  --out-rehab output/hearing_real_demo/rehab_plan.json
```

## 6. 現在的停止點

截至這份報告寫下時：

- 沒有背景中的 Qwen / Whisper / tone batch 在跑。
- real-demo 的患者分析產物都已落盤。
- local Qwen `serena` 8 題 batch 也已跑完，結果是 `1 PASS / 7 FAIL`。
- 所以下次接手時，最合理的第一步不是重做 Stage 2 / 3，而是繼續解 Stage 1 的 speaker / model 選擇問題。
