# 華語聽能複誦 MVP Pipeline

這份文件對應目前 repo 裡新補上的最小可行流程，目標是把以下三段真正落成可跑的 CLI：

1. `Stage 1`：Qwen3-TTS 刺激音自動 QC
2. `Stage 2`：患者複誦結果自動評分
3. `Stage 3`：患者參考報告與復健題目生成

## 新增腳本

- `scripts/demo/run_tts_qc.py`
- `scripts/demo/score_patient_repetition.py`
- `scripts/demo/generate_patient_report.py`
- `scripts/demo/generate_hearing_demo_inputs.py`
- `scripts/common/hearing_pipeline_utils.py`

## Stage 1：Qwen3-TTS 音檔 QC

輸入：

- 題庫 CSV
- TTS candidate CSV
- candidate CSV 需至少提供：
  - `item_id`
  - `audio_path`
  - `asr_text` 或 `asr_pinyin`
  - `predicted_tones`

自動檢查：

- sample rate
- channel count
- duration
- leading / trailing silence
- RMS / peak
- clipping
- speech rate
- ASR 內容比對
- tone sequence 比對

輸出：

- detailed QC report CSV
- summary JSON
- PASS-only approved stimuli CSV

## Stage 2：患者複誦評分

輸入：

- 題庫 CSV
- 患者反應 CSV
- 反應 CSV 需至少提供：
  - `patient_id`
  - `item_id`
  - `asr_text` 或 `asr_pinyin`
  - `predicted_tones`

評分欄位：

- item accuracy
- keyword accuracy
- syllable accuracy
- initial accuracy
- final accuracy
- tone accuracy
- review flags
- confusion summary

輸出：

- `structured_score.json`
- `item_level_results.csv`
- `confusion_summary.json`
- `human_review_items.csv`

## Stage 3：患者參考報告與復健題目

輸入：

- `structured_score.json`
- `confusion_summary.json`
- `item_level_results.csv`
- optional session metadata JSON
- optional template path

輸出：

- markdown 報告
- rehab plan JSON

目前這一版採固定模板與規則式生成，不依賴 LLM 才能跑通。之後若要換成 LLM，只要保留相同的 structured inputs 即可。

## Demo

先產生 demo inputs：

```bash
python -m scripts.demo.generate_hearing_demo_inputs
```

跑 Stage 1：

```bash
python -m scripts.demo.run_tts_qc \
  --items data/hearing_demo/item_bank_demo.csv \
  --candidates data/hearing_demo/tts_candidates_demo.csv \
  --out-csv output/hearing_demo/tts_qc_report.csv \
  --out-summary output/hearing_demo/tts_qc_summary.json \
  --approved-out output/hearing_demo/approved_stimuli.csv
```

跑 Stage 2：

```bash
python -m scripts.demo.score_patient_repetition \
  --items data/hearing_demo/item_bank_demo.csv \
  --responses data/hearing_demo/patient_responses_demo.csv \
  --out-score output/hearing_demo/structured_score.json \
  --out-items output/hearing_demo/item_level_results.csv \
  --out-confusion output/hearing_demo/confusion_summary.json \
  --out-review output/hearing_demo/human_review_items.csv
```

跑 Stage 3：

```bash
python -m scripts.demo.generate_patient_report \
  --score output/hearing_demo/structured_score.json \
  --confusion output/hearing_demo/confusion_summary.json \
  --items output/hearing_demo/item_level_results.csv \
  --session data/hearing_demo/patient_session_demo.json \
  --template '聽能分析報告/AI華語聽能複誦分析參考報告模板.md' \
  --out-report output/hearing_demo/patient_report.md \
  --out-rehab output/hearing_demo/rehab_plan.json
```

## 目前邊界

- 這版 `Stage 1` 與 `Stage 2` 會吃「外部已經跑好的 ASR / tone outputs」，方便先把 QC、評分、報告流程落成。
- `Qwen3-TTS` 真正的生成呼叫，和 `ASR` 真正的模型推論，之後可以再接到這套 schema 上。
- 目前 repo 裡已經有現成 tone model 訓練與 checkpoint；若要把 Stage 2 進一步接成「直接吃患者音檔做 tone inference」，下一步就是補 inference wrapper。
