# Scripts 目錄說明

請在專案根目錄下用模組方式執行腳本：

```bash
rtk python -m scripts.demo.run_tts_qc --help
rtk python -m scripts.training.train_mel_resnet --help
rtk python -m scripts.asr.build_asr_syllable_baseline --help
```

## 子目錄

- `common/`：demo 和評分腳本共用的輔助函式。
- `data/`：AISHELL-3 下載、manifest 建構、子集抽樣和資料切分工具。
- `mfa/`：MFA 語料準備、詞典過濾，以及 TextGrid 邊界轉換。
- `asr/`：基於 ASR 的音節邊界基線，以及 ASR 與 MFA 的時間邊界對比。
- `features/`：utterance 級和 syllable 級特徵擷取。
- `training/`：資料集定義與模型訓練入口。
- `demo/`：聽力 demo、ASR、聲調推斷、質檢、評分和報告腳本。
- `maintenance/`：生成文件或圖片時使用的一次性維護腳本。

## ASR 與 MFA 邊界基線

### Whisper 等時長基線

1. 對 utterance manifest 執行 Whisper ASR：

```bash
rtk python -m scripts.demo.run_local_asr \
  --input-csv data/aishell3/manifest_train_full.csv \
  --output-csv data/aishell3/manifest_train_full_asr.csv
```

2. 將 ASR 輸出轉換為與 MFA 相同的音節邊界 schema：

```bash
rtk python -m scripts.asr.build_asr_syllable_baseline \
  --manifest data/aishell3/manifest_train_full_asr.csv \
  --f0-summary data/aishell3/features/f0_utterance_train_full.csv \
  --out data/aishell3/syllable_manifest_asr_equal_duration.csv
```

3. 與 MFA 邊界對比：

```bash
rtk python -m scripts.asr.compare_syllable_boundaries \
  --reference data/aishell3/syllable_manifest_mfa_train_full_strict.csv \
  --candidate data/aishell3/syllable_manifest_asr_equal_duration.csv \
  --summary-out data/aishell3/asr_vs_mfa_boundary_summary.csv \
  --detail-out data/aishell3/asr_vs_mfa_boundary_detail.csv
```

這個 ASR 基線用 ASR 文本與拼音判斷 utterance 是否可用。如果沒有 ASR 時間戳，它會把整段 utterance，或可選的 voiced F0 區間，平均分配給各個音節。因此對比結果會同時反映 ASR 音節匹配覆蓋率與相對 MFA 的時間誤差。

### WhisperX 時間戳基線

WhisperX 在 Whisper/faster-whisper 轉寫之後增加一次對齊步驟。對普通話資料，優先使用 char timestamp，因為在 AISHELL-3 中一個漢字通常對應一個音節。

本專案已驗證的 WhisperX 環境是：

```bash
/home/sbplab/anaconda3/envs/whisperx
```

執行 WhisperX 時需要把該環境的 `bin` 目錄放到 `PATH` 前面，這樣 WhisperX 才能找到同一環境裡的 `ffmpeg`。

1. 執行 WhisperX ASR 和對齊：

```bash
rtk env PATH=/home/sbplab/anaconda3/envs/whisperx/bin:$PATH \
  /home/sbplab/anaconda3/envs/whisperx/bin/python \
  -m scripts.asr.run_whisperx_asr \
  --input-csv data/aishell3/manifest_train_full.csv \
  --output-csv data/aishell3/manifest_train_full_whisperx.csv \
  --jsonl-out data/aishell3/whisperx_train_full.jsonl \
  --model small \
  --language zh
```

2. 將 WhisperX char timestamps 轉換為相容 MFA 的音節 schema：

```bash
rtk env PATH=/home/sbplab/anaconda3/envs/whisperx/bin:$PATH \
  /home/sbplab/anaconda3/envs/whisperx/bin/python \
  -m scripts.asr.build_whisperx_syllable_baseline \
  --manifest data/aishell3/manifest_train_full_whisperx.csv \
  --whisperx-jsonl data/aishell3/whisperx_train_full.jsonl \
  --out data/aishell3/syllable_manifest_whisperx_char.csv
```

3. 對比 WhisperX 和 MFA：

```bash
rtk python -m scripts.asr.compare_syllable_boundaries \
  --reference data/aishell3/syllable_manifest_mfa_train_full_strict.csv \
  --candidate data/aishell3/syllable_manifest_whisperx_char.csv \
  --summary-out data/aishell3/whisperx_vs_mfa_boundary_summary.csv \
  --detail-out data/aishell3/whisperx_vs_mfa_boundary_detail.csv
```

WhisperX 輸出包括兩類檔案：帶 ASR 欄位的 manifest，以及保存原始對齊結果的 JSONL。轉換腳本會讀取 JSONL 裡的 char 或 word 時間戳；如果某條音訊轉寫或對齊失敗，mismatch CSV 會記錄對應的 `whisperx_error`，避免把執行時錯誤誤判成文本不匹配。
