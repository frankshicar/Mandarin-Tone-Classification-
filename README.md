# Mandarin Tone Classification

這個專案整理華語聲調分類與聽能複誦評分流程，重點包含 AISHELL-3 聲調資料處理、音節邊界建立、F0/log-mel 特徵擷取、聲調模型訓練，以及一組可跑的聽能 demo CLI。

目前最重要的實驗紀錄在 `docs/experiment_summary.md`。腳本使用方式集中在 `scripts/README.md`。

## 專案內容

- `scripts/data/`：AISHELL-3 下載、manifest 建構、資料切分。
- `scripts/mfa/`：MFA forced alignment 語料準備、字典過濾、TextGrid 轉音節邊界。
- `scripts/asr/`：Whisper / WhisperX 音節邊界基線，以及 ASR 與 MFA 邊界比較。
- `scripts/features/`：F0、log-mel、syllable-level 特徵擷取。
- `scripts/training/`：聲調分類資料集與訓練入口。
- `scripts/demo/`：聽能複誦 demo、TTS QC、複誦評分、報告生成。
- `docs/`：實驗計畫、模型報告、MVP pipeline 文件。

## 安裝

建議使用 Python 3.10 或 3.12 的虛擬環境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WhisperX 建議使用獨立 conda 環境。已驗證的本機環境路徑為：

```bash
/home/sbplab/anaconda3/envs/whisperx
```

執行 WhisperX 腳本時，需要把該環境的 `bin` 放在 `PATH` 前面，讓 WhisperX 使用同一環境的 `ffmpeg`：

```bash
env PATH=/home/sbplab/anaconda3/envs/whisperx/bin:$PATH \
  /home/sbplab/anaconda3/envs/whisperx/bin/python \
  -m scripts.asr.run_whisperx_asr --help
```

## 資料與大型檔案

GitHub repo 不包含資料集、模型 checkpoint、產生的特徵、音訊輸出、實驗輸出或論文 PDF。這些檔案已由 `.gitignore` 排除：

- `data/`
- `output/`
- `runs/`
- `checkpoints/`
- `hearing_paper/*.pdf`
- `.venv/`
- `.codegraph/`
- `session.txt`

若要重跑實驗，請先依 `docs/aishell3_tone_plan.md` 和 `scripts/README.md` 準備 AISHELL-3、本機 manifest、MFA/ASR 邊界與特徵檔。

## 常用指令

查看腳本說明：

```bash
python -m scripts.demo.run_tts_qc --help
python -m scripts.training.train_mel_resnet --help
python -m scripts.asr.build_asr_syllable_baseline --help
```

WhisperX char timestamp 轉換為音節邊界：

```bash
env PATH=/home/sbplab/anaconda3/envs/whisperx/bin:$PATH \
  /home/sbplab/anaconda3/envs/whisperx/bin/python \
  -m scripts.asr.build_whisperx_syllable_baseline \
  --manifest data/aishell3/manifest_train_full_whisperx.csv \
  --whisperx-jsonl data/aishell3/whisperx_train_full.jsonl \
  --out data/aishell3/syllable_manifest_whisperx_char.csv
```

比較 WhisperX 與 MFA 音節邊界：

```bash
python -m scripts.asr.compare_syllable_boundaries \
  --reference data/aishell3/syllable_manifest_mfa_train_full_strict.csv \
  --candidate data/aishell3/syllable_manifest_whisperx_char.csv \
  --summary-out data/aishell3/whisperx_vs_mfa_boundary_summary.csv \
  --detail-out data/aishell3/whisperx_vs_mfa_boundary_detail.csv
```

更多命令請看 `scripts/README.md`。

## 目前基線

目前最佳 speaker-disjoint baseline 是 `E007 C1 mel context ResNet + MFA boundaries`：

- Accuracy：`0.8627`
- Macro F1：`0.8360`
- 詳細紀錄：`docs/experiment_summary.md`

WhisperX 與 MFA 的 5 筆 subset 邊界比較已驗證流程可跑通：

- Comparable rows：`65`
- Mean start error：`84.662 ms`
- Mean end error：`120.569 ms`
- ASR position match rate：`0.969231`

## 授權與資料注意事項

請遵守 AISHELL-3、MFA、WhisperX，以及相關論文與模型的原始授權條款。本 repo 只保存程式碼與專案文件，不重新發布資料集或參考論文 PDF。
