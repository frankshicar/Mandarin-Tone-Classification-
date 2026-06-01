# End-to-End-style Mel Context ResNet C1 報告

## 目的

這個實驗測試 End-to-End Mandarin tone classification paper 的核心概念：用鄰近 acoustic segments 的 short-term context 來分類當前 syllable tone。

這是目前 repo 裡最強的 baseline。

## 資料與特徵

- Dataset：local AISHELL-3 train subset。
- Same-speaker utterance split：
  - Split file：`data/aishell3/features/syllable_f0_train_full_split.csv`。
  - Train rows：67,184 syllables。
  - Validation rows：16,881 syllables。
- Speaker-disjoint split：
  - Split file：`data/aishell3/features/syllable_f0_train_full_speaker_split.csv`。
  - Train rows：66,877 syllables。
  - Validation rows：17,188 syllables。
- Boundary source：approximate voiced-span/equal-duration syllable boundaries。
- Context feature source：`data/aishell3/features/syllable_f0_train_full.csv`。
- Log-mel summary：`data/aishell3/features/logmel_utterance_train_full.csv`。
- Log-mel NPZ directory：`data/aishell3/features/logmel_npz_train_full`。
- Input feature：
  - previous syllable log-mel segment
  - current syllable log-mel segment
  - next syllable log-mel segment
  - 80 mel bins
  - 每個 segment resize 到 96 frames
- Segment-level scalar features：
  - `duration_sec`
  - relative syllable index
  - `word_boundary_after`
  - `phrase_boundary_after`
  - previous-context present flag
  - next-context present flag
- Syllable embedding：正式回報的 runs 中停用，避免 lexical shortcut。

## 模型

- Script：`scripts/training/train_mel_context_resnet.py`。
- 架構：
  - 對每個 context slot 使用 shared ResNet-style mel encoder。
  - 將 encoded previous/current/next slot vectors concat。
  - 再 concat segment scalar features。
  - MLP classifier 預測五聲。
- Full-run 預設 hyperparameters：
  - `width=32`
  - `hidden_dim=256`
  - `dropout=0.1`
  - `frames=96`
  - `epochs=20`
  - `batch_size=64`
  - `lr=1e-3`
  - `weight_decay=1e-4`
  - `use_syllable_embedding=false`

## 結果

Same-speaker utterance split：

- Metrics：`runs/mel_context_resnet_train_full/metrics.json`。
- Checkpoint：`runs/mel_context_resnet_train_full/best.pt`。
- Best epoch：19。
- Validation accuracy：`0.6459`。
- Validation macro F1：`0.6182`。
- Per-tone F1：
  - T1：`0.6937`
  - T2：`0.6444`
  - T3：`0.5036`
  - T4：`0.6786`
  - T5：`0.5706`

Speaker-disjoint split：

- Metrics：`runs/mel_context_resnet_train_full_speaker_split/metrics.json`。
- Checkpoint：`runs/mel_context_resnet_train_full_speaker_split/best.pt`。
- Best epoch by macro F1：15。
- Validation accuracy：`0.6172`。
- Validation macro F1：`0.5837`。
- Highest validation accuracy 是 epoch 19 的 `0.6285`，但 macro F1 較低，為 `0.5811`。
- Per-tone F1：
  - T1：`0.6683`
  - T2：`0.6271`
  - T3：`0.4705`
  - T4：`0.6485`
  - T5：`0.5042`

MFA boundary speaker-disjoint split：

- Metrics：`runs/mel_context_resnet_mfa_train_full_strict_speaker_split/metrics.json`。
- Checkpoint：`runs/mel_context_resnet_mfa_train_full_strict_speaker_split/best.pt`。
- Feature split：`data/aishell3/features/syllable_f0_mfa_train_full_strict_speaker_split.csv`。
- Context features：`data/aishell3/features/syllable_f0_mfa_train_full_strict.csv`。
- Boundary source：MFA forced alignment。
- Train rows：62,705 syllables。
- Validation rows：16,212 syllables。
- Best epoch by macro F1：19。
- Validation accuracy：`0.8627`。
- Validation macro F1：`0.8360`。
- Per-tone F1：
  - T1：`0.8781`
  - T2：`0.8658`
  - T3：`0.8017`
  - T4：`0.8827`
  - T5：`0.7517`

和 approximate-boundary speaker-disjoint split 比較：

- Accuracy 從 `0.6172` 提升到 `0.8627`，絕對提升 `+0.2455`。
- Macro F1 從 `0.5837` 提升到 `0.8360`，絕對提升 `+0.2523`。
- T3 F1 從 `0.4705` 提升到 `0.8017`。
- T5 F1 從 `0.5042` 提升到 `0.7517`。

和 log-mel ResNet full run 比較：

- Same-speaker macro F1 從 `0.4747` 提升到 `0.6182`。
- Same-speaker accuracy 從 `0.5234` 提升到 `0.6459`。

Smoke run：

- Metrics：`runs/mel_context_resnet_train200_smoke/metrics.json`。
- Train rows：1,516 syllables。
- Validation rows：338 syllables。
- Best epoch：1。
- Validation accuracy：`0.3018`。
- Validation macro F1：`0.1252`。
- 這個 run 只代表 pipeline check。

## 解讀

C1-style context model 是目前最清楚成功的方向。它顯示 short-term acoustic context 對 tone classification 有明顯幫助。

主要觀察：

- Approximate-boundary speaker-disjoint validation 下仍有合理泛化，只比 same-speaker validation 低約 `0.0345` macro F1。
- MFA boundaries 讓 speaker-disjoint accuracy 達到 `0.8627`，macro F1 達到 `0.8360`。這表示先前和 End-to-End paper 的主要差距，很大部分來自 syllable segmentation quality。
- T3 和 T5 在 MFA boundaries 下大幅改善，但仍是相對較弱的 tones。
- MFA setting 使用 ground-truth transcript 做 forced alignment，和 paper 的 ASR-front-end / deployment-style protocol 不完全相同；報告時要清楚區分。

## 結論

MFA boundary speaker-disjoint C1 是目前新的主 baseline。它證明切分品質是前面低準確率的主要瓶頸。下一步應補跑 same-speaker MFA split，並整理和 End-to-End paper 的 protocol 差異。

## 不要重犯

- 不要只根據 smoke results 評估這個 architecture。
- 不要直接比較 same-speaker validation 和 speaker-disjoint validation。
- 在明確測 lexical leakage 前，不要加入 syllable embeddings。
- 在驗證 MFA boundaries 前，不要先拉長 context window。
