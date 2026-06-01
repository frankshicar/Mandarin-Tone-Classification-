# F0 Structured Transformer 報告

## 目的

這個實驗測試：加入 short-term context、syllable duration 和 boundary indicators，是否能改善 F0-only tone baseline。

背後假設是：聲調不是孤立 contour。Coarticulation、tone sandhi、phrase boundaries 和 neutral-tone reduction 都會影響分類。

## 資料與特徵

- Dataset：local AISHELL-3 train subset。
- Full run split：`data/aishell3/features/syllable_f0_train_full_split.csv`。
- Train rows：67,184 syllables。
- Validation rows：16,881 syllables。
- Boundary source：approximate voiced-span/equal-duration syllable boundaries。
- Dense contours：`data/aishell3/features/syllable_f0_train_full.npz`。
- Input contour context：
  - previous syllable F0
  - current syllable F0
  - next syllable F0
- Structured features：
  - `duration_sec`
  - `relative_syllable_index`
  - `word_boundary_after`
  - `phrase_boundary_after`
  - `has_prev_context`
  - `has_next_context`
- Normalization：
  - F0 statistics 只從 train split 計算。
  - Structured-feature statistics 只從 train split 計算。

## 模型

- Script：`scripts/training/train_f0_structured_transformer.py`。
- 架構：
  - 每個 F0 contour 被投影成一個 context slot。
  - 三個 slots 經過 Transformer encoder。
  - 缺失的 previous/next context 會被 mask 掉。
  - Structured features 由 MLP 編碼。
  - F0 context representation 與 structured representation concat。
  - LayerNorm + linear 5-tone classifier。
- Full-run 預設 hyperparameters：
  - `d_model=64`
  - `nhead=4`
  - `num_layers=2`
  - `dim_feedforward=128`
  - `dropout=0.1`
  - `epochs=20`
  - `batch_size=64`
  - `lr=1e-3`
  - `weight_decay=1e-4`

## 結果

Full run：

- Metrics：`runs/f0_structured_transformer_train_full/metrics.json`。
- Checkpoint：`runs/f0_structured_transformer_train_full/best.pt`。
- Best epoch：9。
- Validation accuracy：`0.4451`。
- Validation macro F1：`0.3794`。
- Per-tone F1：
  - T1：`0.5112`
  - T2：`0.4109`
  - T3：`0.2540`
  - T4：`0.4985`
  - T5：`0.2225`

和 F0-only full run 比較：

- Accuracy 從 `0.4088` 提升到 `0.4451`。
- Macro F1 從 `0.2999` 提升到 `0.3794`。
- T5 從 `0.0000` 提升到 `0.2225`。
- T3 從 `0.1821` 提升到 `0.2540`。

Smoke run：

- Metrics：`runs/f0_structured_transformer_train100_smoke/metrics.json`。
- Train rows：905 syllables。
- Validation rows：292 syllables。
- Best epoch：5。
- Validation accuracy：`0.3596`。
- Validation macro F1：`0.2176`。
- 這個 run 只代表 pipeline check。

## 解讀

Structured F0 model 驗證了 context、duration 和 boundaries 有幫助。相較 F0-only，它對 T3 和 T5 都有改善。

但絕對分數仍明顯低於 end-to-end spectrogram-style baseline。可能限制包括：

- Approximate syllable boundaries 有噪聲。
- F0 extraction 在 unvoiced 或 noisy regions 會丟失資訊。
- 模型只看到 pitch contour 和少數 scalar features，沒有 log-mel input 的 richer spectral information。
- Context 只用三個 F0 slots 表示，對困難的 tone sandhi 和 reduced syllables 可能太淺。

## 結論

這個模型適合作為 ablation，證明 `feature/boundary` information 有幫助。MFA boundaries 完成後應重跑，但它目前不應取代 log-mel context model 作為最強 baseline。

## 不要重犯

- 不要假設 F0 + handcrafted structure 會自動追上 spectrogram models。
- 在修好 alignment quality 前，不要先增加更多 Transformer layers。
- 不要把 smoke run 當成模型正式表現。
