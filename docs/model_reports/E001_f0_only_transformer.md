# F0-only Transformer 報告

## 目的

這個 baseline 測試：只使用目前 syllable-level F0 contour，能恢復多少中文聲調資訊。

這是加入上下文、duration、boundary features 或 spectrogram input 前的最小 pitch-contour baseline。

## 資料與特徵

- Dataset：local AISHELL-3 train subset。
- Full run split：`data/aishell3/features/syllable_f0_train_full_split.csv`。
- Train rows：67,184 syllables。
- Validation rows：16,881 syllables。
- Boundary source：approximate voiced-span/equal-duration syllable boundaries。
- Input feature：單一 current-syllable F0 contour。
- Contour length：40 samples。
- Normalization：使用 raw `f0_hz`，在 `train_f0_transformer.py` 裡用 train split 的 speaker/global statistics 做 normalization。
- Dropped rows：split 前已移除 empty-contour rows。

## 模型

- Script：`scripts/training/train_f0_transformer.py`。
- 架構：
  - 將 1-D F0 values 用 linear projection 投到 `d_model`。
  - 加 sinusoidal positional encoding。
  - Transformer encoder。
  - 沿 contour time axis 做 mean pooling。
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

- Metrics：`runs/f0_transformer_train_full/metrics.json`。
- Checkpoint：`runs/f0_transformer_train_full/best.pt`。
- Best epoch：17。
- Validation accuracy：`0.4088`。
- Validation macro F1：`0.2999`。
- Per-tone F1：
  - T1：`0.4527`
  - T2：`0.3692`
  - T3：`0.1821`
  - T4：`0.4956`
  - T5：`0.0000`

Smoke run：

- Metrics：`runs/f0_transformer_train100_smoke/metrics.json`。
- Train rows：905 syllables。
- Validation rows：292 syllables。
- Best epoch：4。
- Validation accuracy：`0.3459`。
- Validation macro F1：`0.1730`。
- 這個 run 只代表 pipeline check。

## 解讀

F0-only baseline 偏弱。它證明 pipeline 能學到高於 random chance 的訊號，但模型沒有足夠資訊，且 segmentation 不夠乾淨，因此無法穩定辨識五聲。

主要失敗點：

- T3 很弱，可能因為 T3 shape 強烈依賴 context、sandhi、duration 和 syllable reduction。
- Full F0-only run 幾乎沒有學到 T5。
- Approximate boundaries 可能切到相鄰 syllable 或 silence，這會直接污染 F0-only input。
- Pure F0 丟掉了後續 log-mel models 能使用的 spectral 與 voicing cues。

## 結論

F0-only Transformer 應保留為 diagnostic baseline，不應作為主要模型方向。之後若要比較 F0 模型，應等 approximate boundaries 被 MFA/forced-alignment boundaries 取代後再重跑。

## 不要重犯

- 不要把 40% accuracy 解讀成 AISHELL-3 tone recognition 的最終上限。
- 在修正 syllable boundaries 前，不要花太多時間調這個 baseline。
- 不要在沒有註明 input family 差異的情況下，直接把 F0-only 結果和 spectrogram paper 比較。
