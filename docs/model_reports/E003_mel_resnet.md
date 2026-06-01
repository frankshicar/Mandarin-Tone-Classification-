# Log-Mel ResNet 報告

## 目的

這個實驗測試：raw acoustic spectrum features 是否比 F0-only contours 更適合中文聲調辨識。

它是實作更強的 end-to-end-style context model 前，先建立的 ResNet-style spectrogram baseline。

## 資料與特徵

- Dataset：local AISHELL-3 train subset。
- Full run split：`data/aishell3/features/syllable_f0_train_full_split.csv`。
- Train rows：67,184 syllables。
- Validation rows：16,881 syllables。
- Boundary source：approximate voiced-span/equal-duration syllable boundaries。
- Log-mel summary：`data/aishell3/features/logmel_utterance_train_full.csv`。
- Log-mel NPZ directory：`data/aishell3/features/logmel_npz_train_full`。
- Input feature：
  - 80-bin log-mel segment。
  - 96 resized time frames。
  - 預設 tri-tone window：current syllable 加上左右各半個 syllable duration 的 padding。

## 模型

- Script：`scripts/training/train_mel_resnet.py`。
- 架構：
  - Conv2D stem。
  - Small residual CNN body。
  - Adaptive average pooling。
  - Dropout + linear 5-tone classifier。
- Full-run 預設 hyperparameters：
  - `width=32`
  - `dropout=0.1`
  - `frames=96`
  - `epochs=20`
  - `batch_size=64`
  - `lr=1e-3`
  - `weight_decay=1e-4`

## 結果

Full run：

- Metrics：`runs/mel_resnet_train_full/metrics.json`。
- Checkpoint：`runs/mel_resnet_train_full/best.pt`。
- Best epoch：9。
- Validation accuracy：`0.5234`。
- Validation macro F1：`0.4747`。
- Per-tone F1：
  - T1：`0.5796`
  - T2：`0.5153`
  - T3：`0.3699`
  - T4：`0.6011`
  - T5：`0.3074`

和 F0 structured full run 比較：

- Accuracy 從 `0.4451` 提升到 `0.5234`。
- Macro F1 從 `0.3794` 提升到 `0.4747`。

Smoke run：

- Metrics：`runs/mel_resnet_train200_smoke/metrics.json`。
- Train rows：1,516 syllables。
- Validation rows：338 syllables。
- Best epoch：1。
- Validation accuracy：`0.2041`。
- Validation macro F1：`0.0678`。
- 這個 run 只代表 pipeline check。

## 解讀

Log-mel ResNet 明顯優於 F0-only 和 F0-structured models。這支持目前方向：raw spectrum 裡有 extracted F0 以外的有用 tone information。

模型仍有弱點：

- T3 和 T5 仍是最難的 tones。
- Validation performance 在 epoch 9 左右達到 peak，之後出現 overfitting。
- 模型仍使用 approximate syllable boundaries，因此 input segments 仍有噪聲。
- 它還不是 End-to-End paper 的 contextual setup。

## 結論

目前在這個 repo 中，log-mel input 比 F0-only input 強。之後模型修改應把這個視為較低階的 spectrogram baseline，主要比較對象應該是 C1 context model。

## 不要重犯

- 沒有 regularization 或 early stopping 時，不要盲目延長這個模型的 training。
- 不要把 train200 smoke result 當作 mel input 不好的證據。
- 沒有註明 segmentation 是 approximate 時，不要直接和 End-to-End paper 比。
