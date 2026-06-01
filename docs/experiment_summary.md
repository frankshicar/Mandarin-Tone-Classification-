# 實驗總表

這份文件是 AISHELL-3 中文聲調辨識專案的長期實驗紀錄。

每次修改模型前，先閱讀本文件和 `docs/model_reports/` 裡對應的模型報告。每做完一個新實驗，都要在這裡新增一列，並同步更新對應模型報告；如果是新模型，就新增一份模型報告。

## 對應模型報告

| 實驗 | 報告檔案 |
|---|---|
| E001 | `docs/model_reports/E001_f0_only_transformer.md` |
| E002 | `docs/model_reports/E002_f0_structured_transformer.md` |
| E003 | `docs/model_reports/E003_mel_resnet.md` |
| E004 / E005 / E007 | `docs/model_reports/E004_E005_E007_mel_context_resnet_c1.md` |
| E006 | `docs/model_reports/E006_mfa_boundary_pipeline.md` |

## 固定實驗模板

之後每個實驗都使用這個模板：

| 欄位 | 內容 |
|---|---|
| 實驗 ID | 短且穩定的 ID，例如 `E006` |
| 日期 | 實驗執行日期 |
| 動機 | 為什麼需要做這個實驗 |
| 假設 | 預期會改善什麼，以及原因 |
| Dataset / Split | 使用的資料檔、train/val/test 協定、是否 speaker split |
| 邊界來源 | approximate、MFA、ASR、manual 或其他 |
| 輸入特徵 | F0、log-mel、context、duration、boundary、lexical features |
| 模型 | script 路徑與架構摘要 |
| 訓練設定 | epochs、batch size、optimizer、seed、主要 hyperparameters |
| 精確指令 / Config | 可重跑的 command、config file 或 script arguments |
| 輸入 artifacts | manifest、split CSV、feature files、vocabulary、model inputs |
| 輸出 artifacts | metrics JSON、checkpoint、training log、產生的 feature paths |
| 選模標準 | best macro F1、best accuracy、final epoch 或其他 |
| Metrics | accuracy、macro F1、per-tone F1、checkpoint path |
| 結論 | 這次實驗學到什麼 |
| 決策 | 保留、淘汰、重跑，或作為 baseline |
| 下一步 | 一個具體下一步 |
| Caveats | 已知限制，或哪些地方不能過度解讀 |

## 目前 Same-Speaker 排名

這些實驗使用 held-out utterances，但 validation speakers 也出現在 training speakers 裡。這適合做受控 ablation，但不能和 speaker-disjoint validation 直接比較。

| 排名 | 實驗 | Validation Protocol | Accuracy | Macro F1 | 主要結論 |
|---:|---|---|---:|---:|---|
| 1 | E004 C1 mel context ResNet | same-speaker utterance split | 0.6459 | 0.6182 | 目前最強 baseline；短期聲學上下文方向明顯有效。 |
| 2 | E003 tri-tone log-mel ResNet | same-speaker utterance split | 0.5234 | 0.4747 | Spectrogram input 明顯優於 F0-only。 |
| 3 | E002 F0 structured Transformer | same-speaker utterance split | 0.4451 | 0.3794 | context/duration/boundary 對 F0 有幫助，但仍不足。 |
| 4 | E001 F0-only Transformer | same-speaker utterance split | 0.4088 | 0.2999 | 最小診斷 baseline；T3/T5 很弱。 |

## 目前 Speaker-Disjoint 排名

這些實驗的 validation speakers 沒有出現在 training speakers 裡。若要做較嚴格的 generalization claim，優先看這個 protocol。

| 排名 | 實驗 | Validation Protocol | Accuracy | Macro F1 | 主要結論 |
|---:|---|---|---:|---:|---|
| 1 | E007 C1 mel context ResNet + MFA boundaries | speaker-disjoint split | 0.8627 | 0.8360 | MFA boundaries 大幅改善結果，已超過 paper 中 78-79% baseline 的量級。 |
| 2 | E005 C1 mel context ResNet | speaker-disjoint split | 0.6172 | 0.5837 | Approximate boundaries 下仍有效，但 segmentation 明顯限制表現。 |

## 實驗紀錄

| ID | 動機 | 方法 | 結果 | 結論 | 決策 | 下一步 |
|---|---|---|---|---|---|---|
| E000 | 在 full run 前確認資料與訓練 pipeline 能跑通。 | 對 F0-only、F0-structured、mel ResNet、C1 mel context models 做 Train100/Train200 smoke runs。 | Smoke metrics 低且不穩定。 | Smoke run 只驗證程式路徑，不代表模型能力。 | 只保留作為 pipeline check。 | 不要引用 smoke 數字當正式結果。 |
| E001 | 建立最小 F0 contour baseline。 | current-syllable F0 contour 輸入 Transformer encoder；使用 approximate boundaries。 | Accuracy `0.4088`，macro F1 `0.2999`；T5 F1 `0.0000`。 | F0-only 高於 random chance，但不足以作為主方向。 | 保留為診斷 baseline。 | 若要做 ablation，等 MFA boundaries 完成後再重跑。 |
| E002 | 測試 explicit context、duration、boundary features 是否能改善 F0。 | prev/current/next F0 slots + duration/index/boundary flags 輸入 structured Transformer；使用 approximate boundaries。 | Accuracy `0.4451`，macro F1 `0.3794`；T3/T5 比 E001 改善。 | 結構化特徵有幫助，但 F0 仍弱於 spectrogram input。 | 保留為 ablation baseline。 | MFA boundaries 完成後重跑，用來隔離 segmentation 影響。 |
| E003 | 測試 raw spectrum features 是否比 extracted F0 更適合聲調辨識。 | tri-tone 80-bin log-mel segment 輸入 small ResNet；使用 approximate boundaries。 | Accuracy `0.5234`，macro F1 `0.4747`；best epoch 9 後開始 overfit。 | Log-mel input 明顯強於 F0-only。 | 保留為 spectrogram baseline。 | 未來模型優先和 E004 比，不只和 E003 比。 |
| E004 | 重現 End-to-End short-context 論文中最有用的概念。 | prev/current/next log-mel segments + segment scalar features 輸入 C1-style ResNet；same-speaker utterance split。 | Accuracy `0.6459`，macro F1 `0.6182`；T3 F1 `0.5036`，T5 F1 `0.5706`。 | 目前最佳模型；短期聲學上下文很重要。 | 主 baseline。 | 用 MFA boundaries 重跑。 |
| E005 | 評估 E004 是否過度依賴看過相同 speaker。 | 和 E004 相同的 C1-style model，但改成 speaker-disjoint validation split。 | Accuracy `0.6172`，macro F1 `0.5837`；比 E004 約低 `0.0345` macro F1。 | 泛化下降，但仍明顯強於前面 baseline。 | 主要 generalization reference。 | 嚴格 claim 優先使用 speaker-disjoint split。 |
| E006 | 用真實 alignment 取代 noisy approximate syllable boundaries。 | MFA Mandarin forced alignment；先做 train20 smoke，再對 full strict corpus 跑 alignment；用 corpus map parse TextGrid。 | Full strict corpus：9,138 utterances aligned，104,393 syllable rows，0 missing TextGrid，0 label mismatch，0 bad duration。 | MFA boundaries 已可作為下一輪 feature slicing 的輸入；仍要記錄被 skip 的 664 utterances。 | 完成 alignment，進入 feature regeneration。 | 用 MFA boundaries 重新 slice F0/log-mel，然後重跑 E004/E005。 |
| E007 | 驗證 syllable boundary quality 是否是先前低準確率主因。 | C1 mel context ResNet；MFA boundaries；speaker-disjoint split；batch size 64；不使用 syllable embedding。 | Best epoch 19；accuracy `0.8627`，macro F1 `0.8360`；T3 F1 `0.8017`，T5 F1 `0.7517`。 | MFA boundaries 讓 C1 baseline 大幅超越 approximate-boundary 版本，segmentation 是主要瓶頸。 | 新主 baseline。 | 補跑 same-speaker MFA split，並整理和 End-to-End paper 的 protocol 差異。 |

## 跨實驗教訓

- Approximate boundaries 是主要 confound；E007 已證明改用 MFA boundaries 後，speaker-disjoint macro F1 從 `0.5837` 提升到 `0.8360`。
- 目前 log-mel input 優於 F0-only input。這不代表 F0 沒用，而是目前的 F0 extraction 和 segmentation 較弱。
- 短期聲學上下文是目前最有潛力的方向，但 observed gains 仍和 feature/model changes 部分混在一起，還不是純 context ablation。
- T3 和 T5 是關鍵切片。只看 overall accuracy 會掩蓋這兩類錯誤。
- Same-speaker validation 會高估 deployment performance；speaker-disjoint validation 較接近真實泛化。
- Smoke runs 只用來驗證 code path。
- 除非實驗明確要研究 lexical information 或 leakage，否則不要啟用 syllable embeddings。
- 回報時使用 macro F1 選出的 best checkpoint，不要只看 final epoch。

## 標準下一步政策

每次做新模型修改前：

1. 檢查 proposed change 是否對應到已記錄的 failure mode。
2. 決定比較要用 same-speaker 還是 speaker-disjoint validation。
3. 除非實驗目標就是 data 或 segmentation，否則保持 input files 和 split files 固定。
4. 在本 summary 記錄 exact files、command/config、seed、checkpoint、selection criterion 和 metrics。
5. 更新 `docs/model_reports/` 裡對應的模型報告。

## 目前建議下一步

MFA speaker-disjoint C1 baseline 已完成。現在先補齊可比較實驗，不要急著換模型：

1. 補跑 MFA boundaries + C1 mel context ResNet same-speaker split。
2. 整理 E007 和 End-to-End paper 的 protocol 差異。
3. 檢查 664 skipped utterances 的 coverage bias，尤其 `们`。
4. 再考慮是否補 dictionary entries 或改用 ASR boundary 做 deployment-style 實驗。
