# MFA Boundary Pipeline 報告

## 目的

這個 pipeline 的目標是解決目前實驗中最大的已知弱點：approximate syllable boundaries。

我們要用 forced-alignment boundaries 取代 voiced-span/equal-duration boundaries，然後在較乾淨的 segmentation 下重跑相同模型。

## 方法

選擇的 alignment approach：

- Montreal Forced Aligner Mandarin acoustic model。
- 使用 AISHELL-3 已有 transcripts/pinyin 作為已知文字。
- 這是 forced alignment，不是 open-ended ASR。

選擇原因：

- 我們已經有 transcript 和 pinyin labels。
- 目前主要缺的是 syllable start/end time。
- 對 boundary extraction 來說，forced alignment 比 raw ASR 更適合。

## 已實作 scripts

- `scripts/mfa/prepare_mfa_corpus.py`
  - 建立 MFA corpus，並依 speaker 建立 subdirectories。
  - 使用 Hanzi token sequences 寫 `.lab` files。
  - 輸出 corpus map CSV。
- `scripts/mfa/filter_mfa_mandarin_dictionary.py`
  - 將 Mandarin MFA dictionary filter 到 target corpus vocabulary。
  - 移除 acoustic model 不支援 phones 的 pronunciations。
- `scripts/mfa/textgrid_to_syllable_boundaries.py`
  - 將 MFA TextGrid outputs 轉成 repo 使用的 syllable-boundary CSV format。
  - 輸出可接到 downstream F0 slicing 的 rows。

## Smoke Result

Train20 MFA smoke：

- Corpus size：20 aligned utterances。
- Boundary CSV：`data/aishell3/syllable_manifest_mfa_train20.csv`。
- Output rows：253 syllables。
- Processed utterances：20。
- Missing TextGrid utterances：1，這是預期內，因為前面 manifest 中有一列在 corpus preparation 時被 skip。
- Mismatched utterances：0。
- Duration sanity：
  - Minimum syllable duration：`0.06` sec。
  - Maximum syllable duration：`0.92` sec。
  - Bad-duration rows：0。
- Tone counts：
  - T1：66
  - T2：50
  - T3：43
  - T4：88
  - T5：6

第二輪 smoke 改用 MFA corpus map 作為 parser input：

- Boundary CSV：`data/aishell3/syllable_manifest_mfa_train20_review2.csv`。
- Processed utterances：20。
- Missing TextGrid utterances：0。
- Mismatched utterances：0。

## Full Alignment Result

Full strict MFA alignment 已完成：

- Corpus map：`data/aishell3/mfa/corpus_train_full_strict_map.csv`。
- Skipped audit：`data/aishell3/mfa/corpus_train_full_strict_skipped.csv`。
- Dictionary：`data/aishell3/mfa/mandarin_mfa_train_full_strict_filtered.dict`。
- Alignment log：`runs/mfa_align_train_full_strict.log`。
- TextGrid output：`data/aishell3/mfa/aligned_train_full_strict`。
- Boundary CSV：`data/aishell3/syllable_manifest_mfa_train_full_strict.csv`。
- Mismatch audit：`data/aishell3/syllable_manifest_mfa_train_full_strict_mismatches.csv`。

Full strict corpus：

- Written utterances：9,138。
- Skipped utterances：664。
- Skip reasons：
  - `hanzi_pinyin_count_mismatch`：193。
  - `missing_dictionary_word`：471。
- Dictionary coverage after strict filtering：2,893 needed words，2,893 covered words，0 missing words。

Full boundary CSV：

- Rows：104,393 syllables。
- Utterances：9,138。
- Speakers：100。
- Missing TextGrid utterances：0。
- Mismatched utterances：0。
- Bad-duration rows：0。
- Duration range：`0.03` 到 `1.03` sec。
- Word-boundary flags：
  - `true`：36,552
  - `false`：67,841
- Phrase-boundary flags：
  - `true`：10,569
  - `false`：93,824
- Tone counts：
  - T1：22,127
  - T2：24,885
  - T3：16,649
  - T4：34,741
  - T5：5,991

## 重要操作注意

使用 relative temporary paths 時，MFA alignment 失敗。錯誤表面上是 missing SQLite table，但實際原因是 internal alignment symlinks resolve 錯位置。

執行 MFA 時請使用 absolute paths：

- corpus directory
- dictionary path
- output directory
- `--temporary_directory`

## 目前已知問題

- Full strict alignment 為了避免 OOV 和 token mismatch，跳過了 664 utterances。
- `hanzi_pinyin_count_mismatch` 仍有 193 句；已加入簡單兒化音 token merge，但還沒完整解決所有特殊 tokenization。
- `missing_dictionary_word` 造成 471 句被排除；這比使用 `<unk>` 強，因為 full alignment 不會靜默使用不可靠 OOV。
- Dictionary missing 中 `们` 影響最多，之後若要提升 coverage，可以優先補常見 missing 字的 dictionary entries。
- MFA alignment analysis 顯示有低 SNR outliers；之後 feature slicing 或 retraining 時可評估是否排除 SNR < 5 的 utterances。
- 目前 TextGrid parser 是手寫 parser；已做 label consistency 和 strict missing/mismatch fail，但未改用 TextGrid library。

## 結論

MFA full alignment 已成功產生可用的 forced-alignment syllable boundaries。下一步是用這份 boundary CSV 重新產生 syllable-level F0/log-mel features，然後重跑 C1 mel context baseline，檢查 segmentation 改善是否能拉近和 End-to-End paper 的差距。

## 不要重犯

- 不要用 relative temporary paths 跑 full MFA。
- 不要用原始 manifest 直接 parse TextGrid；應使用 MFA corpus map，避免 skipped corpus rows 被誤判成 missing alignment。
- 不要靜默忽略 skipped rows 或 dictionary OOV characters。
