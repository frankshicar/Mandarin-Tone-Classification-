# AISHELL-3 Mandarin Tone Plan

## Goal

Build a Mandarin tone-classification pipeline based on:

```text
F0 contour + short-term syllable context + duration/rhythm + boundary features + Transformer
```

The target model follows the useful part of the end-to-end context paper, but makes F0, duration, and boundary information explicit so the model can better handle coarticulation, tone sandhi, neutral tone, and phrase-boundary F0 reset.

## Current Dataset State

Installed dataset:

- Raw snapshot: `data/aishell3/raw`
- Manifest: `data/aishell3/manifest.csv`
- Train wav files: 9,802
- Test wav files: 1,948
- Total wav files: 11,750
- Note: the annotation files list many more utterances than the local HF wav subset. Current manifests are intentionally built only over wav files that exist locally.

Important limitation:

- The Hugging Face `datasets` interface exposes only `audio` and speaker `label`.
- The raw Hub repository also contains annotation files: `train/content.txt`, `test/content.txt`, `train/label_train-set.txt`, `spk-info.txt`, and `phone_set.txt`.
- The local manifest now merges utterance text, pinyin-with-tone, tone sequence, speaker metadata, and train-set prosody boundaries.
- Exact syllable `start_sec` / `end_sec` / `duration_sec` still requires forced alignment.

## Audit Notes

Checked before forced-alignment work:

- Required raw files exist: `README.md`, `phone_set.txt`, `spk-info.txt`, `train/content.txt`, `test/content.txt`, `train/label_train-set.txt`.
- Local wav counts: 9,802 train and 1,948 test.
- Annotation counts: 63,262 train content rows, 24,773 test content rows, 63,262 train prosody rows, and 218 speaker-info rows.
- Utterance manifest rows: 11,750. All rows have existing audio, transcript, pinyin, tone sequence, sample rate, duration, and speaker metadata.
- Syllable manifest rows: 124,579, exactly matching the sum of utterance syllable counts.
- Syllable tone/context consistency check passes: no tone or prev/next context mismatches.
- F0 smoke test: 100 utterances processed, 100 `.npz` files written, no missing NPZ files.

Known issues to handle:

- The local wav subset is not the full 88,035-utterance AISHELL-3 corpus described in the README.
- `librosa.pyin` can assign very low F0 values near silence/noise; the current smoke pipeline filters by voiced probability and computes speaker/global F0 normalization from the train split only, but full-corpus runs still need the same checks.

## Data Plan

### Stage 1: Audio Manifest

Create a stable manifest with:

- `split`
- `speaker`
- `utt_id`
- `audio_path`
- audio metadata: sample rate, duration, channel count

This is enough for F0 extraction, speaker normalization, and later alignment.

### Stage 2: F0 Feature Extraction

For each utterance:

- Load mono wav.
- Extract F0 using `librosa.pyin` initially.
- Store frame-level F0 with timestamps.
- Keep voiced/unvoiced mask.
- Compute utterance-level summary features.

Output:

- `data/aishell3/features/f0_utterance.csv`
- later: per-utterance `.npz` files for dense F0 contours.

### Stage 3: Transcript And Boundary Enrichment

- Transcript and pinyin-with-tone are available from `content.txt`.
- Train prosody boundaries are available from `label_train-set.txt`.
- Current syllable manifest contains:
  - utterance id
  - syllable index
  - pinyin
  - tone label
  - previous/current/next tone context
  - word/phrase boundary if available
  - empty `start_sec`, `end_sec`, `duration_sec` placeholders for alignment output

Next boundary step:

- Run forced alignment using the pinyin transcript.
- Fill `start_sec`, `end_sec`, and `duration_sec` in `data/aishell3/syllable_manifest.csv`.
- Slice each syllable's F0 contour from the utterance-level F0 `.npz`.

## Model Plan

### Baselines

1. `F0-only Transformer`
   - Input: one syllable normalized F0 sequence.
   - Purpose: measure pure contour signal.

2. `F0 + context`
   - Input: previous/current/next syllable F0.
   - Purpose: reproduce the core idea from the short-term context paper.

3. `F0 + context + duration`
   - Adds syllable duration and relative duration.
   - Purpose: improve T3, T5, reduced syllables, and boundary-adjacent tones.

### Proposed Model

```text
Syllable F0 sequence
  -> contour Transformer encoder

Duration / rhythm tokens
  -> rhythm encoder

Boundary features
  -> boundary embedding

Context window
  -> cross-attention fusion

Classifier
  -> T1/T2/T3/T4/T5
```

Recommended context sizes:

- Start with `left=1, right=1`.
- Test `left=2, right=2`.
- Avoid long context until short context is saturated, because prior work suggests short-term context gives most of the gain.

## Ablation Plan

| Experiment | Input | Purpose |
|---|---|---|
| A0 | F0 only | Minimal pitch contour baseline |
| A1 | F0 + context | Test coarticulation modeling |
| A2 | F0 + duration | Test rhythm/duration contribution |
| A3 | F0 + boundary | Test F0 reset and phrase boundary effects |
| A4 | F0 + context + duration | Strong practical baseline |
| A5 | Full model | Final proposed model |
| A6 | Full model without speaker normalization | Verify normalization importance |

## Evaluation Plan

Primary metrics:

- Overall accuracy
- Macro F1
- Per-tone precision/recall/F1

Critical slices:

- T2 vs T3 confusion
- T3 sandhi candidates
- T5 / neutral tone
- Boundary-adjacent syllables
- Cross-speaker generalization

## Implementation Order

1. Finish utterance manifest with audio metadata and annotations. Done.
2. Extract utterance-level F0 features. Scaffold done; 100-sample run done.
3. Build syllable-level manifest with tone/context/prosody labels. Done.
4. Add forced alignment to fill syllable start/end/duration.
   - Temporary baseline done: `scripts/mfa/build_approx_syllable_boundaries.py`.
   - The baseline trims each utterance to the detected voiced F0 span when an F0 summary is available, then assigns equal-duration syllable windows.
   - This is good enough to validate model input shape, but it should not be treated as final alignment quality.
5. Slice syllable-level F0 contours.
   - Scaffold done: `scripts/features/slice_syllable_f0.py`.
   - Current smoke output: `data/aishell3/features/syllable_f0_train100.csv` and `data/aishell3/features/syllable_f0_train100.npz`.
   - Current train100 result: 1,265 syllable rows, 40-point F0 contour, no NaN, 68 empty-contour rows after voiced-probability filtering.
6. Train F0-only baseline.
   - Scaffold done: `scripts/data/make_tone_splits.py`, `scripts/training/tone_dataset.py`, `scripts/training/train_f0_transformer.py`.
   - Current smoke run: `runs/f0_transformer_train100_smoke/metrics.json` and `runs/f0_transformer_train100_smoke/best.pt`.
   - The training script uses raw `f0_hz` and computes normalization from the train split only. It rejects precomputed `f0_speaker_norm` to avoid validation leakage.
   - Current smoke result, 100 utterances only: best validation accuracy `0.3459`, best macro F1 `0.1730`.
   - Best checkpoint has degenerate per-tone behavior: it predicts only T2/T4, with zero validation F1 for T1/T3/T5. This is only a pipeline check because the current data has one speaker, approximate syllable boundaries, and a small sample.
7. Add context/duration/boundary features.
   - Scaffold done: `scripts/training/train_f0_structured_transformer.py` and `ContextSyllableF0Dataset` in `scripts/training/tone_dataset.py`.
   - Inputs: previous/current/next syllable F0 contours as three context tokens, plus duration, relative syllable position, word boundary, phrase boundary, and prev/next context masks.
   - The model uses a `context_mask` so missing prev/next syllables are excluded from Transformer attention and pooling.
   - Current smoke run: `runs/f0_structured_transformer_train100_smoke/metrics.json` and `runs/f0_structured_transformer_train100_smoke/best.pt`.
   - Current smoke result, 100 utterances only: best-macro-F1 checkpoint has validation accuracy `0.3596` and macro F1 `0.2176`. The highest validation accuracy during this smoke run was `0.3664`.
   - Per-tone F1 remains weak: T1 `0.1818`, T2 `0.4384`, T3 `0.0000`, T4 `0.4679`, T5 `0.0000`. This is still a pipeline check, not a final model result.
8. Run ablations and error analysis.
   - Full local train baseline done on same-speaker utterance split.
   - F0-only best checkpoint: `runs/f0_transformer_train_full/best.pt`, epoch 17, validation accuracy `0.4088`, macro F1 `0.2999`.
   - F0-only per-tone F1: T1 `0.4527`, T2 `0.3692`, T3 `0.1821`, T4 `0.4956`, T5 `0.0000`.
   - F0 + context/duration/boundary best checkpoint: `runs/f0_structured_transformer_train_full/best.pt`, epoch 9, validation accuracy `0.4451`, macro F1 `0.3794`.
   - Structured per-tone F1: T1 `0.5112`, T2 `0.4109`, T3 `0.2540`, T4 `0.4985`, T5 `0.2225`.
   - Structured improves macro F1 by about `+0.0795` absolute over F0-only on this split.
   - Tri-tone log-mel ResNet-style best checkpoint: `runs/mel_resnet_train_full/best.pt`, epoch 9, validation accuracy `0.5234`, macro F1 `0.4747`.
   - Mel ResNet per-tone F1: T1 `0.5796`, T2 `0.5153`, T3 `0.3699`, T4 `0.6011`, T5 `0.3074`.
   - Mel ResNet improves macro F1 by about `+0.0953` absolute over the F0 + context/duration/boundary model on this split.
   - Mel ResNet overfits after epoch 9: training accuracy continues rising, but validation macro F1 drops to `0.4091` by epoch 20. Report `best.pt`, not the final epoch.
   - End-to-End-style C1 mel context ResNet checkpoint: `runs/mel_context_resnet_train_full/best.pt`, epoch 19, validation accuracy `0.6459`, macro F1 `0.6182`.
   - C1 input uses previous/current/next log-mel segments plus duration, relative syllable position, word boundary, phrase boundary, and prev/next presence flags. Syllable embedding was disabled for this run to avoid lexical shortcut.
   - C1 per-tone F1: T1 `0.6937`, T2 `0.6444`, T3 `0.5036`, T4 `0.6786`, T5 `0.5706`.
   - C1 improves macro F1 by about `+0.1435` absolute over the single-syllable/tri-tone mel ResNet baseline on this split, and by about `+0.2388` over the F0 + context/duration/boundary model.
   - C1 still shows the main weak slice at T3: validation recall is `0.4522`, with many T3 tokens predicted as T2 or T4.
   - Speaker-disjoint C1 split done with 80 train speakers and 20 validation speakers, no speaker or utterance overlap.
   - Speaker-disjoint C1 best checkpoint: `runs/mel_context_resnet_train_full_speaker_split/best.pt`, epoch 15 by macro F1, validation accuracy `0.6172`, macro F1 `0.5837`.
   - Speaker-disjoint C1 highest validation accuracy was epoch 19 at `0.6285`, but macro F1 was lower at `0.5811`; report epoch 15 for macro-F1-selected checkpoint.
   - Speaker-disjoint C1 per-tone F1: T1 `0.6683`, T2 `0.6271`, T3 `0.4705`, T4 `0.6485`, T5 `0.5042`.
   - Compared with same-speaker C1, speaker-disjoint C1 drops by about `-0.0287` accuracy and `-0.0345` macro F1. The model still generalizes substantially better than the earlier F0 and mel baselines, but T3 and T5 remain the weakest slices.
   - Caveat: this is a same-speaker held-out-utterance validation split, not speaker-disjoint testing. It is useful for baseline comparison, but it likely overestimates unseen-speaker performance.
   - Caveat: Mel ResNet and C1 use log-mel windows, so they are a different input family from F0-only. C1 is closer to the End-to-End paper baseline, but still uses approximate syllable boundaries rather than ASR/forced-alignment boundaries.

## Current Feature Files

| File | Purpose |
|---|---|
| `data/aishell3/syllable_manifest_approx_train100.csv` | Syllable manifest with approximate voiced-span/equal-duration boundaries for the 100 utterances that currently have F0. |
| `data/aishell3/features/syllable_f0_train100.csv` | One row per syllable with tone, context labels, boundary labels, duration, F0 stats, and contour index. |
| `data/aishell3/features/syllable_f0_train100.npz` | Dense arrays: raw `f0_hz` and speaker-normalized `f0_speaker_norm`, shape `(1265, 40)`. |
| `data/aishell3/features/syllable_f0_train100_split.csv` | Train/validation split by utterance, with empty-contour rows dropped. |
| `runs/f0_transformer_train100_smoke/metrics.json` | F0-only Transformer smoke-run metrics. |
| `runs/f0_transformer_train100_smoke/best.pt` | Best smoke-run checkpoint, including train-only normalization stats. |
| `runs/f0_structured_transformer_train100_smoke/metrics.json` | F0 + context/duration/boundary Transformer smoke-run metrics. |
| `runs/f0_structured_transformer_train100_smoke/best.pt` | Best structured smoke-run checkpoint, including F0 normalization and structured feature metadata. It is not a complete standalone provenance bundle. |
| `data/aishell3/features/syllable_f0_train_full.csv` | Full train syllable-level F0 features before empty-contour dropping. |
| `data/aishell3/features/syllable_f0_train_full.npz` | Full train dense F0 contours, shape `(112353, 40)`. |
| `data/aishell3/features/syllable_f0_train_full_split.csv` | Full train/validation split after dropping empty-contour rows, 67,184 train and 16,881 validation syllables. |
| `runs/f0_transformer_train_full/metrics.json` | Full F0-only Transformer validation metrics. |
| `runs/f0_transformer_train_full/best.pt` | Full F0-only best checkpoint. |
| `runs/f0_structured_transformer_train_full/metrics.json` | Full F0 + context/duration/boundary Transformer validation metrics. |
| `runs/f0_structured_transformer_train_full/best.pt` | Full structured best checkpoint. |
| `data/aishell3/features/logmel_utterance_train_full.csv` | Full train utterance-level log-mel summary, 9,802 utterances. |
| `data/aishell3/features/logmel_npz_train_full` | Full train dense log-mel NPZ files, one per utterance. |
| `runs/mel_resnet_train_full/metrics.json` | Full tri-tone log-mel ResNet-style validation metrics. |
| `runs/mel_resnet_train_full/best.pt` | Best tri-tone log-mel ResNet-style checkpoint. |
| `runs/mel_context_resnet_train_full/metrics.json` | Full End-to-End-style C1 mel context ResNet validation metrics. |
| `runs/mel_context_resnet_train_full/best.pt` | Best End-to-End-style C1 mel context ResNet checkpoint. |
| `data/aishell3/features/syllable_f0_train_full_speaker_split.csv` | Speaker-disjoint full train/validation split after dropping empty-contour rows. |
| `runs/mel_context_resnet_train_full_speaker_split/metrics.json` | Full speaker-disjoint End-to-End-style C1 validation metrics. |
| `runs/mel_context_resnet_train_full_speaker_split/best.pt` | Best speaker-disjoint End-to-End-style C1 checkpoint. |

Next implementation target:

- Build a speaker-disjoint split to estimate unseen-speaker performance.
- Replace approximate syllable boundaries with forced alignment when available.
- Then rerun A0/A1/A2/A3/A4 ablations with the same split and alignment.

## Long-Run Status

- Full local train F0 extraction completed.
- Target manifest: `data/aishell3/manifest_train_full.csv`.
- Target utterances: 9,802 train utterances across 100 speakers, about 9.58 audio hours.
- F0 summary output: `data/aishell3/features/f0_utterance_train_full.csv`.
- Dense F0 output directory: `data/aishell3/features/f0_npz_train_full`.
- Log: `runs/f0_extract_train_full.log`.
- The extraction is resumable and guarded by a metadata sidecar with manifest SHA-256 and F0 extraction parameters.
