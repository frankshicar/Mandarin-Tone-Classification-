#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


FIELDNAMES = [
    "split",
    "speaker",
    "utt_id",
    "audio_path",
    "syllable_index",
    "syllable_count",
    "pinyin",
    "tone",
    "prev_tone",
    "next_tone",
    "tri_tone",
    "word_boundary_after",
    "phrase_boundary_after",
    "has_boundary",
    "start_sec",
    "end_sec",
    "duration_sec",
    "alignment_source",
    "asr_pinyin",
    "asr_base_syllable",
    "reference_base_syllable",
    "asr_position_match",
    "asr_utterance_match",
    "asr_token_count",
    "asr_confidence",
    "asr_model",
]

MISMATCH_FIELDNAMES = [
    "split",
    "speaker",
    "utt_id",
    "reason",
    "reference_count",
    "asr_count",
    "reference_pinyin",
    "asr_pinyin",
    "asr_text",
    "asr_confidence",
    "asr_model",
]


def split_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = re.sub(r"[|,;/]+", " ", value.strip())
    return [token for token in normalized.split() if token]


def normalize_pinyin_token(token: str) -> str:
    cleaned = token.strip().lower().replace("u:", "v").replace("\u00fc", "v")
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def parse_pinyin_tokens(value: str | None) -> list[str]:
    return [token for token in (normalize_pinyin_token(part) for part in split_tokens(value)) if token]


def base_syllable(token: str | None) -> str:
    normalized = normalize_pinyin_token(token or "")
    return re.sub(r"[1-5]$", "", normalized)


def tone_of(pinyin: str) -> str:
    return pinyin[-1] if pinyin and pinyin[-1] in "12345" else ""


def parse_prosody(pinyin_prosody: str, syllable_count: int) -> tuple[list[bool], list[bool]]:
    word_boundary_after = [False] * syllable_count
    phrase_boundary_after = [False] * syllable_count
    current_idx = -1
    for tok in (pinyin_prosody or "").split():
        if tok == "%":
            if current_idx >= 0:
                word_boundary_after[current_idx] = True
            continue
        if tok == "$":
            if current_idx >= 0:
                phrase_boundary_after[current_idx] = True
            continue
        current_idx += 1
    return word_boundary_after, phrase_boundary_after


def load_voiced_spans(
    f0_summary_path: Path | None,
    voiced_prob_threshold: float,
    min_f0: float,
    max_f0: float,
) -> dict[str, tuple[float, float]]:
    if f0_summary_path is None:
        return {}
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("--f0-summary requires numpy; install project dependencies or omit --f0-summary") from exc

    spans: dict[str, tuple[float, float]] = {}
    with f0_summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            npz_path = row.get("f0_npz_path", "")
            if not npz_path:
                continue
            with np.load(npz_path) as data:
                times = data["times"]
                f0_hz = data["f0_hz"]
                voiced = data["voiced"].astype(bool)
                voiced_prob = data["voiced_prob"]
                mask = (
                    voiced
                    & (voiced_prob >= voiced_prob_threshold)
                    & (f0_hz >= min_f0)
                    & (f0_hz <= max_f0)
                )
                if np.any(mask):
                    voiced_times = times[mask]
                    spans[row["utt_id"]] = (float(voiced_times[0]), float(voiced_times[-1]))
    return spans


def interval_span(row: dict[str, str], voiced_spans: dict[str, tuple[float, float]], span_pad_sec: float) -> tuple[float, float] | None:
    try:
        duration = float(row["duration_sec"])
    except (KeyError, ValueError):
        return None
    start = 0.0
    end = duration
    if row["utt_id"] in voiced_spans:
        voiced_start, voiced_end = voiced_spans[row["utt_id"]]
        start = max(0.0, voiced_start - span_pad_sec)
        end = min(duration, voiced_end + span_pad_sec)
        if end <= start:
            start = 0.0
            end = duration
    return start, end


def should_write(match_policy: str, ref_bases: list[str], asr_bases: list[str]) -> tuple[bool, str]:
    if not asr_bases:
        return False, "missing_asr_pinyin"
    if len(ref_bases) != len(asr_bases):
        return False, "syllable_count_mismatch"
    if match_policy == "exact" and ref_bases != asr_bases:
        return False, "syllable_sequence_mismatch"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an ASR-derived syllable-boundary baseline in the same schema as the MFA syllable manifest. "
            "Without ASR timestamps, timing is an utterance/voiced-span equal-duration baseline gated by ASR syllable count."
        )
    )
    parser.add_argument("--manifest", default="data/aishell3/manifest_asr.csv", help="Utterance CSV enriched by scripts.demo.run_local_asr.")
    parser.add_argument("--out", default="data/aishell3/syllable_manifest_asr_equal_duration.csv")
    parser.add_argument("--mismatch-out", default="")
    parser.add_argument("--asr-pinyin-column", default="asr_pinyin")
    parser.add_argument("--match-policy", choices=["count", "exact"], default="count")
    parser.add_argument("--f0-summary", default="", help="Optional utterance F0 summary CSV used to trim the timing span to voiced frames.")
    parser.add_argument("--voiced-prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-f0", type=float, default=70.0)
    parser.add_argument("--max-f0", type=float, default=500.0)
    parser.add_argument("--span-pad-sec", type=float, default=0.05)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out)
    mismatch_out = Path(args.mismatch_out) if args.mismatch_out else out.with_suffix(".mismatches.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    mismatch_out.parent.mkdir(parents=True, exist_ok=True)

    voiced_spans = load_voiced_spans(
        Path(args.f0_summary) if args.f0_summary else None,
        voiced_prob_threshold=args.voiced_prob_threshold,
        min_f0=args.min_f0,
        max_f0=args.max_f0,
    )

    rows_written = 0
    utterances_written = 0
    mismatches = 0
    missing_duration = 0
    with manifest.open(newline="", encoding="utf-8") as f, out.open("w", newline="", encoding="utf-8") as g, mismatch_out.open(
        "w", newline="", encoding="utf-8"
    ) as h:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=FIELDNAMES)
        mismatch_writer = csv.DictWriter(h, fieldnames=MISMATCH_FIELDNAMES)
        writer.writeheader()
        mismatch_writer.writeheader()

        for row in reader:
            ref_pinyins = parse_pinyin_tokens(row.get("pinyin_tone", ""))
            asr_pinyins = parse_pinyin_tokens(row.get(args.asr_pinyin_column, ""))
            ref_bases = [base_syllable(token) for token in ref_pinyins]
            asr_bases = [base_syllable(token) for token in asr_pinyins]
            writable, reason = should_write(args.match_policy, ref_bases, asr_bases)
            span = interval_span(row, voiced_spans, args.span_pad_sec) if writable else None
            if writable and span is None:
                writable = False
                reason = "missing_duration_sec"
                missing_duration += 1

            if not writable:
                mismatches += 1
                mismatch_writer.writerow(
                    {
                        "split": row.get("split", ""),
                        "speaker": row.get("speaker", ""),
                        "utt_id": row.get("utt_id", ""),
                        "reason": reason,
                        "reference_count": len(ref_pinyins),
                        "asr_count": len(asr_pinyins),
                        "reference_pinyin": " ".join(ref_pinyins),
                        "asr_pinyin": " ".join(asr_pinyins),
                        "asr_text": row.get("asr_text", ""),
                        "asr_confidence": row.get("asr_confidence", ""),
                        "asr_model": row.get("asr_model", ""),
                    }
                )
                continue

            start_span, end_span = span
            usable_duration = end_span - start_span
            tones = [tone_of(pinyin) for pinyin in ref_pinyins]
            word_boundary_after, phrase_boundary_after = parse_prosody(row.get("pinyin_prosody", ""), len(ref_pinyins))
            utterance_match = ref_bases == asr_bases
            for idx, (ref_pinyin, ref_base, asr_pinyin, asr_base, tone) in enumerate(
                zip(ref_pinyins, ref_bases, asr_pinyins, asr_bases, tones)
            ):
                start = start_span + usable_duration * idx / len(ref_pinyins)
                end = start_span + usable_duration * (idx + 1) / len(ref_pinyins)
                prev_tone = tones[idx - 1] if idx > 0 else "BOS"
                next_tone = tones[idx + 1] if idx + 1 < len(tones) else "EOS"
                writer.writerow(
                    {
                        "split": row.get("split", ""),
                        "speaker": row.get("speaker", ""),
                        "utt_id": row.get("utt_id", ""),
                        "audio_path": row.get("audio_path", ""),
                        "syllable_index": idx,
                        "syllable_count": len(ref_pinyins),
                        "pinyin": ref_pinyin,
                        "tone": tone,
                        "prev_tone": prev_tone,
                        "next_tone": next_tone,
                        "tri_tone": f"{prev_tone}-{tone}-{next_tone}",
                        "word_boundary_after": str(word_boundary_after[idx]).lower(),
                        "phrase_boundary_after": str(phrase_boundary_after[idx]).lower(),
                        "has_boundary": str(word_boundary_after[idx] or phrase_boundary_after[idx]).lower(),
                        "start_sec": f"{start:.6f}",
                        "end_sec": f"{end:.6f}",
                        "duration_sec": f"{end - start:.6f}",
                        "alignment_source": "asr_voiced_equal_duration" if voiced_spans else "asr_equal_duration",
                        "asr_pinyin": asr_pinyin,
                        "asr_base_syllable": asr_base,
                        "reference_base_syllable": ref_base,
                        "asr_position_match": str(ref_base == asr_base).lower(),
                        "asr_utterance_match": str(utterance_match).lower(),
                        "asr_token_count": len(asr_pinyins),
                        "asr_confidence": row.get("asr_confidence", ""),
                        "asr_model": row.get("asr_model", ""),
                    }
                )
                rows_written += 1
            utterances_written += 1

    print(f"wrote={out}")
    print(f"rows={rows_written}")
    print(f"utterances={utterances_written}")
    print(f"mismatched_utterances={mismatches}")
    print(f"missing_duration_utterances={missing_duration}")
    print(f"wrote_mismatches={mismatch_out}")


if __name__ == "__main__":
    main()
