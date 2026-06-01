#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

from scripts.common.hearing_pipeline_utils import (
    compute_audio_metrics,
    ensure_parent,
    format_ratio,
    load_audio_mono,
    load_item_bank,
    parse_pinyin_tokens,
    parse_tone_sequence,
    read_csv_rows,
    safe_float,
    utc_now_iso,
    write_csv_rows,
    write_json,
)


def evaluate_candidate(candidate: dict[str, str], item: dict | None, args: argparse.Namespace) -> dict:
    result = {
        "item_id": candidate.get("item_id", ""),
        "text": item["text"] if item else "",
        "audio_path": candidate.get("audio_path", ""),
        "tts_engine": candidate.get("tts_engine", "Qwen3-TTS"),
        "asr_text": candidate.get("asr_text", ""),
        "asr_pinyin": candidate.get("asr_pinyin", ""),
        "predicted_tones": candidate.get("predicted_tones", ""),
        "asr_confidence": candidate.get("asr_confidence", ""),
        "tone_confidence": candidate.get("tone_confidence", ""),
        "sample_rate": "",
        "channels": "",
        "audio_subtype": "",
        "duration_sec": "",
        "voiced_duration_sec": "",
        "leading_silence_ms": "",
        "trailing_silence_ms": "",
        "rms_dbfs": "",
        "peak_dbfs": "",
        "clipped_samples": "",
        "clipped_ratio": "",
        "speech_rate_syllable_per_sec": "",
        "content_match_mode": "",
        "content_match": "",
        "tone_match": "",
        "qc_status": "",
        "qc_reasons": "",
    }
    hard_reasons = []
    soft_reasons = []
    expected_syllables = item["syllable_count_value"] if item else 0

    if item is None:
        hard_reasons.append("missing_item_bank_row")
    audio_path = Path(candidate.get("audio_path", ""))
    if not audio_path.exists():
        hard_reasons.append("missing_audio_file")

    metrics = None
    if not hard_reasons:
        samples, sample_rate, channels, subtype = load_audio_mono(audio_path)
        metrics = compute_audio_metrics(
            samples,
            sample_rate,
            silence_threshold_dbfs=args.silence_threshold_dbfs,
            clip_abs=args.clip_abs,
        )
        result["sample_rate"] = str(sample_rate)
        result["channels"] = str(channels)
        result["audio_subtype"] = subtype
        for name, value in metrics.items():
            if isinstance(value, float):
                result[name] = format_ratio(value)
            else:
                result[name] = str(value)

        speech_rate = expected_syllables / metrics["voiced_duration_sec"] if metrics["voiced_duration_sec"] > 0 else 0.0
        result["speech_rate_syllable_per_sec"] = format_ratio(speech_rate)

        if sample_rate != args.expected_sample_rate:
            hard_reasons.append("unexpected_sample_rate")
        if channels != args.expected_channels:
            hard_reasons.append("unexpected_channel_count")
        if metrics["voiced_duration_sec"] <= 0.0:
            hard_reasons.append("no_detected_speech_span")
        if metrics["clipped_ratio"] >= args.fail_clipped_ratio:
            hard_reasons.append("excessive_clipping")
        elif metrics["clipped_ratio"] >= args.review_clipped_ratio:
            soft_reasons.append("mild_clipping")
        if metrics["rms_dbfs"] <= args.fail_min_rms_dbfs:
            hard_reasons.append("audio_too_quiet")
        elif metrics["rms_dbfs"] <= args.review_min_rms_dbfs:
            soft_reasons.append("rms_near_lower_bound")
        if metrics["leading_silence_ms"] >= args.fail_silence_ms or metrics["trailing_silence_ms"] >= args.fail_silence_ms:
            hard_reasons.append("excessive_silence_padding")
        elif metrics["leading_silence_ms"] >= args.review_silence_ms or metrics["trailing_silence_ms"] >= args.review_silence_ms:
            soft_reasons.append("silence_padding_needs_review")

        duration_min = safe_float(item.get("expected_duration_min_sec"), args.default_duration_min_sec) if item else args.default_duration_min_sec
        duration_max = safe_float(item.get("expected_duration_max_sec"), args.default_duration_max_sec) if item else args.default_duration_max_sec
        if duration_min is not None and metrics["duration_sec"] < duration_min:
            hard_reasons.append("duration_too_short")
        if duration_max is not None and metrics["duration_sec"] > duration_max:
            hard_reasons.append("duration_too_long")
        speech_rate_min = safe_float(item.get("expected_speech_rate_min"), args.default_speech_rate_min) if item else args.default_speech_rate_min
        speech_rate_max = safe_float(item.get("expected_speech_rate_max"), args.default_speech_rate_max) if item else args.default_speech_rate_max
        if speech_rate_min is not None and speech_rate < speech_rate_min:
            hard_reasons.append("speech_rate_too_slow")
        if speech_rate_max is not None and speech_rate > speech_rate_max:
            hard_reasons.append("speech_rate_too_fast")

    predicted_pinyin = parse_pinyin_tokens(candidate.get("asr_pinyin"))
    predicted_tones = parse_tone_sequence(candidate.get("predicted_tones"))
    if item:
        if predicted_pinyin:
            result["content_match_mode"] = "pinyin"
            content_match = predicted_pinyin == item["pinyin_tokens"]
            result["content_match"] = str(content_match).lower()
            if not content_match:
                hard_reasons.append("asr_pinyin_mismatch")
        elif candidate.get("asr_text"):
            result["content_match_mode"] = "text_only"
            content_match = candidate.get("asr_text", "").strip() == item["text"]
            result["content_match"] = str(content_match).lower()
            if not content_match:
                hard_reasons.append("asr_text_mismatch")
        else:
            result["content_match_mode"] = "missing_asr"
            result["content_match"] = "unknown"
            soft_reasons.append("missing_asr_verification")

        if predicted_tones:
            tone_match = predicted_tones == item["tones"]
            result["tone_match"] = str(tone_match).lower()
            if not tone_match:
                hard_reasons.append("tone_sequence_mismatch")
        else:
            result["tone_match"] = "unknown"
            soft_reasons.append("missing_tone_verification")

    asr_confidence = safe_float(candidate.get("asr_confidence"))
    if asr_confidence is not None and asr_confidence < args.review_asr_confidence:
        soft_reasons.append("low_asr_confidence")
    tone_confidence = safe_float(candidate.get("tone_confidence"))
    if tone_confidence is not None and tone_confidence < args.review_tone_confidence:
        soft_reasons.append("low_tone_confidence")

    if hard_reasons:
        status = "FAIL"
    elif soft_reasons:
        status = "REVIEW"
    else:
        status = "PASS"
    result["qc_status"] = status
    result["qc_reasons"] = ";".join(hard_reasons + soft_reasons)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated QC for Qwen3-TTS stimulus audio.")
    parser.add_argument("--item-bank", default="data/hearing_demo/item_bank_demo.csv")
    parser.add_argument("--candidates", default="data/hearing_demo/tts_candidates_demo.csv")
    parser.add_argument("--out-csv", default="output/hearing_demo/tts_qc_report.csv")
    parser.add_argument("--out-summary", default="output/hearing_demo/tts_qc_summary.json")
    parser.add_argument("--out-approved", default="output/hearing_demo/approved_stimuli.csv")
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument("--expected-channels", type=int, default=1)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-40.0)
    parser.add_argument("--clip-abs", type=float, default=0.999)
    parser.add_argument("--review-clipped-ratio", type=float, default=0.0001)
    parser.add_argument("--fail-clipped-ratio", type=float, default=0.0050)
    parser.add_argument("--review-min-rms-dbfs", type=float, default=-30.0)
    parser.add_argument("--fail-min-rms-dbfs", type=float, default=-40.0)
    parser.add_argument("--review-silence-ms", type=float, default=300.0)
    parser.add_argument("--fail-silence-ms", type=float, default=600.0)
    parser.add_argument("--default-duration-min-sec", type=float, default=0.6)
    parser.add_argument("--default-duration-max-sec", type=float, default=2.5)
    parser.add_argument("--default-speech-rate-min", type=float, default=1.5)
    parser.add_argument("--default-speech-rate-max", type=float, default=6.5)
    parser.add_argument("--review-asr-confidence", type=float, default=0.80)
    parser.add_argument("--review-tone-confidence", type=float, default=0.75)
    args = parser.parse_args()

    items = load_item_bank(args.item_bank)
    candidates = read_csv_rows(args.candidates)
    report_rows = [evaluate_candidate(candidate, items.get(candidate.get("item_id", "").strip()), args) for candidate in candidates]

    fieldnames = [
        "item_id",
        "text",
        "audio_path",
        "tts_engine",
        "asr_text",
        "asr_pinyin",
        "predicted_tones",
        "asr_confidence",
        "tone_confidence",
        "sample_rate",
        "channels",
        "audio_subtype",
        "duration_sec",
        "voiced_duration_sec",
        "leading_silence_ms",
        "trailing_silence_ms",
        "rms_dbfs",
        "peak_dbfs",
        "clipped_samples",
        "clipped_ratio",
        "speech_rate_syllable_per_sec",
        "content_match_mode",
        "content_match",
        "tone_match",
        "qc_status",
        "qc_reasons",
    ]
    write_csv_rows(args.out_csv, fieldnames, report_rows)

    approved_rows = [row for row in report_rows if row["qc_status"] == "PASS"]
    write_csv_rows(args.out_approved, ["item_id", "text", "audio_path", "tts_engine"], approved_rows)

    status_counts = Counter(row["qc_status"] for row in report_rows)
    reason_counts = Counter()
    for row in report_rows:
        for reason in row["qc_reasons"].split(";"):
            if reason:
                reason_counts[reason] += 1

    summary = {
        "generated_at": utc_now_iso(),
        "item_bank": str(Path(args.item_bank)),
        "candidates": str(Path(args.candidates)),
        "rows_total": len(report_rows),
        "status_counts": dict(status_counts),
        "approved_count": len(approved_rows),
        "reason_counts": dict(reason_counts.most_common()),
        "outputs": {
            "report_csv": str(ensure_parent(args.out_csv)),
            "summary_json": str(ensure_parent(args.out_summary)),
            "approved_csv": str(ensure_parent(args.out_approved)),
        },
    }
    write_json(args.out_summary, summary)
    print(f"wrote={args.out_csv}")
    print(f"approved={len(approved_rows)}")
    print(f"status_counts={dict(status_counts)}")


if __name__ == "__main__":
    main()
