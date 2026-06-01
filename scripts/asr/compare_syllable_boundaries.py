#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


SUMMARY_FIELDNAMES = [
    "metric",
    "value",
]

DETAIL_FIELDNAMES = [
    "utt_id",
    "syllable_index",
    "speaker",
    "reference_start_sec",
    "candidate_start_sec",
    "start_error_ms",
    "abs_start_error_ms",
    "reference_end_sec",
    "candidate_end_sec",
    "end_error_ms",
    "abs_end_error_ms",
    "reference_duration_sec",
    "candidate_duration_sec",
    "duration_error_ms",
    "abs_duration_error_ms",
    "reference_pinyin",
    "candidate_pinyin",
    "asr_pinyin",
    "asr_position_match",
    "reference_source",
    "candidate_source",
]


def normalize_pinyin_token(token: str) -> str:
    cleaned = token.strip().lower().replace("u:", "v").replace("\u00fc", "v")
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def base_syllable(token: str | None) -> str:
    normalized = normalize_pinyin_token(token or "")
    return re.sub(r"[1-5]$", "", normalized)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[str, int]:
    return row["utt_id"], int(row["syllable_index"])


def safe_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(values: list[float], prefix: str) -> list[tuple[str, str]]:
    if not values:
        return [(f"{prefix}_count", "0")]
    sorted_values = sorted(values)
    p90_index = min(len(sorted_values) - 1, int(round(0.90 * (len(sorted_values) - 1))))
    p95_index = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
    return [
        (f"{prefix}_count", str(len(values))),
        (f"{prefix}_mean_ms", f"{statistics.fmean(values):.3f}"),
        (f"{prefix}_median_ms", f"{statistics.median(values):.3f}"),
        (f"{prefix}_p90_ms", f"{sorted_values[p90_index]:.3f}"),
        (f"{prefix}_p95_ms", f"{sorted_values[p95_index]:.3f}"),
        (f"{prefix}_max_ms", f"{max(values):.3f}"),
    ]


def ratio(numerator: int, denominator: int) -> str:
    return f"{(numerator / denominator) if denominator else 0.0:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate syllable timing against an MFA syllable manifest.")
    parser.add_argument("--reference", default="data/aishell3/syllable_manifest_mfa_train_full_strict.csv")
    parser.add_argument("--candidate", default="data/aishell3/syllable_manifest_asr_equal_duration.csv")
    parser.add_argument("--summary-out", default="data/aishell3/asr_vs_mfa_boundary_summary.csv")
    parser.add_argument("--detail-out", default="data/aishell3/asr_vs_mfa_boundary_detail.csv")
    parser.add_argument("--threshold-ms", type=float, action="append", default=[20.0, 50.0, 100.0])
    args = parser.parse_args()

    reference_rows = load_rows(Path(args.reference))
    candidate_rows = load_rows(Path(args.candidate))
    reference_by_key = {key(row): row for row in reference_rows}
    candidate_by_key = {key(row): row for row in candidate_rows}
    shared_keys = sorted(set(reference_by_key) & set(candidate_by_key))

    details: list[dict[str, str]] = []
    abs_start_errors: list[float] = []
    abs_end_errors: list[float] = []
    abs_duration_errors: list[float] = []
    position_matches = 0
    comparable_rows = 0

    for item_key in shared_keys:
        ref = reference_by_key[item_key]
        cand = candidate_by_key[item_key]
        ref_start = safe_float(ref, "start_sec")
        ref_end = safe_float(ref, "end_sec")
        ref_duration = safe_float(ref, "duration_sec")
        cand_start = safe_float(cand, "start_sec")
        cand_end = safe_float(cand, "end_sec")
        cand_duration = safe_float(cand, "duration_sec")
        if None in {ref_start, ref_end, ref_duration, cand_start, cand_end, cand_duration}:
            continue

        start_error = (cand_start - ref_start) * 1000.0
        end_error = (cand_end - ref_end) * 1000.0
        duration_error = (cand_duration - ref_duration) * 1000.0
        abs_start = abs(start_error)
        abs_end = abs(end_error)
        abs_duration = abs(duration_error)
        abs_start_errors.append(abs_start)
        abs_end_errors.append(abs_end)
        abs_duration_errors.append(abs_duration)
        comparable_rows += 1

        asr_pinyin = cand.get("asr_pinyin", "")
        asr_position_match = cand.get("asr_position_match", "")
        if asr_position_match == "":
            asr_position_match = str(base_syllable(ref.get("pinyin", "")) == base_syllable(asr_pinyin or cand.get("pinyin", ""))).lower()
        if asr_position_match.lower() == "true":
            position_matches += 1

        details.append(
            {
                "utt_id": ref["utt_id"],
                "syllable_index": ref["syllable_index"],
                "speaker": ref.get("speaker", ""),
                "reference_start_sec": f"{ref_start:.6f}",
                "candidate_start_sec": f"{cand_start:.6f}",
                "start_error_ms": f"{start_error:.3f}",
                "abs_start_error_ms": f"{abs_start:.3f}",
                "reference_end_sec": f"{ref_end:.6f}",
                "candidate_end_sec": f"{cand_end:.6f}",
                "end_error_ms": f"{end_error:.3f}",
                "abs_end_error_ms": f"{abs_end:.3f}",
                "reference_duration_sec": f"{ref_duration:.6f}",
                "candidate_duration_sec": f"{cand_duration:.6f}",
                "duration_error_ms": f"{duration_error:.3f}",
                "abs_duration_error_ms": f"{abs_duration:.3f}",
                "reference_pinyin": ref.get("pinyin", ""),
                "candidate_pinyin": cand.get("pinyin", ""),
                "asr_pinyin": asr_pinyin,
                "asr_position_match": asr_position_match,
                "reference_source": ref.get("alignment_source", ""),
                "candidate_source": cand.get("alignment_source", ""),
            }
        )

    reference_utts = {row["utt_id"] for row in reference_rows}
    candidate_utts = {row["utt_id"] for row in candidate_rows}
    shared_utts = reference_utts & candidate_utts

    summary: list[tuple[str, str]] = [
        ("reference_rows", str(len(reference_rows))),
        ("candidate_rows", str(len(candidate_rows))),
        ("shared_rows", str(len(shared_keys))),
        ("comparable_rows", str(comparable_rows)),
        ("row_coverage", ratio(len(shared_keys), len(reference_rows))),
        ("reference_utterances", str(len(reference_utts))),
        ("candidate_utterances", str(len(candidate_utts))),
        ("shared_utterances", str(len(shared_utts))),
        ("utterance_coverage", ratio(len(shared_utts), len(reference_utts))),
        ("asr_position_match_rate", ratio(position_matches, comparable_rows)),
    ]
    summary.extend(summarize(abs_start_errors, "abs_start_error"))
    summary.extend(summarize(abs_end_errors, "abs_end_error"))
    summary.extend(summarize(abs_duration_errors, "abs_duration_error"))
    for threshold in args.threshold_ms:
        summary.append((f"start_within_{threshold:g}ms", ratio(sum(v <= threshold for v in abs_start_errors), len(abs_start_errors))))
        summary.append((f"end_within_{threshold:g}ms", ratio(sum(v <= threshold for v in abs_end_errors), len(abs_end_errors))))

    summary_out = Path(args.summary_out)
    detail_out = Path(args.detail_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for metric, value in summary:
            writer.writerow({"metric": metric, "value": value})
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(details)

    print(f"wrote={summary_out}")
    print(f"wrote_detail={detail_out}")
    print(f"comparable_rows={comparable_rows}")
    if abs_start_errors:
        print(f"abs_start_error_mean_ms={statistics.fmean(abs_start_errors):.3f}")
        print(f"abs_end_error_mean_ms={statistics.fmean(abs_end_errors):.3f}")
        print(f"asr_position_match_rate={ratio(position_matches, comparable_rows)}")


if __name__ == "__main__":
    main()
