#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def load_durations(manifest_path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("utt_id") and row.get("duration_sec"):
                durations[row["utt_id"]] = float(row["duration_sec"])
    return durations


def load_voiced_spans(f0_summary_path: Path | None, voiced_prob_threshold: float, min_f0: float, max_f0: float):
    if f0_summary_path is None:
        return {}

    spans = {}
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
                if not np.any(mask):
                    continue
                voiced_times = times[mask]
                spans[row["utt_id"]] = (float(voiced_times[0]), float(voiced_times[-1]))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill syllable start/end times with an equal-duration AISHELL-3 baseline."
    )
    parser.add_argument("--syllable-manifest", default="data/aishell3/syllable_manifest.csv")
    parser.add_argument("--utterance-manifest", default="data/aishell3/manifest.csv")
    parser.add_argument("--out", default="data/aishell3/syllable_manifest_approx.csv")
    parser.add_argument("--f0-summary", default=None, help="Optional F0 summary CSV used to trim to voiced span.")
    parser.add_argument(
        "--filter-to-f0-summary",
        action="store_true",
        help="Only write syllables whose utterance appears in --f0-summary.",
    )
    parser.add_argument("--voiced-prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-f0", type=float, default=70.0)
    parser.add_argument("--max-f0", type=float, default=500.0)
    parser.add_argument("--span-pad-sec", type=float, default=0.05)
    args = parser.parse_args()

    syllable_manifest = Path(args.syllable_manifest)
    utterance_manifest = Path(args.utterance_manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.filter_to_f0_summary and not args.f0_summary:
        raise ValueError("--filter-to-f0-summary requires --f0-summary")

    durations = load_durations(utterance_manifest)
    voiced_spans = load_voiced_spans(
        Path(args.f0_summary) if args.f0_summary else None,
        voiced_prob_threshold=args.voiced_prob_threshold,
        min_f0=args.min_f0,
        max_f0=args.max_f0,
    )
    f0_utt_ids = set(voiced_spans)
    if args.filter_to_f0_summary and args.f0_summary:
        with Path(args.f0_summary).open(newline="", encoding="utf-8") as f:
            f0_utt_ids = {row["utt_id"] for row in csv.DictReader(f)}
    missing_duration = 0
    rows_written = 0
    voiced_span_rows = 0
    skipped_by_f0_filter = 0

    with syllable_manifest.open(newline="", encoding="utf-8") as f, out.open(
        "w", newline="", encoding="utf-8"
    ) as g:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        writer = csv.DictWriter(g, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if args.filter_to_f0_summary and row["utt_id"] not in f0_utt_ids:
                skipped_by_f0_filter += 1
                continue
            duration = durations.get(row["utt_id"])
            if duration is None:
                missing_duration += 1
                writer.writerow(row)
                rows_written += 1
                continue

            idx = int(row["syllable_index"])
            count = int(row["syllable_count"])
            utterance_start = 0.0
            utterance_end = duration
            if row["utt_id"] in voiced_spans:
                span_start, span_end = voiced_spans[row["utt_id"]]
                utterance_start = max(0.0, span_start - args.span_pad_sec)
                utterance_end = min(duration, span_end + args.span_pad_sec)
                if utterance_end <= utterance_start:
                    utterance_start = 0.0
                    utterance_end = duration
                else:
                    voiced_span_rows += 1

            usable_duration = utterance_end - utterance_start
            start = utterance_start + usable_duration * idx / count
            end = utterance_start + usable_duration * (idx + 1) / count
            row["start_sec"] = f"{start:.6f}"
            row["end_sec"] = f"{end:.6f}"
            row["duration_sec"] = f"{(end - start):.6f}"
            writer.writerow(row)
            rows_written += 1

    print(f"wrote={out}")
    print(f"rows={rows_written}")
    print(f"missing_duration_rows={missing_duration}")
    print(f"skipped_by_f0_filter={skipped_by_f0_filter}")
    print(f"voiced_span_rows={voiced_span_rows}")
    print("boundary_source=voiced_span_equal_duration_baseline" if voiced_spans else "boundary_source=equal_duration_baseline")


if __name__ == "__main__":
    main()
