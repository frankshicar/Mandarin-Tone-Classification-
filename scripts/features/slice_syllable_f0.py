#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_f0_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["utt_id"]: row for row in csv.DictReader(f)}


def interpolate_contour(times: np.ndarray, values: np.ndarray, start: float, end: float, length: int) -> np.ndarray:
    targets = np.linspace(start, end, num=length, endpoint=True, dtype=np.float32)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    if values.size == 1:
        return np.full(length, float(values[0]), dtype=np.float32)
    return np.interp(targets, times.astype(np.float64), values.astype(np.float64)).astype(np.float32)


def safe_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice utterance-level F0 into syllable-level contour features.")
    parser.add_argument("--syllable-manifest", default="data/aishell3/syllable_manifest_approx.csv")
    parser.add_argument("--f0-summary", default="data/aishell3/features/f0_utterance_train100.csv")
    parser.add_argument("--out", default="data/aishell3/features/syllable_f0_train100.csv")
    parser.add_argument("--npz-out", default="data/aishell3/features/syllable_f0_train100.npz")
    parser.add_argument("--contour-length", type=int, default=40)
    parser.add_argument("--voiced-prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-f0", type=float, default=70.0)
    parser.add_argument("--max-f0", type=float, default=500.0)
    args = parser.parse_args()

    syllable_manifest = Path(args.syllable_manifest)
    f0_summary = Path(args.f0_summary)
    out = Path(args.out)
    npz_out = Path(args.npz_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_out.parent.mkdir(parents=True, exist_ok=True)

    f0_index = load_f0_index(f0_summary)
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    speaker_values: dict[str, list[float]] = defaultdict(list)
    feature_rows: list[dict[str, str]] = []
    contours: list[np.ndarray] = []
    missing_f0 = 0
    empty_contour = 0

    fieldnames = [
        "split",
        "speaker",
        "utt_id",
        "syllable_index",
        "syllable_count",
        "pinyin",
        "tone",
        "prev_tone",
        "next_tone",
        "tri_tone",
        "word_boundary_after",
        "phrase_boundary_after",
        "start_sec",
        "end_sec",
        "duration_sec",
        "voiced_frames",
        "voiced_ratio",
        "f0_mean_hz",
        "f0_median_hz",
        "f0_min_hz",
        "f0_max_hz",
        "f0_std_hz",
        "contour_index",
    ]

    with syllable_manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            f0_row = f0_index.get(row["utt_id"])
            start = safe_float(row.get("start_sec", ""))
            end = safe_float(row.get("end_sec", ""))
            if f0_row is None or start is None or end is None:
                missing_f0 += 1
                continue

            npz_path = f0_row["f0_npz_path"]
            if npz_path not in npz_cache:
                with np.load(npz_path) as data:
                    npz_cache[npz_path] = {key: data[key] for key in data.files}
            data = npz_cache[npz_path]

            times = data["times"]
            f0_hz = data["f0_hz"]
            voiced = data["voiced"].astype(bool)
            voiced_prob = data["voiced_prob"]
            mask = (
                (times >= start)
                & (times < end)
                & voiced
                & (voiced_prob >= args.voiced_prob_threshold)
                & (f0_hz >= args.min_f0)
                & (f0_hz <= args.max_f0)
            )
            values = f0_hz[mask]
            value_times = times[mask]
            contour = interpolate_contour(value_times, values, start, end, args.contour_length)
            contour_index = len(contours)
            contours.append(contour)

            voiced_frames = int(values.size)
            total_frames = int(np.sum((times >= start) & (times < end)))
            if voiced_frames == 0:
                empty_contour += 1
            else:
                speaker_values[row["speaker"]].extend(float(v) for v in values)

            stats = {
                "voiced_frames": str(voiced_frames),
                "voiced_ratio": f"{(voiced_frames / total_frames) if total_frames else 0.0:.6f}",
                "f0_mean_hz": "",
                "f0_median_hz": "",
                "f0_min_hz": "",
                "f0_max_hz": "",
                "f0_std_hz": "",
            }
            if voiced_frames:
                stats.update(
                    {
                        "f0_mean_hz": f"{float(np.mean(values)):.6f}",
                        "f0_median_hz": f"{float(np.median(values)):.6f}",
                        "f0_min_hz": f"{float(np.min(values)):.6f}",
                        "f0_max_hz": f"{float(np.max(values)):.6f}",
                        "f0_std_hz": f"{float(np.std(values)):.6f}",
                    }
                )

            feature_rows.append(
                {
                    "split": row["split"],
                    "speaker": row["speaker"],
                    "utt_id": row["utt_id"],
                    "syllable_index": row["syllable_index"],
                    "syllable_count": row["syllable_count"],
                    "pinyin": row["pinyin"],
                    "tone": row["tone"],
                    "prev_tone": row["prev_tone"],
                    "next_tone": row["next_tone"],
                    "tri_tone": row["tri_tone"],
                    "word_boundary_after": row["word_boundary_after"],
                    "phrase_boundary_after": row["phrase_boundary_after"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "duration_sec": row["duration_sec"],
                    "contour_index": str(contour_index),
                    **stats,
                }
            )

    speaker_norm = {}
    for speaker, values in speaker_values.items():
        arr = np.asarray(values, dtype=np.float32)
        speaker_norm[speaker] = (float(np.mean(arr)), float(np.std(arr) if np.std(arr) > 1e-6 else 1.0))

    contours_arr = np.stack(contours).astype(np.float32) if contours else np.empty((0, args.contour_length), dtype=np.float32)
    contours_norm = np.zeros_like(contours_arr)
    for row in feature_rows:
        speaker = row["speaker"]
        idx = int(row["contour_index"])
        mean, std = speaker_norm.get(speaker, (0.0, 1.0))
        contours_norm[idx] = (contours_arr[idx] - mean) / std if mean else contours_arr[idx]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    np.savez_compressed(npz_out, f0_hz=contours_arr, f0_speaker_norm=contours_norm)

    print(f"wrote={out}")
    print(f"wrote_npz={npz_out}")
    print(f"rows={len(feature_rows)}")
    print(f"missing_f0_or_boundary_rows={missing_f0}")
    print(f"empty_contour_rows={empty_contour}")
    print(f"speakers_with_norm={len(speaker_norm)}")


if __name__ == "__main__":
    main()
