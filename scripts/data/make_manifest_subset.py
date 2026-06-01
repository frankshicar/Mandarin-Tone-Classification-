#!/usr/bin/env python3
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced AISHELL-3 manifest subset by speaker.")
    parser.add_argument("--manifest", default="data/aishell3/manifest.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    manifest = Path(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row["split"] == args.split]

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    rng = random.Random(args.seed)
    selected = []
    for speaker in sorted(by_speaker):
        speaker_rows = sorted(by_speaker[speaker], key=lambda row: row["utt_id"])
        rng.shuffle(speaker_rows)
        count = max(1, round(len(speaker_rows) * args.fraction))
        selected.extend(speaker_rows[:count])

    selected.sort(key=lambda row: (row["speaker"], row["utt_id"]))

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    duration_hours = sum(float(row["duration_sec"]) for row in selected) / 3600.0
    print(f"wrote={out}")
    print(f"rows={len(selected)}")
    print(f"speakers={len(by_speaker)}")
    print(f"duration_hours={duration_hours:.3f}")
    print(f"fraction={args.fraction}")


if __name__ == "__main__":
    main()
