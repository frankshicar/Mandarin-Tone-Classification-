#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


TONES = ["1", "2", "3", "4", "5"]


def make_group_split(groups: list[str], val_ratio: float, seed: int) -> dict[str, str]:
    group_ids = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    val_count = max(1, round(len(group_ids) * val_ratio))
    val_groups = set(group_ids[:val_count])
    return {group_id: "val" if group_id in val_groups else "train" for group_id in group_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation split for tone baseline.")
    parser.add_argument("--features", default="data/aishell3/features/syllable_f0_train100.csv")
    parser.add_argument("--out", default="data/aishell3/features/syllable_f0_train100_split.csv")
    parser.add_argument("--meta-out", default="data/aishell3/features/syllable_f0_train100_split_meta.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--drop-empty-contour", action="store_true")
    parser.add_argument("--split-unit", choices=["utterance", "speaker"], default="utterance")
    args = parser.parse_args()

    features = Path(args.features)
    out = Path(args.out)
    meta_out = Path(args.meta_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    with features.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.drop_empty_contour:
        rows = [row for row in rows if int(row["voiced_frames"]) > 0]
    if not rows:
        raise ValueError(f"no usable rows in {features}; check input path or empty-contour filtering")

    bad_tones = sorted({row["tone"] for row in rows if row["tone"] not in TONES})
    if bad_tones:
        raise ValueError(f"unexpected tone labels: {bad_tones}")

    split_key = "speaker" if args.split_unit == "speaker" else "utt_id"
    missing_split_keys = [i for i, row in enumerate(rows) if not row.get(split_key)]
    if missing_split_keys:
        raise ValueError(f"{len(missing_split_keys)} rows are missing {split_key}; first={missing_split_keys[:5]}")
    split_by_group = make_group_split([row[split_key] for row in rows], val_ratio=args.val_ratio, seed=args.seed)
    fieldnames = [name for name in rows[0].keys() if name != "data_split"] + ["data_split"]
    split_counts = Counter()
    tone_by_split = {"train": Counter(), "val": Counter()}
    utt_by_split = {"train": set(), "val": set()}
    speaker_by_split = {"train": set(), "val": set()}

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data_split = split_by_group[row[split_key]]
            split_counts[data_split] += 1
            tone_by_split[data_split][row["tone"]] += 1
            utt_by_split[data_split].add(row["utt_id"])
            speaker_by_split[data_split].add(row.get("speaker", ""))
            clean_row = {name: row[name] for name in fieldnames if name != "data_split"}
            writer.writerow({**clean_row, "data_split": data_split})

    meta = {
        "features": str(features),
        "out": str(out),
        "rows": len(rows),
        "split_unit": args.split_unit,
        "split_groups": len(split_by_group),
        "utterances": len({row["utt_id"] for row in rows}),
        "speakers": len({row["speaker"] for row in rows if row.get("speaker")}),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "drop_empty_contour": args.drop_empty_contour,
        "split_counts": dict(split_counts),
        "utterance_counts": {key: len(value) for key, value in utt_by_split.items()},
        "speaker_counts": {key: len(value) for key, value in speaker_by_split.items()},
        "speaker_overlap": len(speaker_by_split["train"] & speaker_by_split["val"]),
        "tone_by_split": {key: dict(value) for key, value in tone_by_split.items()},
        "tone_to_class": {tone: idx for idx, tone in enumerate(TONES)},
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote={out}")
    print(f"wrote_meta={meta_out}")
    print(f"rows={len(rows)}")
    print(f"split_unit={args.split_unit}")
    print(f"split_groups={len(split_by_group)}")
    print(f"utterances={meta['utterances']}")
    print(f"speakers={meta['speakers']}")
    print(f"speaker_overlap={meta['speaker_overlap']}")
    print(f"split_counts={dict(split_counts)}")
    print(f"tone_by_split={meta['tone_by_split']}")


if __name__ == "__main__":
    main()
