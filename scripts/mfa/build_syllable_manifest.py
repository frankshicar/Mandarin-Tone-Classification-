#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def tone_of(pinyin: str) -> str:
    match = re.search(r"([1-5])$", pinyin)
    return match.group(1) if match else ""


def parse_prosody(pinyin_prosody: str):
    if not pinyin_prosody:
        return [], [], []
    pinyins = []
    word_boundary_after = []
    phrase_boundary_after = []
    current_idx = -1
    for tok in pinyin_prosody.split():
        if tok == "%":
            if current_idx >= 0:
                word_boundary_after[current_idx] = True
            continue
        if tok == "$":
            if current_idx >= 0:
                phrase_boundary_after[current_idx] = True
            continue
        pinyins.append(tok)
        word_boundary_after.append(False)
        phrase_boundary_after.append(False)
        current_idx += 1
    return pinyins, word_boundary_after, phrase_boundary_after


def main() -> None:
    parser = argparse.ArgumentParser(description="Build syllable-level manifest from AISHELL-3 utterance manifest.")
    parser.add_argument("--manifest", default="data/aishell3/manifest.csv")
    parser.add_argument("--out", default="data/aishell3/syllable_manifest.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    ]

    total = 0
    with open(args.manifest, newline="", encoding="utf-8") as f, out.open("w", newline="", encoding="utf-8") as g:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            pinyins = row["pinyin_tone"].split()
            word_boundary_after = [False] * len(pinyins)
            phrase_boundary_after = [False] * len(pinyins)
            if row.get("pinyin_prosody"):
                prosody_pinyins, word_boundary_after, phrase_boundary_after = parse_prosody(row["pinyin_prosody"])
                if len(prosody_pinyins) != len(pinyins):
                    word_boundary_after = [False] * len(pinyins)
                    phrase_boundary_after = [False] * len(pinyins)

            tones = [tone_of(pinyin) for pinyin in pinyins]
            for idx, (pinyin, tone) in enumerate(zip(pinyins, tones)):
                prev_tone = tones[idx - 1] if idx > 0 else "BOS"
                next_tone = tones[idx + 1] if idx + 1 < len(tones) else "EOS"
                writer.writerow(
                    {
                        "split": row["split"],
                        "speaker": row["speaker"],
                        "utt_id": row["utt_id"],
                        "audio_path": row["audio_path"],
                        "syllable_index": idx,
                        "syllable_count": len(pinyins),
                        "pinyin": pinyin,
                        "tone": tone,
                        "prev_tone": prev_tone,
                        "next_tone": next_tone,
                        "tri_tone": f"{prev_tone}-{tone}-{next_tone}",
                        "word_boundary_after": str(word_boundary_after[idx]).lower(),
                        "phrase_boundary_after": str(phrase_boundary_after[idx]).lower(),
                        "has_boundary": row.get("has_boundary", "false"),
                        "start_sec": "",
                        "end_sec": "",
                        "duration_sec": "",
                    }
                )
                total += 1

    print(f"wrote={out}")
    print(f"rows={total}")


if __name__ == "__main__":
    main()
