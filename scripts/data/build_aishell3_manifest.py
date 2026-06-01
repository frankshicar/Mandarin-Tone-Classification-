#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import soundfile as sf


def load_content(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            utt_file, text = line.split("\t", 1)
            utt_id = Path(utt_file).stem
            tokens = text.split()
            chars = tokens[0::2]
            pinyins = tokens[1::2]
            rows[utt_id] = {
                "hanzi": "".join(chars),
                "pinyin_tone": " ".join(pinyins),
                "syllable_count": str(len(pinyins)),
                "tone_sequence": " ".join(extract_tone(pinyin) for pinyin in pinyins),
            }
    return rows


def load_prosody(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            utt_id, pinyin_prosody, hanzi_prosody = parts[:3]
            pinyins = [tok for tok in pinyin_prosody.split() if tok not in {"%", "$"}]
            rows[utt_id] = {
                "pinyin_prosody": pinyin_prosody,
                "hanzi_prosody": hanzi_prosody,
                "prosody_tone_sequence": " ".join(extract_tone(pinyin) for pinyin in pinyins),
                "prosody_syllable_count": str(len(pinyins)),
                "prosody_word_boundary_count": str(pinyin_prosody.count("%")),
                "prosody_phrase_boundary_count": str(pinyin_prosody.count("$")),
            }
    return rows


def load_speaker_info(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            speaker, age_group, gender, accent = line.split()
            rows[speaker] = {
                "age_group": age_group,
                "gender": gender,
                "accent": accent,
            }
    return rows


def extract_tone(pinyin: str) -> str:
    match = re.search(r"([1-5])$", pinyin)
    return match.group(1) if match else ""


def iter_wavs(root: Path, with_audio_info: bool):
    content = {
        "train": load_content(root / "train" / "content.txt"),
        "test": load_content(root / "test" / "content.txt"),
    }
    train_prosody = load_prosody(root / "train" / "label_train-set.txt")
    speaker_info = load_speaker_info(root / "spk-info.txt")

    for split in ("train", "test"):
        wav_root = root / split / "wav"
        for wav in sorted(wav_root.glob("*/*.wav")):
            speaker = wav.parent.name
            utt_id = wav.stem
            text_row = content.get(split, {}).get(utt_id, {})
            prosody_row = train_prosody.get(utt_id, {}) if split == "train" else {}
            spk_row = speaker_info.get(speaker, {})
            row = {
                "split": split,
                "speaker": speaker,
                "utt_id": utt_id,
                "audio_path": str(wav),
                "hanzi": text_row.get("hanzi", ""),
                "pinyin_tone": text_row.get("pinyin_tone", ""),
                "tone_sequence": text_row.get("tone_sequence", ""),
                "syllable_count": text_row.get("syllable_count", ""),
                "pinyin_prosody": prosody_row.get("pinyin_prosody", ""),
                "hanzi_prosody": prosody_row.get("hanzi_prosody", ""),
                "prosody_tone_sequence": prosody_row.get("prosody_tone_sequence", ""),
                "prosody_syllable_count": prosody_row.get("prosody_syllable_count", ""),
                "prosody_word_boundary_count": prosody_row.get("prosody_word_boundary_count", ""),
                "prosody_phrase_boundary_count": prosody_row.get("prosody_phrase_boundary_count", ""),
                "age_group": spk_row.get("age_group", ""),
                "gender": spk_row.get("gender", ""),
                "accent": spk_row.get("accent", ""),
                "has_transcript": str(bool(text_row)).lower(),
                "has_pinyin": str(bool(text_row.get("pinyin_tone"))).lower(),
                "has_tone_label": str(bool(text_row.get("tone_sequence"))).lower(),
                "has_boundary": str(bool(prosody_row)).lower(),
            }
            if with_audio_info:
                info = sf.info(wav)
                row.update(
                    {
                        "sample_rate": info.samplerate,
                        "num_frames": info.frames,
                        "duration_sec": f"{info.frames / info.samplerate:.6f}",
                        "channels": info.channels,
                    }
                )
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV manifest for AISHELL-3 wav files.")
    parser.add_argument("--root", default="data/aishell3/raw", help="AISHELL-3 raw snapshot directory")
    parser.add_argument("--out", default="data/aishell3/manifest.csv", help="Output CSV path")
    parser.add_argument("--no-audio-info", action="store_true", help="Skip reading wav metadata")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_wavs(root, with_audio_info=not args.no_audio_info))
    fieldnames = [
        "split",
        "speaker",
        "utt_id",
        "audio_path",
        "sample_rate",
        "num_frames",
        "duration_sec",
        "channels",
        "hanzi",
        "pinyin_tone",
        "tone_sequence",
        "syllable_count",
        "pinyin_prosody",
        "hanzi_prosody",
        "prosody_tone_sequence",
        "prosody_syllable_count",
        "prosody_word_boundary_count",
        "prosody_phrase_boundary_count",
        "age_group",
        "gender",
        "accent",
        "has_transcript",
        "has_pinyin",
        "has_tone_label",
        "has_boundary",
    ]
    if args.no_audio_info:
        fieldnames = [name for name in fieldnames if name not in {"sample_rate", "num_frames", "duration_sec", "channels"}]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    speaker_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        speaker_counts.setdefault(row["split"], set()).add(row["speaker"])

    print(f"wrote={out}")
    print(f"rows={len(rows)}")
    for split in ("train", "test"):
        print(f"{split}_rows={split_counts.get(split, 0)}")
        print(f"{split}_speakers={len(speaker_counts.get(split, set()))}")


if __name__ == "__main__":
    main()
