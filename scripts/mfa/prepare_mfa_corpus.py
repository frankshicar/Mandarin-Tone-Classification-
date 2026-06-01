#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from pathlib import Path


def char_tokens(text: str) -> list[str]:
    return [char for char in text.strip() if not char.isspace()]


def hanzi_tokens_for_pinyin(text: str, pinyin_count: int) -> list[str]:
    tokens = char_tokens(text)
    if len(tokens) == pinyin_count:
        return tokens
    merged: list[str] = []
    for token in tokens:
        if token in {"儿", "兒"} and merged and len(tokens) - len(merged) >= pinyin_count:
            merged[-1] += token
        else:
            merged.append(token)
    return merged


def write_audio_link(src: Path, dst: Path, copy_audio: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_audio:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an AISHELL-3 corpus directory for Montreal Forced Aligner.")
    parser.add_argument("--manifest", default="data/aishell3/manifest_train_full.csv")
    parser.add_argument("--out", default="data/aishell3/mfa/corpus_train_full")
    parser.add_argument("--map-out", default="data/aishell3/mfa/corpus_train_full_map.csv")
    parser.add_argument("--skipped-out", default="")
    parser.add_argument("--allowed-words", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--copy-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out)
    map_out = Path(args.map_out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    map_out.parent.mkdir(parents=True, exist_ok=True)
    skipped_out = Path(args.skipped_out) if args.skipped_out else map_out.with_suffix(".skipped.csv")
    skipped_out.parent.mkdir(parents=True, exist_ok=True)
    allowed_words = None
    if args.allowed_words:
        allowed_words = {line.strip() for line in Path(args.allowed_words).read_text(encoding="utf-8").splitlines() if line.strip()}

    map_fields = [
        "split",
        "speaker",
        "utt_id",
        "audio_path",
        "mfa_wav_path",
        "mfa_lab_path",
        "hanzi",
        "mfa_text",
        "pinyin_tone",
        "pinyin_prosody",
        "syllable_count",
    ]
    skipped_fields = [
        "split",
        "speaker",
        "utt_id",
        "audio_path",
        "hanzi",
        "pinyin_tone",
        "hanzi_token_count",
        "pinyin_count",
        "reason",
        "detail",
    ]
    written = 0
    skipped = 0
    with (
        manifest.open(newline="", encoding="utf-8") as f,
        map_out.open("w", newline="", encoding="utf-8") as g,
        skipped_out.open("w", newline="", encoding="utf-8") as h,
    ):
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=map_fields)
        skipped_writer = csv.DictWriter(h, fieldnames=skipped_fields)
        writer.writeheader()
        skipped_writer.writeheader()
        for row in reader:
            if args.limit and written >= args.limit:
                break
            audio_path = Path(row["audio_path"])
            pinyins = row["pinyin_tone"].split()
            hanzi_tokens = hanzi_tokens_for_pinyin(row["hanzi"], len(pinyins))

            reason = ""
            detail = ""
            if not audio_path.exists():
                reason = "missing_audio"
                detail = str(audio_path)
            elif len(hanzi_tokens) != len(pinyins):
                reason = "hanzi_pinyin_count_mismatch"
                detail = f"hanzi_tokens={' '.join(hanzi_tokens)}"
            elif allowed_words is not None:
                missing_words = [token for token in hanzi_tokens if token not in allowed_words]
                if missing_words:
                    reason = "missing_dictionary_word"
                    detail = " ".join(sorted(set(missing_words)))

            if reason:
                skipped += 1
                skipped_writer.writerow(
                    {
                        "split": row["split"],
                        "speaker": row["speaker"],
                        "utt_id": row["utt_id"],
                        "audio_path": row["audio_path"],
                        "hanzi": row["hanzi"],
                        "pinyin_tone": row["pinyin_tone"],
                        "hanzi_token_count": len(hanzi_tokens),
                        "pinyin_count": len(pinyins),
                        "reason": reason,
                        "detail": detail,
                    }
                )
                continue

            speaker_dir = out / row["speaker"]
            speaker_dir.mkdir(parents=True, exist_ok=True)
            wav_dst = speaker_dir / f"{row['utt_id']}.wav"
            lab_dst = speaker_dir / f"{row['utt_id']}.lab"
            write_audio_link(audio_path, wav_dst, args.copy_audio)
            mfa_text = " ".join(hanzi_tokens)
            lab_dst.write_text(mfa_text + "\n", encoding="utf-8")
            writer.writerow(
                {
                    "split": row["split"],
                    "speaker": row["speaker"],
                    "utt_id": row["utt_id"],
                    "audio_path": row["audio_path"],
                    "mfa_wav_path": str(wav_dst),
                    "mfa_lab_path": str(lab_dst),
                        "hanzi": row["hanzi"],
                        "mfa_text": mfa_text,
                        "pinyin_tone": row["pinyin_tone"],
                        "pinyin_prosody": row.get("pinyin_prosody", ""),
                        "syllable_count": row["syllable_count"],
                    }
                )
            written += 1

    print(f"wrote_corpus={out}")
    print(f"wrote_map={map_out}")
    print(f"wrote_skipped={skipped_out}")
    print(f"written={written}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
