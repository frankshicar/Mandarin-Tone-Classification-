#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def char_tokens(text: str) -> list[str]:
    return [char for char in text.strip() if not char.isspace()]


def expected_tokens_for_row(row: dict[str, str]) -> list[str]:
    if row.get("mfa_text"):
        return row["mfa_text"].split()
    return char_tokens(row.get("hanzi", ""))


def tone_of(pinyin: str) -> str:
    match = re.search(r"([1-5])$", pinyin)
    return match.group(1) if match else ""


def parse_prosody(pinyin_prosody: str, syllable_count: int) -> tuple[list[bool], list[bool]]:
    word_boundary_after = [False] * syllable_count
    phrase_boundary_after = [False] * syllable_count
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
        current_idx += 1
    return word_boundary_after, phrase_boundary_after


def parse_textgrid_intervals(path: Path, tier_names: set[str]) -> list[tuple[float, float, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    intervals: list[tuple[float, float, str]] = []
    in_tier = False
    current_name = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        name_match = re.match(r'name = "(.*)"', line)
        if name_match:
            current_name = name_match.group(1)
            in_tier = current_name in tier_names
            i += 1
            continue
        if in_tier and line.startswith("intervals ["):
            xmin = xmax = text = None
            i += 1
            while i < len(lines):
                item = lines[i].strip()
                if item.startswith("intervals [") or item.startswith("item ["):
                    i -= 1
                    break
                if item.startswith("xmin ="):
                    xmin = float(item.split("=", 1)[1].strip())
                elif item.startswith("xmax ="):
                    xmax = float(item.split("=", 1)[1].strip())
                elif item.startswith("text ="):
                    text = item.split("=", 1)[1].strip().strip('"')
                    break
                i += 1
            if xmin is not None and xmax is not None and text is not None:
                intervals.append((xmin, xmax, text))
        i += 1
    if intervals:
        return intervals

    # Minimal fallback for short TextGrid files: xmin/xmax/text triples without labels.
    triples = []
    for idx, line in enumerate(lines):
        if line.strip().startswith('"') and idx >= 2:
            try:
                xmin = float(lines[idx - 2].strip())
                xmax = float(lines[idx - 1].strip())
            except ValueError:
                continue
            triples.append((xmin, xmax, line.strip().strip('"')))
    return triples


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def textgrid_path(textgrid_dir: Path, speaker: str, utt_id: str) -> Path:
    nested = textgrid_dir / speaker / f"{utt_id}.TextGrid"
    if nested.exists():
        return nested
    return textgrid_dir / f"{utt_id}.TextGrid"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MFA TextGrid word intervals into AISHELL-3 syllable boundary CSV.")
    parser.add_argument("--manifest", default="data/aishell3/mfa/corpus_train_full_strict_map.csv")
    parser.add_argument("--textgrid-dir", default="data/aishell3/mfa/aligned_train_full")
    parser.add_argument("--out", default="data/aishell3/syllable_manifest_mfa_train_full.csv")
    parser.add_argument("--tier", action="append", default=["words", "word"])
    parser.add_argument("--utterance-limit", type=int, default=0)
    parser.add_argument("--mismatch-out", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tier_names = set(args.tier)
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
        "alignment_source",
    ]

    total = 0
    processed_utterances = 0
    missing = 0
    mismatched = 0
    mismatch_out = Path(args.mismatch_out) if args.mismatch_out else out.with_suffix(".mismatches.csv")
    mismatch_out.parent.mkdir(parents=True, exist_ok=True)
    mismatch_fields = [
        "speaker",
        "utt_id",
        "reason",
        "expected_count",
        "actual_count",
        "expected_tokens",
        "actual_tokens",
        "textgrid_path",
    ]
    with out.open("w", newline="", encoding="utf-8") as g, mismatch_out.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(g, fieldnames=fieldnames)
        mismatch_writer = csv.DictWriter(h, fieldnames=mismatch_fields)
        writer.writeheader()
        mismatch_writer.writeheader()
        for row in manifest:
            if args.utterance_limit and processed_utterances >= args.utterance_limit:
                break
            pinyins = row["pinyin_tone"].split()
            tones = [tone_of(pinyin) for pinyin in pinyins]
            expected_tokens = expected_tokens_for_row(row)
            tg_path = textgrid_path(Path(args.textgrid_dir), row["speaker"], row["utt_id"])
            if not tg_path.exists():
                missing += 1
                mismatch_writer.writerow(
                    {
                        "speaker": row["speaker"],
                        "utt_id": row["utt_id"],
                        "reason": "missing_textgrid",
                        "expected_count": len(pinyins),
                        "actual_count": 0,
                        "expected_tokens": " ".join(expected_tokens),
                        "actual_tokens": "",
                        "textgrid_path": str(tg_path),
                    }
                )
                continue
            intervals = [
                (start, end, text)
                for start, end, text in parse_textgrid_intervals(tg_path, tier_names)
                if text.strip() and text.strip() not in {"<eps>", "sil", "sp", "spn"}
            ]
            if len(intervals) != len(pinyins):
                mismatched += 1
                mismatch_writer.writerow(
                    {
                        "speaker": row["speaker"],
                        "utt_id": row["utt_id"],
                        "reason": "pinyin_interval_count_mismatch",
                        "expected_count": len(pinyins),
                        "actual_count": len(intervals),
                        "expected_tokens": " ".join(expected_tokens),
                        "actual_tokens": " ".join(text for _, _, text in intervals),
                        "textgrid_path": str(tg_path),
                    }
                )
                continue
            actual_tokens = [text.strip() for _, _, text in intervals]
            if expected_tokens and actual_tokens != expected_tokens:
                mismatched += 1
                mismatch_writer.writerow(
                    {
                        "speaker": row["speaker"],
                        "utt_id": row["utt_id"],
                        "reason": "textgrid_label_mismatch",
                        "expected_count": len(expected_tokens),
                        "actual_count": len(actual_tokens),
                        "expected_tokens": " ".join(expected_tokens),
                        "actual_tokens": " ".join(actual_tokens),
                        "textgrid_path": str(tg_path),
                    }
                )
                continue
            word_boundary_after, phrase_boundary_after = parse_prosody(row.get("pinyin_prosody", ""), len(pinyins))
            for idx, ((start, end, _text), pinyin, tone) in enumerate(zip(intervals, pinyins, tones)):
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
                        "has_boundary": str(word_boundary_after[idx] or phrase_boundary_after[idx]).lower(),
                        "start_sec": f"{start:.6f}",
                        "end_sec": f"{end:.6f}",
                        "duration_sec": f"{max(0.0, end - start):.6f}",
                        "alignment_source": "mfa",
                    }
                )
                total += 1
            processed_utterances += 1

    print(f"wrote={out}")
    print(f"rows={total}")
    print(f"processed_utterances={processed_utterances}")
    print(f"missing_textgrid_utterances={missing}")
    print(f"mismatched_utterances={mismatched}")
    print(f"wrote_mismatches={mismatch_out}")
    if args.strict and (missing or mismatched):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
