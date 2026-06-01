#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDNAMES = [
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
    "asr_pinyin",
    "asr_base_syllable",
    "reference_base_syllable",
    "asr_position_match",
    "asr_utterance_match",
    "asr_token_count",
    "asr_confidence",
    "asr_model",
    "asr_text",
    "asr_unit",
]

MISMATCH_FIELDNAMES = [
    "split",
    "speaker",
    "utt_id",
    "reason",
    "reference_count",
    "asr_count",
    "timed_unit_count",
    "reference_pinyin",
    "asr_pinyin",
    "asr_text",
    "asr_confidence",
    "asr_model",
    "whisperx_error",
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_whisperx_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    by_utt: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            utt_id = str(payload.get("utt_id", ""))
            if utt_id:
                by_utt[utt_id] = payload
    return by_utt


def tone_of(pinyin: str) -> str:
    return pinyin[-1] if pinyin and pinyin[-1] in "12345" else ""


def split_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = re.sub(r"[|,;/]+", " ", value.strip())
    return [token for token in normalized.split() if token]


def normalize_pinyin_token(token: str) -> str:
    cleaned = token.strip().lower().replace("u:", "v").replace("\u00fc", "v")
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def parse_pinyin_tokens(value: str | None) -> list[str]:
    return [token for token in (normalize_pinyin_token(part) for part in split_tokens(value)) if token]


def base_syllable(token: str | None) -> str:
    normalized = normalize_pinyin_token(token or "")
    return re.sub(r"[1-5]$", "", normalized)


def parse_prosody(pinyin_prosody: str, syllable_count: int) -> tuple[list[bool], list[bool]]:
    word_boundary_after = [False] * syllable_count
    phrase_boundary_after = [False] * syllable_count
    current_idx = -1
    for tok in (pinyin_prosody or "").split():
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


def text_to_pinyin(text: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pypinyin is required to convert WhisperX timed text to numbered pinyin.") from exc

    tokens = lazy_pinyin(
        text or "",
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    return " ".join(parse_pinyin_tokens(" ".join(tokens)))


def timed_unit_text(unit: dict[str, Any]) -> str:
    return str(unit.get("char") or unit.get("word") or unit.get("text") or "").strip()


def valid_timed_unit(unit: dict[str, Any]) -> bool:
    if unit.get("start") is None or unit.get("end") is None:
        return False
    text = timed_unit_text(unit)
    return bool(text and not re.fullmatch(r"[\s,.;:!?，。！？；：「」『』（）()【】\[\]{}<>\"'`~_|/-]+", text))


def iter_segment_chars(payload: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for segment in payload.get("segments", []):
        for key in ("chars", "char_segments"):
            for char in segment.get(key, []) or []:
                if valid_timed_unit(char):
                    units.append(char)
    if not units:
        for char in payload.get("char_segments", []) or []:
            if valid_timed_unit(char):
                units.append(char)
    return units


def iter_word_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    units = [unit for unit in payload.get("word_segments", []) or [] if valid_timed_unit(unit)]
    if units:
        return units
    for segment in payload.get("segments", []):
        for word in segment.get("words", []) or []:
            if valid_timed_unit(word):
                units.append(word)
    return units


def timed_units(payload: dict[str, Any], timing_source: str) -> tuple[list[dict[str, Any]], str]:
    if timing_source == "char":
        return iter_segment_chars(payload), "char"
    if timing_source == "word":
        return iter_word_segments(payload), "word"
    chars = iter_segment_chars(payload)
    if chars:
        return chars, "char"
    return iter_word_segments(payload), "word"


def write_mismatch(
    writer: csv.DictWriter,
    row: dict[str, str],
    reason: str,
    ref_pinyins: list[str],
    asr_pinyins: list[str],
    timed_count: int,
) -> None:
    writer.writerow(
        {
            "split": row.get("split", ""),
            "speaker": row.get("speaker", ""),
            "utt_id": row.get("utt_id", ""),
            "reason": reason,
            "reference_count": len(ref_pinyins),
            "asr_count": len(asr_pinyins),
            "timed_unit_count": timed_count,
            "reference_pinyin": " ".join(ref_pinyins),
            "asr_pinyin": " ".join(asr_pinyins),
            "asr_text": row.get("asr_text", ""),
            "asr_confidence": row.get("asr_confidence", ""),
            "asr_model": row.get("asr_model", ""),
            "whisperx_error": row.get("whisperx_error", ""),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert WhisperX char/word alignment JSONL into the MFA-compatible syllable manifest schema. "
            "For Mandarin, char timing is preferred because one Hanzi usually maps to one syllable."
        )
    )
    parser.add_argument("--manifest", default="data/aishell3/manifest_train_full_whisperx.csv")
    parser.add_argument("--whisperx-jsonl", default="data/aishell3/whisperx_train_full.jsonl")
    parser.add_argument("--out", default="data/aishell3/syllable_manifest_whisperx_char.csv")
    parser.add_argument("--mismatch-out", default="")
    parser.add_argument("--match-policy", choices=["count", "exact"], default="count")
    parser.add_argument("--timing-source", choices=["auto", "char", "word"], default="auto")
    args = parser.parse_args()

    manifest_rows = load_csv_rows(Path(args.manifest))
    whisperx_by_utt = load_whisperx_jsonl(Path(args.whisperx_jsonl))
    out = Path(args.out)
    mismatch_out = Path(args.mismatch_out) if args.mismatch_out else out.with_suffix(".mismatches.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    mismatch_out.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    utterances_written = 0
    mismatches = 0
    with out.open("w", newline="", encoding="utf-8") as g, mismatch_out.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(g, fieldnames=FIELDNAMES)
        mismatch_writer = csv.DictWriter(h, fieldnames=MISMATCH_FIELDNAMES)
        writer.writeheader()
        mismatch_writer.writeheader()

        for row in manifest_rows:
            utt_id = row.get("utt_id", "")
            payload = whisperx_by_utt.get(utt_id)
            ref_pinyins = parse_pinyin_tokens(row.get("pinyin_tone", ""))
            ref_bases = [base_syllable(token) for token in ref_pinyins]
            if not payload:
                mismatches += 1
                write_mismatch(mismatch_writer, row, "missing_whisperx_jsonl", ref_pinyins, [], 0)
                continue
            if payload.get("status") and payload.get("status") != "ok":
                mismatches += 1
                row_with_error = dict(row)
                row_with_error["whisperx_error"] = str(payload.get("error", ""))
                write_mismatch(mismatch_writer, row_with_error, f"whisperx_{payload['status']}", ref_pinyins, [], 0)
                continue

            units, source = timed_units(payload, args.timing_source)
            asr_text = str(payload.get("text") or row.get("asr_text", ""))
            unit_pinyins = [text_to_pinyin(timed_unit_text(unit)) for unit in units]
            asr_pinyins = parse_pinyin_tokens(" ".join(unit_pinyins)) or parse_pinyin_tokens(payload.get("pinyin") or row.get("asr_pinyin", ""))
            asr_bases = [base_syllable(token) for token in asr_pinyins]
            reason = ""
            if not asr_pinyins:
                reason = "missing_asr_pinyin"
            elif len(ref_pinyins) != len(asr_pinyins):
                reason = "syllable_count_mismatch"
            elif len(units) != len(asr_pinyins):
                reason = "timed_unit_count_mismatch"
            elif args.match_policy == "exact" and ref_bases != asr_bases:
                reason = "syllable_sequence_mismatch"

            if reason:
                mismatches += 1
                write_mismatch(mismatch_writer, row, reason, ref_pinyins, asr_pinyins, len(units))
                continue

            tones = [tone_of(pinyin) for pinyin in ref_pinyins]
            word_boundary_after, phrase_boundary_after = parse_prosody(row.get("pinyin_prosody", ""), len(ref_pinyins))
            utterance_match = ref_bases == asr_bases
            for idx, (ref_pinyin, ref_base, asr_pinyin, asr_base, tone, unit) in enumerate(
                zip(ref_pinyins, ref_bases, asr_pinyins, asr_bases, tones, units)
            ):
                start = float(unit["start"])
                end = float(unit["end"])
                prev_tone = tones[idx - 1] if idx > 0 else "BOS"
                next_tone = tones[idx + 1] if idx + 1 < len(tones) else "EOS"
                writer.writerow(
                    {
                        "split": row.get("split", ""),
                        "speaker": row.get("speaker", ""),
                        "utt_id": utt_id,
                        "audio_path": row.get("audio_path", ""),
                        "syllable_index": idx,
                        "syllable_count": len(ref_pinyins),
                        "pinyin": ref_pinyin,
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
                        "alignment_source": f"whisperx_{source}",
                        "asr_pinyin": asr_pinyin,
                        "asr_base_syllable": asr_base,
                        "reference_base_syllable": ref_base,
                        "asr_position_match": str(ref_base == asr_base).lower(),
                        "asr_utterance_match": str(utterance_match).lower(),
                        "asr_token_count": len(asr_pinyins),
                        "asr_confidence": row.get("asr_confidence", ""),
                        "asr_model": row.get("asr_model", payload.get("model", "")),
                        "asr_text": asr_text,
                        "asr_unit": timed_unit_text(unit),
                    }
                )
                rows_written += 1
            utterances_written += 1

    print(f"wrote={out}")
    print(f"rows={rows_written}")
    print(f"utterances={utterances_written}")
    print(f"mismatched_utterances={mismatches}")
    print(f"wrote_mismatches={mismatch_out}")


if __name__ == "__main__":
    main()
