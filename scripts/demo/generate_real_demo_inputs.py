#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from scripts.common.hearing_pipeline_utils import read_csv_rows, write_csv_rows


DEFAULT_UTT_IDS = [
    "SSB04340409",  # 烟火
    "SSB01930055",  # 地图
    "SSB04070277",  # 在乎
    "SSB00180299",  # 关机
    "SSB04340151",  # 女人
    "SSB04070042",  # 李静
    "SSB04070450",  # 波浪
    "SSB06060485",  # 今日
]


def select_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_utt = {row["utt_id"]: row for row in rows}
    selected = [by_utt[utt_id] for utt_id in DEFAULT_UTT_IDS if utt_id in by_utt]
    if len(selected) >= limit:
        return selected[:limit]
    fallback = [
        row
        for row in rows
        if row["utt_id"] not in {picked["utt_id"] for picked in selected}
        and int(row.get("syllable_count") or 0) == 2
        and float(row.get("duration_sec") or 0.0) <= 1.4
    ]
    return (selected + fallback)[:limit]


def duration_bounds(duration_sec: float) -> tuple[str, str]:
    min_sec = max(0.60, duration_sec * 0.70)
    max_sec = max(1.20, duration_sec * 1.35)
    return f"{min_sec:.2f}", f"{max_sec:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real AISHELL-3 based demo item bank and patient audio manifest.")
    parser.add_argument("--manifest", default="data/aishell3/manifest.csv")
    parser.add_argument("--out-dir", default="data/hearing_real_demo")
    parser.add_argument("--patient-id", default="P_REAL_001")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    rows = [
        row
        for row in read_csv_rows(args.manifest)
        if row.get("split") == args.split
        and int(row.get("syllable_count") or 0) == 2
        and row.get("audio_path")
        and Path(row["audio_path"]).exists()
    ]
    selected = select_rows(rows, args.limit)
    if not selected:
        raise ValueError("no AISHELL-3 rows matched the requested criteria")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    item_rows = []
    patient_rows = []
    for row in selected:
        duration_min, duration_max = duration_bounds(float(row["duration_sec"]))
        item_rows.append(
            {
                "item_id": row["utt_id"],
                "text": row["hanzi"],
                "pinyin": row["pinyin_tone"],
                "surface_tones": row["tone_sequence"],
                "keywords": row["hanzi"],
                "material_type": "雙音節詞",
                "difficulty": "real_demo",
                "syllable_count": row["syllable_count"],
                "expected_duration_min_sec": duration_min,
                "expected_duration_max_sec": duration_max,
                "expected_speech_rate_min": "1.4",
                "expected_speech_rate_max": "5.5",
                "source_utt_id": row["utt_id"],
                "source_audio_path": row["audio_path"],
            }
        )
        patient_rows.append(
            {
                "patient_id": args.patient_id,
                "item_id": row["utt_id"],
                "audio_path": row["audio_path"],
                "recording_quality": "good",
            }
        )

    write_csv_rows(
        out_dir / "item_bank_real.csv",
        [
            "item_id",
            "text",
            "pinyin",
            "surface_tones",
            "keywords",
            "material_type",
            "difficulty",
            "syllable_count",
            "expected_duration_min_sec",
            "expected_duration_max_sec",
            "expected_speech_rate_min",
            "expected_speech_rate_max",
            "source_utt_id",
            "source_audio_path",
        ],
        item_rows,
    )
    write_csv_rows(
        out_dir / "patient_responses_seed.csv",
        ["patient_id", "item_id", "audio_path", "recording_quality"],
        patient_rows,
    )

    session_payload = {
        "patient_id": args.patient_id,
        "test_date": "2026-05-27",
        "stimulus_source": "Qwen3-TTS",
        "materials": "AISHELL-3 雙音節詞",
        "condition": "安靜",
        "snr": "N/A",
        "playback_level": "65 dB SPL",
        "input_mode": "口語複誦",
        "analysis_model": "Local Whisper ASR + Mel Context Tone Model",
        "source_corpus": "AISHELL-3",
        "selected_utt_ids": [row["utt_id"] for row in selected],
    }
    (out_dir / "patient_session_real.json").write_text(
        json.dumps(session_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote={out_dir}")
    print(f"items={len(item_rows)}")


if __name__ == "__main__":
    main()
