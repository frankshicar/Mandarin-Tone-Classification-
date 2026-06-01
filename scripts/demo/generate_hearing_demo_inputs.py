#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def synth_wave(
    path: Path,
    *,
    speech_duration_sec: float,
    leading_silence_sec: float,
    trailing_silence_sec: float,
    amplitude: float,
    frequency_hz: float,
    clip_boost: float = 1.0,
    sample_rate: int = 16000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, speech_duration_sec, int(sample_rate * speech_duration_sec), endpoint=False, dtype=np.float32)
    envelope = np.linspace(0.35, 1.0, len(t), dtype=np.float32)
    speech = amplitude * envelope * np.sin(2.0 * np.pi * frequency_hz * t)
    speech = np.clip(speech * clip_boost, -1.0, 1.0)
    full = np.concatenate(
        [
            np.zeros(int(sample_rate * leading_silence_sec), dtype=np.float32),
            speech.astype(np.float32),
            np.zeros(int(sample_rate * trailing_silence_sec), dtype=np.float32),
        ]
    )
    sf.write(path, full, sample_rate)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo inputs for the Mandarin hearing pipeline.")
    parser.add_argument("--out-dir", default="data/hearing_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio" / "tts"

    synth_wave(audio_dir / "W001.wav", speech_duration_sec=0.95, leading_silence_sec=0.03, trailing_silence_sec=0.05, amplitude=0.22, frequency_hz=190.0)
    synth_wave(audio_dir / "W002.wav", speech_duration_sec=1.05, leading_silence_sec=0.05, trailing_silence_sec=0.42, amplitude=0.18, frequency_hz=240.0)
    synth_wave(audio_dir / "W003.wav", speech_duration_sec=0.88, leading_silence_sec=0.04, trailing_silence_sec=0.04, amplitude=0.25, frequency_hz=300.0, clip_boost=1.7)
    synth_wave(audio_dir / "W004.wav", speech_duration_sec=0.92, leading_silence_sec=0.02, trailing_silence_sec=0.03, amplitude=0.24, frequency_hz=170.0)

    item_bank_rows = [
        {
            "item_id": "W001",
            "text": "青菜",
            "pinyin": "qing1 cai4",
            "surface_tones": "1 4",
            "keywords": "青菜",
            "material_type": "雙音節詞",
            "difficulty": "easy",
            "syllable_count": "2",
            "expected_duration_min_sec": "0.70",
            "expected_duration_max_sec": "1.40",
            "expected_speech_rate_min": "1.6",
            "expected_speech_rate_max": "4.5",
        },
        {
            "item_id": "W002",
            "text": "白兔",
            "pinyin": "bai2 tu4",
            "surface_tones": "2 4",
            "keywords": "白兔",
            "material_type": "雙音節詞",
            "difficulty": "easy",
            "syllable_count": "2",
            "expected_duration_min_sec": "0.70",
            "expected_duration_max_sec": "1.80",
            "expected_speech_rate_min": "1.6",
            "expected_speech_rate_max": "4.5",
        },
        {
            "item_id": "W003",
            "text": "老師",
            "pinyin": "lao3 shi1",
            "surface_tones": "3 1",
            "keywords": "老師",
            "material_type": "雙音節詞",
            "difficulty": "medium",
            "syllable_count": "2",
            "expected_duration_min_sec": "0.70",
            "expected_duration_max_sec": "1.40",
            "expected_speech_rate_min": "1.6",
            "expected_speech_rate_max": "4.5",
        },
        {
            "item_id": "W004",
            "text": "安靜",
            "pinyin": "an1 jing4",
            "surface_tones": "1 4",
            "keywords": "安靜",
            "material_type": "雙音節詞",
            "difficulty": "medium",
            "syllable_count": "2",
            "expected_duration_min_sec": "0.70",
            "expected_duration_max_sec": "1.40",
            "expected_speech_rate_min": "1.6",
            "expected_speech_rate_max": "4.5",
        },
    ]
    write_csv(
        out_dir / "item_bank_demo.csv",
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
        ],
        item_bank_rows,
    )

    tts_rows = [
        {
            "item_id": "W001",
            "audio_path": str(audio_dir / "W001.wav"),
            "tts_engine": "Qwen3-TTS",
            "asr_text": "青菜",
            "asr_pinyin": "qing1 cai4",
            "asr_confidence": "0.96",
            "predicted_tones": "1 4",
            "tone_confidence": "0.93",
        },
        {
            "item_id": "W002",
            "audio_path": str(audio_dir / "W002.wav"),
            "tts_engine": "Qwen3-TTS",
            "asr_text": "白兔",
            "asr_pinyin": "bai2 tu4",
            "asr_confidence": "0.79",
            "predicted_tones": "2 4",
            "tone_confidence": "0.81",
        },
        {
            "item_id": "W003",
            "audio_path": str(audio_dir / "W003.wav"),
            "tts_engine": "Qwen3-TTS",
            "asr_text": "老西",
            "asr_pinyin": "lao3 xi1",
            "asr_confidence": "0.90",
            "predicted_tones": "3 1",
            "tone_confidence": "0.84",
        },
        {
            "item_id": "W004",
            "audio_path": str(audio_dir / "W004.wav"),
            "tts_engine": "Qwen3-TTS",
            "asr_text": "安靜",
            "asr_pinyin": "an1 jing4",
            "asr_confidence": "0.95",
            "predicted_tones": "2 4",
            "tone_confidence": "0.90",
        },
    ]
    write_csv(
        out_dir / "tts_candidates_demo.csv",
        ["item_id", "audio_path", "tts_engine", "asr_text", "asr_pinyin", "asr_confidence", "predicted_tones", "tone_confidence"],
        tts_rows,
    )

    patient_rows = [
        {
            "patient_id": "P001",
            "item_id": "W001",
            "asr_text": "青菜",
            "asr_pinyin": "qing1 cai4",
            "asr_confidence": "0.95",
            "predicted_tones": "1 4",
            "tone_confidence": "0.93",
            "recording_quality": "good",
        },
        {
            "patient_id": "P001",
            "item_id": "W002",
            "asr_text": "白兔",
            "asr_pinyin": "bai3 tu4",
            "asr_confidence": "0.89",
            "predicted_tones": "3 4",
            "tone_confidence": "0.71",
            "recording_quality": "good",
        },
        {
            "patient_id": "P001",
            "item_id": "W003",
            "asr_text": "老西",
            "asr_pinyin": "lao3 xi1",
            "asr_confidence": "0.61",
            "predicted_tones": "3 1",
            "tone_confidence": "0.76",
            "recording_quality": "good",
        },
        {
            "patient_id": "P001",
            "item_id": "W004",
            "asr_text": "安靜",
            "asr_pinyin": "ang1 jing4",
            "asr_confidence": "0.82",
            "predicted_tones": "1 4",
            "tone_confidence": "0.87",
            "recording_quality": "good",
        },
    ]
    write_csv(
        out_dir / "patient_responses_demo.csv",
        ["patient_id", "item_id", "asr_text", "asr_pinyin", "asr_confidence", "predicted_tones", "tone_confidence", "recording_quality"],
        patient_rows,
    )

    session_payload = {
        "patient_id": "P001",
        "test_date": "2026-05-27",
        "stimulus_source": "Qwen3-TTS",
        "materials": "雙音節詞",
        "condition": "安靜",
        "snr": "N/A",
        "playback_level": "65 dB SPL",
        "input_mode": "口語複誦",
        "analysis_model": "ASR + 聲調辨識模型",
    }
    (out_dir / "patient_session_demo.json").write_text(
        json.dumps(session_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={out_dir}")


if __name__ == "__main__":
    main()
