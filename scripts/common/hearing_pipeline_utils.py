#!/usr/bin/env python3
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


INITIALS = sorted(
    [
        "zh",
        "ch",
        "sh",
        "b",
        "p",
        "m",
        "f",
        "d",
        "t",
        "n",
        "l",
        "g",
        "k",
        "h",
        "j",
        "q",
        "x",
        "r",
        "z",
        "c",
        "s",
        "y",
        "w",
    ],
    key=len,
    reverse=True,
)
RETROFLEX_INITIALS = {"zh", "ch", "sh"}
PALATAL_INITIALS = {"j", "q", "x"}
ASPIRATION_GROUPS = [
    {"b", "p"},
    {"d", "t"},
    {"g", "k"},
    {"j", "q"},
    {"zh", "ch"},
    {"z", "c"},
]
TEXT_NORMALIZE_RE = re.compile(r"[\s,.;:!?，。！？；：「」『』（）()【】\[\]{}<>\"'`~_|/-]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    path = ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: str | Path, payload: dict | list) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def format_ratio(value: float) -> str:
    return f"{value:.4f}"


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return TEXT_NORMALIZE_RE.sub("", text).lower()


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


def parse_tone_sequence(value: str | None) -> list[str]:
    parts = split_tokens(value)
    if parts and all(re.fullmatch(r"[1-5]", part) for part in parts):
        return parts
    return [tone_of_token(token) for token in parse_pinyin_tokens(value)]


def parse_keywords(value: str | None) -> list[str]:
    parts = split_tokens(value)
    if not parts and value:
        normalized = value.strip()
        return [normalized] if normalized else []
    return parts


def tone_of_token(token: str | None) -> str:
    if not token:
        return ""
    match = re.search(r"([1-5])$", token)
    return match.group(1) if match else ""


def base_syllable(token: str | None) -> str:
    normalized = normalize_pinyin_token(token or "")
    return re.sub(r"[1-5]$", "", normalized)


def split_initial_final(token: str | None) -> tuple[str, str]:
    base = base_syllable(token)
    for initial in INITIALS:
        if base.startswith(initial):
            final = base[len(initial) :]
            return initial, final
    return "", base


def compute_position_accuracy(
    expected: list[str],
    predicted: list[str],
    transform=None,
) -> tuple[int, int, list[tuple[str | None, str | None, bool]]]:
    transform = transform or (lambda value: value)
    total = max(len(expected), len(predicted))
    pairs = []
    correct = 0
    for index in range(total):
        exp = expected[index] if index < len(expected) else None
        pred = predicted[index] if index < len(predicted) else None
        match = exp is not None and pred is not None and transform(exp) == transform(pred)
        if match:
            correct += 1
        pairs.append((exp, pred, match))
    return correct, total, pairs


def keyword_accuracy(keywords: list[str], predicted_text: str) -> tuple[int, int]:
    if not keywords:
        return 0, 0
    normalized_text = normalize_text(predicted_text)
    hits = 0
    for keyword in keywords:
        if normalize_text(keyword) and normalize_text(keyword) in normalized_text:
            hits += 1
    return hits, len(keywords)


def load_audio_mono(path: str | Path) -> tuple[np.ndarray, int, int, str]:
    info = sf.info(str(path))
    data, sample_rate = sf.read(str(path), always_2d=True)
    channels = data.shape[1]
    mono = data.astype(np.float32).mean(axis=1)
    return mono, sample_rate, channels, info.subtype


def compute_audio_metrics(
    samples: np.ndarray,
    sample_rate: int,
    silence_threshold_dbfs: float = -40.0,
    clip_abs: float = 0.999,
) -> dict[str, float | int]:
    abs_samples = np.abs(samples)
    duration_sec = float(len(samples) / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    peak_abs = float(np.max(abs_samples)) if len(samples) else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    peak_dbfs = 20.0 * math.log10(max(peak_abs, 1e-12))
    clipped_samples = int(np.count_nonzero(abs_samples >= clip_abs))
    clipped_ratio = float(clipped_samples / len(samples)) if len(samples) else 0.0

    silence_threshold = 10 ** (silence_threshold_dbfs / 20.0)
    active = abs_samples >= silence_threshold
    if np.any(active):
        indices = np.flatnonzero(active)
        first = int(indices[0])
        last = int(indices[-1])
        leading_silence_ms = 1000.0 * first / sample_rate
        trailing_silence_ms = 1000.0 * (len(samples) - last - 1) / sample_rate
        voiced_duration_sec = max(0.0, (last - first + 1) / sample_rate)
    else:
        leading_silence_ms = duration_sec * 1000.0
        trailing_silence_ms = duration_sec * 1000.0
        voiced_duration_sec = 0.0

    return {
        "duration_sec": duration_sec,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "clipped_samples": clipped_samples,
        "clipped_ratio": clipped_ratio,
        "leading_silence_ms": leading_silence_ms,
        "trailing_silence_ms": trailing_silence_ms,
        "voiced_duration_sec": voiced_duration_sec,
    }


def is_n_ng_confusion(expected_final: str, predicted_final: str) -> bool:
    if expected_final == predicted_final:
        return False
    if expected_final.endswith("n") and predicted_final.endswith("ng"):
        return expected_final[:-1] == predicted_final[:-2]
    if expected_final.endswith("ng") and predicted_final.endswith("n"):
        return expected_final[:-2] == predicted_final[:-1]
    return False


def is_retroflex_palatal_confusion(expected_initial: str, predicted_initial: str) -> bool:
    if expected_initial == predicted_initial:
        return False
    return (
        expected_initial in RETROFLEX_INITIALS
        and predicted_initial in PALATAL_INITIALS
        or expected_initial in PALATAL_INITIALS
        and predicted_initial in RETROFLEX_INITIALS
    )


def is_aspiration_confusion(expected_initial: str, predicted_initial: str) -> bool:
    if expected_initial == predicted_initial:
        return False
    pair = {expected_initial, predicted_initial}
    return any(pair == group for group in ASPIRATION_GROUPS)


def top_counter_rows(counter: Counter, limit: int = 10, label_names: tuple[str, str] = ("from", "to")) -> list[dict]:
    rows = []
    for key, count in counter.most_common(limit):
        if isinstance(key, tuple) and len(key) == 2:
            rows.append({label_names[0]: key[0], label_names[1]: key[1], "count": count})
        else:
            rows.append({"label": str(key), "count": count})
    return rows


def load_item_bank(path: str | Path) -> dict[str, dict]:
    items = {}
    for row in read_csv_rows(path):
        item_id = (row.get("item_id") or "").strip()
        if not item_id:
            continue
        pinyin_tokens = parse_pinyin_tokens(
            row.get("pinyin")
            or row.get("pinyin_tone")
            or row.get("pinyin_tokens")
        )
        tones = parse_tone_sequence(
            row.get("surface_tones")
            or row.get("tone_sequence")
            or row.get("tones")
            or row.get("lexical_tones")
            or row.get("pinyin")
        )
        initials = []
        finals = []
        for token in pinyin_tokens:
            initial, final = split_initial_final(token)
            initials.append(initial)
            finals.append(final)
        items[item_id] = {
            **row,
            "item_id": item_id,
            "text": row.get("text") or row.get("hanzi") or "",
            "pinyin_tokens": pinyin_tokens,
            "tones": tones,
            "initials": initials,
            "finals": finals,
            "keywords_list": parse_keywords(row.get("keywords") or row.get("keyword_list")),
            "syllable_count_value": safe_int(row.get("syllable_count"), len(pinyin_tokens)) or len(pinyin_tokens),
        }
    return items
