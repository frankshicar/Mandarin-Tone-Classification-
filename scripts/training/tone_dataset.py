import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


TONE_TO_CLASS = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
CLASS_TO_TONE = {value: key for key, value in TONE_TO_CLASS.items()}


def parse_bool(value: str) -> float:
    return 1.0 if value.lower() == "true" else 0.0


class SyllableF0Dataset(Dataset):
    def __init__(
        self,
        features_csv: str | Path,
        contour_npz: str | Path,
        split: str,
        contour_key: str = "f0_hz",
        norm_stats: dict[str, tuple[float, float]] | None = None,
        drop_empty_contour: bool = True,
    ) -> None:
        self.features_csv = Path(features_csv)
        self.contour_npz = Path(contour_npz)
        self.split = split
        self.contour_key = contour_key

        with self.features_csv.open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row.get("data_split") == split]
        if drop_empty_contour:
            rows = [row for row in rows if int(row["voiced_frames"]) > 0]
        if not rows:
            raise ValueError(f"no rows for split={split} in {self.features_csv}")

        with np.load(self.contour_npz) as data:
            if contour_key not in data.files:
                raise ValueError(f"{contour_key} not found in {self.contour_npz}; available={data.files}")
            contours = data[contour_key].astype(np.float32)

        contour_indices = [int(row["contour_index"]) for row in rows]
        bad_indices = [idx for idx in contour_indices if idx < 0 or idx >= len(contours)]
        if bad_indices:
            raise ValueError(
                f"{len(bad_indices)} contour_index values are outside {contour_key} shape {contours.shape}"
            )

        self.rows = rows
        self.contours = contours
        self.norm_stats = norm_stats or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        contour_index = int(row["contour_index"])
        x = self.contours[contour_index].copy()
        if self.norm_stats:
            mean, std = self.norm_stats.get(row["speaker"], self.norm_stats["__global__"])
            x = (x - mean) / std
        y = TONE_TO_CLASS[row["tone"]]
        return {
            "f0": torch.from_numpy(x).unsqueeze(-1),
            "label": torch.tensor(y, dtype=torch.long),
            "tone": row["tone"],
            "utt_id": row["utt_id"],
            "syllable_index": int(row["syllable_index"]),
        }


class ContextSyllableF0Dataset(Dataset):
    def __init__(
        self,
        features_csv: str | Path,
        contour_npz: str | Path,
        split: str,
        contour_key: str = "f0_hz",
        norm_stats: dict[str, tuple[float, float]] | None = None,
        structured_stats: dict[str, tuple[float, float]] | None = None,
        left_context: int = 1,
        right_context: int = 1,
        drop_empty_contour: bool = True,
    ) -> None:
        if left_context != 1 or right_context != 1:
            raise ValueError("ContextSyllableF0Dataset currently supports left_context=1 and right_context=1")
        self.features_csv = Path(features_csv)
        self.contour_npz = Path(contour_npz)
        self.split = split
        self.contour_key = contour_key
        self.norm_stats = norm_stats or {}
        self.structured_stats = structured_stats or {}

        with self.features_csv.open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row.get("data_split") == split]
        if drop_empty_contour:
            rows = [row for row in rows if int(row["voiced_frames"]) > 0]
        if not rows:
            raise ValueError(f"no rows for split={split} in {self.features_csv}")

        with np.load(self.contour_npz) as data:
            if contour_key not in data.files:
                raise ValueError(f"{contour_key} not found in {self.contour_npz}; available={data.files}")
            contours = data[contour_key].astype(np.float32)

        contour_indices = [int(row["contour_index"]) for row in rows]
        bad_indices = [idx for idx in contour_indices if idx < 0 or idx >= len(contours)]
        if bad_indices:
            raise ValueError(
                f"{len(bad_indices)} contour_index values are outside {contour_key} shape {contours.shape}"
            )

        self.rows = rows
        self.contours = contours
        self.row_by_position = {
            (row["utt_id"], int(row["syllable_index"])): row
            for row in rows
        }

    def __len__(self) -> int:
        return len(self.rows)

    def normalized_contour(self, row: dict[str, str] | None, like: np.ndarray) -> tuple[np.ndarray, float]:
        if row is None:
            return np.zeros_like(like, dtype=np.float32), 0.0
        contour = self.contours[int(row["contour_index"])].copy()
        if self.norm_stats:
            mean, std = self.norm_stats.get(row["speaker"], self.norm_stats["__global__"])
            contour = (contour - mean) / std
        return contour.astype(np.float32), 1.0

    def normalize_structured(self, values: dict[str, float]) -> np.ndarray:
        ordered = [
            "duration_sec",
            "relative_syllable_index",
            "word_boundary_after",
            "phrase_boundary_after",
            "has_prev_context",
            "has_next_context",
        ]
        normalized = []
        for key in ordered:
            value = values[key]
            if key in self.structured_stats:
                mean, std = self.structured_stats[key]
                value = (value - mean) / std
            normalized.append(value)
        return np.asarray(normalized, dtype=np.float32)

    def __getitem__(self, index: int):
        row = self.rows[index]
        current_idx = int(row["syllable_index"])
        current_contour = self.contours[int(row["contour_index"])]
        prev_row = self.row_by_position.get((row["utt_id"], current_idx - 1))
        next_row = self.row_by_position.get((row["utt_id"], current_idx + 1))

        left, has_prev = self.normalized_contour(prev_row, current_contour)
        current, _ = self.normalized_contour(row, current_contour)
        right, has_next = self.normalized_contour(next_row, current_contour)
        f0_context = np.stack([left, current, right], axis=-1)
        context_mask = np.asarray([has_prev, 1.0, has_next], dtype=np.float32)

        syllable_count = max(1, int(row["syllable_count"]))
        relative_index = current_idx / max(1, syllable_count - 1)
        structured = self.normalize_structured(
            {
                "duration_sec": float(row["duration_sec"]),
                "relative_syllable_index": relative_index,
                "word_boundary_after": parse_bool(row["word_boundary_after"]),
                "phrase_boundary_after": parse_bool(row["phrase_boundary_after"]),
                "has_prev_context": has_prev,
                "has_next_context": has_next,
            }
        )
        y = TONE_TO_CLASS[row["tone"]]
        return {
            "f0": torch.from_numpy(f0_context),
            "context_mask": torch.from_numpy(context_mask),
            "structured": torch.from_numpy(structured),
            "label": torch.tensor(y, dtype=torch.long),
            "tone": row["tone"],
            "utt_id": row["utt_id"],
            "syllable_index": current_idx,
        }


def compute_train_structured_stats(features_csv: str | Path) -> dict[str, tuple[float, float]]:
    features_csv = Path(features_csv)
    continuous = {
        "duration_sec": [],
        "relative_syllable_index": [],
    }
    with features_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("data_split") != "train" or int(row["voiced_frames"]) <= 0:
                continue
            syllable_count = max(1, int(row["syllable_count"]))
            relative_index = int(row["syllable_index"]) / max(1, syllable_count - 1)
            continuous["duration_sec"].append(float(row["duration_sec"]))
            continuous["relative_syllable_index"].append(relative_index)

    stats = {}
    for key, values in continuous.items():
        if not values:
            raise ValueError(f"no train values available for structured feature {key}")
        arr = np.asarray(values, dtype=np.float32)
        std = float(np.std(arr))
        stats[key] = (float(np.mean(arr)), std if std > 1e-6 else 1.0)
    return stats


def compute_train_norm_stats(
    features_csv: str | Path,
    contour_npz: str | Path,
    contour_key: str = "f0_hz",
    voiced_only: bool = True,
) -> dict[str, tuple[float, float]]:
    features_csv = Path(features_csv)
    contour_npz = Path(contour_npz)
    with features_csv.open(newline="", encoding="utf-8") as f:
        train_rows = [
            row
            for row in csv.DictReader(f)
            if row.get("data_split") == "train" and (not voiced_only or int(row["voiced_frames"]) > 0)
        ]
    if not train_rows:
        raise ValueError(f"no training rows in {features_csv}")

    with np.load(contour_npz) as data:
        contours = data[contour_key].astype(np.float32)

    by_speaker: dict[str, list[np.ndarray]] = {}
    all_values = []
    for row in train_rows:
        contour = contours[int(row["contour_index"])]
        voiced_values = contour[contour > 0.0]
        if voiced_values.size == 0:
            continue
        by_speaker.setdefault(row["speaker"], []).append(voiced_values)
        all_values.append(voiced_values)

    if not all_values:
        raise ValueError("no non-zero F0 values available for train-only normalization")

    def mean_std(parts: list[np.ndarray]) -> tuple[float, float]:
        values = np.concatenate(parts)
        std = float(np.std(values))
        return float(np.mean(values)), std if std > 1e-6 else 1.0

    stats = {"__global__": mean_std(all_values)}
    for speaker, parts in by_speaker.items():
        stats[speaker] = mean_std(parts)
    return stats
