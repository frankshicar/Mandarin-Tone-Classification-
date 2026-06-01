#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.features.extract_logmel_utterance import extract_logmel  # noqa: E402
from scripts.common.hearing_pipeline_utils import load_audio_mono, load_item_bank, parse_pinyin_tokens, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from scripts.training.tone_dataset import CLASS_TO_TONE  # noqa: E402
from scripts.training.train_mel_context_resnet import MelContextResNet  # noqa: E402
from scripts.training.train_mel_resnet import resize_time  # noqa: E402


def choose_device(device_arg: str, gpu_min_free_mib: int) -> str:
    if device_arg != "auto":
        return device_arg
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_mib = free_bytes / (1024 * 1024)
        if free_mib >= gpu_min_free_mib:
            return "cuda:0"
    except Exception:
        return "cpu"
    return "cpu"


def tone_base(token: str) -> str:
    return re.sub(r"[1-5]$", "", token)


def detect_active_span(samples: np.ndarray, sample_rate: int, silence_threshold_dbfs: float) -> tuple[float, float]:
    if len(samples) == 0:
        return 0.0, 0.0
    threshold = 10 ** (silence_threshold_dbfs / 20.0)
    active = np.abs(samples) >= threshold
    if not np.any(active):
        return 0.0, float(len(samples) / sample_rate)
    indices = np.flatnonzero(active)
    start_sec = float(indices[0] / sample_rate)
    end_sec = float((indices[-1] + 1) / sample_rate)
    return start_sec, end_sec


def build_boundaries(
    syllable_count: int,
    start_sec: float,
    end_sec: float,
    span_pad_sec: float,
) -> list[dict[str, float | int | bool]]:
    if syllable_count <= 0:
        return []
    utterance_start = max(0.0, start_sec - span_pad_sec)
    utterance_end = max(utterance_start, end_sec + span_pad_sec)
    usable_duration = utterance_end - utterance_start
    if usable_duration <= 0.0:
        usable_duration = 1e-3
    rows = []
    for idx in range(syllable_count):
        part_start = utterance_start + usable_duration * idx / syllable_count
        part_end = utterance_start + usable_duration * (idx + 1) / syllable_count
        rows.append(
            {
                "syllable_index": idx,
                "syllable_count": syllable_count,
                "start_sec": part_start,
                "end_sec": part_end,
                "duration_sec": part_end - part_start,
                "word_boundary_after": idx == syllable_count - 1,
                "phrase_boundary_after": idx == syllable_count - 1,
            }
        )
    return rows


def segment_for_boundary(
    logmel: np.ndarray,
    times: np.ndarray,
    boundary: dict[str, float | int | bool] | None,
    frames: int,
) -> tuple[np.ndarray, float]:
    if boundary is None:
        return np.zeros((80, frames), dtype=np.float32), 0.0
    start = float(boundary["start_sec"])
    end = float(boundary["end_sec"])
    duration = max(0.0, end - start)
    start = max(0.0, start - duration * 0.5)
    end = end + duration * 0.5
    segment = logmel[:, (times >= start) & (times < end)]
    segment = resize_time(segment, frames)
    segment = np.clip((segment + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    return segment, 1.0


def build_model(checkpoint_path: Path, device: str) -> tuple[MelContextResNet, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    vocab_path = Path(train_args["vocab"])
    vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.exists() else {"<unk>": 0}
    model = MelContextResNet(
        vocab_size=max(1, len(vocab)),
        width=train_args.get("width", 32),
        dropout=train_args.get("dropout", 0.1),
        use_syllable_embedding=train_args.get("use_syllable_embedding", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, train_args, vocab


def prepare_row_inputs(
    logmel: np.ndarray,
    times: np.ndarray,
    pinyin_tokens: list[str],
    boundaries: list[dict[str, float | int | bool]],
    frames: int,
    vocab: dict[str, int],
    use_syllable_embedding: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mel_context_rows = []
    feature_rows = []
    syllable_ids = []
    for idx, boundary in enumerate(boundaries):
        prev_boundary = boundaries[idx - 1] if idx > 0 else None
        next_boundary = boundaries[idx + 1] if idx + 1 < len(boundaries) else None
        prev_segment, has_prev = segment_for_boundary(logmel, times, prev_boundary, frames)
        current_segment, _ = segment_for_boundary(logmel, times, boundary, frames)
        next_segment, has_next = segment_for_boundary(logmel, times, next_boundary, frames)
        mel_context = np.stack([prev_segment, current_segment, next_segment], axis=0).astype(np.float32)
        mel_context_rows.append(mel_context)

        syllable_count = max(1, int(boundary["syllable_count"]))
        relative_index = idx / max(1, syllable_count - 1)
        feature_rows.append(
            [
                float(boundary["duration_sec"]),
                relative_index,
                1.0 if boundary["word_boundary_after"] else 0.0,
                1.0 if boundary["phrase_boundary_after"] else 0.0,
                has_prev,
                has_next,
            ]
        )
        if use_syllable_embedding and idx < len(pinyin_tokens):
            syllable_ids.append(vocab.get(tone_base(pinyin_tokens[idx]), 0))
        else:
            syllable_ids.append(0)

    mel_tensor = torch.from_numpy(np.asarray(mel_context_rows, dtype=np.float32)).unsqueeze(2)
    feature_tensor = torch.from_numpy(np.asarray(feature_rows, dtype=np.float32))
    syllable_tensor = torch.tensor(syllable_ids, dtype=torch.long)
    return mel_tensor, feature_tensor, syllable_tensor


def run_inference(
    model: MelContextResNet,
    device: str,
    mel_tensor: torch.Tensor,
    feature_tensor: torch.Tensor,
    syllable_tensor: torch.Tensor,
) -> tuple[list[str], list[float]]:
    with torch.no_grad():
        logits = model(
            mel_tensor.to(device),
            feature_tensor.to(device),
            syllable_tensor.to(device),
        )
        probs = torch.softmax(logits, dim=-1)
        best_probs, best_idx = probs.max(dim=-1)
    tones = [CLASS_TO_TONE[int(index)] for index in best_idx.cpu().tolist()]
    confidences = [float(value) for value in best_probs.cpu().tolist()]
    return tones, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Mandarin tone sequences from word-level audio.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--item-bank", default=None)
    parser.add_argument("--pinyin-column", default="asr_pinyin")
    parser.add_argument("--checkpoint", default="runs/mel_context_resnet_mfa_train_full_strict_speaker_split/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-min-free-mib", type=int, default=2048)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-40.0)
    parser.add_argument("--span-pad-sec", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_csv_rows(args.input_csv)
    if not rows:
        raise ValueError(f"no rows found in {args.input_csv}")

    item_bank = load_item_bank(args.item_bank) if args.item_bank else {}
    checkpoint_path = Path(args.checkpoint)
    device = choose_device(args.device, args.gpu_min_free_mib)
    model, train_args, vocab = build_model(checkpoint_path, device)

    meta_path = Path(train_args["logmel_summary"]).with_suffix(Path(train_args["logmel_summary"]).suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = int(train_args.get("frames", 96))
    use_syllable_embedding = bool(train_args.get("use_syllable_embedding", False))

    output_rows = []
    confidence_values = []
    predicted_rows = 0
    missing_audio_rows = 0
    skipped_rows = 0

    for row in rows:
        output = dict(row)
        item = item_bank.get((row.get("item_id") or "").strip())
        pinyin_tokens = item["pinyin_tokens"] if item else parse_pinyin_tokens(row.get(args.pinyin_column))
        audio_path = Path(row.get(args.audio_column, "")).expanduser()

        if not audio_path.exists():
            missing_audio_rows += 1
            output["predicted_tones"] = ""
            output["tone_confidence"] = ""
            output["tone_model"] = str(checkpoint_path)
            output["tone_boundary_source"] = "missing_audio"
            output_rows.append(output)
            continue

        if not pinyin_tokens:
            skipped_rows += 1
            output["predicted_tones"] = ""
            output["tone_confidence"] = ""
            output["tone_model"] = str(checkpoint_path)
            output["tone_boundary_source"] = "missing_pinyin"
            output_rows.append(output)
            continue

        samples, sample_rate, _, _ = load_audio_mono(audio_path)
        active_start, active_end = detect_active_span(samples, sample_rate, args.silence_threshold_dbfs)
        boundaries = build_boundaries(
            syllable_count=len(pinyin_tokens),
            start_sec=active_start,
            end_sec=active_end,
            span_pad_sec=args.span_pad_sec,
        )

        _, logmel, times = extract_logmel(
            str(audio_path),
            sr=int(meta["sr"]),
            n_fft=int(meta["n_fft"]),
            hop_length=int(meta["hop_length"]),
            n_mels=int(meta["n_mels"]),
            fmin=float(meta["fmin"]),
            fmax=float(meta["fmax"]),
        )
        mel_tensor, feature_tensor, syllable_tensor = prepare_row_inputs(
            logmel=logmel,
            times=times,
            pinyin_tokens=pinyin_tokens,
            boundaries=boundaries,
            frames=frames,
            vocab=vocab,
            use_syllable_embedding=use_syllable_embedding,
        )
        tones, tone_probs = run_inference(
            model=model,
            device=device,
            mel_tensor=mel_tensor,
            feature_tensor=feature_tensor,
            syllable_tensor=syllable_tensor,
        )

        predicted_rows += 1
        confidence = float(sum(tone_probs) / len(tone_probs)) if tone_probs else 0.0
        confidence_values.append(confidence)
        output["predicted_tones"] = " ".join(tones)
        output["tone_confidence"] = f"{confidence:.4f}"
        output["tone_model"] = str(checkpoint_path)
        output["tone_boundary_source"] = "equal_duration_active_span"
        output["tone_syllable_confidence"] = json.dumps([round(value, 4) for value in tone_probs], ensure_ascii=False)
        output_rows.append(output)

    fieldnames = list(output_rows[0].keys())
    for extra in ["predicted_tones", "tone_confidence", "tone_model", "tone_boundary_source", "tone_syllable_confidence"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv_rows(args.output_csv, fieldnames, output_rows)

    if args.summary_json:
        summary = {
            "input_csv": args.input_csv,
            "output_csv": args.output_csv,
            "checkpoint": str(checkpoint_path),
            "device": device,
            "rows_total": len(rows),
            "predicted_rows": predicted_rows,
            "missing_audio_rows": missing_audio_rows,
            "skipped_rows": skipped_rows,
            "mean_tone_confidence": round(float(sum(confidence_values) / len(confidence_values)), 4) if confidence_values else 0.0,
        }
        write_json(args.summary_json, summary)

    print(f"wrote={args.output_csv}")
    print(f"rows={len(output_rows)}")
    print(f"device={device}")


if __name__ == "__main__":
    main()
