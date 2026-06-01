#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def text_to_pinyin(text: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pypinyin is required to convert WhisperX text to numbered pinyin.") from exc

    tokens = lazy_pinyin(
        text or "",
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    return " ".join(parse_pinyin_tokens(" ".join(tokens)))


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


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def confidence_from_segments(segments: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for segment in segments:
        if "avg_logprob" in segment and segment["avg_logprob"] is not None:
            try:
                values.append(float(segment["avg_logprob"]))
            except (TypeError, ValueError):
                pass
    if not values:
        return None
    return float(max(0.0, min(1.0, math.exp(sum(values) / len(values)))))


def timed_unit_count(aligned: dict[str, Any]) -> int:
    count = 0
    for segment in aligned.get("segments", []) or []:
        for key in ("chars", "char_segments", "words"):
            for unit in segment.get(key, []) or []:
                if unit.get("start") is not None and unit.get("end") is not None:
                    count += 1
    for key in ("char_segments", "word_segments"):
        for unit in aligned.get(key, []) or []:
            if unit.get("start") is not None and unit.get("end") is not None:
                count += 1
    return count


def make_failure_payload(
    row: dict[str, str],
    audio_path: Path,
    model_name: str,
    language: str,
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "utt_id": row.get("utt_id", ""),
        "speaker": row.get("speaker", ""),
        "audio_path": str(audio_path),
        "model": model_name,
        "language": language,
        "status": status,
        "error": error,
        "text": "",
        "pinyin": "",
        "segments": [],
        "word_segments": [],
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def transcribe_and_align(
    whisperx: Any,
    audio_path: Path,
    model: Any,
    align_model: Any,
    align_metadata: Any,
    device: str,
    batch_size: int,
    language: str,
) -> dict[str, Any]:
    audio = whisperx.load_audio(str(audio_path))
    try:
        result = model.transcribe(audio, batch_size=batch_size, language=language)
    except TypeError:
        result = model.transcribe(audio, batch_size=batch_size)
    if not result.get("segments"):
        raise ValueError("whisperx_transcribe_returned_no_segments")
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=True,
    )
    aligned["language"] = result.get("language", language)
    if not timed_unit_count(aligned):
        raise ValueError("whisperx_alignment_returned_no_timed_units")
    return aligned


def load_whisperx_model(whisperx: Any, model_name: str, device: str, compute_type: str, language: str, model_dir: str) -> Any:
    kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type, "language": language}
    if model_dir:
        kwargs["download_root"] = model_dir
    try:
        return whisperx.load_model(model_name, **kwargs)
    except TypeError:
        kwargs.pop("language", None)
        try:
            return whisperx.load_model(model_name, **kwargs)
        except TypeError:
            kwargs.pop("download_root", None)
            return whisperx.load_model(model_name, **kwargs)


def load_whisperx_align_model(whisperx: Any, language: str, device: str, model_dir: str, local_files_only: bool) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {"language_code": language, "device": device}
    if model_dir:
        kwargs["model_dir"] = model_dir
    if local_files_only:
        kwargs["model_cache_only"] = True
    try:
        return whisperx.load_align_model(**kwargs)
    except TypeError:
        kwargs.pop("model_cache_only", None)
        try:
            return whisperx.load_align_model(**kwargs)
        except TypeError:
            kwargs.pop("model_dir", None)
            return whisperx.load_align_model(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run WhisperX ASR + alignment on an utterance manifest. "
            "Writes an enriched manifest plus JSONL with raw aligned segments."
        )
    )
    parser.add_argument("--input-csv", default="data/aishell3/manifest_train_full.csv")
    parser.add_argument("--output-csv", default="data/aishell3/manifest_train_full_whisperx.csv")
    parser.add_argument("--jsonl-out", default="data/aishell3/whisperx_train_full.jsonl")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="", help="WhisperX compute type. Defaults to float16 on CUDA and int8 on CPU.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-dir", default="", help="Optional cache/download directory for WhisperX ASR and align models.")
    parser.add_argument("--local-files-only", action="store_true", help="Only use cached alignment models. WhisperX ASR model loading still depends on its installed backend cache behavior.")
    args = parser.parse_args()

    try:
        import whisperx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "whisperx is not installed. Install it before running this script, e.g. `pip install whisperx`."
        ) from exc

    rows = load_rows(Path(args.input_csv))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no rows found in {args.input_csv}")

    device = choose_device(args.device)
    compute_type = args.compute_type or ("float16" if device.startswith("cuda") else "int8")
    model = load_whisperx_model(
        whisperx=whisperx,
        model_name=args.model,
        device=device,
        compute_type=compute_type,
        language=args.language,
        model_dir=args.model_dir,
    )
    align_model, align_metadata = load_whisperx_align_model(
        whisperx=whisperx,
        language=args.language,
        device=device,
        model_dir=args.model_dir,
        local_files_only=args.local_files_only,
    )

    jsonl_path = Path(args.jsonl_out)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    model_label = f"whisperx/{args.model}"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            output = dict(row)
            audio_path = Path(row.get(args.audio_column, "")).expanduser()
            if not audio_path.exists():
                payload = make_failure_payload(
                    row=row,
                    audio_path=audio_path,
                    model_name=model_label,
                    language=args.language,
                    status="missing_audio",
                    error=f"audio file not found: {audio_path}",
                )
                jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.update(
                    {
                        "asr_text": "",
                        "asr_pinyin": "",
                        "asr_confidence": "",
                        "asr_model": model_label,
                        "whisperx_jsonl": str(jsonl_path),
                        "whisperx_status": "missing_audio",
                        "whisperx_error": payload["error"],
                    }
                )
                output_rows.append(output)
                continue

            try:
                aligned = transcribe_and_align(
                    whisperx=whisperx,
                    audio_path=audio_path,
                    model=model,
                    align_model=align_model,
                    align_metadata=align_metadata,
                    device=device,
                    batch_size=args.batch_size,
                    language=args.language,
                )
                text = "".join(segment.get("text", "") for segment in aligned.get("segments", [])).strip()
                confidence = confidence_from_segments(aligned.get("segments", []))
                payload = {
                    "utt_id": row.get("utt_id", ""),
                    "speaker": row.get("speaker", ""),
                    "audio_path": str(audio_path),
                    "model": model_label,
                    "language": aligned.get("language", args.language),
                    "status": "ok",
                    "error": "",
                    "text": text,
                    "pinyin": text_to_pinyin(text),
                    "segments": aligned.get("segments", []),
                    "word_segments": aligned.get("word_segments", []),
                }
                jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.update(
                    {
                        "asr_text": text,
                        "asr_pinyin": payload["pinyin"],
                        "asr_confidence": f"{confidence:.4f}" if confidence is not None else "",
                        "asr_model": model_label,
                        "whisperx_jsonl": str(jsonl_path),
                        "whisperx_status": "ok",
                        "whisperx_error": "",
                    }
                )
            except Exception as exc:
                status = f"error:{type(exc).__name__}"
                error = str(exc)
                payload = make_failure_payload(
                    row=row,
                    audio_path=audio_path,
                    model_name=model_label,
                    language=args.language,
                    status=status,
                    error=error,
                )
                jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.update(
                    {
                        "asr_text": "",
                        "asr_pinyin": "",
                        "asr_confidence": "",
                        "asr_model": model_label,
                        "whisperx_jsonl": str(jsonl_path),
                        "whisperx_status": status,
                        "whisperx_error": error,
                    }
                )
            output_rows.append(output)

    fieldnames = list(output_rows[0].keys())
    for extra in ["asr_text", "asr_pinyin", "asr_confidence", "asr_model", "whisperx_jsonl", "whisperx_status", "whisperx_error"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv_rows(args.output_csv, fieldnames, output_rows)
    print(f"wrote={args.output_csv}")
    print(f"wrote_jsonl={jsonl_path}")
    print(f"rows={len(output_rows)}")
    print(f"device={device}")
    print(f"compute_type={compute_type}")


if __name__ == "__main__":
    main()
