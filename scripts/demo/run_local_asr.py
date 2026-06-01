#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from pypinyin import Style, lazy_pinyin
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.common.hearing_pipeline_utils import parse_pinyin_tokens, read_csv_rows, write_csv_rows  # noqa: E402


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


def choose_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def resample_audio(audio: torch.Tensor | list | tuple, sr: int, target_sr: int) -> tuple[list[float], int]:
    if sr == target_sr:
        return audio, sr
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return resampled, target_sr


def text_to_pinyin(text: str) -> str:
    tokens = lazy_pinyin(
        text or "",
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    return " ".join(parse_pinyin_tokens(" ".join(tokens)))


def confidence_from_mean_logprob(mean_logprob: float | None) -> float | None:
    if mean_logprob is None:
        return None
    return float(max(0.0, min(1.0, math.exp(mean_logprob))))


def load_model(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool,
) -> tuple[AutoProcessor, AutoModelForSpeechSeq2Seq]:
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return processor, model


def transcribe_one(
    audio_path: Path,
    processor: AutoProcessor,
    model: AutoModelForSpeechSeq2Seq,
    device: str,
    language: str,
    max_new_tokens: int,
) -> tuple[str, float | None, float | None]:
    audio, sr = sf.read(str(audio_path), always_2d=True)
    audio = audio.mean(axis=1).astype("float32")
    audio, sr = resample_audio(audio, sr=sr, target_sr=16000)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)
    outputs = model.generate(
        input_features,
        forced_decoder_ids=processor.get_decoder_prompt_ids(language=language, task="transcribe"),
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    mean_logprob = None
    if getattr(outputs, "scores", None):
        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        values = [float(value) for value in transition_scores[0].tolist() if value != 0.0]
        if values:
            mean_logprob = sum(values) / len(values)
    return text, confidence_from_mean_logprob(mean_logprob), mean_logprob


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Whisper ASR and enrich a CSV manifest in place.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--gpu-min-free-mib", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    rows = read_csv_rows(args.input_csv)
    if not rows:
        raise ValueError(f"no rows found in {args.input_csv}")

    device = choose_device(args.device, args.gpu_min_free_mib)
    dtype = choose_dtype(args.dtype, device)
    processor, model = load_model(args.model, device, dtype, args.local_files_only)

    output_rows = []
    for row in rows:
        output = dict(row)
        audio_path = Path(row.get(args.audio_column, "")).expanduser()
        if not audio_path.exists():
            output["asr_text"] = ""
            output["asr_pinyin"] = ""
            output["asr_confidence"] = ""
            output["asr_mean_logprob"] = ""
            output["asr_model"] = args.model
            output_rows.append(output)
            continue

        text, confidence, mean_logprob = transcribe_one(
            audio_path=audio_path,
            processor=processor,
            model=model,
            device=device,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
        )
        output["asr_text"] = text
        output["asr_pinyin"] = text_to_pinyin(text)
        output["asr_confidence"] = f"{confidence:.4f}" if confidence is not None else ""
        output["asr_mean_logprob"] = f"{mean_logprob:.4f}" if mean_logprob is not None else ""
        output["asr_model"] = args.model
        output_rows.append(output)

    fieldnames = list(output_rows[0].keys())
    for extra in ["asr_text", "asr_pinyin", "asr_confidence", "asr_mean_logprob", "asr_model"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv_rows(args.output_csv, fieldnames, output_rows)
    print(f"wrote={args.output_csv}")
    print(f"rows={len(output_rows)}")
    print(f"device={device}")


if __name__ == "__main__":
    main()
