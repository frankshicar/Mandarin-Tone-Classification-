#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.common.hearing_pipeline_utils import load_item_bank, write_csv_rows  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402


PREFERRED_SPEAKERS = ("aiden", "serena", "eric", "dylan")


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
        return torch.bfloat16
    return torch.float32


def maybe_resample(wav: np.ndarray, source_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if source_sr == target_sr:
        return wav.astype(np.float32), source_sr
    resampled = librosa.resample(wav.astype(np.float32), orig_sr=source_sr, target_sr=target_sr)
    return resampled.astype(np.float32), target_sr


def parse_item_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_speaker_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    seen = set()
    speakers = []
    for part in value.split(","):
        speaker = part.strip()
        if not speaker or speaker in seen:
            continue
        seen.add(speaker)
        speakers.append(speaker)
    return speakers or None


def choose_default_speaker(tts: Qwen3TTSModel) -> str | None:
    supported = tts.get_supported_speakers() or []
    if not supported:
        return None
    for speaker in PREFERRED_SPEAKERS:
        if speaker in supported:
            return speaker
    return supported[0]


def build_generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {}
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    if args.do_sample is not None:
        kwargs["do_sample"] = args.do_sample
        kwargs["subtalker_dosample"] = args.do_sample
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
        kwargs["subtalker_temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
        kwargs["subtalker_top_p"] = args.top_p
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
        kwargs["subtalker_top_k"] = args.top_k
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    return kwargs


def sanitize_generate_defaults(tts: Qwen3TTSModel, generation_kwargs: dict) -> None:
    if generation_kwargs.get("do_sample") is False:
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "subtalker_temperature",
            "subtalker_top_p",
            "subtalker_top_k",
        ):
            tts.generate_defaults.pop(key, None)


def resolve_speakers(tts: Qwen3TTSModel, args: argparse.Namespace, model_type: str) -> list[str | None]:
    if model_type != "custom_voice":
        return [None]

    supported = tts.get_supported_speakers() or []
    speaker_list = parse_speaker_list(args.speaker_list)
    if speaker_list:
        unsupported = [speaker for speaker in speaker_list if speaker not in supported]
        if unsupported:
            raise ValueError(f"unsupported speakers requested: {', '.join(unsupported)}")
        return speaker_list

    if args.speaker:
        if supported and args.speaker not in supported:
            raise ValueError(f"unsupported speaker requested: {args.speaker}")
        return [args.speaker]

    speaker = choose_default_speaker(tts)
    if not speaker:
        raise ValueError("custom voice model did not expose supported speakers; pass --speaker explicitly")
    return [speaker]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3-TTS WAVs and candidate CSV rows for Stage 1 QC.")
    parser.add_argument("--item-bank", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--speaker-list", default=None, help="Comma-separated speaker list for custom voice sweeps.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--gpu-min-free-mib", type=int, default=12288)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--item-ids", default=None)
    parser.add_argument("--output-sample-rate", type=int, default=16000)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()

    items = list(load_item_bank(args.item_bank).values())
    wanted_ids = parse_item_ids(args.item_ids)
    if wanted_ids is not None:
        items = [item for item in items if item["item_id"] in wanted_ids]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("no items selected for generation")

    device = choose_device(args.device, args.gpu_min_free_mib)
    dtype = choose_dtype(args.dtype, device)
    tts = Qwen3TTSModel.from_pretrained(
        args.model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    generation_kwargs = build_generation_kwargs(args)
    sanitize_generate_defaults(tts, generation_kwargs)

    model_type = tts.model.tts_model_type
    speakers = resolve_speakers(tts, args, model_type)

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    multi_speaker = len(speakers) > 1
    for speaker in speakers:
        speaker_audio_dir = audio_dir / speaker if multi_speaker and speaker else audio_dir
        speaker_audio_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            if model_type == "custom_voice":
                wavs, sample_rate = tts.generate_custom_voice(
                    text=item["text"],
                    speaker=speaker,
                    language=args.language,
                    instruct=args.instruct,
                    **generation_kwargs,
                )
            elif model_type == "voice_design":
                if not args.instruct:
                    raise ValueError("voice design generation requires --instruct")
                wavs, sample_rate = tts.generate_voice_design(
                    text=item["text"],
                    instruct=args.instruct,
                    language=args.language,
                    **generation_kwargs,
                )
            elif model_type == "base":
                if not args.ref_audio:
                    raise ValueError("base model generation requires --ref-audio")
                wavs, sample_rate = tts.generate_voice_clone(
                    text=item["text"],
                    language=args.language,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    x_vector_only_mode=not bool(args.ref_text),
                    **generation_kwargs,
                )
            else:
                raise ValueError(f"unsupported qwen model type: {model_type}")

            wav = np.asarray(wavs[0], dtype=np.float32)
            wav, sample_rate = maybe_resample(wav, sample_rate, args.output_sample_rate)
            audio_path = speaker_audio_dir / f"{item['item_id']}.wav"
            sf.write(audio_path, wav, sample_rate)
            output_rows.append(
                {
                    "item_id": item["item_id"],
                    "audio_path": str(audio_path),
                    "tts_engine": f"{args.model_id}:{speaker}" if speaker else args.model_id,
                    "asr_text": "",
                    "asr_pinyin": "",
                    "asr_confidence": "",
                    "predicted_tones": "",
                    "tone_confidence": "",
                }
            )

    fieldnames = ["item_id", "audio_path", "tts_engine", "asr_text", "asr_pinyin", "asr_confidence", "predicted_tones", "tone_confidence"]
    write_csv_rows(args.output_csv, fieldnames, output_rows)
    print(f"wrote={args.output_csv}")
    print(f"rows={len(output_rows)}")
    print(f"device={device}")
    print(f"model_type={model_type}")
    if len(speakers) == 1 and speakers[0]:
        print(f"speaker={speakers[0]}")
    elif speakers and speakers[0]:
        print(f"speakers={','.join(str(speaker) for speaker in speakers if speaker)}")


if __name__ == "__main__":
    main()
