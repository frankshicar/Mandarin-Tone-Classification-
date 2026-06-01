#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_logmel(path: str, sr: int, n_fft: int, hop_length: int, n_mels: int, fmin: float, fmax: float):
    y, actual_sr = librosa.load(path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=actual_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    times = librosa.frames_to_time(
        np.arange(logmel.shape[1]),
        sr=actual_sr,
        hop_length=hop_length,
        n_fft=n_fft,
    ).astype(np.float32)
    return actual_sr, logmel, times


def valid_npz(path: Path, n_mels: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return (
                "logmel" in data.files
                and "times" in data.files
                and data["logmel"].ndim == 2
                and data["logmel"].shape[0] == n_mels
                and data["times"].ndim == 1
                and data["times"].shape[0] == data["logmel"].shape[1]
            )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract utterance-level log-mel spectrograms for AISHELL-3.")
    parser.add_argument("--manifest", default="data/aishell3/manifest_train_full.csv")
    parser.add_argument("--out", default="data/aishell3/features/logmel_utterance_train_full.csv")
    parser.add_argument("--npz-dir", default="data/aishell3/features/logmel_npz_train_full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=7600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out)
    npz_dir = Path(args.npz_dir)
    meta_out = out.with_suffix(out.suffix + ".meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    run_meta = {
        "schema_version": 2,
        "extractor": "extract_logmel_utterance.py",
        "librosa_version": librosa.__version__,
        "numpy_version": np.__version__,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "limit": args.limit,
        "rows_requested": len(rows),
        "npz_dir": str(npz_dir),
        "sr": args.sr,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "fmin": args.fmin,
        "fmax": args.fmax,
    }
    if args.resume and meta_out.exists():
        existing_meta = json.loads(meta_out.read_text(encoding="utf-8"))
        if existing_meta != run_meta:
            raise ValueError(f"resume metadata mismatch for {out}; existing={existing_meta}; requested={run_meta}")
    elif args.resume and out.exists() and not meta_out.exists():
        raise ValueError(f"cannot resume {out} because metadata sidecar is missing: {meta_out}")

    processed = set()
    if args.resume and out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                npz_path = row.get("logmel_npz_path", "")
                if row.get("utt_id") and npz_path and valid_npz(Path(npz_path), args.n_mels):
                    processed.add(row["utt_id"])

    rows_to_process = [row for row in rows if row["utt_id"] not in processed]
    mode = "a" if args.resume and out.exists() else "w"
    write_header = mode == "w" or out.stat().st_size == 0
    if write_header:
        meta_out.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "split",
        "speaker",
        "utt_id",
        "audio_path",
        "sample_rate",
        "duration_sec",
        "logmel_npz_path",
        "logmel_num_mels",
        "logmel_num_frames",
        "logmel_frame_hop_sec",
    ]

    with out.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(rows_to_process, start=1):
            actual_sr, logmel, times = extract_logmel(
                row["audio_path"],
                sr=args.sr,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            npz_path = npz_dir / row["split"] / row["speaker"] / f"{row['utt_id']}.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_npz_path = npz_path.with_suffix(".npz.tmp")
            with tmp_npz_path.open("wb") as tmp_f:
                np.savez_compressed(tmp_f, logmel=logmel, times=times)
            tmp_npz_path.replace(npz_path)
            writer.writerow(
                {
                    "split": row["split"],
                    "speaker": row["speaker"],
                    "utt_id": row["utt_id"],
                    "audio_path": row["audio_path"],
                    "sample_rate": actual_sr,
                    "duration_sec": row.get("duration_sec", ""),
                    "logmel_npz_path": str(npz_path),
                    "logmel_num_mels": logmel.shape[0],
                    "logmel_num_frames": logmel.shape[1],
                    "logmel_frame_hop_sec": f"{args.hop_length / actual_sr:.6f}",
                }
            )
            f.flush()
            if idx % args.progress_every == 0:
                print(f"processed_new={idx}/{len(rows_to_process)} total_done={len(processed) + idx}/{len(rows)}")

    print(f"wrote={out}")
    print(f"rows_requested={len(rows)}")
    print(f"rows_previously_done={len(processed)}")
    print(f"rows_new={len(rows_to_process)}")


if __name__ == "__main__":
    main()
