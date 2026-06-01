#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np


def summarize_f0(path: str, frame_length: int, hop_length: int, fmin: float, fmax: float):
    y, sr = librosa.load(path, sr=None, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    voiced_f0 = f0[np.isfinite(f0)]

    summary = {
        "f0_num_frames": len(f0),
        "f0_voiced_frames": int(voiced_f0.size),
        "f0_voiced_ratio": f"{(voiced_f0.size / len(f0)) if len(f0) else 0.0:.6f}",
        "f0_mean_hz": "",
        "f0_median_hz": "",
        "f0_min_hz": "",
        "f0_max_hz": "",
        "f0_std_hz": "",
    }
    if voiced_f0.size:
        summary.update(
            {
                "f0_mean_hz": f"{float(np.mean(voiced_f0)):.6f}",
                "f0_median_hz": f"{float(np.median(voiced_f0)):.6f}",
                "f0_min_hz": f"{float(np.min(voiced_f0)):.6f}",
                "f0_max_hz": f"{float(np.max(voiced_f0)):.6f}",
                "f0_std_hz": f"{float(np.std(voiced_f0)):.6f}",
            }
        )

    dense = {
        "times": times.astype(np.float32),
        "f0_hz": np.nan_to_num(f0, nan=0.0).astype(np.float32),
        "voiced": voiced_flag.astype(np.bool_),
        "voiced_prob": voiced_prob.astype(np.float32),
    }
    return summary, dense


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract utterance-level F0 summaries for AISHELL-3.")
    parser.add_argument("--manifest", default="data/aishell3/manifest.csv")
    parser.add_argument("--out", default="data/aishell3/features/f0_utterance.csv")
    parser.add_argument("--npz-dir", default="data/aishell3/features/f0_npz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--fmin", type=float, default=float(librosa.note_to_hz("C2")))
    parser.add_argument("--fmax", type=float, default=float(librosa.note_to_hz("C7")))
    parser.add_argument("--resume", action="store_true", help="Append new rows and skip utterances already in --out with existing NPZ files.")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.out)
    npz_dir = Path(args.npz_dir)
    meta_out = out.with_suffix(out.suffix + ".meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.split:
        rows = [row for row in rows if row["split"] == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]

    run_meta = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split": args.split,
        "limit": args.limit,
        "rows_requested": len(rows),
        "npz_dir": str(npz_dir),
        "frame_length": args.frame_length,
        "hop_length": args.hop_length,
        "fmin": args.fmin,
        "fmax": args.fmax,
    }

    if args.resume and meta_out.exists():
        existing_meta = json.loads(meta_out.read_text(encoding="utf-8"))
        if existing_meta != run_meta:
            raise ValueError(
                f"resume metadata mismatch for {out}; existing={existing_meta}; requested={run_meta}"
            )
    elif args.resume and out.exists() and not meta_out.exists():
        raise ValueError(f"cannot resume {out} because metadata sidecar is missing: {meta_out}")

    fieldnames = [
        "split",
        "speaker",
        "utt_id",
        "audio_path",
        "sample_rate",
        "duration_sec",
        "f0_npz_path",
        "f0_num_frames",
        "f0_voiced_frames",
        "f0_voiced_ratio",
        "f0_mean_hz",
        "f0_median_hz",
        "f0_min_hz",
        "f0_max_hz",
        "f0_std_hz",
    ]

    processed: set[str] = set()
    if args.resume and out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                npz_path = row.get("f0_npz_path", "")
                if row.get("utt_id") and npz_path and Path(npz_path).exists():
                    processed.add(row["utt_id"])

    mode = "a" if args.resume and out.exists() else "w"
    write_header = mode == "w" or out.stat().st_size == 0
    rows_to_process = [row for row in rows if row["utt_id"] not in processed]
    requested_utt_ids = {row["utt_id"] for row in rows}
    stale_utt_ids = processed - requested_utt_ids
    if stale_utt_ids:
        raise ValueError(f"{out} contains {len(stale_utt_ids)} utterances not present in current manifest")

    if write_header:
        meta_out.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with out.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(rows_to_process, start=1):
            summary, dense = summarize_f0(
                row["audio_path"],
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            npz_path = npz_dir / row["split"] / row["speaker"] / f"{row['utt_id']}.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **dense)
            writer.writerow(
                {
                    "split": row["split"],
                    "speaker": row["speaker"],
                    "utt_id": row["utt_id"],
                    "audio_path": row["audio_path"],
                    "sample_rate": row.get("sample_rate", ""),
                    "duration_sec": row.get("duration_sec", ""),
                    "f0_npz_path": str(npz_path),
                    **summary,
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
