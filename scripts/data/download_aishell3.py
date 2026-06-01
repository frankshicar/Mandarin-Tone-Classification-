#!/usr/bin/env python3
import argparse
from pathlib import Path

from datasets import Audio, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AISHELL-3 from Hugging Face.")
    parser.add_argument("--cache-dir", default="data/hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--output-dir", default="data/aishell3", help="Directory for save_to_disk output")
    parser.add_argument("--split", default=None, choices=[None, "train", "test"], help="Optional split to download")
    parser.add_argument("--quick", action="store_true", help="Download only a small subset for verification")
    parser.add_argument("--decode-audio", action="store_true", help="Decode audio arrays while loading examples")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = args.split
    if args.quick:
        split = f"{split or 'train'}[:5]"

    ds = load_dataset(
        "AISHELL/AISHELL-3",
        "default",
        split=split,
        cache_dir=str(cache_dir),
    )
    if not args.decode_audio:
        if hasattr(ds, "cast_column"):
            ds = ds.cast_column("audio", Audio(decode=False))
        else:
            for key in ds:
                ds[key] = ds[key].cast_column("audio", Audio(decode=False))

    target = output_dir / ("quick" if args.quick else (args.split or "all"))
    ds.save_to_disk(str(target))

    print(f"saved_to={target}")
    print(ds)
    if args.quick:
        first = ds[0]
        print("first_keys=", list(first.keys()))
        print("first_audio=", first["audio"])
        print("first_label=", first.get("label"))


if __name__ == "__main__":
    main()
