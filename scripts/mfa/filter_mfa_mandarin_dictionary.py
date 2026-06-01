#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


TONE_REPLACEMENTS = {
    "˥˥": "˥",
}


def load_phone_set(path: Path) -> set[str]:
    phones = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            phones.add(line.split()[0])
    return phones


def corpus_words(map_csv: Path) -> set[str]:
    words = {"<eps>", "<unk>"}
    with map_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words.update(token for token in row["mfa_text"].split() if token)
    return words


def remap_phone(phone: str) -> str:
    for old, new in TONE_REPLACEMENTS.items():
        phone = phone.replace(old, new)
    return phone


def split_dictionary_line(line: str) -> tuple[str, list[str], list[str]]:
    parts = line.rstrip("\n").split()
    if not parts:
        return "", [], []
    word = parts[0]
    idx = 1
    while idx < len(parts):
        try:
            float(parts[idx])
            idx += 1
        except ValueError:
            break
    return word, parts[1:idx], parts[idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter/remap MFA Mandarin dictionary to the acoustic model phone set.")
    parser.add_argument("--dictionary", default=str(Path.home() / "Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict"))
    parser.add_argument("--phones", default=str(Path.home() / "Documents/MFA/extracted_models/acoustic/mandarin_mfa_acoustic/phones.txt"))
    parser.add_argument("--map-csv", default="data/aishell3/mfa/corpus_train20_map.csv")
    parser.add_argument("--out", default="data/aishell3/mfa/mandarin_mfa_train20_filtered.dict")
    parser.add_argument("--missing-out", default="")
    parser.add_argument("--covered-out", default="")
    parser.add_argument("--bad-phone-out", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    phone_set = load_phone_set(Path(args.phones))
    needed_words = corpus_words(Path(args.map_csv))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_word = 0
    skipped_phone = 0
    covered_words = set()
    seen_prons = set()
    bad_phone_rows = []
    with Path(args.dictionary).open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            word, probabilities, phones = split_dictionary_line(line)
            if not word:
                continue
            if word not in needed_words:
                skipped_word += 1
                continue
            remapped = [remap_phone(phone) for phone in phones]
            missing_phones = sorted({phone for phone in remapped if phone not in phone_set})
            if missing_phones:
                skipped_phone += 1
                bad_phone_rows.append(
                    {
                        "word": word,
                        "probabilities": " ".join(probabilities),
                        "original_phones": " ".join(phones),
                        "remapped_phones": " ".join(remapped),
                        "missing_phones": " ".join(missing_phones),
                    }
                )
                continue
            key = (word, tuple(remapped))
            if key in seen_prons:
                continue
            seen_prons.add(key)
            covered_words.add(word)
            if probabilities:
                g.write("\t".join([word, *probabilities, *remapped]) + "\n")
            else:
                g.write("\t".join([word, *remapped]) + "\n")
            written += 1

    missing_words = sorted(needed_words - covered_words - {"<eps>"})
    if "<unk>" not in covered_words:
        with out.open("a", encoding="utf-8") as g:
            g.write("<unk>\t0.99\t0.3\t1.73\t0.87\tspn\n")
        covered_words.add("<unk>")
        written += 1
        missing_words = [word for word in missing_words if word != "<unk>"]

    print(f"wrote={out}")
    print(f"needed_words={len(needed_words)}")
    print(f"covered_words={len(covered_words)}")
    print(f"written_pronunciations={written}")
    print(f"skipped_not_needed={skipped_word}")
    print(f"skipped_bad_phone={skipped_phone}")
    print(f"missing_words={len(missing_words)}")
    if missing_words:
        print("missing_first=" + " ".join(missing_words[:50]))
    if args.missing_out:
        missing_out = Path(args.missing_out)
        missing_out.parent.mkdir(parents=True, exist_ok=True)
        missing_out.write_text("\n".join(missing_words) + ("\n" if missing_words else ""), encoding="utf-8")
        print(f"wrote_missing={missing_out}")
    if args.covered_out:
        covered_out = Path(args.covered_out)
        covered_out.parent.mkdir(parents=True, exist_ok=True)
        covered_out.write_text("\n".join(sorted(covered_words - {'<eps>', '<unk>'})) + "\n", encoding="utf-8")
        print(f"wrote_covered={covered_out}")
    if args.bad_phone_out:
        bad_phone_out = Path(args.bad_phone_out)
        bad_phone_out.parent.mkdir(parents=True, exist_ok=True)
        with bad_phone_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["word", "probabilities", "original_phones", "remapped_phones", "missing_phones"],
            )
            writer.writeheader()
            writer.writerows(bad_phone_rows)
        print(f"wrote_bad_phone={bad_phone_out}")
    if args.strict and missing_words:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
