#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

from scripts.common.hearing_pipeline_utils import (
    base_syllable,
    compute_position_accuracy,
    is_aspiration_confusion,
    is_n_ng_confusion,
    is_retroflex_palatal_confusion,
    keyword_accuracy,
    load_item_bank,
    parse_pinyin_tokens,
    parse_tone_sequence,
    read_csv_rows,
    safe_float,
    split_initial_final,
    top_counter_rows,
    utc_now_iso,
    write_csv_rows,
    write_json,
)


def accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def evaluate_response(response: dict[str, str], item: dict | None, args: argparse.Namespace) -> tuple[dict, dict]:
    row = {
        "patient_id": response.get("patient_id", ""),
        "item_id": response.get("item_id", ""),
        "text": item["text"] if item else "",
        "expected_pinyin": " ".join(item["pinyin_tokens"]) if item else "",
        "predicted_pinyin": " ".join(parse_pinyin_tokens(response.get("asr_pinyin"))),
        "expected_tones": " ".join(item["tones"]) if item else "",
        "predicted_tones": " ".join(parse_tone_sequence(response.get("predicted_tones"))),
        "asr_text": response.get("asr_text", ""),
        "asr_confidence": response.get("asr_confidence", ""),
        "tone_confidence": response.get("tone_confidence", ""),
        "recording_quality": response.get("recording_quality", ""),
        "item_correct": "",
        "text_match": "",
        "keyword_accuracy": "",
        "syllable_accuracy": "",
        "initial_accuracy": "",
        "final_accuracy": "",
        "tone_accuracy": "",
        "review_required": "",
        "review_reasons": "",
    }
    detail = {
        "keyword_correct": 0,
        "keyword_total": 0,
        "syllable_correct": 0,
        "syllable_total": 0,
        "initial_correct": 0,
        "initial_total": 0,
        "final_correct": 0,
        "final_total": 0,
        "tone_correct": 0,
        "tone_total": 0,
        "item_correct": 0,
        "text_match": 0,
        "ai_readable": 0,
        "review_required": 0,
        "tone_confusions": [],
        "initial_confusions": [],
        "final_confusions": [],
        "pattern_counts": Counter(),
    }
    review_reasons = []

    if item is None:
        row["review_required"] = "true"
        row["review_reasons"] = "missing_item_bank_row"
        detail["review_required"] = 1
        return row, detail

    expected_pinyin = item["pinyin_tokens"]
    expected_tones = item["tones"]
    expected_initials = item["initials"]
    expected_finals = item["finals"]
    predicted_pinyin = parse_pinyin_tokens(response.get("asr_pinyin"))
    predicted_tones = parse_tone_sequence(response.get("predicted_tones")) or [token[-1] for token in predicted_pinyin if token and token[-1].isdigit()]
    predicted_initials = []
    predicted_finals = []
    for token in predicted_pinyin:
        initial, final = split_initial_final(token)
        predicted_initials.append(initial)
        predicted_finals.append(final)

    if response.get("asr_text"):
        detail["ai_readable"] = 1
        text_match = response.get("asr_text", "").strip() == item["text"]
        row["text_match"] = str(text_match).lower()
        detail["text_match"] = int(text_match)
    elif predicted_pinyin:
        detail["ai_readable"] = 1
        row["text_match"] = "unknown"
    else:
        row["text_match"] = "false"
        review_reasons.append("missing_asr_output")

    keyword_correct, keyword_total = keyword_accuracy(item["keywords_list"], response.get("asr_text", ""))
    if keyword_total == 0 and item["text"]:
        keyword_correct = int(response.get("asr_text", "").strip() == item["text"])
        keyword_total = 1
    detail["keyword_correct"] = keyword_correct
    detail["keyword_total"] = keyword_total
    row["keyword_accuracy"] = f"{accuracy(keyword_correct, keyword_total):.4f}"

    syllable_correct, syllable_total, syllable_pairs = compute_position_accuracy(
        expected_pinyin,
        predicted_pinyin,
        transform=base_syllable,
    )
    detail["syllable_correct"] = syllable_correct
    detail["syllable_total"] = syllable_total
    row["syllable_accuracy"] = f"{accuracy(syllable_correct, syllable_total):.4f}"

    initial_correct, initial_total, initial_pairs = compute_position_accuracy(expected_initials, predicted_initials)
    final_correct, final_total, final_pairs = compute_position_accuracy(expected_finals, predicted_finals)
    tone_correct, tone_total, tone_pairs = compute_position_accuracy(expected_tones, predicted_tones)
    detail["initial_correct"] = initial_correct
    detail["initial_total"] = initial_total
    detail["final_correct"] = final_correct
    detail["final_total"] = final_total
    detail["tone_correct"] = tone_correct
    detail["tone_total"] = tone_total
    row["initial_accuracy"] = f"{accuracy(initial_correct, initial_total):.4f}"
    row["final_accuracy"] = f"{accuracy(final_correct, final_total):.4f}"
    row["tone_accuracy"] = f"{accuracy(tone_correct, tone_total):.4f}"

    exact_pinyin_match = predicted_pinyin == expected_pinyin if predicted_pinyin else False
    exact_tone_match = predicted_tones == expected_tones if predicted_tones else False
    item_correct = int(exact_pinyin_match and exact_tone_match)
    detail["item_correct"] = item_correct
    row["item_correct"] = str(bool(item_correct)).lower()

    for expected, predicted, match in tone_pairs:
        if not match and expected and predicted:
            detail["tone_confusions"].append((expected, predicted))
    for expected, predicted, match in initial_pairs:
        if not match and expected is not None and predicted is not None:
            detail["initial_confusions"].append((expected, predicted))
            if is_retroflex_palatal_confusion(expected, predicted):
                detail["pattern_counts"]["retroflex_vs_palatal"] += 1
            if is_aspiration_confusion(expected, predicted):
                detail["pattern_counts"]["aspiration"] += 1
    for expected, predicted, match in final_pairs:
        if not match and expected is not None and predicted is not None:
            detail["final_confusions"].append((expected, predicted))
            if is_n_ng_confusion(expected, predicted):
                detail["pattern_counts"]["n_ng"] += 1

    asr_confidence = safe_float(response.get("asr_confidence"))
    if asr_confidence is not None and asr_confidence < args.review_asr_confidence:
        review_reasons.append("low_asr_confidence")
    tone_confidence = safe_float(response.get("tone_confidence"))
    if tone_confidence is not None and tone_confidence < args.review_tone_confidence:
        review_reasons.append("low_tone_confidence")
    if response.get("recording_quality", "").lower() in {"poor", "bad", "noisy"}:
        review_reasons.append("poor_recording_quality")
    if len(expected_pinyin) != len(predicted_pinyin):
        review_reasons.append("syllable_count_mismatch")
    if not predicted_tones:
        review_reasons.append("missing_tone_output")

    detail["review_required"] = int(bool(review_reasons))
    row["review_required"] = str(bool(review_reasons)).lower()
    row["review_reasons"] = ";".join(review_reasons)
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Score patient repetition outputs using ASR and tone predictions.")
    parser.add_argument("--item-bank", default="data/hearing_demo/item_bank_demo.csv")
    parser.add_argument("--responses", default="data/hearing_demo/patient_responses_demo.csv")
    parser.add_argument("--out-score", default="output/hearing_demo/structured_score.json")
    parser.add_argument("--out-items", default="output/hearing_demo/item_level_results.csv")
    parser.add_argument("--out-confusion", default="output/hearing_demo/confusion_summary.json")
    parser.add_argument("--out-review", default="output/hearing_demo/human_review_items.csv")
    parser.add_argument("--review-asr-confidence", type=float, default=0.80)
    parser.add_argument("--review-tone-confidence", type=float, default=0.75)
    args = parser.parse_args()

    items = load_item_bank(args.item_bank)
    responses = read_csv_rows(args.responses)
    item_rows = []
    details = []
    for response in responses:
        row, detail = evaluate_response(response, items.get(response.get("item_id", "").strip()), args)
        item_rows.append(row)
        details.append(detail)

    item_fieldnames = [
        "patient_id",
        "item_id",
        "text",
        "expected_pinyin",
        "predicted_pinyin",
        "expected_tones",
        "predicted_tones",
        "asr_text",
        "asr_confidence",
        "tone_confidence",
        "recording_quality",
        "item_correct",
        "text_match",
        "keyword_accuracy",
        "syllable_accuracy",
        "initial_accuracy",
        "final_accuracy",
        "tone_accuracy",
        "review_required",
        "review_reasons",
    ]
    write_csv_rows(args.out_items, item_fieldnames, item_rows)

    review_rows = [row for row in item_rows if row["review_required"] == "true"]
    write_csv_rows(args.out_review, item_fieldnames, review_rows)

    tone_confusions = Counter()
    initial_confusions = Counter()
    final_confusions = Counter()
    pattern_counts = Counter()
    review_reason_counts = Counter()
    totals = Counter()

    for row, detail in zip(item_rows, details):
        tone_confusions.update(detail["tone_confusions"])
        initial_confusions.update(detail["initial_confusions"])
        final_confusions.update(detail["final_confusions"])
        pattern_counts.update(detail["pattern_counts"])
        totals["items"] += 1
        totals["item_correct"] += detail["item_correct"]
        totals["text_match"] += detail["text_match"]
        totals["keyword_correct"] += detail["keyword_correct"]
        totals["keyword_total"] += detail["keyword_total"]
        totals["syllable_correct"] += detail["syllable_correct"]
        totals["syllable_total"] += detail["syllable_total"]
        totals["initial_correct"] += detail["initial_correct"]
        totals["initial_total"] += detail["initial_total"]
        totals["final_correct"] += detail["final_correct"]
        totals["final_total"] += detail["final_total"]
        totals["tone_correct"] += detail["tone_correct"]
        totals["tone_total"] += detail["tone_total"]
        totals["ai_readable"] += detail["ai_readable"]
        totals["review_required"] += detail["review_required"]
        for reason in row["review_reasons"].split(";"):
            if reason:
                review_reason_counts[reason] += 1

    score_payload = {
        "generated_at": utc_now_iso(),
        "item_bank": str(Path(args.item_bank)),
        "responses": str(Path(args.responses)),
        "counts": {
            "items": totals["items"],
            "review_required_items": totals["review_required"],
            "ai_readable_items": totals["ai_readable"],
        },
        "metrics": {
            "item_accuracy": accuracy(totals["item_correct"], totals["items"]),
            "text_match_rate": accuracy(totals["text_match"], totals["items"]),
            "keyword_accuracy": accuracy(totals["keyword_correct"], totals["keyword_total"]),
            "syllable_accuracy": accuracy(totals["syllable_correct"], totals["syllable_total"]),
            "initial_accuracy": accuracy(totals["initial_correct"], totals["initial_total"]),
            "final_accuracy": accuracy(totals["final_correct"], totals["final_total"]),
            "tone_accuracy": accuracy(totals["tone_correct"], totals["tone_total"]),
            "asr_readable_rate": accuracy(totals["ai_readable"], totals["items"]),
            "review_rate": accuracy(totals["review_required"], totals["items"]),
        },
        "outputs": {
            "item_level_results_csv": str(Path(args.out_items)),
            "human_review_items_csv": str(Path(args.out_review)),
            "confusion_summary_json": str(Path(args.out_confusion)),
        },
    }
    write_json(args.out_score, score_payload)

    confusion_payload = {
        "generated_at": utc_now_iso(),
        "tone_confusions": top_counter_rows(tone_confusions, limit=10, label_names=("expected_tone", "predicted_tone")),
        "initial_confusions": top_counter_rows(initial_confusions, limit=10, label_names=("expected_initial", "predicted_initial")),
        "final_confusions": top_counter_rows(final_confusions, limit=10, label_names=("expected_final", "predicted_final")),
        "pattern_counts": dict(pattern_counts),
        "review_reason_counts": dict(review_reason_counts.most_common()),
    }
    write_json(args.out_confusion, confusion_payload)
    print(f"wrote={args.out_score}")
    print(f"review_items={len(review_rows)}")
    print(f"tone_accuracy={score_payload['metrics']['tone_accuracy']:.4f}")


if __name__ == "__main__":
    main()
