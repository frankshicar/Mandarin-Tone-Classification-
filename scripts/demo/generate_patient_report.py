#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.common.hearing_pipeline_utils import read_csv_rows, write_json


TONE_REHAB_LIBRARY = {
    ("2", "3"): ["麻 / 馬", "白 / 擺", "姨 / 椅"],
    ("3", "2"): ["馬 / 麻", "椅 / 姨", "擺 / 白"],
    ("3", "4"): ["好 / 號", "米 / 密", "請 / 慶"],
    ("4", "3"): ["號 / 好", "密 / 米", "慶 / 請"],
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_sentence(metrics: dict[str, float]) -> str:
    overall = metrics["item_accuracy"]
    tone = metrics["tone_accuracy"]
    if overall >= 0.85:
        level = "本次整體複誦表現穩定。"
    elif overall >= 0.65:
        level = "本次整體複誦表現中等，已有主要題型的基礎辨識能力。"
    elif overall >= 0.45:
        level = "本次整體複誦表現顯示仍有明顯的聽辨困難。"
    else:
        level = "本次整體複誦表現偏弱，建議優先聚焦在核心聽辨對比訓練。"
    if tone < 0.70:
        return level + " 其中聲調辨識是本次最需要優先處理的面向。"
    return level


def build_rehab_priorities(metrics: dict[str, float], confusion: dict, item_rows: list[dict]) -> list[dict]:
    priorities = []
    tone_confusions = confusion.get("tone_confusions", [])
    if metrics["tone_accuracy"] < 0.85 or tone_confusions:
        top = tone_confusions[0] if tone_confusions else None
        examples = ["媽 / 麻 / 馬 / 罵", "青 / 情 / 請 / 慶"]
        reason = f"聲調正確率為 {pct(metrics['tone_accuracy'])}。"
        if top:
            pair = (top.get("expected_tone", ""), top.get("predicted_tone", ""))
            examples = TONE_REHAB_LIBRARY.get(pair, examples)
            reason += f" 最常見混淆為 {pair[0]} 聲 -> {pair[1]} 聲。"
        priorities.append(
            {
                "focus_area": "聲調最小對比聽辨",
                "reason": reason,
                "suggested_tasks": [
                    "先做單音節最小對比辨識，再進入雙音節詞複誦。",
                    "每回合先聽 2~3 組固定對比，再加入隨機順序。",
                ],
                "example_items": examples,
            }
        )

    if confusion.get("pattern_counts", {}).get("retroflex_vs_palatal", 0) > 0 or metrics["initial_accuracy"] < 0.85:
        priorities.append(
            {
                "focus_area": "捲舌音與舌面音聽辨",
                "reason": f"聲母正確率為 {pct(metrics['initial_accuracy'])}，且出現 zh/ch/sh 與 j/q/x 混淆。",
                "suggested_tasks": [
                    "以單音節最小對比練習 `sh` vs `x`、`zh` vs `j`。",
                    "加入圖片或語意配對，避免只靠記憶猜題。",
                ],
                "example_items": ["師 / 西", "知 / 雞", "吃 / 七"],
            }
        )

    if confusion.get("pattern_counts", {}).get("n_ng", 0) > 0 or metrics["final_accuracy"] < 0.85:
        priorities.append(
            {
                "focus_area": "鼻音韻尾 -n / -ng 聽辨",
                "reason": f"韻母正確率為 {pct(metrics['final_accuracy'])}，且觀察到鼻音韻尾混淆。",
                "suggested_tasks": [
                    "先做 `an` / `ang`、`in` / `ing` 對比，再進入詞級辨識。",
                    "搭配慢速播放與重複播放，確認尾音差異。",
                ],
                "example_items": ["安 / 昂", "新 / 星", "班 / 幫"],
            }
        )

    if metrics["keyword_accuracy"] < 0.85:
        priorities.append(
            {
                "focus_area": "關鍵字辨識與短詞複誦",
                "reason": f"關鍵字正確率為 {pct(metrics['keyword_accuracy'])}，建議先鞏固詞級辨識穩定度。",
                "suggested_tasks": [
                    "先用高頻雙音節詞做封閉式聽辨，再進入開放式複誦。",
                    "逐步加入噪音條件，檢查關鍵字是否仍能保留。",
                ],
                "example_items": ["青菜", "白兔", "安靜", "老師"],
            }
        )

    if not priorities:
        priorities.append(
            {
                "focus_area": "維持性聽辨訓練",
                "reason": "本次主要指標整體穩定，可維持多題型與多說話者語音暴露。",
                "suggested_tasks": [
                    "維持多說話者、多語速條件下的複誦練習。",
                    "定期抽查聲調與關鍵字辨識，避免特定對比退步。",
                ],
                "example_items": ["多說話者短詞辨識", "安靜與噪音條件交替練習"],
            }
        )

    return priorities[:4]


def representative_items(item_rows: list[dict], limit: int = 3) -> list[dict]:
    sorted_rows = sorted(
        item_rows,
        key=lambda row: (
            row.get("review_required") != "true",
            -(1.0 - float(row.get("tone_accuracy", "0"))),
            -(1.0 - float(row.get("syllable_accuracy", "0"))),
        ),
    )
    return sorted_rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a patient-facing reference report and rehab plan.")
    parser.add_argument("--score", default="output/hearing_demo/structured_score.json")
    parser.add_argument("--confusion", default="output/hearing_demo/confusion_summary.json")
    parser.add_argument("--items", default="output/hearing_demo/item_level_results.csv")
    parser.add_argument("--session", default="data/hearing_demo/patient_session_demo.json")
    parser.add_argument("--template", default="聽能分析報告/AI華語聽能複誦分析參考報告模板.md")
    parser.add_argument("--out-report", default="output/hearing_demo/patient_report.md")
    parser.add_argument("--out-rehab", default="output/hearing_demo/rehab_plan.json")
    args = parser.parse_args()

    score = load_json(args.score)
    confusion = load_json(args.confusion)
    session = load_json(args.session) if Path(args.session).exists() else {}
    item_rows = read_csv_rows(args.items)
    metrics = score["metrics"]
    priorities = build_rehab_priorities(metrics, confusion, item_rows)
    examples = representative_items(item_rows)

    patient_id = session.get("patient_id") or (item_rows[0]["patient_id"] if item_rows else "")
    test_date = session.get("test_date") or datetime.now(timezone.utc).date().isoformat()
    report_lines = [
        "# AI 華語聽能複誦分析參考報告",
        "",
        "## 一、測驗資訊",
        f"- 受試者編號：{patient_id}",
        f"- 測驗日期：{test_date}",
        f"- 刺激來源：{session.get('stimulus_source', 'Qwen3-TTS')}",
        f"- 測驗材料：{session.get('materials', '雙音節詞')}",
        f"- 測驗條件：{session.get('condition', '安靜')}",
        f"- SNR：{session.get('snr', 'N/A')}",
        f"- 題數：{score['counts']['items']}",
        f"- 播放音量：{session.get('playback_level', '未記錄')}",
        f"- 輸入方式：{session.get('input_mode', '口語複誦')}",
        f"- 分析模型：{session.get('analysis_model', 'ASR + 聲調辨識模型')}",
        "",
        "## 二、整體表現摘要",
        f"- 整體複誦正確率：{pct(metrics['item_accuracy'])}",
        f"- 關鍵字正確率：{pct(metrics['keyword_accuracy'])}",
        f"- 聲母聽辨正確率：{pct(metrics['initial_accuracy'])}",
        f"- 韻母聽辨正確率：{pct(metrics['final_accuracy'])}",
        f"- 聲調聽辨正確率：{pct(metrics['tone_accuracy'])}",
        f"- ASR 可判讀比例：{pct(metrics['asr_readable_rate'])}",
        f"- 低信心 / 人工複核比例：{pct(metrics['review_rate'])}",
        "",
        summary_sentence(metrics),
        "",
        "## 三、主要聽辨混淆型態",
    ]
    if confusion.get("tone_confusions"):
        top = confusion["tone_confusions"][0]
        report_lines.append(
            f"- 聲調混淆：最常見為 {top['expected_tone']} 聲 -> {top['predicted_tone']} 聲，共 {top['count']} 次。"
        )
    if confusion.get("initial_confusions"):
        top = confusion["initial_confusions"][0]
        report_lines.append(
            f"- 聲母混淆：最常見為 `{top['expected_initial']}` -> `{top['predicted_initial']}`，共 {top['count']} 次。"
        )
    if confusion.get("final_confusions"):
        top = confusion["final_confusions"][0]
        report_lines.append(
            f"- 韻母混淆：最常見為 `{top['expected_final']}` -> `{top['predicted_final']}`，共 {top['count']} 次。"
        )
    if not any(confusion.get(name) for name in ("tone_confusions", "initial_confusions", "final_confusions")):
        report_lines.append("- 本次未觀察到明顯集中於單一類型的混淆。")

    report_lines.extend(
        [
            "",
            "代表題目：",
        ]
    )
    for row in examples:
        report_lines.append(
            f"- {row['item_id']} `{row['text']}`：預期 `{row['expected_pinyin']}`，患者 `{row['predicted_pinyin'] or row['asr_text']}`，"
            f" 聲調正確率 {pct(float(row['tone_accuracy']))}。"
        )

    report_lines.extend(
        [
            "",
            "## 四、AI 判讀信心與人工複核建議",
            "- 以下情況建議人工複核：錄音品質不佳、ASR 信心低、ASR 與聲調結果不一致、患者多次自我修正。",
            f"- 本次需人工複核題數：{score['counts']['review_required_items']}",
        ]
    )
    if confusion.get("review_reason_counts"):
        report_lines.append(f"- 常見複核原因：{', '.join(confusion['review_reason_counts'].keys())}")

    report_lines.extend(
        [
            "",
            "## 五、聽能訓練建議與復健題目",
        ]
    )
    for index, priority in enumerate(priorities, start=1):
        report_lines.append(f"{index}. {priority['focus_area']}")
        report_lines.append(f"   原因：{priority['reason']}")
        report_lines.append(f"   建議活動：{'；'.join(priority['suggested_tasks'])}")
        report_lines.append(f"   參考題目：{'、'.join(priority['example_items'])}")

    report_lines.extend(
        [
            "",
            "## 六、備註",
            "- 本報告為 AI 自動分析產生之參考結果，不能單獨作為臨床診斷依據。",
            "- 結果可能受注意力、短期記憶、口語輸出、錄音品質、ASR 辨識誤差與模型不確定性影響。",
            f"- 模板參考：{args.template}",
        ]
    )

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    rehab_payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "patient_id": patient_id,
        "source_score": str(Path(args.score)),
        "source_confusion": str(Path(args.confusion)),
        "priorities": priorities,
    }
    write_json(args.out_rehab, rehab_payload)
    print(f"wrote={args.out_report}")
    print(f"priorities={len(priorities)}")


if __name__ == "__main__":
    main()
