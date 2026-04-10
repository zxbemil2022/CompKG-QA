"""QA regression gate.

Usage:
  python examples/cs408/eval/run_qa_regression_gate.py \
    --dataset examples/cs408/eval/qa_regression_dataset.jsonl \
    --predictions examples/cs408/eval/predictions.template.jsonl \
    --min-overall 0.65 --min-citation 0.6
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    return all(k.lower() in text.lower() for k in keywords)


def has_citation(text: str) -> bool:
    # 支持 evidence_id 形态，如 G001 / V001
    upper = text.upper()
    return any(f"{prefix}{i:03d}" in upper for prefix in ["G", "V"] for i in range(1, 200))


def evaluate(dataset: list[dict], preds: dict[str, dict]) -> dict:
    by_category: dict[str, dict[str, int]] = {}
    total = 0
    passed = 0
    citation_total = 0
    citation_passed = 0
    evidence_truth_total = 0
    evidence_truth_passed = 0

    for item in dataset:
        total += 1
        pred = preds.get(item["id"], {})
        answer = str(pred.get("answer", ""))
        category = item.get("category", "unknown")
        by_category.setdefault(category, {"total": 0, "passed": 0})
        by_category[category]["total"] += 1

        keyword_ok = contains_all_keywords(answer, item.get("must_include", []))
        citation_ok = True
        if item.get("must_cite"):
            citation_total += 1
            citation_ok = has_citation(answer)
            if citation_ok:
                citation_passed += 1
        gold_evidence_ids = set(item.get("gold_evidence_ids", []))
        pred_evidence_ids = set(pred.get("evidence_ids", []))
        if gold_evidence_ids:
            evidence_truth_total += 1
            if pred_evidence_ids and (pred_evidence_ids & gold_evidence_ids):
                evidence_truth_passed += 1

        if keyword_ok and citation_ok:
            passed += 1
            by_category[category]["passed"] += 1

    overall = passed / total if total else 0.0
    citation_rate = citation_passed / citation_total if citation_total else 1.0
    category_scores = {
        c: (v["passed"] / v["total"] if v["total"] else 0.0)
        for c, v in by_category.items()
    }
    category_std = round(statistics.pstdev(category_scores.values()), 4) if len(category_scores) > 1 else 0.0
    evidence_truth_rate = evidence_truth_passed / evidence_truth_total if evidence_truth_total else 1.0
    return {
        "overall": round(overall, 4),
        "citation_rate": round(citation_rate, 4),
        "evidence_truth_rate": round(evidence_truth_rate, 4),
        "category_scores": category_scores,
        "category_balance_std": category_std,
        "total": total,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--min-overall", type=float, default=0.65)
    parser.add_argument("--min-citation", type=float, default=0.6)
    parser.add_argument("--max-category-std", type=float, default=0.25)
    parser.add_argument("--min-evidence-truth", type=float, default=0.5)
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    pred_rows = load_jsonl(Path(args.predictions))
    preds = {row["id"]: row for row in pred_rows}

    report = evaluate(dataset, preds)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["overall"] < args.min_overall:
        raise SystemExit(f"[GATE_FAIL] overall={report['overall']} < {args.min_overall}")
    if report["citation_rate"] < args.min_citation:
        raise SystemExit(f"[GATE_FAIL] citation_rate={report['citation_rate']} < {args.min_citation}")
    if report["category_balance_std"] > args.max_category_std:
        raise SystemExit(f"[GATE_FAIL] category_balance_std={report['category_balance_std']} > {args.max_category_std}")
    if report["evidence_truth_rate"] < args.min_evidence_truth:
        raise SystemExit(f"[GATE_FAIL] evidence_truth_rate={report['evidence_truth_rate']} < {args.min_evidence_truth}")
    print("[GATE_PASS] regression gate passed.")


if __name__ == "__main__":
    main()

