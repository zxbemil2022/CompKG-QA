from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit_cs408_dataset(path: str) -> dict[str, Any]:
    dataset_path = Path(path)
    rows = _load_jsonl(dataset_path)
    issues: list[dict[str, Any]] = []
    relation_counter: Counter[str] = Counter()
    subject_counter: Counter[str] = Counter()
    seen = set()

    for idx, row in enumerate(rows):
        rid = row.get("id") or f"row-{idx}"
        head = str(row.get("head", "")).strip()
        relation = str(row.get("relation", "")).strip()
        tail = str(row.get("tail", "")).strip()
        subject = str(row.get("subject", "")).strip() or "unknown"

        subject_counter[subject] += 1
        if relation:
            relation_counter[relation] += 1

        if not head or not relation or not tail:
            issues.append({"id": rid, "type": "missing_field", "detail": "head/relation/tail 不能为空"})

        dedupe_key = (head.lower(), relation.lower(), tail.lower(), subject.lower())
        if dedupe_key in seen:
            issues.append({"id": rid, "type": "duplicate", "detail": "重复三元组"})
        seen.add(dedupe_key)

        if relation in {"HAS_COMPLEXITY", "has_complexity"} and "O(" not in tail:
            issues.append({"id": rid, "type": "complexity_format", "detail": "复杂度关系未使用 O() 形式"})

        if relation in {"PREREQUISITE", "prerequisite_of"} and head == tail:
            issues.append({"id": rid, "type": "prerequisite_loop", "detail": "先修关系出现自环"})

    issue_counter = Counter(item["type"] for item in issues)
    score = max(0.0, 1.0 - len(issues) / max(1, len(rows)))
    subject_counts = dict(subject_counter)
    category_std = 0.0
    if subject_counts:
        vals = list(subject_counts.values())
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        category_std = round(variance ** 0.5, 4)

    return {
        "dataset_path": str(dataset_path),
        "sample_count": len(rows),
        "quality_score": round(score, 4),
        "issue_count": len(issues),
        "issue_breakdown": dict(issue_counter),
        "subject_distribution": subject_counts,
        "subject_balance_std": category_std,
        "top_relations": relation_counter.most_common(12),
        "issues": issues[:500],
    }


def upgrade_cs408_dataset(path: str, output_path: str | None = None) -> dict[str, Any]:
    dataset_path = Path(path)
    rows = _load_jsonl(dataset_path)

    normalized_rows: list[dict[str, Any]] = []
    seen = set()
    deduped_count = 0

    for row in rows:
        normalized = dict(row)
        normalized["head"] = str(normalized.get("head", "")).strip()
        normalized["relation"] = str(normalized.get("relation", "")).strip().lower()
        normalized["tail"] = str(normalized.get("tail", "")).strip()
        normalized["subject"] = str(normalized.get("subject", "computer")).strip() or "computer"

        key = (
            normalized["head"].lower(),
            normalized["relation"],
            normalized["tail"].lower(),
            normalized["subject"].lower(),
        )
        if key in seen:
            deduped_count += 1
            continue
        seen.add(key)
        normalized_rows.append(normalized)

    target = Path(output_path) if output_path else dataset_path
    _dump_jsonl(target, normalized_rows)

    return {
        "input_path": str(dataset_path),
        "output_path": str(target),
        "input_count": len(rows),
        "output_count": len(normalized_rows),
        "deduped_count": deduped_count,
    }
