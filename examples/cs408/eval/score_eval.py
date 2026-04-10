"""Offline evaluation scorer: accuracy / hallucination / evidence coverage."""

import json
import statistics
from pathlib import Path


def token_overlap(a: str, b: str) -> float:
    sa = set([x for x in a.lower().replace('，', ' ').replace('。', ' ').split() if x])
    sb = set([x for x in b.lower().replace('，', ' ').replace('。', ' ').split() if x])
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def load_jsonl(path: Path):
    items = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            obj = json.loads(line)
            items[obj['id']] = obj
    return items


def main():
    base = Path(__file__).resolve().parent
    gold = load_jsonl(base / 'offline_eval_dataset.jsonl')
    pred_file = base / 'predictions.jsonl'
    if not pred_file.exists():
        print('predictions.jsonl not found, copy from predictions.template.jsonl first')
        raise SystemExit(1)
    pred = load_jsonl(pred_file)

    acc_scores = []
    hallucinations = 0
    coverage_scores = []
    contradiction_hits = 0

    for qid, g in gold.items():
        p = pred.get(qid, {'answer': '', 'evidence_ids': [], 'source_kbs': []})
        acc = token_overlap(p.get('answer', ''), g.get('reference_answer', ''))
        acc_scores.append(acc)

        evidence_ids = set(p.get('evidence_ids', []))
        if not evidence_ids and p.get('answer', '').strip():
            hallucinations += 1

        gold_e = set(g.get('gold_evidence_ids', []))
        coverage = len(evidence_ids & gold_e) / len(gold_e) if gold_e else 0
        coverage_scores.append(coverage)
        if g.get("negative_markers"):
            if any(marker in p.get("answer", "") for marker in g["negative_markers"]):
                contradiction_hits += 1

    n = len(gold)
    result = {
        'sample_count': n,
        'accuracy': round(sum(acc_scores) / n, 4),
        'hallucination_rate': round(hallucinations / n, 4),
        'evidence_coverage': round(sum(coverage_scores) / n, 4),
        'accuracy_std': round(statistics.pstdev(acc_scores), 4) if len(acc_scores) > 1 else 0.0,
        'contradiction_flag_rate': round(contradiction_hits / n, 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

