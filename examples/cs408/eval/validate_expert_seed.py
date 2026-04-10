"""Validate cs408 expert seed triples against relation ontology."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SEED = ROOT / "cs408_expert_seed.jsonl"
ONTOLOGY = ROOT / "cs408_relation_ontology.yaml"


def main():
    rel_conf = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))
    allowed_relations = set((rel_conf.get("relations") or {}).keys())

    errors = []
    total = 0
    for idx, line in enumerate(SEED.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        item = json.loads(line)
        h = str(item.get("h", "")).strip()
        r = str(item.get("r", "")).strip()
        t = str(item.get("t", "")).strip()
        subject = str(item.get("subject", "")).strip()

        if not h or not t:
            errors.append(f"line {idx}: empty head/tail")
        if r not in allowed_relations:
            errors.append(f"line {idx}: relation `{r}` not in ontology")
        if not subject:
            errors.append(f"line {idx}: missing subject")

    print(f"total triples: {total}")
    print(f"errors: {len(errors)}")
    if errors:
        for e in errors[:20]:
            print(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
