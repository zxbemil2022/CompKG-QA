"""Validate cs408 expert seed triples against paper-grade ontology/schema."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SEED = ROOT / "cs408_expert_seed.jsonl"
ONTOLOGY = ROOT / "cs408_relation_ontology.yaml"


def main() -> None:
    conf = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8")) or {}
    allowed_relations = set((conf.get("relations") or {}).keys())
    allowed_entity_types = set(conf.get("entity_types") or [])

    errors: list[str] = []
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
        h_type = str(item.get("h_type", "Concept")).strip() or "Concept"
        t_type = str(item.get("t_type", "Concept")).strip() or "Concept"

        if not h or not t:
            errors.append(f"line {idx}: empty head/tail")
        if r not in allowed_relations:
            errors.append(f"line {idx}: relation `{r}` not in ontology")
        if not subject:
            errors.append(f"line {idx}: missing subject")
        if h_type not in allowed_entity_types:
            errors.append(f"line {idx}: h_type `{h_type}` invalid")
        if t_type not in allowed_entity_types:
            errors.append(f"line {idx}: t_type `{t_type}` invalid")

    print(f"total triples: {total}")
    print(f"errors: {len(errors)}")
    if errors:
        for e in errors[:30]:
            print(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
