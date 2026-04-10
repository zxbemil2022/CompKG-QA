"""Generate CS408 full KG dataset aligned with paper-grade schema.

Entity types: Course/Chapter/Concept/Method/Property/Formula/Example
Relations: BELONGS_TO/HAS_SUB/HAS_METHOD/HAS_PROPERTY/HAS_COMPLEXITY/
          PREREQUISITE/CAUSE/USED_IN/EQUIVALENT
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


RELATIONS = [
    "HAS_SUB",
    "HAS_METHOD",
    "HAS_PROPERTY",
    "HAS_COMPLEXITY",
    "PREREQUISITE",
    "CAUSE",
    "USED_IN",
    "EQUIVALENT",
]
ALLOWED_TYPES = ["Concept", "Method", "Property", "Formula", "Example"]


def _node(nid: str, name: str, subject: str, chapter: str, ntype: str) -> dict:
    return {
        "id": nid,
        "name": name,
        "subject": subject,
        "chapter": chapter,
        "type": ntype,
    }


def main() -> None:
    random.seed(408)
    base = Path(__file__).resolve().parent

    subjects = [
        ("数据结构", ["线性表", "栈与队列", "树与图", "查找排序"]),
        ("操作系统", ["进程线程", "内存管理", "文件系统", "并发同步"]),
        ("计算机网络", ["应用层", "传输层", "网络层", "数据链路层"]),
        ("计算机组成原理", ["数据表示", "指令系统", "存储系统", "流水线"]),
    ]

    nodes: list[dict] = []
    edges: list[dict] = []
    node_id = 1

    for subject, chapters in subjects:
        sid = f"N{node_id:05d}"
        node_id += 1
        nodes.append(_node(sid, subject, subject, "", "Course"))

        for chapter in chapters:
            cid = f"N{node_id:05d}"
            node_id += 1
            nodes.append(_node(cid, f"{subject}-{chapter}", subject, chapter, "Chapter"))
            edges.append({"source": cid, "target": sid, "relation": "BELONGS_TO", "subject": subject, "weight": 1.0})

            for i in range(1, 61):
                nid = f"N{node_id:05d}"
                node_id += 1
                ntype = random.choice(ALLOWED_TYPES)
                name = f"{subject}-{chapter}-{ntype}{i:02d}"
                nodes.append(_node(nid, name, subject, chapter, ntype))
                edges.append({"source": nid, "target": cid, "relation": "BELONGS_TO", "subject": subject, "weight": 1.0})

    chapter_nodes: dict[tuple[str, str], list[dict]] = {}
    for n in nodes:
        if n["type"] in ALLOWED_TYPES:
            chapter_nodes.setdefault((n["subject"], n["chapter"]), []).append(n)

    for (subject, _chapter), arr in chapter_nodes.items():
        for i, n in enumerate(arr):
            if i + 1 < len(arr):
                edges.append({
                    "source": n["id"],
                    "target": arr[i + 1]["id"],
                    "relation": "PREREQUISITE",
                    "subject": subject,
                    "weight": round(random.uniform(0.62, 0.95), 3),
                })
            if i + 3 < len(arr):
                edges.append({
                    "source": n["id"],
                    "target": arr[i + 3]["id"],
                    "relation": random.choice(RELATIONS),
                    "subject": subject,
                    "weight": round(random.uniform(0.52, 0.99), 3),
                })

    by_subject: dict[str, list[dict]] = {s: [] for s, _ in subjects}
    for n in nodes:
        if n["type"] in ALLOWED_TYPES:
            by_subject[n["subject"]].append(n)

    for subject, arr in by_subject.items():
        for _ in range(320):
            a, b = random.sample(arr, 2)
            edges.append({
                "source": a["id"],
                "target": b["id"],
                "relation": random.choice(RELATIONS),
                "subject": subject,
                "weight": round(random.uniform(0.4, 0.98), 3),
            })

    subject_names = list(by_subject.keys())
    for _ in range(700):
        s1, s2 = random.sample(subject_names, 2)
        a = random.choice(by_subject[s1])
        b = random.choice(by_subject[s2])
        edges.append({
            "source": a["id"],
            "target": b["id"],
            "relation": random.choice(["USED_IN", "EQUIVALENT", "CAUSE"]),
            "subject": f"{s1}|{s2}",
            "weight": round(random.uniform(0.35, 0.9), 3),
        })

    uniq = {(e["source"], e["target"], e["relation"]): e for e in edges}
    edges = list(uniq.values())

    id_to_name = {n["id"]: n["name"] for n in nodes}
    id_to_type = {n["id"]: n["type"] for n in nodes}
    triples = [
        {
            "h": id_to_name[e["source"]],
            "h_type": id_to_type[e["source"]],
            "r": e["relation"],
            "t": id_to_name[e["target"]],
            "t_type": id_to_type[e["target"]],
            "subject": e["subject"],
        }
        for e in edges
    ]

    (base / "cs408_full_kg_nodes.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    with (base / "cs408_full_kg_edges.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "relation", "subject", "weight"])
        writer.writeheader()
        writer.writerows(edges)

    (base / "cs408_full_kg_triples.json").write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (base / "cs408_full_kg_triples.jsonl").open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    meta = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "subject_count": len(subjects),
        "subjects": [s for s, _ in subjects],
        "allowed_entity_types": ["Course", "Chapter", *ALLOWED_TYPES],
        "allowed_relations": ["BELONGS_TO", *RELATIONS],
    }
    (base / "cs408_full_kg_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()


