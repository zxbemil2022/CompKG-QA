from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class KGEnhancementResult:
    fused_triples: list[tuple[str, str, str]]
    inferred_triples: list[tuple[str, str, str]]
    provenance: dict[str, list[str]]


class KGCompleterAndFusion:
    """图谱补全与多源融合（轻量规则版）。"""

    _SYMMETRIC_RELS = {"COMPATIBLE_WITH", "ALTERNATIVE_TO", "RELATED_TO"}
    _INVERSE_REL = {
        "DEPENDS_ON": "REQUIRED_BY",
        "PART_OF": "HAS_PART",
        "USES": "USED_BY",
    }

    @staticmethod
    def _triple_to_tuple(item: Any) -> tuple[str, str, str] | None:
        if item is None:
            return None
        if isinstance(item, tuple) and len(item) >= 3:
            return str(item[0]).strip(), str(item[1]).strip().upper(), str(item[2]).strip()
        sub = str(getattr(item, "subject", "")).strip()
        pred = str(getattr(item, "predicate", "")).strip().upper()
        obj = str(getattr(item, "obj", "")).strip()
        if sub and pred and obj:
            return sub, pred, obj
        return None

    def merge_sources(self, source_to_triples: dict[str, list[Any]]) -> KGEnhancementResult:
        fused: list[tuple[str, str, str]] = []
        inferred: list[tuple[str, str, str]] = []
        provenance: dict[str, list[str]] = defaultdict(list)
        seen = set()

        for source, triples in (source_to_triples or {}).items():
            for t in triples or []:
                key = self._triple_to_tuple(t)
                if not key or not all(key):
                    continue
                if key not in seen:
                    seen.add(key)
                    fused.append(key)
                provenance[str(key)].append(source)

        for t in list(fused):
            if t[1] in self._SYMMETRIC_RELS:
                sym = (t[2], t[1], t[0])
                if sym not in seen and t[0] != t[2]:
                    seen.add(sym)
                    inferred.append(sym)
                    provenance[str(sym)].append("inferred:symmetric")

            if t[1] in self._INVERSE_REL:
                inv = (t[2], self._INVERSE_REL[t[1]], t[0])
                if inv not in seen and t[0] != t[2]:
                    seen.add(inv)
                    inferred.append(inv)
                    provenance[str(inv)].append("inferred:inverse")

        dep_edges = [(s, o) for s, p, o in fused if p == "DEPENDS_ON"]
        for a, b in dep_edges:
            for x, c in dep_edges:
                if b == x and a != c:
                    tri = (a, "DEPENDS_ON", c)
                    if tri not in seen:
                        seen.add(tri)
                        inferred.append(tri)
                        provenance[str(tri)].append("inferred:transitive")

        return KGEnhancementResult(fused_triples=fused, inferred_triples=inferred, provenance=dict(provenance))


def visualize_reasoning_paths(triples: list[Any], max_paths: int = 12) -> dict:
    """输出可视化友好的节点-边结构 + 简单路径字符串。"""
    nodes = {}
    edges = []
    paths = []

    normalized = []
    for t in triples:
        row = KGCompleterAndFusion._triple_to_tuple(t)
        if row:
            normalized.append(row)

    for sub, pred, obj in normalized[: max_paths * 3]:
        nodes[sub] = {"id": sub, "label": sub, "type": "entity"}
        nodes[obj] = {"id": obj, "label": obj, "type": "entity"}
        edges.append({"source": sub, "target": obj, "label": pred})
        if len(paths) < max_paths:
            paths.append(f"{sub} --{pred}--> {obj}")

    for a, r1, b in [(e["source"], e["label"], e["target"]) for e in edges]:
        for x, r2, c in [(e["source"], e["label"], e["target"]) for e in edges]:
            if b == x and len(paths) < max_paths:
                paths.append(f"{a} --{r1}--> {b} --{r2}--> {c}")

    return {"nodes": list(nodes.values()), "edges": edges, "reasoning_paths": paths}
