from __future__ import annotations

import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


from src.knowledge.entity_linking import OntologyEntityLinker
from src.knowledge.kg_enhancement import KGCompleterAndFusion

@dataclass
class SPOTriple:
    subject: str
    predicate: str
    obj: str


@dataclass
class KGExtractionResult:
    cleaned_text: str
    segments: list[str]
    entities: list[str]
    triples: list[SPOTriple]
    linked_entities: list[dict] = field(default_factory=list)
    inferred_triples: list[SPOTriple] = field(default_factory=list)
    provenance: dict[str, list[str]] = field(default_factory=dict)


class BaseNERPlugin(ABC):
    @abstractmethod
    def extract(self, segments: list[str]) -> list[str]:
        raise NotImplementedError


class BaseREPlugin(ABC):
    @abstractmethod
    def extract(self, segments: list[str], entities: list[str]) -> list[SPOTriple]:
        raise NotImplementedError


class RuleNERPlugin(BaseNERPlugin):
    def extract(self, segments: list[str]) -> list[str]:
        entities = set()
        for seg in segments:
            for m in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,20}", seg):
                entities.add(m)
        return list(entities)[:300]


class RuleREPlugin(BaseREPlugin):
    _REL_PATTERNS = [
        (r"(.{1,30})是(.{1,30})的一种", "IS_A"),
        (r"(.{1,30})包括(.{1,30})", "INCLUDES"),
        (r"(.{1,30})由(.{1,30})组成", "COMPOSED_OF"),
        (r"(.{1,30})用于(.{1,30})", "USED_FOR"),
        (r"(.{1,30})依赖(.{1,30})", "DEPENDS_ON"),
    ]

    def extract(self, segments: list[str], entities: list[str]) -> list[SPOTriple]:
        triples: list[SPOTriple] = []
        for seg in segments:
            matched = False
            for pattern, rel in self._REL_PATTERNS:
                m = re.search(pattern, seg)
                if not m:
                    continue
                sub = m.group(1).strip(" ，,：:")[:60]
                obj = m.group(2).strip(" ，,：:")[:60]
                if sub and obj and sub != obj:
                    triples.append(SPOTriple(sub, rel, obj))
                matched = True
                break

            if not matched and "是" in seg:
                parts = seg.split("是", 1)
                if len(parts) == 2:
                    sub = parts[0].strip()[:60]
                    obj = parts[1].strip()[:60]
                    if sub and obj and sub != obj:
                        triples.append(SPOTriple(sub, "RELATED_TO", obj))
        return triples


NER_PLUGIN_REGISTRY: dict[str, type[BaseNERPlugin]] = {
    "rule": RuleNERPlugin,
}
RE_PLUGIN_REGISTRY: dict[str, type[BaseREPlugin]] = {
    "rule": RuleREPlugin,
}


def register_ner_plugin(name: str, plugin_cls: type[BaseNERPlugin]) -> None:
    NER_PLUGIN_REGISTRY[name] = plugin_cls


def register_re_plugin(name: str, plugin_cls: type[BaseREPlugin]) -> None:
    RE_PLUGIN_REGISTRY[name] = plugin_cls


class UnstructuredToKGPipeline:
    """
    GraphRAG 风格的最小可用流水线：
    文档清洗 -> 分段 -> NER(规则) -> 关系抽取(规则) -> 三元组
    """

    def __init__(
        self,
        ner_plugin: str = "rule",
        re_plugin: str = "rule",
        ner_kwargs: dict | None = None,
        re_kwargs: dict | None = None,
        enable_entity_linking: bool = True,
        enable_kg_enhancement: bool = True,
    ):
        ner_cls = NER_PLUGIN_REGISTRY.get(ner_plugin, RuleNERPlugin)
        re_cls = RE_PLUGIN_REGISTRY.get(re_plugin, RuleREPlugin)
        self.ner_plugin_name = ner_plugin
        self.re_plugin_name = re_plugin
        ner_kwargs = ner_kwargs or {}
        re_kwargs = re_kwargs or {}
        try:
            self.ner_plugin: BaseNERPlugin = ner_cls(**ner_kwargs)
        except TypeError:
            self.ner_plugin = ner_cls()
        try:
            self.re_plugin: BaseREPlugin = re_cls(**re_kwargs)
        except TypeError:
            self.re_plugin = re_cls()

        self.enable_entity_linking = enable_entity_linking
        self.enable_kg_enhancement = enable_kg_enhancement
        self.entity_linker = OntologyEntityLinker() if enable_entity_linking else None
        self.kg_enhancer = KGCompleterAndFusion() if enable_kg_enhancement else None

    def preprocess(self, text: str) -> tuple[str, list[str]]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return "", []
        segments = [seg.strip(" ，,：:") for seg in re.split(r"[。！？\n;；]", cleaned) if seg.strip()]
        return cleaned, segments

    def extract_entities(self, segments: list[str]) -> list[str]:
        return self.ner_plugin.extract(segments)

    def extract_relations(self, segments: list[str], entities: list[str]) -> list[SPOTriple]:
        return self.re_plugin.extract(segments, entities)

    def run(self, text: str, max_triples: int = 80) -> KGExtractionResult:
        cleaned, segments = self.preprocess(text)
        entities = self.extract_entities(segments)
        triples = self.extract_relations(segments, entities)

        dedup = []
        seen = set()
        for tri in triples:
            key = (tri.subject, tri.predicate, tri.obj)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(tri)
            if len(dedup) >= max_triples:
                break

        linked_entities = []
        if self.entity_linker is not None:
            linked_entities = [
                {
                    "mention": le.mention,
                    "canonical_name": le.canonical_name,
                    "ontology_id": le.ontology_id,
                    "ontology_source": le.ontology_source,
                    "confidence": le.confidence,
                }
                for le in self.entity_linker.link(entities)
            ]

        inferred_triples: list[SPOTriple] = []
        provenance: dict[str, list[str]] = {}
        if self.kg_enhancer is not None:
            enhancement = self.kg_enhancer.merge_sources({"primary": dedup})
            dedup = [SPOTriple(subject=s, predicate=p, obj=o) for s, p, o in (enhancement.fused_triples + enhancement.inferred_triples)]
            inferred_triples = [SPOTriple(subject=s, predicate=p, obj=o) for s, p, o in enhancement.inferred_triples]
            provenance = enhancement.provenance

        return KGExtractionResult(
            cleaned_text=cleaned,
            segments=segments,
            entities=entities,
            triples=dedup,
            linked_entities=linked_entities,
            inferred_triples=inferred_triples,
            provenance=provenance,
        )
