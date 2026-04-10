from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LinkedEntity:
    mention: str
    canonical_name: str
    ontology_id: str
    ontology_source: str
    confidence: float


class OntologyEntityLinker:
    """轻量实体链接器（可对接专业本体库）。

    当前实现支持：
    - 内置计算机领域小型本体映射；
    - 规则归一化（大小写、分隔符、常见别名）；
    - 未命中回退到规范化 mention。

    后续可扩展为远程本体 API（UMLS/Wikidata/DBpedia/自建术语库）。
    """

    def __init__(self, custom_ontology: dict[str, dict] | None = None):
        base = {
            "tcp": {"canonical": "Transmission Control Protocol", "id": "IETF:TCP", "source": "IETF"},
            "udp": {"canonical": "User Datagram Protocol", "id": "IETF:UDP", "source": "IETF"},
            "http": {"canonical": "Hypertext Transfer Protocol", "id": "IETF:HTTP", "source": "IETF"},
            "dns": {"canonical": "Domain Name System", "id": "IETF:DNS", "source": "IETF"},
            "bfs": {"canonical": "Breadth-First Search", "id": "CS:ALG:BFS", "source": "CSOntology"},
            "dfs": {"canonical": "Depth-First Search", "id": "CS:ALG:DFS", "source": "CSOntology"},
            "dijkstra": {"canonical": "Dijkstra Algorithm", "id": "CS:ALG:DIJKSTRA", "source": "CSOntology"},
            "red black tree": {"canonical": "Red-Black Tree", "id": "CS:DS:RBTREE", "source": "CSOntology"},
            "redis": {"canonical": "Redis", "id": "TECH:REDIS", "source": "TechOntology"},
            "mysql": {"canonical": "MySQL", "id": "TECH:MYSQL", "source": "TechOntology"},
            "neo4j": {"canonical": "Neo4j", "id": "TECH:NEO4J", "source": "TechOntology"},
            "rag": {"canonical": "Retrieval-Augmented Generation", "id": "LLM:RAG", "source": "LLMOntology"},
            "graph rag": {"canonical": "Graph Retrieval-Augmented Generation", "id": "LLM:GRAPH_RAG", "source": "LLMOntology"},
            "langgraph": {"canonical": "LangGraph", "id": "TOOL:LANGGRAPH", "source": "ToolOntology"},
            "mcp": {"canonical": "Model Context Protocol", "id": "PROTOCOL:MCP", "source": "Anthropic/OpenProtocol"},
        }
        self.ontology = {**base, **(custom_ontology or {})}

    @staticmethod
    def _normalize(text: str) -> str:
        t = (text or "").strip().lower()
        t = t.replace("_", " ").replace("-", " ")
        t = re.sub(r"\s+", " ", t)
        return t

    def link(self, entities: list[str]) -> list[LinkedEntity]:
        linked: list[LinkedEntity] = []
        for e in entities or []:
            mention = (e or "").strip()
            if not mention:
                continue
            key = self._normalize(mention)
            hit = self.ontology.get(key)
            if hit:
                linked.append(
                    LinkedEntity(
                        mention=mention,
                        canonical_name=hit["canonical"],
                        ontology_id=hit["id"],
                        ontology_source=hit["source"],
                        confidence=0.95,
                    )
                )
            else:
                linked.append(
                    LinkedEntity(
                        mention=mention,
                        canonical_name=mention,
                        ontology_id=f"LOCAL:{self._normalize(mention).replace(' ', '_')[:48]}",
                        ontology_source="LOCAL",
                        confidence=0.5,
                    )
                )
        return linked
