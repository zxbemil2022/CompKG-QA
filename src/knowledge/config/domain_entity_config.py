"""知识图谱领域配置（默认：计算机领域）。"""

from __future__ import annotations

COMPUTER_ENTITY_TYPES = [
    "Technology",  # 技术/概念（如虚拟内存、微服务）
    "Algorithm",  # 算法（如Dijkstra、Transformer）
    "DataStructure",  # 数据结构（如B-Tree、HashMap）
    "ProgrammingLanguage",  # 编程语言（如Python、Rust）
    "Framework",  # 框架（如Spring、Vue）
    "Library",  # 库（如NumPy、Pandas）
    "Protocol",  # 协议（如HTTP、TCP）
    "Database",  # 数据库（如MySQL、Redis）
    "Tool",  # 工具（如Docker、Git）
    "Platform",  # 平台（如Linux、Kubernetes）
    "Organization",  # 组织（如Apache Foundation）
    "Person",  # 人物（如Alan Turing）
]

COMPUTER_RELATION_TYPES = [
    "is_a",  # A 是 B 的一种
    "part_of",  # A 是 B 的一部分
    "depends_on",  # A 依赖 B
    "implements",  # A 实现了 B
    "uses",  # A 使用 B
    "compatible_with",  # A 与 B 兼容
    "compared_with",  # A 与 B 可比较
    "proposed_by",  # A 由 B 提出
    "developed_by",  # A 由 B 开发
    "runs_on",  # A 运行在 B 上
    "stores_in",  # A 存储于 B
    "communicates_via",  # A 通过 B 通信
    "extends",  # A 扩展了 B
    "optimized_for",  # A 针对 B 优化
    "alternative_to",  # A 可替代 B
    "introduced_in",  # A 在 B（版本/时期）中引入
    "measured_by",  # A 由 B 指标衡量
]

MUSEUM_ENTITY_TYPES = [
    "Artifact",
    "Period",
    "Site",
    "Category",
    "Material",
    "Function",
    "Person",
    "State",
    "Exhibition",
    "Theme",
    "Ritual",
]

MUSEUM_RELATION_TYPES = [
    "belongs_to",
    "created_in",
    "discovered_at",
    "made_of",
    "used_for",
    "related_to",
    "exhibited_in",
    "part_of",
    "influenced_by",
    "represents",
    "restored_by",
    "located_in",
    "documented_in",
]

CS408_ENTITY_TYPES = [
    "Course",
    "Chapter",
    "Concept",
    "Algorithm",
    "DataStructure",
    "Protocol",
    "SystemModule",
    "Formula",
    "Complexity",
    "Scenario",
    "Pitfall",
]

CS408_RELATION_TYPES = [
    "belongs_to",
    "has_subtopic",
    "prerequisite_of",
    "implemented_by",
    "optimized_by",
    "uses",
    "depends_on",
    "compared_with",
    "causes",
    "solves",
    "measured_by",
    "verified_by",
    "applies_to",
]

DOMAIN_CONFIGS = {
    "computer": {
        "label": "计算机领域",
        "entity_types": COMPUTER_ENTITY_TYPES,
        "relation_types": COMPUTER_RELATION_TYPES,
    },
    "museum": {
        "label": "博物馆领域",
        "entity_types": MUSEUM_ENTITY_TYPES,
        "relation_types": MUSEUM_RELATION_TYPES,
    },
    "cs408": {
        "label": "408计算机考研领域",
        "entity_types": CS408_ENTITY_TYPES,
        "relation_types": CS408_RELATION_TYPES,
    },
}


def get_domain_entity_relation_config(domain: str | None) -> dict[str, list[str]]:
    """
    获取领域配置：
    - computer（默认）
    - museum（兼容旧配置）
    - cs408（408 专项知识图谱
    """
    normalized_domain = (domain or "computer").strip().lower()
    if normalized_domain not in DOMAIN_CONFIGS:
        normalized_domain = "computer"
    config = DOMAIN_CONFIGS[normalized_domain]
    return {"entity_types": config["entity_types"], "relation_types": config["relation_types"]}


def get_supported_domains() -> dict[str, dict]:
        """返回系统支持的领域本体配置。"""
        return DOMAIN_CONFIGS
