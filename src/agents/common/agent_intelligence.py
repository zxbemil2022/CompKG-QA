from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AnyMessage


def compress_context_memory(messages: list[AnyMessage], short_window: int = 8) -> dict[str, Any]:
    """短期/长期记忆分离：
    - short_term: 最近 N 条；
    - long_term_summary: 更早消息压缩摘要（轻量规则）。
    """
    if not messages:
        return {"short_term": [], "long_term_summary": ""}

    short_term = messages[-short_window:]
    long_term = messages[:-short_window]

    bullet_pool: list[str] = []
    for msg in long_term[-24:]:
        content = getattr(msg, "content", "") or ""
        content = re.sub(r"\s+", " ", str(content)).strip()
        if content:
            bullet_pool.append(content[:100])

    summary = "；".join(bullet_pool[:6])
    return {"short_term": short_term, "long_term_summary": summary}


def build_multistep_plan(user_query: str) -> list[str]:
    """多任务分层规划（启发式）。"""
    q = (user_query or "").strip()
    if not q:
        return []

    base_plan = [
        "澄清目标与边界条件",
        "拆解子任务并确定依赖顺序",
        "为每个子任务选择合适工具与知识源",
        "执行并聚合中间结果",
        "进行自检与冲突消解，输出结论与后续建议",
    ]
    if any(k in q.lower() for k in ["图", "image", "截图", "pdf", "文档"]):
        base_plan.insert(2, "优先执行多模态解析（OCR/VL）并提取证据")
    return base_plan


def auto_select_tools(user_query: str, tools: list[Any]) -> list[Any]:
    """自动工具选择：基于关键词选择子集，避免全量绑定。"""
    q = (user_query or "").lower()
    if not q or not tools:
        return tools

    keep = []
    for t in tools:
        name = getattr(t, "name", "").lower()
        desc = (getattr(t, "description", "") or "").lower()
        if any(k in q for k in ["图谱", "关系", "推理", "路径", "知识"]):
            if any(k in name + desc for k in ["graph", "knowledge", "kg", "reason", "path"]):
                keep.append(t)
                continue
        if any(k in q for k in ["图片", "图像", "截图", "ocr", "image"]):
            if any(k in name + desc for k in ["image", "ocr", "multimodal", "vision"]):
                keep.append(t)
                continue
        if any(k in q for k in ["搜索", "联网", "最新", "web"]):
            if any(k in name + desc for k in ["search", "tavily"]):
                keep.append(t)
                continue

    return keep or tools


def should_self_reflect(user_query: str, response_text: str) -> bool:
    q = (user_query or "").lower()
    r = (response_text or "").lower()
    return any(k in q for k in ["检查", "反思", "复核", "verify", "double check"]) or len(r) > 1200
