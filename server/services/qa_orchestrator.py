import re
import os
import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.agents.common.tools import adaptive_graph_rag_qa
from src.utils.logging_config import logger
from server.services.retrieval_cache import get_retrieval_cache


RELATION_PRIORITY_CUES = ["关系", "因果", "比较", "流程", "推导", "链路", "依赖", "区别", "为什么", "原理"]


@dataclass
class OrchestratorPlan:
    raw_query: str
    strategy: str = "default"
    should_prioritize_graph: bool = False
    sub_queries: list[str] = field(default_factory=list)
    subject: str = ""


class QAOrchestrator:
    """统一检索编排器：query解析 -> 检索编排 -> 证据融合 -> 验证契约。"""

    def parse_query(self, query: str, subject: str = "") -> OrchestratorPlan:
        normalized_query = (query or "").strip()
        lower_query = normalized_query.lower()
        should_prioritize_graph = any(cue in lower_query for cue in RELATION_PRIORITY_CUES)
        sub_queries = self._decompose_query(normalized_query)
        strategy = "graph_hybrid_priority" if should_prioritize_graph else "default"
        return OrchestratorPlan(
            raw_query=normalized_query,
            strategy=strategy,
            should_prioritize_graph=should_prioritize_graph,
            sub_queries=sub_queries,
            subject=subject,
        )

    async def prepare_context(self, query: str, subject: str = "", image_evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """提前执行可控检索（图谱优先策略），生成可解释上下文。"""
        plan = self.parse_query(query=query, subject=subject)
        image_evidence = image_evidence or []
        retrieval_bundle = {
            "plan": {
                "strategy": plan.strategy,
                "should_prioritize_graph": plan.should_prioritize_graph,
                "sub_queries": plan.sub_queries,
                "subject": subject or "",
            },
            "evidence_bundle": image_evidence[:10],
            "conflict_flags": [],
        }
        context_lines = []

        if plan.should_prioritize_graph:
            retrieval_result = await self._run_hybrid_retrieval(plan)
            evidence_bundle = retrieval_result.get("evidence_bundle", [])
            retrieval_bundle["evidence_bundle"] = (image_evidence[:10] + evidence_bundle)[:30]
            retrieval_bundle["conflict_flags"] = self._detect_conflicts(evidence_bundle)
            context_lines = self._format_grounding_context(retrieval_result, plan)
        elif image_evidence:
            context_lines = [
                "[系统编排提示] 当前问题包含图片输入，已启用图像理解双通路（VL + OCR）。",
                "[证据引用要求] 最终答案至少引用 1 个图像证据ID（例如 IMG001）。",
            ]
            for item in image_evidence[:3]:
                context_lines.append(f"- {item.get('evidence_id', 'IMGUNK')} @ {item.get('image_type', 'unknown')}")

        if image_evidence and not retrieval_bundle["conflict_flags"]:
            retrieval_bundle["conflict_flags"] = self._detect_conflicts(retrieval_bundle["evidence_bundle"])
        confidence_builder = getattr(self, "_estimate_confidence", None)
        if callable(confidence_builder):
            retrieval_bundle["confidence"] = confidence_builder(retrieval_bundle)
        else:
            # 兜底保护：避免线上热更新/老版本混用时因缺少方法导致流式响应崩溃
            retrieval_bundle["confidence"] = {
                "score": 0.5,
                "level": "medium",
                "evidence_count": len(retrieval_bundle.get("evidence_bundle", [])),
                "reasoning_path_count": 0,
                "has_conflict": bool(retrieval_bundle.get("conflict_flags")),
            }

        return {
            "plan": retrieval_bundle["plan"],
            "retrieval_bundle": retrieval_bundle,
            "grounding_context": "\n".join(context_lines).strip(),
        }

    async def _run_hybrid_retrieval(self, plan: OrchestratorPlan) -> dict[str, Any]:
        """图谱+向量混合检索。"""
        max_parallel = int(os.getenv("ORCHESTRATOR_MAX_PARALLEL", "3"))
        cache = get_retrieval_cache()
        cache_key = f"retrieval:query:{plan.subject}:{plan.raw_query}"
        cache_item = cache.get(cache_key)
        if cache_item:
            return cache_item

        aggregated_evidence = []
        used_kbs = set()
        reasoning_paths = []
        derivation_chain = ""

        query_candidates = plan.sub_queries or [plan.raw_query]
        semaphore = asyncio.Semaphore(max_parallel)

        async def _invoke_one(sub_query: str):
            try:
                async with semaphore:
                    return await adaptive_graph_rag_qa.ainvoke(
                        {"query": sub_query, "subject": plan.subject, "memory_messages": []}
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"orchestrator hybrid retrieval failed for sub-query={sub_query}: {exc}")
                return {}

        results = await asyncio.gather(*[_invoke_one(sub_query) for sub_query in query_candidates[:3]])
        for result in results:
            if not isinstance(result, dict):
                continue
            sub_query = str(result.get("query", "") or "")
            evidence_bundle = result.get("evidence_bundle", [])
            cache.set(f"retrieval:chunk:{plan.subject}:{sub_query}", evidence_bundle[:8])
            aggregated_evidence.extend(evidence_bundle)
            used_kbs.update(result.get("used_knowledge_bases", []))
            reasoning_paths.extend(result.get("reasoning_path", []))
            if not derivation_chain and result.get("derivation_chain"):
                derivation_chain = result["derivation_chain"]

        value = {
            "evidence_bundle": aggregated_evidence[:20],
            "used_knowledge_bases": sorted(list(used_kbs)),
            "reasoning_path": reasoning_paths[:10],
            "derivation_chain": derivation_chain or "暂无",
        }
        cache.set(cache_key, value)
        cache.set(f"retrieval:graph_path:{plan.subject}:{plan.raw_query}", value.get("reasoning_path", []))
        return value

    def _decompose_query(self, query: str) -> list[str]:
        """问题分解：并列拆分 + 轻量多跳链路构造。"""
        if not query:
            return []
        separators = r"[？?。；;，,\n]|以及|并且|同时|然后|再"
        chunks = [chunk.strip() for chunk in re.split(separators, query) if chunk and chunk.strip()]
        deduped = []
        seen = set()
        for chunk in chunks:
            if len(chunk) < 4:
                continue
            if chunk in seen:
                continue
            seen.add(chunk)
            deduped.append(chunk)
            # 多跳构造：把前两个子问题组合成“先A后B”的链式查询，帮助关系推理
            if len(deduped) >= 2:
                chained = f"{deduped[0]} -> {deduped[1]}"
                if chained not in seen:
                    deduped.append(chained)
            return deduped[:5]

        def _estimate_confidence(self, retrieval_bundle: dict[str, Any]) -> dict[str, Any]:
            """置信度估计：证据覆盖 + 推理路径 + 冲突惩罚。"""
            evidence_count = len(retrieval_bundle.get("evidence_bundle", []))
            reasoning_count = len(retrieval_bundle.get("reasoning_path", []))
            has_conflict = bool(retrieval_bundle.get("conflict_flags"))

            score = min(1.0, 0.35 + evidence_count * 0.03 + reasoning_count * 0.05)
            if has_conflict:
                score = max(0.0, score - 0.25)

            level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
            return {
                "score": round(score, 3),
                "level": level,
                "evidence_count": evidence_count,
                "reasoning_path_count": reasoning_count,
                "has_conflict": has_conflict,
            }

    def _detect_conflicts(self, evidence_bundle: list[dict[str, Any]]) -> list[str]:
        """证据冲突检测（P1基础版）：简单检查同主题正反结论并存。"""
        if not evidence_bundle:
            return []

        positive_markers = ["可以", "支持", "能够", "是", "需要"]
        negative_markers = ["不能", "不支持", "无法", "不是", "不需要"]
        has_positive = False
        has_negative = False
        for evidence in evidence_bundle:
            text = str(evidence.get("content") or evidence.get("triple") or "")
            if any(marker in text for marker in positive_markers):
                has_positive = True
            if any(marker in text for marker in negative_markers):
                has_negative = True

        if has_positive and has_negative:
            return ["possible_conflict_detected"]
        return []

    def _format_grounding_context(self, retrieval_result: dict[str, Any], plan: OrchestratorPlan) -> list[str]:
        evidence_bundle = retrieval_result.get("evidence_bundle", [])
        used_kbs = retrieval_result.get("used_knowledge_bases", [])
        reasoning_path = retrieval_result.get("reasoning_path", [])
        derivation_chain = retrieval_result.get("derivation_chain", "暂无")

        lines = [
            "[系统编排提示] 当前问题命中关系/因果/比较/流程/推导意图，已优先执行图谱+向量混合检索。",
            f"[检索策略] {plan.strategy}",
            f"[使用知识库] {', '.join(used_kbs) if used_kbs else '未命中'}",
            f"[推导链] {derivation_chain}",
            "[证据引用要求] 最终答案至少引用 1 个 evidence_id（例如 G001/V001）。",
        ]

        if reasoning_path:
            lines.append("[关键推理路径]")
            lines.extend([f"- {step}" for step in reasoning_path[:4]])

        if evidence_bundle:
            lines.append("[证据样例]")
            for item in evidence_bundle[:5]:
                evidence_id = item.get("evidence_id", "UNKNOWN")
                source_kb = item.get("source_kb") or item.get("source_db") or "unknown"
                lines.append(f"- {evidence_id} @ {source_kb}")

        return lines

    def validate_answer_contract(
        self,
        final_answer: str,
        quality_report: dict[str, Any] | None,
        source_refs: list[dict[str, Any]] | None,
        retrieval_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """强制回答契约验证。"""
        answer = (final_answer or "").strip()
        quality_report = quality_report or {}
        source_refs = source_refs or []
        retrieval_bundle = retrieval_bundle or {}
        evidence_mentions = re.findall(r"\b(?:G|V)\d{3}\b", answer)
        has_evidence_ref = len(evidence_mentions) > 0
        confidence_level = (quality_report.get("confidence_level") or "").lower()
        quality_flags = []
        remediation_notes = []

        if not has_evidence_ref:
            quality_flags.append("missing_evidence_citation")
            if source_refs:
                hint_refs = ", ".join([ref["evidence_id"] for ref in source_refs[:3] if ref.get("evidence_id")])
                remediation_notes.append(f"补充证据引用：请在答案中显式引用 {hint_refs} 等 evidence_id。")
            else:
                remediation_notes.append(
                    "当前知识库中未检索到与该问题直接相关的有效证据，因此暂无法生成具有依据的回答。\n\n"
                    "为提升回答质量，建议您：\n"
                    "1. 明确问题涉及的具体领域或对象\n"
                    "2. 提供关键词或相关背景信息\n"
                    "3. 将复杂问题拆分为多个子问题\n\n"
                    "系统将在获取更精确信息后，结合知识库进行检索并给出可解释的回答（含来源依据）。"
                )

        if confidence_level == "low":
            quality_flags.append("low_confidence_needs_clarification")
            remediation_notes.append("当前结论置信度较低，建议补充检索并向用户追问关键约束条件（场景/时间/范围）。")

        conflict_flags = retrieval_bundle.get("conflict_flags", [])
        if conflict_flags:
            quality_flags.append("evidence_conflict")
            remediation_notes.append("检测到证据可能冲突，请在答案中明确不确定项并给出验证建议。")

        return {
            "passed": len(quality_flags) == 0,
            "quality_flags": quality_flags,
            "remediation_message": "\n".join(
                [note if "\n" in note else f"- {note}" for note in remediation_notes]
            ).strip(),
            "evidence_mentions": evidence_mentions,
        }
