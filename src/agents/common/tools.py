import asyncio
import re
import traceback
from typing import Annotated, Any

from langchain_core.tools import StructuredTool, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src import config, graph_base, knowledge_base
from src.models.chat import select_model
from src.models.rerank import get_reranker
from src.utils import logger


@tool
def query_knowledge_graph(
    query: Annotated[str, "The keyword to query knowledge graph."],
    subject: Annotated[str, "Optional 408 subject for targeted QA, e.g. 数据结构/操作系统/计算机网络/计算机组成原理."] = "",
) -> Any:
    """Use this to query the knowledge graph for technical/computer-domain facts and relations. If subject is provided, return subject-focused results."""
    try:
        logger.debug(f"Querying knowledge graph with: {query}, subject={subject}")
        result = graph_base.query_node(query, hops=2, return_format="triples", subject=subject or None)
        logger.debug(
            f"Knowledge graph query returned "
            f"{len(result.get('triples', [])) if isinstance(result, dict) else 'N/A'} triples"
        )
        return result
    except Exception as e:
        logger.error(f"Knowledge graph query error: {e}, {traceback.format_exc()}")
        return f"知识图谱查询失败: {str(e)}"


def get_static_tools() -> list:
    """注册静态工具"""
    static_tools = [
        query_knowledge_graph,
        adaptive_graph_rag_qa,
    ]

    # 检查是否启用网页搜索
    if config.enable_web_search:
        static_tools.append(TavilySearch(max_results=10))

    return static_tools


class KnowledgeRetrieverModel(BaseModel):
    query_text: str = Field(
        default="",
        description=(
            "当用户提供的输入中包含关键词时，请提供一个查询的关键词，查询的时候，应该尽量以可能帮助回答这个问题的关键词进行查询，不要直接使用用户的原始输入去查询。如果没有请忽略这个字段。"
        )
    )
    query_img: str = Field(
        default="",
        description=(
            "当用户提供的输入中包含图片url时，则请提供图片的URL去查询,否则请忽略这个字段。"
        )
    )
    query_desc: str = Field(
        default="",
        description=(
            "当用户输入包含技术概念、算法特征、协议行为、系统架构或错误现象等描述时，请提炼为可检索的技术描述。没有则忽略。"
        )
    )


class AdaptiveGraphRAGModel(BaseModel):
    query: str = Field(default="", description="用户问题（必填）")
    subject: str = Field(
        default="",
        description="可选：408 学科（数据结构/操作系统/计算机网络/计算机组成原理），为空则走总图谱。",
    )
    user_profile: dict[str, Any] = Field(
        default_factory=dict,
        description="可选：用户画像，例如 {'level':'考研','goal':'408冲刺','weakness':['操作系统']}。",
    )
    memory_messages: list[str] = Field(
        default_factory=list,
        description="可选：最近多轮对话摘要（字符串列表），用于上下文记忆。",
    )


def _route_query_strategy(query: str, subject: str = "") -> str:
    """
    Query Router: 自动选择检索策略（graph / vector / hybrid）
    """
    q = (query or "").strip().lower()
    if not q:
        return "hybrid"
    graph_cues = ["关系", "依赖", "流程", "推导", "区别", "比较", "为什么", "原理", "链路"]
    vector_cues = ["定义", "概念", "是什么", "举例", "总结", "说明"]
    if any(cue in q for cue in graph_cues):
        return "hybrid" if subject else "graph"
    if any(cue in q for cue in vector_cues):
        return "hybrid" if subject else "vector"
    return "hybrid"


def _route_subject_chapter(query: str, subject: str = "") -> dict[str, str]:
    text = (query or "").lower()
    chapter_map = {
        "计算机网络": {
            "传输层": ["tcp", "udp", "拥塞", "滑动窗口", "重传"],
            "网络层": ["ip", "路由", "子网", "icmp"],
            "应用层": ["http", "dns", "smtp", "ftp"],
        },
        "计算机组成原理": {
            "流水线": ["流水线", "冒险", "hazard"],
            "存储系统": ["cache", "tlb", "主存", "替换"],
            "指令系统": ["指令", "寻址", "译码"],
        },
        "操作系统": {
            "进程线程": ["进程", "线程", "调度", "上下文"],
            "内存管理": ["页表", "虚拟内存", "缺页", "置换"],
            "文件系统": ["inode", "目录", "磁盘", "日志"],
        },
        "数据结构": {
            "树与图": ["树", "图", "最小生成树", "最短路", "dfs", "bfs"],
            "查找排序": ["排序", "查找", "二分", "堆", "快排"],
            "线性结构": ["栈", "队列", "链表", "数组"],
        },
    }

    selected_subject = subject or "综合"
    selected_chapter = "综合"
    for s, chapters in chapter_map.items():
        for c, kws in chapters.items():
            if any(k in text for k in kws):
                selected_subject = s
                selected_chapter = c
                return {"subject": selected_subject, "chapter": selected_chapter}
    return {"subject": selected_subject, "chapter": selected_chapter}


def _safe_snippet(text: Any, max_len: int = 400) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:max_len]


def _fallback_rerank_score(query: str, text: str) -> float:
    """无 cross-encoder 时的兜底 lexical score。"""
    q_tokens = set([t for t in re.split(r"[\s,，。；;:：]+", query.lower()) if t])
    d_tokens = set([t for t in re.split(r"[\s,，。；;:：]+", text.lower()) if t])
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def _build_prompt_template(
    query: str,
    strategy: str,
    subject: str,
    user_profile: dict[str, Any],
    memory_messages: list[str],
    graph_triples: list[tuple],
    vector_docs: list[dict[str, Any]],
):
    """
    Prompt 模板优化 + 多轮记忆 + 用户画像
    """
    memory_text = "\n".join([f"- {m}" for m in memory_messages[:6]]) if memory_messages else "- 无"
    profile_text = ", ".join([f"{k}={v}" for k, v in (user_profile or {}).items()]) if user_profile else "无"
    triples_text = "\n".join([f"- {h} --{r}--> {t}" for h, r, t in graph_triples[:12]]) if graph_triples else "- 无"
    docs_text = "\n".join([f"- {item.get('content', '')}" for item in vector_docs[:6]]) if vector_docs else "- 无"
    subject_text = subject or "总图谱（不限定学科）"

    return (
        "你是计算机408问答助手。请基于给定证据回答，禁止编造。\n"
        f"[检索策略] {strategy}\n"
        f"[学科范围] {subject_text}\n"
        f"[用户画像] {profile_text}\n"
        f"[多轮记忆]\n{memory_text}\n"
        f"[图谱证据]\n{triples_text}\n"
        f"[向量证据]\n{docs_text}\n\n"
        f"用户问题：{query}\n\n"
        "请输出：\n"
        "1) 最终答案（简洁准确）\n"
        "2) 推理路径（步骤化）\n"
        "3) 知识点推导链（A -> B -> C）\n"
        "4) 证据来源（图谱/向量）"
    )


async def _hybrid_retrieve(query: str, subject: str = "") -> dict[str, Any]:
    """
    Graph + Vector 融合 + Rerank（cross-encoder）
    """
    graph_result = graph_base.query_node(query, hops=2, return_format="triples", subject=subject or None)
    graph_triples = graph_result.get("triples", []) if isinstance(graph_result, dict) else []

    vector_docs: list[dict[str, Any]] = []
    retrievers = knowledge_base.get_retrievers()
    for db_id, info in retrievers.items():
        retriever = info.get("retriever")
        if retriever is None:
            continue
        try:
            if asyncio.iscoroutinefunction(retriever):
                docs = await retriever(query, "", "")
            else:
                docs = retriever(query, "", "")
            if isinstance(docs, list):
                for d in docs[:8]:
                    vector_docs.append(
                        {"source_db": db_id, "content": _safe_snippet(d), "raw": d}
                    )
        except Exception as e:
            logger.warning(f"Hybrid retrieve vector db failed: {db_id}, error={e}")

    # Rerank
    reranked_docs = []
    if vector_docs:
        scores = []
        if config.enable_reranker:
            try:
                reranker = get_reranker(config.reranker)
                docs_text = [d["content"] for d in vector_docs]
                scores = reranker.compute_score((query, docs_text), normalize=True)
            except Exception as e:
                logger.warning(f"Cross-encoder rerank failed, fallback lexical score: {e}")
                scores = [_fallback_rerank_score(query, d["content"]) for d in vector_docs]
        else:
            scores = [_fallback_rerank_score(query, d["content"]) for d in vector_docs]

        for item, score in zip(vector_docs, scores):
            reranked_docs.append({**item, "score": float(score)})
        reranked_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)

    # Agent 自动推理路径 + 知识点推导链（可解释结构）
    reasoning_path = []
    for idx, triple in enumerate(graph_triples[:6], 1):
        try:
            h, r, t = triple
            reasoning_path.append(f"Step {idx}: {h} --{r}--> {t}")
        except Exception:
            continue

    derivation_chain = " -> ".join(
        [str(t[0]) for t in graph_triples[:1]] + [str(t[2]) for t in graph_triples[:3] if len(t) >= 3]
    ).strip(" ->")

    graph_evidence = []
    for idx, triple in enumerate(graph_triples[:20], 1):
        graph_evidence.append({"evidence_id": f"G{idx:03d}", "triple": triple, "source_kb": "neo4j_graph"})

    vector_evidence = []
    for idx, doc in enumerate(reranked_docs[:8], 1):
        vector_evidence.append({"evidence_id": f"V{idx:03d}", **doc})

    return {
        "graph_triples": graph_triples[:20],
        "vector_docs": reranked_docs[:8],
        "graph_evidence": graph_evidence,
        "vector_evidence": vector_evidence,
        "used_knowledge_bases": sorted(list({d["source_db"] for d in reranked_docs})),
        "reasoning_path": reasoning_path,
        "derivation_chain": derivation_chain or "暂无可推导链",
    }


def _analyze_answer_confidence(graph_triples: list[tuple], vector_docs: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_count = len(graph_triples) + len(vector_docs)
    top_rerank = vector_docs[0]["score"] if vector_docs else 0.0
    if evidence_count >= 12 and top_rerank >= 0.6:
        level = "high"
    elif evidence_count >= 6:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "evidence_count": evidence_count,
        "top_rerank_score": round(float(top_rerank), 4),
        "advice": "建议补充更多专业资料后再确认结论" if level == "low" else "当前证据支持度较好",
    }


def _llm_fallback_answer(query: str, subject: str, prompt: str) -> str:
    try:
        provider, model_name = config.default_model.split("/", 1)
        model = select_model(provider, model_name)
        message = [
            {"role": "system", "content": "你是计算机408专业辅导助手，请给出专业准确的回答并标注不确定性。"},
            {"role": "user", "content": f"学科：{subject or '综合'}\n问题：{query}\n参考提示：{prompt}"},
        ]
        response = model.call(message, stream=False)
        return getattr(response, "content", str(response))
    except Exception as e:
        logger.warning(f"LLM fallback failed: {e}")
        return "知识库证据不足，且大模型兜底回答失败，请稍后重试。"


@tool(args_schema=AdaptiveGraphRAGModel)
async def adaptive_graph_rag_qa(
    query: str,
    subject: str = "",
    user_profile: dict[str, Any] | None = None,
    memory_messages: list[str] | None = None,
) -> Any:
    """
    综合能力工具：
    - Query Router（自动策略选择）
    - Graph+Vector 融合检索
    - Cross-encoder Rerank
    - Prompt 模板优化
    - 多轮记忆 + 用户画像
    - GraphRAG 可解释输出（推理路径 / 知识点推导链）
    """
    try:
        strategy = _route_query_strategy(query=query, subject=subject)
        chapter_route = _route_subject_chapter(query=query, subject=subject)
        retrieval = await _hybrid_retrieve(query=query, subject=chapter_route["subject"] if chapter_route["subject"] != "综合" else subject)
        prompt = _build_prompt_template(
            query=query,
            strategy=strategy,
            subject=subject,
            user_profile=user_profile or {},
            memory_messages=memory_messages or [],
            graph_triples=retrieval["graph_triples"],
            vector_docs=retrieval["vector_docs"],
        )

        return {
            "strategy": strategy,
            "subject": subject or "all",
            "chapter_route": chapter_route,
            "graph_triples": retrieval["graph_triples"],
            "vector_docs": retrieval["vector_docs"],
            "evidence_bundle": retrieval.get("graph_evidence", []) + retrieval.get("vector_evidence", []),
            "used_knowledge_bases": retrieval.get("used_knowledge_bases", []),
            "reasoning_path": retrieval["reasoning_path"],
            "derivation_chain": retrieval["derivation_chain"],
            "accuracy_analysis": _analyze_answer_confidence(
                retrieval["graph_triples"], retrieval["vector_docs"]
            ),
            "learning_path_recommendation": graph_base.recommend_learning_path(
                concept=query,
                subject=chapter_route["subject"] if chapter_route["subject"] != "综合" else (subject or None),
            ),
            "knowledge_association_analysis": graph_base.analyze_knowledge_association(
                concept=query,
                subject=chapter_route["subject"] if chapter_route["subject"] != "综合" else (subject or None),
            ),
            "llm_fallback_answer": _llm_fallback_answer(query, subject, prompt)
            if len(retrieval["graph_triples"]) == 0 and len(retrieval["vector_docs"]) == 0
            else "",
            "answer_contract": "最终回答必须至少引用1个evidence_id，并给出source_kb",
            "optimized_prompt_template": prompt,
        }
    except Exception as e:
        logger.error(f"adaptive_graph_rag_qa error: {e}, {traceback.format_exc()}")
        return {"error": f"adaptive_graph_rag_qa failed: {str(e)}"}


def get_kb_based_tools() -> list:
    """获取所有知识库基于的工具"""
    # 获取所有知识库
    kb_tools = []
    retrievers = knowledge_base.get_retrievers()

    def _create_retriever_wrapper(db_id: str, retriever_info: dict[str, Any]):
        """创建检索器包装函数的工厂函数，避免闭包变量捕获问题"""

        async def async_retriever_wrapper(query_text: str = "", query_img: str = "", query_desc: str = "") -> Any:
            """异步检索器包装函数"""
            retriever = retriever_info["retriever"]
            try:
                logger.debug(f"Retrieving from database {db_id} with query: {query_text}, query_img: {query_img}, query_desc: {query_desc}")
                if asyncio.iscoroutinefunction(retriever):
                    result = await retriever(query_text, query_img, query_desc)
                else:
                    result = retriever(query_text, query_img, query_desc)
                logger.debug(f"Retrieved {len(result) if isinstance(result, list) else 'N/A'} results from {db_id}")
                return result
            except Exception as e:
                logger.error(f"Error in retriever {db_id}: {e}")
                return f"检索失败: {str(e)}"

        return async_retriever_wrapper

    for db_id, retrieve_info in retrievers.items():
        try:
            # 使用改进的工具ID生成策略
            tool_id = f"query_{db_id[:8]}"

            # 构建工具描述
            description = (
                f"使用 {retrieve_info['name']} 知识库进行检索（支持文本+图像）。\n"
                f"- query_text: 文本问题\n"
                f"- query_img: 图片URL/路径（可选）\n"
                f"- query_desc: OCR/图像摘要（可选）\n"
                f"知识库描述：{retrieve_info['description'] or '没有描述。'}"
            )

            # 使用工厂函数创建检索器包装函数，避免闭包问题
            retriever_wrapper = _create_retriever_wrapper(db_id, retrieve_info)

            # 使用 StructuredTool.from_function 创建异步工具
            tool = StructuredTool.from_function(
                coroutine=retriever_wrapper,
                name=tool_id,
                description=description,
                args_schema=KnowledgeRetrieverModel,
                metadata=retrieve_info["metadata"] | {"tag": ["knowledgebase"]},
            )

            kb_tools.append(tool)
            # logger.debug(f"Successfully created tool {tool_id} for database {db_id}")

        except Exception as e:
            logger.error(f"Failed to create tool for database {db_id}: {e}, \n{traceback.format_exc()}")
            continue

    return kb_tools


def get_buildin_tools() -> list:
    """获取所有可运行的工具（给大模型使用）"""
    tools = []

    try:
        # 获取所有知识库基于的工具
        tools.extend(get_kb_based_tools())
        tools.extend(get_static_tools())

        # from src.agents.common.toolkits.mysql.tools import get_mysql_tools

        # tools.extend(get_mysql_tools())

    except Exception as e:
        logger.error(f"Failed to get knowledge base retrievers: {e}")

    return tools


def gen_tool_info(tools) -> list[dict[str, Any]]:
    """获取所有工具的信息（用于前端展示）"""
    tools_info = []

    try:
        # 获取注册的工具信息
        for tool_obj in tools:
            try:
                metadata = getattr(tool_obj, "metadata", {}) or {}
                info = {
                    "id": tool_obj.name,
                    "name": metadata.get("name", tool_obj.name),
                    "description": tool_obj.description,
                    "metadata": metadata,
                    "args": [],
                    # "is_async": is_async  # Include async information
                }

                if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    schema = tool_obj.args_schema.schema()
                    for arg_name, arg_info in schema.get("properties", {}).items():
                        info["args"].append(
                            {
                                "name": arg_name,
                                "type": arg_info.get("type", ""),
                                "description": arg_info.get("description", ""),
                            }
                        )

                tools_info.append(info)
                # logger.debug(f"Successfully processed tool info for {tool_obj.name}")

            except Exception as e:
                logger.error(
                    f"Failed to process tool {getattr(tool_obj, 'name', 'unknown')}: {e}\n{traceback.format_exc()}"
                )
                continue

    except Exception as e:
        logger.error(f"Failed to get tools info: {e}\n{traceback.format_exc()}")
        return []

    logger.info(f"Successfully extracted info for {len(tools_info)} tools")
    return tools_info
