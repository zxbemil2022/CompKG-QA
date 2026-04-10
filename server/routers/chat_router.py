import asyncio
import json
import traceback
import uuid
import yaml
import os
import base64
from pathlib import Path
from typing import Any
from time import monotonic

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.storage.db.models import User, MessageFeedback, Message, Conversation
from src.storage.conversation import ConversationManager
from src.storage.db.manager import db_manager
from server.routers.auth_router import get_admin_user
from server.utils.auth_middleware import get_db, get_required_user
from src import executor
from src import config as conf
from src.agents import agent_manager
from src.agents.common.tools import gen_tool_info, get_buildin_tools
from src.models import select_model
from src.plugins.guard import content_guard
from src.utils.logging_config import logger
from src.utils.error_codes import STREAM_ERROR_CODE_BY_TYPE
from src.knowledge.utils.image_embedding_utils import build_image_evidence_bundle
from server.services.qa_orchestrator import QAOrchestrator
from server.services.breaker_provider import get_global_breaker
from server.services.observability import get_observability_registry

chat = APIRouter(prefix="/chat", tags=["chat"])

BREAKER = get_global_breaker()
OBSERVABILITY = get_observability_registry()

MODEL_NOT_OPEN_CACHE: set[str] = set()


def _provider_is_available(provider: str) -> bool:
    if not provider:
        return False
    info = (conf.model_names or {}).get(provider, {})
    env_var = info.get("env", "")
    if not env_var:
        return False
    if env_var == "NO_API_KEY":
        return True
    return bool(os.getenv(env_var))


def _normalize_model_spec(model_spec: str | None) -> str:
    if not model_spec:
        return ""
    spec = str(model_spec).strip()
    if "/" in spec:
        return spec
    # 兼容仅传 provider 的场景
    provider = spec
    provider_info = (conf.model_names or {}).get(provider, {})
    default_model = provider_info.get("default", "")
    return f"{provider}/{default_model}" if provider and default_model else ""


def _candidate_runtime_models(requested: str, prefer_vision: bool) -> list[str]:
    candidates: list[str] = []
    normalized_requested = _normalize_model_spec(requested)
    if normalized_requested:
        candidates.append(normalized_requested)

    # 图片场景优先考虑视觉模型
    if prefer_vision:
        vl_model = _normalize_model_spec(getattr(conf, "vl_model", ""))
        if vl_model:
            candidates.append(vl_model)

        for provider, info in (conf.model_names or {}).items():
            for model_name in info.get("models", []) or []:
                if _is_vision_capable_model(model_name):
                    candidates.append(f"{provider}/{model_name}")

    for spec in [getattr(conf, "fast_model", ""), getattr(conf, "default_model", "")]:
        normalized = _normalize_model_spec(spec)
        if normalized:
            candidates.append(normalized)

    dedup: list[str] = []
    seen = set()
    for spec in candidates:
        if spec and spec not in seen:
            seen.add(spec)
            dedup.append(spec)
    return dedup


def _select_best_model_spec(requested: str, prefer_vision: bool) -> str:
    for spec in _candidate_runtime_models(requested, prefer_vision):
        if spec in MODEL_NOT_OPEN_CACHE:
            continue
        provider = spec.split("/", 1)[0] if "/" in spec else ""
        if _provider_is_available(provider):
            return spec

    fallback = _normalize_model_spec(getattr(conf, "default_model", ""))
    return fallback or _normalize_model_spec(requested)




def _get_provider_cfg(model_provider: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """兼容 dict/object 两种配置形态，返回 (provider_cfg_dict, model_names_dict)。"""
    raw_model_names = getattr(conf, "model_names", None)
    if raw_model_names is None and isinstance(conf, dict):
        raw_model_names = conf.get("model_names")

    if hasattr(raw_model_names, "items"):
        model_names_map = dict(raw_model_names)
    else:
        model_names_map = {}

    raw_provider_cfg = model_names_map.get(model_provider)
    if raw_provider_cfg is None:
        return {}, model_names_map

    if isinstance(raw_provider_cfg, dict):
        return dict(raw_provider_cfg), model_names_map

    # 兜底：支持对象风格（历史遗留）
    provider_cfg = {
        "name": getattr(raw_provider_cfg, "name", ""),
        "url": getattr(raw_provider_cfg, "url", ""),
        "base_url": getattr(raw_provider_cfg, "base_url", ""),
        "default": getattr(raw_provider_cfg, "default", ""),
        "env": getattr(raw_provider_cfg, "env", ""),
        "models": list(getattr(raw_provider_cfg, "models", []) or []),
    }
    return provider_cfg, model_names_map

def _is_vision_capable_model(model_name: str) -> bool:
    """根据模型名称进行启发式判断，避免把图片直接发送给纯文本模型。"""
    lowered = (model_name or "").lower()
    vision_keywords = (
        "vision",
        "vl",
        "vlm",
        "gpt-4o",
        "gemini",
        "claude-3",
        "llava",
        "qwen-vl",
        "doubao-seed",
    )
    return any(keyword in lowered for keyword in vision_keywords)


def _normalize_text_content(content: Any) -> str:
    """将任意消息内容规范化为字符串，便于质量评估。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join([_normalize_text_content(item) for item in content])
    if isinstance(content, dict):
        text_parts = []
        for value in content.values():
            text_parts.append(_normalize_text_content(value))
        return " ".join([part for part in text_parts if part]).strip()
    return str(content)


def _safe_json_loads(raw_text: Any) -> Any:
    """安全解析 JSON 字符串。"""
    if not isinstance(raw_text, str):
        return None
    try:
        return json.loads(raw_text)
    except Exception:
        return None


def _extract_source_refs_from_state_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """从工具消息中抽取 evidence_id/source_kb，形成可溯源引用列表。"""
    refs: list[dict[str, Any]] = []
    seen_ref_keys: set[str] = set()

    for msg in messages:
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else {}
        if msg_dict.get("type") != "tool":
            continue

        tool_name = msg_dict.get("name", "unknown")
        parsed = _safe_json_loads(msg_dict.get("content"))
        if not isinstance(parsed, dict):
            continue

        evidence_items = parsed.get("evidence_bundle", [])
        if not isinstance(evidence_items, list):
            continue

        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue
            evidence_id = evidence.get("evidence_id")
            source_kb = evidence.get("source_kb") or evidence.get("source_db")
            if not evidence_id or not source_kb:
                continue
            ref_key = f"{evidence_id}:{source_kb}"
            if ref_key in seen_ref_keys:
                continue
            seen_ref_keys.add(ref_key)
            refs.append({"evidence_id": evidence_id, "source_kb": source_kb, "tool_name": tool_name})

    return refs


def _build_answer_quality_report(
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    source_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建统一的答案质量报告（置信度、可溯源性、工具使用情况）。"""
    normalized_answer = (final_answer or "").strip()
    char_count = len(normalized_answer)
    evidence_count = len(source_refs)
    tool_call_count = len(tool_calls)

    has_citations = evidence_count > 0
    uncertainty_markers = ["可能", "不确定", "无法确认", "猜测", "大概", "建议核实"]
    uncertainty_hits = sum(1 for marker in uncertainty_markers if marker in normalized_answer)

    confidence_score = 0.25
    if char_count >= 80:
        confidence_score += 0.2
    if tool_call_count > 0:
        confidence_score += min(0.25, tool_call_count * 0.08)
    if has_citations:
        confidence_score += min(0.25, evidence_count * 0.06)
    if uncertainty_hits > 0:
        confidence_score -= min(0.2, uncertainty_hits * 0.05)
    confidence_score = max(0.0, min(1.0, round(confidence_score, 3)))

    if confidence_score >= 0.75:
        confidence_level = "high"
    elif confidence_score >= 0.45:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    quality_flags = []
    if not has_citations:
        quality_flags.append("missing_source_refs")
    if tool_call_count == 0:
        quality_flags.append("no_tool_reasoning")
    if char_count < 40:
        quality_flags.append("answer_too_short")

    return {
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "answer_length": char_count,
        "tool_call_count": tool_call_count,
        "source_ref_count": evidence_count,
        "quality_flags": quality_flags,
    }


# =============================================================================
# > === 智能体管理分组 ===
# =============================================================================


@chat.get("/default_agent")
async def get_default_agent(current_user: User = Depends(get_required_user)):
    """获取默认智能体ID（需要登录）"""
    try:
        default_agent_id = conf.default_agent_id
        # 如果没有设置默认智能体，尝试获取第一个可用的智能体
        if not default_agent_id:
            agents = await agent_manager.get_agents_info()
            if agents:
                default_agent_id = agents[0].get("id", "")

        return {"default_agent_id": default_agent_id}
    except Exception as e:
        logger.error(f"获取默认智能体出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取默认智能体出错: {str(e)}")


@chat.post("/set_default_agent")
async def set_default_agent(request_data: dict = Body(...), current_user=Depends(get_admin_user)):
    """设置默认智能体ID (仅管理员)"""
    try:
        agent_id = request_data.get("agent_id")
        if not agent_id:
            raise HTTPException(status_code=422, detail="缺少必需的 agent_id 字段")

        # 验证智能体是否存在
        agents = await agent_manager.get_agents_info()
        agent_ids = [agent.get("id", "") for agent in agents]

        if agent_id not in agent_ids:
            raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

        # 设置默认智能体ID
        conf.default_agent_id = agent_id
        # 保存配置
        conf.save()

        return {"success": True, "default_agent_id": agent_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"设置默认智能体出错: {e}")
        raise HTTPException(status_code=500, detail=f"设置默认智能体出错: {str(e)}")


# =============================================================================
# > === 对话分组 ===
# =============================================================================


@chat.post("/call")
async def call(query: str = Body(...), meta: dict = Body(None), current_user: User = Depends(get_required_user)):
    """调用模型进行简单问答（需要登录）"""
    meta = meta or {}
    model = select_model(
        model_provider=meta.get("model_provider"),
        model_name=meta.get("model_name"),
        model_spec=meta.get("model_spec") or meta.get("model"),
    )

    async def call_async(query):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, model.call, query)

    response = await call_async(query)
    logger.debug({"query": query, "response": response.content})

    return {"response": response.content}


@chat.get("/agent")
async def get_agent(current_user: User = Depends(get_required_user)):
    """获取所有可用智能体（需要登录）"""
    agents = await agent_manager.get_agents_info()
    # logger.debug(f"agents: {agents}")
    metadata = {}
    if Path("src/config/static/agents_meta.yaml").exists():
        with open("src/config/static/agents_meta.yaml") as f:
            metadata = yaml.safe_load(f)
    return {"agents": agents, "metadata": metadata}


# TODO:[未完成]这个thread_id在前端是直接生成的1234，最好传入thread_id时做校验只允许uuid4
@chat.post("/agent/{agent_id}")
async def chat_agent(
    agent_id: str,
    query: str = Body(...),
    images: list = Body([]),
    config: dict = Body({}),
    meta: dict = Body({}),
    current_user: User = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """使用特定智能体进行对话（需要登录）"""
    start_time = asyncio.get_event_loop().time()
    orchestrator = QAOrchestrator()

    logger.info(f"agent_id: {agent_id}, query: {query}, images_count: {len(images)}, config: {config}, meta: {meta}")

    # 保存图片到服务器并获取保存路径
    image_paths = []
    for image in images:
        image_path = image["url"]
        image_paths.append(image_path)

    logger.debug(f"image_paths: {image_paths}")

    requested_model = (config or {}).get("model") or (meta or {}).get("model_spec") or (meta or {}).get("model")
    selected_model = _select_best_model_spec(str(requested_model or ""), prefer_vision=bool(image_paths))

    meta.update(
        {
            "query": query,
            "images": image_paths,
            "agent_id": agent_id,
            "server_model_name": selected_model or str(requested_model or agent_id),
            "thread_id": config.get("thread_id"),
            "use_kb_image_retrieval": bool((meta or {}).get("use_kb_image_retrieval", True)),
            "user_id": current_user.id,
            "model_auto_selected": bool(selected_model),
        }
    )

    # 将meta和thread_id整合到config中
    def make_chunk(content=None, **kwargs):
        return (
            json.dumps(
                {"request_id": meta.get("request_id"), "response": content, **kwargs}, ensure_ascii=False
            ).encode("utf-8")
            + b"\n"
        )

    def make_error_chunk(message_text: str, error_type: str = "model", **kwargs):
        return make_chunk(
            message=message_text,
            status="error",
            error_type=error_type,
            error_code=STREAM_ERROR_CODE_BY_TYPE.get(error_type, "UNKNOWN_ERROR"),
            **kwargs,
        )

    async def save_messages_from_langgraph_state(
        agent_instance,
        thread_id,
        conv_mgr,
        config_dict,
        image_source_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        从 LangGraph state 中读取完整消息并保存到数据库
        这样可以获得完整的 tool_calls 参数
        """
        try:
            graph = await agent_instance.get_graph()
            state = await graph.aget_state(config_dict)

            if not state or not state.values:
                logger.warning("No state found in LangGraph")
                return {"quality_report": {}, "source_refs": [], "tool_trace": []}

            messages = state.values.get("messages", [])
            logger.debug(f"Retrieved {len(messages)} messages from LangGraph state")
            extracted_refs = _extract_source_refs_from_state_messages(messages)
            for ref in extracted_refs:
                source_kb = str(ref.get("source_kb") or "")
                if "mode" not in ref:
                    ref["mode"] = "temp_chat_image" if source_kb == "multimodal_image" else "kb_image"
            source_refs = (image_source_refs or []) + extracted_refs
            dedup_refs = []
            seen_ref = set()
            for ref in source_refs:
                key = f"{ref.get('evidence_id')}:{ref.get('source_kb')}"
                if key in seen_ref:
                    continue
                seen_ref.add(key)
                dedup_refs.append(ref)
            source_refs = dedup_refs
            tool_trace: list[dict[str, Any]] = []
            final_ai_answer = ""
            final_ai_msg_id = None

            # 获取已保存的消息数量，避免重复保存
            existing_messages = conv_mgr.get_messages_by_thread_id(thread_id)
            existing_ids = {
                msg.extra_metadata["id"]
                for msg in existing_messages
                if msg.extra_metadata and "id" in msg.extra_metadata
            }

            for msg in messages:
                msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else {}
                msg_type = msg_dict.get("type", "unknown")

                if msg_type == "human" or msg.id in existing_ids:
                    continue

                elif msg_type == "ai":
                    # AI 消息
                    content = msg_dict.get("content", "")
                    tool_calls_data = msg_dict.get("tool_calls", [])

                    # 格式清洗
                    if finish_reason := msg_dict.get("response_metadata", {}).get("finish_reason"):
                        if "tool_call" in finish_reason and len(finish_reason) > len("tool_call"):
                            model_name = msg_dict.get("response_metadata", {}).get("model_name", "")
                            repeat_count = len(finish_reason) // len("tool_call")
                            msg_dict["response_metadata"]["finish_reason"] = "tool_call"
                            msg_dict["response_metadata"]["model_name"] = model_name[: len(model_name) // repeat_count]

                    # 保存 AI 消息
                    ai_msg = conv_mgr.add_message_by_thread_id(
                        thread_id=thread_id,
                        role="assistant",
                        content=content,
                        message_type="text",
                        extra_metadata=msg_dict,  # 保存原始 model_dump
                    )

                    # 保存 tool_calls（如果有）- 使用 LangGraph 的 tool_call_id
                    if tool_calls_data:
                        logger.debug(f"Saving {len(tool_calls_data)} tool calls from AI message")
                        for tc in tool_calls_data:
                            conv_mgr.add_tool_call(
                                message_id=ai_msg.id,
                                tool_name=tc.get("name", "unknown"),
                                tool_input=tc.get("args", {}),  # 完整的参数
                                status="pending",  # 工具还未执行
                                langgraph_tool_call_id=tc.get("id"),  # 保存 LangGraph tool_call_id
                            )
                            tool_trace.append(
                                {
                                    "tool_name": tc.get("name", "unknown"),
                                    "tool_call_id": tc.get("id"),
                                    "status": "pending",
                                }
                            )

                    logger.debug(f"Saved AI message {ai_msg.id} with {len(tool_calls_data)} tool calls")
                    final_ai_answer = _normalize_text_content(content)
                    final_ai_msg_id = ai_msg.id

                elif msg_type == "tool":
                    # 工具执行结果消息 - 使用 tool_call_id 精确匹配
                    tool_call_id = msg_dict.get("tool_call_id")
                    content = msg_dict.get("content", "")
                    name = msg_dict.get("name", "")

                    if tool_call_id:
                        # 确保tool_output是字符串类型，避免SQLite不支持列表类型
                        if isinstance(content, list):
                            tool_output = json.dumps(content) if content else ""
                        else:
                            tool_output = str(content)

                        # 通过 LangGraph tool_call_id 精确匹配并更新
                        updated_tc = conv_mgr.update_tool_call_output(
                            langgraph_tool_call_id=tool_call_id,
                            tool_output=tool_output,
                            status="success",
                        )
                        if updated_tc:
                            logger.debug(f"Updated tool_call {tool_call_id} ({name}) with output")
                            tool_trace.append(
                                {
                                    "tool_name": name or "unknown",
                                    "tool_call_id": tool_call_id,
                                    "status": "success",
                                }
                            )
                        else:
                             logger.warning(f"Tool call {tool_call_id} not found for update")

                else:
                    logger.warning(f"Unknown message type: {msg_type}, skipping")
                    continue

                logger.debug(f"Processed message type={msg_type}")

            logger.info("Saved messages from LangGraph state")
            quality_report = _build_answer_quality_report(
                final_answer=final_ai_answer,
                tool_calls=tool_trace,
                source_refs=source_refs,
            )

            # 将质量报告追加到最后一条 AI 消息，便于后续前端读取/评估
            if final_ai_msg_id:
                final_ai_message = conv_mgr.db.query(Message).filter(Message.id == final_ai_msg_id).first()
                if final_ai_message:
                    extra_metadata = final_ai_message.extra_metadata or {}
                    extra_metadata["quality_report"] = quality_report
                    extra_metadata["source_refs"] = source_refs[:20]
                    final_ai_message.extra_metadata = extra_metadata
                    conv_mgr.db.commit()

            return {
                "quality_report": quality_report,
                "source_refs": source_refs[:20],
                "tool_trace": tool_trace,
            }

        except Exception as e:
            logger.error(f"Error saving messages from LangGraph state: {e}")
            logger.error(traceback.format_exc())
            return {"quality_report": {}, "source_refs": [], "tool_trace": []}

    # TODO:[功能建议]针对需要人工审批后再执行的工具，
    # 可以使用langgraph的interrupt方法中断对话，等待用户输入后再使用command跳转回去
    async def stream_messages():
        def build_image_grounding_text(evidence_bundle: list[dict[str, Any]], max_items: int = 3) -> str:
            if not evidence_bundle:
                return ""
            lines = [
                "[图像证据摘要] 以下是已提取的图像证据，请优先基于这些证据回答，并显式引用 evidence_id："
            ]
            for item in evidence_bundle[:max_items]:
                eid = item.get("evidence_id", "IMGUNK")
                image_type = item.get("image_type", "unknown")
                description = str(item.get("description", "")).strip().replace("\n", " ")
                ocr_text = str(item.get("ocr_text", "")).strip().replace("\n", " ")
                if len(description) > 220:
                    description = description[:220] + "..."
                if len(ocr_text) > 160:
                    ocr_text = ocr_text[:160] + "..."
                lines.append(
                    f"- {eid} ({image_type}) 描述: {description or '无'}；OCR: {ocr_text or '无'}"
                )
            return "\n".join(lines)

        # 代表服务端已经收到了请求
        processed_query = query  # 使用外部函数的query变量
        image_evidence_bundle = (
            await asyncio.to_thread(build_image_evidence_bundle, image_paths, (meta or {}).get("subject", ""))
            if image_paths else []
        )
        use_kb_image_retrieval = bool((meta or {}).get("use_kb_image_retrieval", True))
        orchestrator_context = await orchestrator.prepare_context(
            query=query,
            subject=(meta or {}).get("subject", ""),
            image_evidence=image_evidence_bundle,
        )
        if orchestrator_context.get("grounding_context"):
            processed_query = f"{processed_query}\n\n{orchestrator_context['grounding_context']}"
        if image_evidence_bundle:
            processed_query = f"{processed_query}\n\n{build_image_grounding_text(image_evidence_bundle)}"
        multimodal_user_content: Any = processed_query
        model_name = str(meta.get("server_model_name", ""))
        vision_capable = _is_vision_capable_model(model_name)
        if image_paths and vision_capable:
            multimodal_user_content = [{"type": "text", "text": processed_query}] + [
                {"type": "image_url", "image_url": {"url": img}} for img in image_paths
            ]
        elif image_paths and not vision_capable:
            logger.warning("模型 %s 不支持视觉输入，已退化为文本+图像证据模式", model_name)
            processed_query = (
                f"{processed_query}\n\n"
                "[系统提示] 当前对话模型不支持直接图像输入，"
                "请基于已提取的 OCR 与图像描述证据进行分析。"
            )
            multimodal_user_content = processed_query
            meta["vision_fallback"] = True
        yield make_chunk(status="init", meta=meta, msg=HumanMessage(content=multimodal_user_content).model_dump())

        # Input guard
        if conf.enable_content_guard and await content_guard.check(query):
            yield make_error_chunk("输入内容包含敏感词", error_type="guard", meta=meta)
            return

        try:
            agent = agent_manager.get_agent(agent_id)
        except Exception as e:
            logger.error(f"Error getting agent {agent_id}: {e}, {traceback.format_exc()}")
            yield make_error_chunk(f"Error getting agent {agent_id}: {e}", error_type="model")
            return

        # 构造包含图片的消息
        messages = [{"role": "user", "content": multimodal_user_content}]

        # 构造运行时配置，如果没有thread_id则生成一个
        user_id = str(current_user.id)
        thread_id = config.get("thread_id")
        input_context = {"user_id": user_id, "thread_id": thread_id}
        if selected_model:
            input_context["model"] = selected_model
        if isinstance((config or {}).get("tools"), list):
            input_context["tools"] = config.get("tools")
        if isinstance((config or {}).get("mcps"), list):
            input_context["mcps"] = config.get("mcps")

        if not thread_id:
            thread_id = str(uuid.uuid4())
            logger.warning(f"No thread_id provided, generated new thread_id: {thread_id}")

        # Initialize conversation manager
        conv_manager = ConversationManager(db)

        # Save user message
        try:
            conv_manager.add_message_by_thread_id(
                thread_id=thread_id,
                role="user",
                content=processed_query,
                message_type="text",
                extra_metadata={
                    "raw_message": HumanMessage(content=multimodal_user_content).model_dump(),
                    "image_evidence_refs": image_evidence_bundle[:10],
                },
            )
        except Exception as e:
            logger.error(f"Error saving user message: {e}")

        try:
            full_msg = None
            stream_started_at = monotonic()
            last_chunk_at = stream_started_at
            total_timeout_sec = float(os.getenv("CHAT_STREAM_TOTAL_TIMEOUT_SEC", "300"))
            idle_timeout_sec = float(os.getenv("CHAT_STREAM_IDLE_TIMEOUT_SEC", "120"))
            if "thinking" in str(meta.get("server_model_name", "")).lower():
                total_timeout_sec = max(total_timeout_sec, 420.0)
            model_key = f"{agent_id}:{meta.get('server_model_name', agent_id)}"
            if not BREAKER.allow(model_key):
                yield make_error_chunk(
                    "模型服务短时不可用（熔断保护中），请稍后重试或切换模型。",
                    error_type="model",
                    meta=meta | {"error_code": "MODEL_CIRCUIT_OPEN"},
                )
                return
            async for msg, metadata in agent.stream_messages(messages, input_context=input_context):
                now = monotonic()
                if now - stream_started_at > total_timeout_sec:
                    raise TimeoutError(f"stream total timeout exceeded ({total_timeout_sec}s)")
                if now - last_chunk_at > idle_timeout_sec:
                    raise TimeoutError(f"stream idle timeout exceeded ({idle_timeout_sec}s)")
                if isinstance(msg, AIMessageChunk):
                    last_chunk_at = monotonic()
                    full_msg = msg if not full_msg else full_msg + msg
                    if conf.enable_content_guard and await content_guard.check_with_keywords(full_msg.content[-20:]):
                        logger.warning("Sensitive content detected in stream")
                        yield make_error_chunk("检测到敏感内容，已中断输出", error_type="guard")
                        return

                    yield make_chunk(content=msg.content, msg=msg.model_dump(), metadata=metadata, status="loading")

                else:
                    last_chunk_at = monotonic()
                    yield make_chunk(msg=msg.model_dump(), metadata=metadata, status="loading")

            if (
                conf.enable_content_guard
                and hasattr(full_msg, "content")
                and await content_guard.check(full_msg.content)
            ):
                logger.warning("Sensitive content detected in final message")
                yield make_error_chunk("检测到敏感内容，已中断输出", error_type="guard")
                return

            # After streaming finished, save all messages from LangGraph state
            langgraph_config = {"configurable": input_context}
            quality_artifacts = await save_messages_from_langgraph_state(
                agent_instance=agent,
                thread_id=thread_id,
                conv_mgr=conv_manager,
                config_dict=langgraph_config,
                image_source_refs=[
                    {
                        "evidence_id": item.get("evidence_id"),
                        "source_kb": item.get("source_kb"),
                        "mode": item.get("mode", "temp_chat_image"),
                        "tool_name": "multimodal_image_pipeline",
                        "image_type": item.get("image_type"),
                        "image_url": item.get("image_url"),
                    }
                    for item in image_evidence_bundle
                ],
            )
            contract_validation = orchestrator.validate_answer_contract(
                final_answer=full_msg.content if full_msg and hasattr(full_msg, "content") else "",
                quality_report=quality_artifacts.get("quality_report", {}),
                source_refs=quality_artifacts.get("source_refs", []),
                retrieval_bundle=orchestrator_context.get("retrieval_bundle", {}),
            )
            quality_flags = set(contract_validation.get("quality_flags", []))
            hard_block_triggered = {"missing_evidence_citation", "low_confidence_needs_clarification"}.issubset(
                quality_flags
            )
            if hard_block_triggered:
                OBSERVABILITY.record_failed_sample(
                    {
                        "type": "answer_contract_hard_block",
                        "agent_id": agent_id,
                        "thread_id": thread_id,
                        "query": query,
                        "quality_report": quality_artifacts.get("quality_report", {}),
                        "contract_validation": contract_validation,
                    }
                )
                clarification_template = (
                    "当前问题需要更多约束信息才能给出可靠答案。请补充以下任一信息后我再继续：\n"
                    "1) 具体适用场景/边界条件；\n"
                    "2) 你期望的比较维度（性能/复杂度/实现难度）；\n"
                    "3) 是否限定教材版本或课程章节。"
                )
                yield make_chunk(content=f"\n\n【澄清问题】\n{clarification_template}", status="loading")
                yield make_chunk(
                    status="error",
                    message="回答未满足最低证据与置信度要求，已进入澄清模式。",
                    error_type="quality",
                    error_code="ANSWER_CONTRACT_HARD_BLOCK",
                    meta=meta | {"contract_validation": contract_validation},
                )
                return
            if not contract_validation.get("passed") and contract_validation.get("remediation_message"):
                remediation_text = (
                    "\n\n【回答质量补充】\n"
                    f"{contract_validation['remediation_message']}"
                )
                yield make_chunk(content=remediation_text, status="loading",metadata={"contract_validation": contract_validation})
                try:
                    conv_manager.add_message_by_thread_id(
                        thread_id=thread_id,
                        role="assistant",
                        content=remediation_text,
                        message_type="text",
                        extra_metadata={"contract_validation": contract_validation,"message_kind": "quality_remediation"},
                    )
                except Exception as remediation_err:
                    logger.warning(f"Failed to persist remediation message: {remediation_err}")

            meta["time_cost"] = asyncio.get_event_loop().time() - start_time
            meta["quality_report"] = quality_artifacts.get("quality_report", {})
            meta["source_refs"] = quality_artifacts.get("source_refs", [])
            meta["tool_trace"] = quality_artifacts.get("tool_trace", [])
            meta["image_evidence_count"] = len(image_evidence_bundle)
            meta["use_kb_image_retrieval"] = use_kb_image_retrieval
            meta["contract_validation"] = contract_validation
            meta["orchestrator_plan"] = orchestrator_context.get("plan", {})
            meta["retrieval_conflict_flags"] = orchestrator_context.get("retrieval_bundle", {}).get("conflict_flags", [])
            yield make_chunk(status="finished", meta=meta)
            BREAKER.record_success(model_key)

        except (asyncio.CancelledError, ConnectionError) as e:
            # 客户端主动中断连接，尝试保存已生成的部分内容
            logger.warning(f"Client disconnected, cancelling stream: {e}")
            OBSERVABILITY.record_failed_sample(
                {
                    "type": "stream_interrupted",
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "query": query,
                    "error": str(e),
                }
            )
            if full_msg:
                # 创建新的 db session，因为原 session 可能已关闭
                new_db = db_manager.get_session()
                try:
                    new_conv_manager = ConversationManager(new_db)
                    msg_dict = full_msg.model_dump() if hasattr(full_msg, "model_dump") else {}
                    content = full_msg.content if hasattr(full_msg, "content") else str(full_msg)
                    new_conv_manager.add_message_by_thread_id(
                        thread_id=thread_id,
                        role="assistant",
                        content=content,
                        message_type="text",
                        extra_metadata=msg_dict | {"error_type": "interrupted"},  # 保存原始 model_dump
                    )
                finally:
                    new_db.close()

            # 通知前端中断（可能发送不到，但用于一致性）
            yield make_chunk(status="interrupted", message="对话已中断", meta=meta)

        except Exception as e:
            err_text = str(e)
            model_key = f"{agent_id}:{meta.get('server_model_name', agent_id)}"
            if "AuthenticationError" in err_text or "401" in err_text or "Unauthorized" in err_text:
                logger.warning(f"Model authentication failed while streaming: {err_text}")
                BREAKER.record_failure(model_key)
                OBSERVABILITY.record_failed_sample(
                    {
                        "type": "model_auth_error",
                        "agent_id": agent_id,
                        "thread_id": thread_id,
                        "query": query,
                        "error": err_text,
                    }
                )
                friendly = (
                    "模型鉴权失败（401）。请在设置页检查并更新对应模型提供商的 API Key，"
                    "确认 default_model 的 provider 与已配置密钥一致后重试。"
                )
                yield make_error_chunk(friendly, error_type="auth", meta=meta)
                return
            if isinstance(e, TimeoutError):
                logger.warning(f"Model stream timeout: {err_text}")
                BREAKER.record_failure(model_key)
                OBSERVABILITY.record_failed_sample(
                    {
                        "type": "model_timeout",
                        "agent_id": agent_id,
                        "thread_id": thread_id,
                        "query": query,
                        "error": err_text,
                    }
                )
                yield make_error_chunk(
                    "模型响应超时，请重试或切换 Instruct/非Thinking 模型；若工具链较长可适当提高流式超时阈值。",
                    error_type="timeout",
                    meta=meta,
                )
                return
            if any(k in err_text for k in ["ModelNotOpen", "has not activated the model"]):
                logger.warning(f"Model not open while streaming: {err_text}")
                if meta.get("server_model_name"):
                    MODEL_NOT_OPEN_CACHE.add(str(meta.get("server_model_name")))
                BREAKER.record_failure(model_key)
                OBSERVABILITY.record_failed_sample(
                    {
                        "type": "model_not_open",
                        "agent_id": agent_id,
                        "thread_id": thread_id,
                        "query": query,
                        "error": err_text,
                    }
                )
                yield make_error_chunk(
                    "当前所选模型未开通（ModelNotOpen）。请在模型控制台开通该模型，或在智能体配置中切换到已开通模型后重试。",
                    error_type="model",
                    meta=meta | {"error_code": "MODEL_NOT_OPEN"},
                )
                return
            logger.error(f"Error streaming messages: {e}, {traceback.format_exc()}")
            BREAKER.record_failure(model_key)
            OBSERVABILITY.record_failed_sample(
                {
                    "type": "stream_unknown_error",
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "query": query,
                    "error": err_text,
                }
            )

            if full_msg:
                # 创建新的 db session，因为原 session 可能已关闭
                new_db = db_manager.get_session()
                try:
                    new_conv_manager = ConversationManager(new_db)
                    msg_dict = full_msg.model_dump() if hasattr(full_msg, "model_dump") else {}
                    content = full_msg.content if hasattr(full_msg, "content") else str(full_msg)
                    new_conv_manager.add_message_by_thread_id(
                        thread_id=thread_id,
                        role="assistant",
                        content=content,
                        message_type="text",
                        extra_metadata=msg_dict | {"error_type": "unexpect"},  # 保存原始 model_dump
                    )
                finally:
                    new_db.close()
            yield make_error_chunk(f"Error streaming messages: {e}", error_type="model")

    return StreamingResponse(stream_messages(), media_type="application/json")


# =============================================================================
# > === 模型管理分组 ===
# =============================================================================


@chat.get("/models")
async def get_chat_models(model_provider: str, current_user: User = Depends(get_admin_user)):
    """获取指定模型提供商的模型列表（需要登录）"""
    model = select_model(model_provider=model_provider)
    return {"models": model.get_models()}


@chat.post("/models/update")
async def update_chat_models(model_provider: str, model_names: list[str], current_user=Depends(get_admin_user)):
    """更新指定模型提供商的模型列表 (仅管理员)"""
    provider_cfg, model_names_map = _get_provider_cfg(model_provider)
    if not provider_cfg:
        raise HTTPException(status_code=404, detail=f"模型提供商不存在: {model_provider}")

    provider_cfg["models"] = list(model_names or [])

    # 如果 default 不在新列表里，自动回退为首个模型，避免配置失效
    current_default = str(provider_cfg.get("default") or "")
    if provider_cfg["models"] and current_default not in provider_cfg["models"]:
        provider_cfg["default"] = provider_cfg["models"][0]

    model_names_map[model_provider] = provider_cfg
    conf.model_names = model_names_map
    conf._save_models_to_file()

    # 同步更新全局 default_model（仅当当前 default_model 属于该 provider 且不再可用）
    global_default = str(getattr(conf, "default_model", ""))
    if "/" in global_default:
        gp, gm = global_default.split("/", 1)
        if gp == model_provider and provider_cfg.get("models") and gm not in provider_cfg["models"]:
            conf.default_model = f"{model_provider}/{provider_cfg['default']}"
            conf.save()

    return {"models": provider_cfg.get("models", []), "default": provider_cfg.get("default", "")}


@chat.get("/tools")
async def get_tools(agent_id: str, current_user: User = Depends(get_required_user)):
    """获取所有可用工具（需要登录）"""
    # 获取Agent实例和配置类
    if not (agent := agent_manager.get_agent(agent_id)):
        raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

    if hasattr(agent, "get_tools"):
        tools = agent.get_tools()
    else:
        tools = get_buildin_tools()

    tools_info = gen_tool_info(tools)
    return {"tools": {tool["id"]: tool for tool in tools_info}}


@chat.post("/agent/{agent_id}/config")
async def save_agent_config(agent_id: str, config: dict = Body(...), current_user: User = Depends(get_required_user)):
    """保存智能体配置到YAML文件（需要登录）"""
    try:
        # 获取Agent实例和配置类
        if not (agent := agent_manager.get_agent(agent_id)):
            raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

        # 使用配置类的save_to_file方法保存配置
        result = agent.context_schema.save_to_file(config, agent.module_name)

        if result:
            return {"success": True, "message": f"智能体 {agent.name} 配置已保存"}
        else:
            raise HTTPException(status_code=500, detail="保存智能体配置失败")

    except Exception as e:
        logger.error(f"保存智能体配置出错: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"保存智能体配置出错: {str(e)}")


@chat.get("/agent/{agent_id}/history")
async def get_agent_history(
    agent_id: str, thread_id: str, current_user: User = Depends(get_required_user), db: Session = Depends(get_db)
):
    """获取智能体历史消息（需要登录）- NEW STORAGE ONLY"""
    try:
        # 获取Agent实例验证
        if not agent_manager.get_agent(agent_id):
            raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

        # Use new storage system ONLY
        conv_manager = ConversationManager(db)
        messages = conv_manager.get_messages_by_thread_id(thread_id)

        # Convert to frontend-compatible format
        history = []
        for msg in messages:
            # Map role to type that frontend expects
            role_type_map = {"user": "human", "assistant": "ai", "tool": "tool", "system": "system"}

            msg_dict = {
                "id": msg.id,  # Include message ID for feedback
                "type": role_type_map.get(msg.role, msg.role),  # human/ai/tool/system
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "error_type": msg.extra_metadata.get("error_type") if msg.extra_metadata else None,
            }

            # Add tool calls if present (for AI messages)
            if msg.tool_calls and len(msg.tool_calls) > 0:
                msg_dict["tool_calls"] = [
                    {
                        "id": str(tc.id),
                        "name": tc.tool_name,
                        "function": {"name": tc.tool_name},  # Frontend compatibility
                        "args": tc.tool_input or {},
                        "tool_call_result": {"content": tc.tool_output} if tc.tool_output else None,
                        "status": tc.status,
                    }
                    for tc in msg.tool_calls
                ]

            history.append(msg_dict)

        logger.info(f"Loaded {len(history)} messages from new storage for thread {thread_id}")
        return {"history": history}

    except Exception as e:
        logger.error(f"获取智能体历史消息出错: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取智能体历史消息出错: {str(e)}")


@chat.get("/agent/{agent_id}/config")
async def get_agent_config(agent_id: str, current_user: User = Depends(get_required_user)):
    """从YAML文件加载智能体配置（需要登录）"""
    try:
        # 检查智能体是否存在
        if not (agent := agent_manager.get_agent(agent_id)):
            raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

        config = await agent.get_config()
        logger.debug(f"config: {config}, ContextClass: {agent.context_schema=}")
        return {"success": True, "config": config}

    except Exception as e:
        logger.error(f"加载智能体配置出错: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"加载智能体配置出错: {str(e)}")


# ==================== 线程管理 API ====================


class ThreadCreate(BaseModel):
    title: str | None = None
    agent_id: str
    metadata: dict | None = None


class ThreadResponse(BaseModel):
    id: str
    user_id: str
    agent_id: str
    title: str | None = None
    created_at: str
    updated_at: str


# =============================================================================
# > === 会话管理分组 ===
# =============================================================================


@chat.post("/thread", response_model=ThreadResponse)
async def create_thread(
    thread: ThreadCreate, db: Session = Depends(get_db), current_user: User = Depends(get_required_user)
):
    """创建新对话线程 (使用新存储系统)"""
    thread_id = str(uuid.uuid4())
    logger.debug(f"thread.agent_id: {thread.agent_id}")

    # Create conversation using new storage system
    conv_manager = ConversationManager(db)
    conversation = conv_manager.create_conversation(
        user_id=str(current_user.id),
        agent_id=thread.agent_id,
        title=thread.title or "新的对话",
        thread_id=thread_id,
        metadata=thread.metadata,
    )

    logger.info(f"Created conversation with thread_id: {thread_id}")

    return {
        "id": conversation.thread_id,
        "user_id": conversation.user_id,
        "agent_id": conversation.agent_id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


@chat.get("/threads", response_model=list[ThreadResponse])
async def list_threads(agent_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_required_user)):
    """获取用户的所有对话线程 (使用新存储系统)"""
    assert agent_id, "agent_id 不能为空"

    logger.debug(f"agent_id: {agent_id}")

    # Use new storage system
    conv_manager = ConversationManager(db)
    conversations = conv_manager.list_conversations(
        user_id=str(current_user.id),
        agent_id=agent_id,
        status="active",
    )

    return [
        {
            "id": conv.thread_id,
            "user_id": conv.user_id,
            "agent_id": conv.agent_id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        }
        for conv in conversations
    ]


@chat.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_required_user)):
    """删除对话线程 (使用新存储系统)"""
    # Use new storage system
    conv_manager = ConversationManager(db)
    conversation = conv_manager.get_conversation_by_thread_id(thread_id)

    if not conversation or conversation.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="对话线程不存在")

    # Soft delete
    success = conv_manager.delete_conversation(thread_id, soft_delete=True)

    if not success:
        raise HTTPException(status_code=500, detail="删除失败")

    return {"message": "删除成功"}


class ThreadUpdate(BaseModel):
    title: str | None = None


@chat.put("/thread/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: str,
    thread_update: ThreadUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """更新对话线程信息 (使用新存储系统)"""
    # Use new storage system
    conv_manager = ConversationManager(db)
    conversation = conv_manager.get_conversation_by_thread_id(thread_id)

    if not conversation or conversation.user_id != str(current_user.id) or conversation.status == "deleted":
        raise HTTPException(status_code=404, detail="对话线程不存在")

    # Update conversation
    updated_conv = conv_manager.update_conversation(
        thread_id=thread_id,
        title=thread_update.title,
    )

    if not updated_conv:
        raise HTTPException(status_code=500, detail="更新失败")

    return {
        "id": updated_conv.thread_id,
        "user_id": updated_conv.user_id,
        "agent_id": updated_conv.agent_id,
        "title": updated_conv.title,
        "created_at": updated_conv.created_at.isoformat(),
        "updated_at": updated_conv.updated_at.isoformat(),
    }


# =============================================================================
# > === 消息反馈分组 ===
# =============================================================================


class MessageFeedbackRequest(BaseModel):
    rating: str  # 'like' or 'dislike'
    reason: str | None = None  # Optional reason for dislike


class MessageFeedbackResponse(BaseModel):
    id: int
    message_id: int
    rating: str
    reason: str | None
    created_at: str


class FeedbackListItem(BaseModel):
    id: int
    message_id: int
    user_id: str
    rating: str
    reason: str | None
    created_at: str
    message_content: str
    conversation_title: str | None
    agent_id: str


@chat.post("/message/{message_id}/feedback", response_model=MessageFeedbackResponse)
async def submit_message_feedback(
    message_id: int,
    feedback_data: MessageFeedbackRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Submit user feedback for a specific message"""
    try:
        # Validate rating
        if feedback_data.rating not in ["like", "dislike"]:
            raise HTTPException(status_code=422, detail="Rating must be 'like' or 'dislike'")

        # Verify message exists and get conversation to check permissions
        message = db.query(Message).filter_by(id=message_id).first()

        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # Verify user has access to this message (through conversation)
        conversation = db.query(Conversation).filter_by(id=message.conversation_id).first()
        if not conversation or conversation.user_id != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if feedback already exists (user can only submit once)
        existing_feedback = (
            db.query(MessageFeedback).filter_by(message_id=message_id, user_id=str(current_user.id)).first()
        )

        if existing_feedback:
            raise HTTPException(status_code=409, detail="Feedback already submitted for this message")

        # Create new feedback
        new_feedback = MessageFeedback(
            message_id=message_id,
            user_id=str(current_user.id),
            rating=feedback_data.rating,
            reason=feedback_data.reason,
        )

        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback)

        logger.info(f"User {current_user.id} submitted {feedback_data.rating} feedback for message {message_id}")

        return MessageFeedbackResponse(
            id=new_feedback.id,
            message_id=new_feedback.message_id,
            rating=new_feedback.rating,
            reason=new_feedback.reason,
            created_at=new_feedback.created_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting message feedback: {e}, {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@chat.get("/message/{message_id}/feedback")
async def get_message_feedback(
    message_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get feedback status for a specific message (for current user)"""
    try:
        # Get user's feedback for this message
        feedback = db.query(MessageFeedback).filter_by(message_id=message_id, user_id=str(current_user.id)).first()

        if not feedback:
            return {"has_feedback": False, "feedback": None}

        return {
            "has_feedback": True,
            "feedback": {
                "id": feedback.id,
                "rating": feedback.rating,
                "reason": feedback.reason,
                "created_at": feedback.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"Error getting message feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")


@chat.get("/feedbacks", response_model=list[FeedbackListItem])
async def list_my_feedbacks(
    rating: str | None = None,
    agent_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """List feedback records for current user."""
    try:
        query = (
            db.query(MessageFeedback, Message, Conversation)
            .join(Message, MessageFeedback.message_id == Message.id)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .filter(MessageFeedback.user_id == str(current_user.id))
            .order_by(MessageFeedback.created_at.desc())
        )

        if rating in ["like", "dislike"]:
            query = query.filter(MessageFeedback.rating == rating)
        if agent_id:
            query = query.filter(Conversation.agent_id == agent_id)

        rows = query.all()
        return [
            {
                "id": feedback.id,
                "message_id": feedback.message_id,
                "user_id": feedback.user_id,
                "rating": feedback.rating,
                "reason": feedback.reason,
                "created_at": feedback.created_at.isoformat(),
                "message_content": message.content[:120] + ("..." if len(message.content) > 120 else ""),
                "conversation_title": conversation.title,
                "agent_id": conversation.agent_id,
            }
            for feedback, message, conversation in rows
        ]
    except Exception as e:
        logger.error(f"Error listing feedbacks for user {current_user.id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to list feedbacks: {str(e)}")


# =============================================================================
# > === 图片上传分组 ===
# =============================================================================


@chat.post("/upload-image")
async def upload_chat_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_required_user),
):
    """上传聊天图片到服务器"""
    try:
        # 检查文件类型
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只能上传图片文件")

        # 检查文件大小（10MB限制）
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="文件大小不能超过10MB")

        # 创建图片保存目录
        images_dir = Path("saves/chat_images")
        images_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一的文件名
        file_extension = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "jpg"
        filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = images_dir / filename

        # 保存图片文件
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 构建完整的图片访问URL
        # 这里假设服务器运行在 localhost:5050，实际部署时需要根据环境配置
        image_url = f"http://localhost:5050/api/system/images/{filename}"

        logger.info(f"用户 {current_user.id} 上传图片: {filename}, 大小: {file_size} bytes")

        return {
            "success": True,
            "image_url": image_url,
            "filename": filename,
            "message": "图片上传成功"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图片上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片上传失败: {str(e)}")
