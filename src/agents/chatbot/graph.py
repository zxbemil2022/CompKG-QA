import os
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from src import config as sys_config
from src.agents.common.base import BaseAgent
from src.agents.common.agent_intelligence import (
    auto_select_tools,
    build_multistep_plan,
    compress_context_memory,
    should_self_reflect,
)
from src.agents.common.mcp import get_mcp_tools
from src.agents.common.models import load_chat_model
from src.utils import logger

from .context import Context
from .state import State
from .tools import get_tools


class ChatbotAgent(BaseAgent):
    name = "智能体助手"
    description = "基础的对话机器人，可以回答问题，默认不使用任何工具，可在配置中启用需要的工具。"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = None
        self.checkpointer = None
        self.context_schema = Context
        self.agent_tools = None
        self._tool_call_unsupported_models: set[str] = set()

    def get_tools(self):
        # 获取所有工具
        return get_tools()

    def _is_model_not_open_error(self, err: Exception) -> bool:
        text = str(err)
        return any(k in text for k in ["ModelNotOpen", "has not activated the model", "model_not_found"])

    def _is_tool_call_unsupported_error(self, err: Exception) -> bool:
        """检测模型是否不支持 function/tool calling。"""
        signals = [
            "Function call is not supported for this model",
            "tool calls are not supported",
            "function_call is not supported",
            "code': 20037",
            '"code": 20037',
            "code: 20037",
        ]

        def _has_signal(text: str) -> bool:
            return any(sig in text for sig in signals)

        if _has_signal(str(err)):
            return True

        # openai.BadRequestError 在不同版本里可能把结构化错误放在 body/response 等字段
        for attr in ("body", "response", "message", "args"):
            try:
                value = getattr(err, attr, None)
            except Exception:
                value = None
            if value is None:
                continue
            if _has_signal(str(value)):
                return True

        cause = getattr(err, "__cause__", None)
        if cause and cause is not err:
            return _has_signal(str(cause))

        return False

    def _is_vision_model(self, model_spec: str) -> bool:
        lowered = (model_spec or "").lower()
        return any(
            k in lowered
            for k in ["vision", "qwen-vl", "vl", "gpt-4o", "gemini", "doubao-seed", "claude-3"]
        )

    def _messages_require_vision(self, messages: list[dict | BaseMessage]) -> bool:
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
        return False

    def _strip_vision_messages(self, messages: list[dict | BaseMessage]) -> list[dict | BaseMessage]:
        """当 fallback 到纯文本模型时，自动把多模态消息降级为文本消息。"""
        converted: list[dict | BaseMessage] = []
        for msg in messages:
            raw_role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "human")
            role_map = {"human": "user", "ai": "assistant"}
            role = role_map.get(str(raw_role), str(raw_role))
            if role == "tool":
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

            text_content = ""
            if isinstance(content, list):
                text_parts: list[str] = []
                image_count = 0
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")).strip())
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        image_count += 1
                text_content = "\n".join([t for t in text_parts if t]).strip()
                if image_count and not text_content:
                    text_content = f"[用户上传了 {image_count} 张图片，请基于现有OCR/图像证据回答]"
            elif isinstance(content, str):
                text_content = content

            elif content is not None:
                text_content = str(content)
            converted.append({"role": role, "content": text_content or "请基于现有上下文继续回答。"})
        return converted

    def _strip_tool_messages(self, messages: list[dict | BaseMessage]) -> list[dict | BaseMessage]:
        """在模型不支持 function calling 时，移除 tool 角色与 tool_call 元数据。"""
        normalized: list[dict | BaseMessage] = []
        for msg in messages:
            raw_role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "human")
            role_map = {"human": "user", "ai": "assistant"}
            role = role_map.get(str(raw_role), str(raw_role))
            if role == "tool":
                continue

            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                text_bits = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_bits.append(str(item.get("text", "")).strip())
                content = "\n".join([t for t in text_bits if t]).strip()
            elif not isinstance(content, str):
                content = str(content or "")

            normalized.append({"role": role, "content": content or "请基于已知信息继续回答。"})
        return normalized

    def _candidate_fallback_models(self, current_model: str, require_vision: bool = False) -> list[str]:
        candidates: list[str] = []
        env_vlm_fallbacks = os.getenv("AGENT_VLM_FALLBACKS", "")
        if require_vision and env_vlm_fallbacks:
            candidates.extend([x.strip() for x in env_vlm_fallbacks.split(",") if x.strip()])

        env_fallbacks = os.getenv("AGENT_MODEL_FALLBACKS", "")
        if env_fallbacks:
            candidates.extend([x.strip() for x in env_fallbacks.split(",") if x.strip()])

        for spec in [
            getattr(sys_config, "vl_model", ""),
            getattr(sys_config, "fast_model", ""),
            getattr(sys_config, "default_model", ""),
        ]:
            if spec and spec not in candidates:
                candidates.append(spec)

        normalized = [m for m in candidates if m and m != current_model]
        if require_vision:
            normalized = [m for m in normalized if self._is_vision_model(m)]
        return normalized

    async def _ainvoke_with_fallback(
        self, model_spec: str, messages: list[dict | BaseMessage], available_tools: list[Any]
    ) -> AIMessage:
        require_vision = self._messages_require_vision(messages)
        model = load_chat_model(model_spec)
        use_tools = bool(available_tools) and model_spec not in self._tool_call_unsupported_models
        if use_tools:
            model = model.bind_tools(available_tools)
        try:
            return cast(AIMessage, await model.ainvoke(messages))
        except Exception as err:
            err_text = str(err)
            tool_call_unsupported = self._is_tool_call_unsupported_error(err)
            if self._is_tool_call_unsupported_error(err):
                self._tool_call_unsupported_models.add(model_spec)
                logger.warning(f"Model {model_spec} doesn't support function calling, retry without tools.")
                plain_model = load_chat_model(model_spec)
                try:
                    plain_messages = self._strip_tool_messages(messages)
                    return cast(AIMessage, await plain_model.ainvoke(plain_messages))
                except Exception as plain_err:
                    err_text = str(plain_err)
                    err = plain_err
                    tool_call_unsupported = self._is_tool_call_unsupported_error(plain_err)

            is_not_open = self._is_model_not_open_error(err)
            is_not_vlm = "not a VLM" in err_text or "Please use text-only prompts" in err_text
            if not (is_not_open or is_not_vlm or tool_call_unsupported):
                raise
            fallback_tools = [] if tool_call_unsupported else available_tools
            fallback_messages = self._strip_tool_messages(messages) if tool_call_unsupported else messages
            if is_not_vlm:
                # 当前模型不支持视觉，先尝试视觉模型；若无可用视觉模型则自动降级为文本输入
                fallback_messages = messages

            for fallback_spec in self._candidate_fallback_models(model_spec, require_vision=require_vision):
                try:
                    logger.warning(
                        f"Primary model unavailable ({model_spec}), fallback to {fallback_spec}"
                    )
                    fb_model = load_chat_model(fallback_spec)
                    if fallback_tools and fallback_spec not in self._tool_call_unsupported_models:
                        fb_model = fb_model.bind_tools(fallback_tools)
                    try:
                        return cast(AIMessage, await fb_model.ainvoke(fallback_messages))
                    except Exception as fb_err:
                        if self._is_tool_call_unsupported_error(fb_err):
                            self._tool_call_unsupported_models.add(fallback_spec)
                            logger.warning(
                                f"Fallback model {fallback_spec} doesn't support function calling, retry plain."
                            )
                            fb_plain_model = load_chat_model(fallback_spec)
                            return cast(AIMessage, await fb_plain_model.ainvoke(fallback_messages))
                        raise
                except Exception as fb_err:
                    logger.warning(f"Fallback model failed: {fallback_spec}, error={fb_err}")
                    continue

            if tool_call_unsupported:
                logger.warning(
                    "Function calling unsupported for %s and all fallbacks, force tool-less completion.",
                    model_spec,
                )
                text_only_messages = self._strip_vision_messages(messages) if require_vision else messages
                text_only_messages = self._strip_tool_messages(text_only_messages)
                last_plain_error: Exception | None = None
                for plain_spec in [model_spec, *self._candidate_fallback_models(model_spec, require_vision=False)]:
                    try:
                        plain_model = load_chat_model(plain_spec)
                        invoke_messages = (
                            self._strip_tool_messages(messages)
                            if (require_vision and self._is_vision_model(plain_spec))
                            else text_only_messages
                        )
                        return cast(AIMessage, await plain_model.ainvoke(invoke_messages))
                    except Exception as plain_err:
                        last_plain_error = plain_err
                        logger.warning(
                            "Tool-less retry failed for %s, error=%s",
                            plain_spec,
                            plain_err,
                        )
                        continue
                if last_plain_error:
                    raise last_plain_error
                raise err

            # 没有可用视觉模型时，降级为文本提示，避免全链路报错
            if require_vision:
                logger.warning("No available VLM fallback found, degrade to text-only prompt.")
                text_only_messages = self._strip_vision_messages(messages)
                for fallback_spec in self._candidate_fallback_models(model_spec, require_vision=False):
                    try:
                        fb_model = load_chat_model(fallback_spec)
                        if fallback_tools and fallback_spec not in self._tool_call_unsupported_models:
                            fb_model = fb_model.bind_tools(fallback_tools)
                        try:
                            return cast(AIMessage, await fb_model.ainvoke(text_only_messages))
                        except Exception as fb_err:
                            if self._is_tool_call_unsupported_error(fb_err):
                                self._tool_call_unsupported_models.add(fallback_spec)
                                fb_plain_model = load_chat_model(fallback_spec)
                                return cast(AIMessage, await fb_plain_model.ainvoke(text_only_messages))
                            raise
                    except Exception as fb_err:
                        logger.warning(f"Text-only fallback model failed: {fallback_spec}, error={fb_err}")
                        continue
            raise

    async def _get_invoke_tools(self, selected_tools: list[str], selected_mcps: list[str]):
        """根据配置获取工具。
        默认不使用任何工具。
        如果配置为列表，则使用列表中的工具。
        """
        enabled_tools = []
        # 如果agent_tools为空，则获取所有工具，否则使用agent_tools
        self.agent_tools = self.agent_tools or self.get_tools()
        if selected_tools and isinstance(selected_tools, list) and len(selected_tools) > 0:
            enabled_tools = [tool for tool in self.agent_tools if tool.name in selected_tools]
        else:
            enabled_tools = list(self.agent_tools)

        if selected_mcps and isinstance(selected_mcps, list) and len(selected_mcps) > 0:
            for mcp in selected_mcps:
                enabled_tools.extend(await get_mcp_tools(mcp))

        return enabled_tools

    async def llm_call(self, state: State, runtime: Runtime[Context] = None) -> dict[str, Any]:
        """调用 llm 模型 - 异步版本以支持异步工具"""
        current_model_spec = runtime.context.model

        available_tools = await self._get_invoke_tools(runtime.context.tools, runtime.context.mcps)

        latest_user_query = ""
        for msg in reversed(state.messages):
            if getattr(msg, "type", "") == "human":
                latest_user_query = str(getattr(msg, "content", "") or "")
                break

        if runtime.context.enable_tool_auto_routing:
            available_tools = auto_select_tools(latest_user_query, available_tools)

        logger.info(f"LLM binded ({len(available_tools)}) available_tools: {[tool.name for tool in available_tools]}")

        system_notes = [runtime.context.system_prompt]
        work_messages = list(state.messages)

        if runtime.context.enable_memory_manager:
            memory_pack = compress_context_memory(work_messages)
            work_messages = memory_pack["short_term"]
            if memory_pack["long_term_summary"]:
                system_notes.append(f"[长期记忆压缩] {memory_pack['long_term_summary']}")

        if runtime.context.enable_task_planning and latest_user_query:
            plan_steps = build_multistep_plan(latest_user_query)
            if plan_steps:
                plan_text = "\n".join([f"- {s}" for s in plan_steps])
                system_notes.append(f"[多步规划]\n{plan_text}")

        response = await self._ainvoke_with_fallback(
            current_model_spec,
            [{"role": "system", "content": "\n\n".join(system_notes)}, *work_messages],
            available_tools,
        )

        if runtime.context.enable_self_reflection and should_self_reflect(latest_user_query, str(response.content)):
            reflection_msg = await self._ainvoke_with_fallback(
                current_model_spec,
                [
                    {"role": "system","content": "你是回答质量审查器。请在不改变事实的前提下，修复歧义与遗漏，输出更稳健版本。"},
                    {"role": "user", "content": str(response.content)},
                ],
                available_tools,
            )
            return {"messages": [reflection_msg]}
        return {"messages": [response]}

    async def dynamic_tools_node(self, state: State, runtime: Runtime[Context]) -> dict[str, list[ToolMessage]]:
        """Execute tools dynamically based on configuration.

        This function gets the available tools based on the current configuration
        and executes the requested tool calls from the last message.
        """
        # Get available tools based on configuration
        available_tools = await self._get_invoke_tools(runtime.context.tools, runtime.context.mcps)

        # Create a ToolNode with the available tools
        tool_node = ToolNode(available_tools)

        # Execute the tool node
        result = await tool_node.ainvoke(state)

        return cast(dict[str, list[ToolMessage]], result)

    async def get_graph(self, **kwargs):
        """构建图"""
        if self.graph:
            return self.graph

        builder = StateGraph(State, context_schema=self.context_schema)
        builder.add_node("chatbot", self.llm_call)
        builder.add_node("tools", self.dynamic_tools_node)
        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        builder.add_edge("tools", "chatbot")
        builder.add_edge("chatbot", END)

        self.checkpointer = await self._get_checkpointer()
        graph = builder.compile(checkpointer=self.checkpointer, name=self.name)
        self.graph = graph
        return graph


def main():
    pass


if __name__ == "__main__":
    main()
    # asyncio.run(main())
