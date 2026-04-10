"""统一错误码定义与说明。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCode:
    code: str
    message: str
    category: str


ERROR_CODE_REGISTRY: dict[str, ErrorCode] = {
    "AUTH_INVALID_KEY": ErrorCode("AUTH_INVALID_KEY", "模型鉴权失败，请检查 API Key 与 provider 配置。", "auth"),
    "MODEL_STREAM_FAILED": ErrorCode("MODEL_STREAM_FAILED", "模型流式调用失败，请稍后重试。", "model"),
    "MODEL_STREAM_TIMEOUT": ErrorCode("MODEL_STREAM_TIMEOUT", "模型响应超时，请重试或切换轻量模型。", "model"),
    "MODEL_CIRCUIT_OPEN": ErrorCode("MODEL_CIRCUIT_OPEN", "模型服务短时不可用（熔断保护中）。", "model"),
    "TOOL_EXECUTION_FAILED": ErrorCode("TOOL_EXECUTION_FAILED", "工具执行失败。", "tool"),
    "STORAGE_WRITE_FAILED": ErrorCode("STORAGE_WRITE_FAILED", "写入存储失败。", "storage"),
    "CONTENT_GUARD_BLOCKED": ErrorCode("CONTENT_GUARD_BLOCKED", "触发内容安全策略。", "guard"),
    "ANSWER_CONTRACT_HARD_BLOCK": ErrorCode("ANSWER_CONTRACT_HARD_BLOCK", "回答未满足最低证据与置信度要求。", "quality"),
}


STREAM_ERROR_CODE_BY_TYPE = {
    "auth": "AUTH_INVALID_KEY",
    "model": "MODEL_STREAM_FAILED",
    "timeout": "MODEL_STREAM_TIMEOUT",
    "tool": "TOOL_EXECUTION_FAILED",
    "storage": "STORAGE_WRITE_FAILED",
    "guard": "CONTENT_GUARD_BLOCKED",
}
