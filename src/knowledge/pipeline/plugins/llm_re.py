from __future__ import annotations

import os
import json
import re
from typing import Any

from src.knowledge.pipeline.unstructured_to_kg import BaseREPlugin, SPOTriple, register_re_plugin
from src.models.chat import select_model


class LLMBasedREStubPlugin(BaseREPlugin):
    """
    LLM-RE 插件占位实现（可复制模板）。

    目标：
    1) 演示“注册 -> 配置 -> 调用”的完整链路；
    2) 在未接入真实 LLM 时保持可运行（退化到简易规则）。

    可配置环境变量：
    - KG_RE_PLUGIN_LLM_ENABLED: true/false，默认 false
    - KG_RE_PLUGIN_MODEL: 仅占位记录
    """

    def __init__(self, enabled: bool | None = None, model: str | None = None, model_spec: str | None = None) -> None:
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("KG_RE_PLUGIN_LLM_ENABLED", "false").lower() == "true"
        )
        self.model = model or os.getenv("KG_RE_PLUGIN_MODEL", "stub-model")
        self.model_spec = model_spec or os.getenv("KG_RE_PLUGIN_MODEL_SPEC", "")

    def _rule_fallback(self, segments: list[str]) -> list[SPOTriple]:
        triples: list[SPOTriple] = []
        for seg in segments:
            if "依赖" in seg:
                parts = seg.split("依赖", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    triples.append(SPOTriple(parts[0].strip()[:60], "DEPENDS_ON", parts[1].strip()[:60]))
            elif "用于" in seg:
                parts = seg.split("用于", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    triples.append(SPOTriple(parts[0].strip()[:60], "USED_FOR", parts[1].strip()[:60]))
        return triples

    def _call_llm_re(self, segments: list[str], entities: list[str]) -> list[SPOTriple]:
        payload: dict[str, Any] = {
            "segments": segments[:20],
            "entities": entities[:120],
            "model": self.model,
        }
        try:
            model = select_model(model_spec=self.model_spec) if self.model_spec else select_model(model_name=self.model)
            prompt = (
                "你是关系抽取器。请从给定文本片段中抽取三元组，返回 JSON 数组，元素格式为"
                '{"subject":"...","predicate":"...","object":"..."}。'
                "不要输出任何额外说明。\n"
                f"输入: {json.dumps(payload, ensure_ascii=False)}"
            )
            response = model.call([{"role": "user", "content": prompt}], stream=False)
            text = (response.content or "").strip()
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S)
            rows = json.loads(text)
            triples: list[SPOTriple] = []
            if isinstance(rows, list):
                for row in rows[:80]:
                    if not isinstance(row, dict):
                        continue
                    sub = str(row.get("subject", "")).strip()[:60]
                    pred = str(row.get("predicate", "")).strip()[:40]
                    obj = str(row.get("object", "")).strip()[:60]
                    if sub and pred and obj and sub != obj:
                        triples.append(SPOTriple(sub, pred, obj))
            return triples or self._rule_fallback(segments)
        except Exception:
            return self._rule_fallback(segments)

    def extract(self, segments: list[str], entities: list[str]) -> list[SPOTriple]:
        if self.enabled:
            return self._call_llm_re(segments, entities)
        return self._rule_fallback(segments)


def register() -> None:
    register_re_plugin("llm_stub", LLMBasedREStubPlugin)
    register_re_plugin("llm", LLMBasedREStubPlugin)
