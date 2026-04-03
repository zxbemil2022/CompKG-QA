from __future__ import annotations

import json
import os
import re

from src.knowledge.pipeline.unstructured_to_kg import BaseNERPlugin, register_ner_plugin
from src.models.chat import select_model


class LLMBasedNERPlugin(BaseNERPlugin):
    """模型版 NER 插件（可在现有部署中按开关启用）。"""

    def __init__(self, enabled: bool | None = None, model: str | None = None, model_spec: str | None = None) -> None:
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("KG_NER_PLUGIN_LLM_ENABLED", "false").lower() == "true"
        )
        self.model = model or os.getenv("KG_NER_PLUGIN_MODEL", "stub-model")
        self.model_spec = model_spec or os.getenv("KG_NER_PLUGIN_MODEL_SPEC", "")

    def _rule_fallback(self, segments: list[str]) -> list[str]:
        entities = set()
        for seg in segments:
            for m in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,20}", seg):
                entities.add(m)
        return list(entities)[:300]

    def _call_llm_ner(self, segments: list[str]) -> list[str]:
        payload = {"segments": segments[:30]}
        try:
            model = select_model(model_spec=self.model_spec) if self.model_spec else select_model(model_name=self.model)
            prompt = (
                "你是实体识别器。请提取输入片段中的关键实体，返回 JSON 数组字符串，例如"
                '["实体A","实体B"]。不要输出额外说明。\n'
                f"输入: {json.dumps(payload, ensure_ascii=False)}"
            )
            response = model.call([{"role": "user", "content": prompt}], stream=False)
            text = (response.content or "").strip()
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S)
            rows = json.loads(text)
            if isinstance(rows, list):
                entities = [str(x).strip()[:60] for x in rows if str(x).strip()]
                return entities[:300] if entities else self._rule_fallback(segments)
            return self._rule_fallback(segments)
        except Exception:
            return self._rule_fallback(segments)

    def extract(self, segments: list[str]) -> list[str]:
        if self.enabled:
            return self._call_llm_ner(segments)
        return self._rule_fallback(segments)


def register() -> None:
    register_ner_plugin("llm", LLMBasedNERPlugin)
