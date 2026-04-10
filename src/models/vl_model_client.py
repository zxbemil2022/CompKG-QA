import os
import base64
from typing import Any, Optional
import requests


from src.config.app import config
from src.utils.logging_config import logger


class VLModelClient:
    """视觉语言模型客户端（支持自动候选与回退）。"""

    VISION_KEYWORDS = (
        "vision",
        "qwen-vl",
        "vl",
        "gpt-4o",
        "gemini",
        "doubao-seed",
        "claude-3",
    )
    
    def __init__(self):
        self.provider: str | None = None
        self.model_name: str | None = None
        self.base_url: str | None = None
        self.api_key: str | None = None
        self.candidates: list[dict[str, str | None]] = []
        self._setup_model()
    
    def _is_vision_name(self, model_name: str) -> bool:
        lower = (model_name or "").lower()
        return any(k in lower for k in self.VISION_KEYWORDS)

    def _resolve_provider_config(self, provider: str) -> dict[str, Any] | None:
        if provider in (config.vl_model_names or {}):
            return config.vl_model_names[provider]
        if provider in (config.model_names or {}):
            return config.model_names[provider]
        return None

    def _append_candidate(self, candidates: list[dict[str, str | None]], provider: str, model_name: str | None):
        provider_cfg = self._resolve_provider_config(provider)
        if not provider_cfg:
            return

        model = model_name or provider_cfg.get("default")
        if not model:
            return

        env_var = provider_cfg.get("env")
        api_key = None if env_var == "NO_API_KEY" else os.getenv(env_var or "")
        if env_var not in (None, "NO_API_KEY") and not api_key:
            return

        base_url = provider_cfg.get("base_url")
        if not base_url:
            return

        row = {
            "provider": provider,
            "model_name": model,
            "base_url": base_url,
            "api_key": api_key,
        }
        if row not in candidates:
            candidates.append(row)

    def _setup_model(self):
        """构建候选 VL 模型列表，并默认选择首个可用项。"""
        candidates: list[dict[str, str | None]] = []

        # 1) 显式配置的 vl_model
        vl_model_spec = str(getattr(config, "vl_model", "") or "").strip()
        if vl_model_spec:
            if "/" in vl_model_spec:
                p, m = vl_model_spec.split("/", 1)
                self._append_candidate(candidates, p, m)
            else:
                self._append_candidate(candidates, vl_model_spec, None)

                # 2) VL_MODEL_INFO 中的默认与模型列表
                for provider, info in (config.vl_model_names or {}).items():
                    self._append_candidate(candidates, provider, info.get("default"))
                    for m in info.get("models", []) or []:
                        self._append_candidate(candidates, provider, m)

                # 3) MODEL_NAMES 中视觉模型（支持豆包/Qwen-VL 等）
                for provider, info in (config.model_names or {}).items():
                    for m in info.get("models", []) or []:
                        if self._is_vision_name(m):
                            self._append_candidate(candidates, provider, m)

                self.candidates = candidates

                if candidates:
                    first = candidates[0]
                    self.provider = str(first["provider"])
                    self.model_name = str(first["model_name"])
                    self.base_url = str(first["base_url"])
                    self.api_key = first["api_key"]
                    logger.info(
                        f"VL模型客户端初始化成功，候选数={len(candidates)}，默认={self.provider}/{self.model_name}"
                    )
                else:
                    self.provider = None
                    self.model_name = None
                    self.base_url = None
                    self.api_key = None
                    logger.warning("未发现可用VL模型（检查 VL_MODEL_INFO / MODEL_NAMES 与对应 API Key）")
    
    def is_available(self) -> bool:
        # 配置可能被运行时更新，因此每次检查时重建一次候选
        self._setup_model()
        return len(self.candidates) > 0
    
    def _image_to_base64(self, image_path: str) -> str:
        try:
            if image_path.startswith(("http://", "https://")):
                if image_path.startswith("http://localhost:5050/api/system/images/"):
                    filename = image_path.split("/")[-1]
                    local_path = os.path.join("saves", "chat_images", filename)
                    if os.path.exists(local_path):
                        # 直接从本地文件读取，避免网络请求
                        with open(local_path, "rb") as f:
                            image_data = f.read()
                    else:
                        # 如果本地文件不存在，回退到网络下载
                        response = requests.get(image_path, timeout=10)
                        response.raise_for_status()
                        image_data = response.content
                else:
                    # 其他网络URL，正常下载
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()
                    image_data = response.content
            else:
                # 从本地文件读取图片
                with open(image_path, 'rb') as f:
                    image_data = f.read()

            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            raise ValueError(f"无法加载图片: {image_path}，错误: {str(e)}")

    def _call_vl_model(self, candidate: dict[str, str | None], image_base64: str, prompt: str) -> str:
        payload = {
            "model": candidate["model_name"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if candidate.get("api_key"):
            headers["Authorization"] = f"Bearer {candidate['api_key']}"

        response = requests.post(
            f"{candidate['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return str(result["choices"][0]["message"]["content"]).strip()

    def get_image_description(self, image_path: str, prompt: Optional[str] = None) -> str:

        if not self.is_available():
            raise RuntimeError("VL模型不可用，请检查配置")

        image_base64 = self._image_to_base64(image_path)
        user_prompt = prompt or "请详细描述这张图片的内容、场景、物体、颜色、风格等特征"

        last_err = None
        for candidate in self.candidates:
            try:
                description = self._call_vl_model(candidate, image_base64, user_prompt)
                self.provider = str(candidate["provider"])
                self.model_name = str(candidate["model_name"])
                self.base_url = str(candidate["base_url"])
                self.api_key = candidate.get("api_key")
                logger.info(f"VL模型图片描述生成成功: {self.provider}/{self.model_name}")
                return description
            except Exception as e:
                last_err = e
                logger.warning(
                        f"VL候选调用失败: {candidate.get('provider')}/{candidate.get('model_name')}, error={e}"
            )
                continue

                raise RuntimeError(f"所有VL候选模型均调用失败: {last_err}")

# 全局VL模型客户端实例
vl_client = VLModelClient()