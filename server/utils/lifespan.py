from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from server.services import tasker
from src import config, graph_base
from src.models.chat import test_chat_model_status
from src.utils.logging_config import logger

# TODO:[已完成]使用lifespan进行统一生命周期管理


@asynccontextmanager
async def lifespan(app: FastAPI):
    precheck_enabled = os.getenv("STARTUP_PRECHECK_ENABLED", "true").lower() == "true"
    block_on_precheck_fail = os.getenv("STARTUP_BLOCK_ON_PRECHECK_FAIL", "false").lower() == "true"

    if precheck_enabled:
        precheck_errors = []
        # 默认模型连通性预热
        try:
            provider, model_name = str(getattr(config, "default_model", "")).split("/", 1)
            model_status = await test_chat_model_status(provider, model_name)
            if model_status.get("status") not in ["ok", "healthy", "available"]:
                precheck_errors.append(f"default model unavailable: {model_status}")
        except Exception as exc:  # noqa: BLE001
            precheck_errors.append(f"default model precheck failed: {exc}")

        # Neo4j 连通性预热
        try:
            if not (graph_base and graph_base.is_running()):
                precheck_errors.append("neo4j is not running")
        except Exception as exc:  # noqa: BLE001
            precheck_errors.append(f"neo4j precheck failed: {exc}")

        if precheck_errors:
            logger.warning(f"Startup precheck found issues: {precheck_errors}")
            if block_on_precheck_fail:
                raise RuntimeError(f"Startup blocked by precheck failures: {precheck_errors}")

    await tasker.start()
    """FastAPI lifespan事件管理器"""
    yield
    await tasker.shutdown()
