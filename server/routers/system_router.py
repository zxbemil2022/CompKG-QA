import os
import hashlib
from collections import deque
from pathlib import Path

import requests
import yaml
from fastapi import APIRouter, Body, Depends, HTTPException

from src.storage.db.models import User
from server.utils.auth_middleware import get_admin_user, get_superadmin_user
from server.services.breaker_provider import get_global_breaker
from server.services.observability import get_observability_registry
from server.services.retrieval_cache import get_retrieval_cache
from src import config, graph_base
from src.models.chat import test_chat_model_status, test_all_chat_models_status
from src.utils.logging_config import logger

system = APIRouter(prefix="/system", tags=["system"])

# =============================================================================
# === 健康检查分组 ===
# =============================================================================


@system.get("/health")
async def health_check():
    """系统健康检查接口（公开接口）"""
    return {"status": "ok", "message": "服务正常运行"}


@system.get("/health/dependencies")
async def health_check_dependencies(current_user: User = Depends(get_admin_user)):
    """依赖健康探针：模型配置/模型连通性/图数据库状态。"""
    dependency_status = {
        "config": {"status": "ok", "message": "config loaded"},
        "chat_model": {"status": "unknown", "message": ""},
        "neo4j": {"status": "unknown", "message": ""},
        "breaker": {"status": "healthy", "message": "breaker metrics available"},
        "retrieval_cache": {"status": "healthy", "message": "retrieval cache metrics available"},
    }

    # 1) 模型配置与可达性校验
    default_model = getattr(config, "default_model", "")
    try:
        provider, model_name = default_model.split("/", 1)
        model_status = await test_chat_model_status(provider, model_name)
        dependency_status["chat_model"] = model_status
    except Exception as exc:  # noqa: BLE001
        dependency_status["chat_model"] = {"status": "error", "message": f"default_model 校验失败: {exc}"}

    # 2) Neo4j 状态校验
    try:
        is_running = bool(graph_base and graph_base.is_running())
        dependency_status["neo4j"] = {
            "status": "healthy" if is_running else "unhealthy",
            "message": "neo4j connected" if is_running else "neo4j not connected",
        }
    except Exception as exc:  # noqa: BLE001
        dependency_status["neo4j"] = {"status": "error", "message": f"neo4j health check failed: {exc}"}

    overall = "healthy"
    if any(item.get("status") in ["error", "unhealthy", "unavailable"] for item in dependency_status.values()):
        overall = "degraded"

    dependency_status["breaker"]["metrics"] = get_global_breaker().get_metrics()
    dependency_status["retrieval_cache"]["metrics"] = get_retrieval_cache().get_metrics()

    return {"overall_status": overall, "dependencies": dependency_status}


# =============================================================================
# === 配置管理分组 ===
# =============================================================================


@system.get("/config")
def get_config(current_user: User = Depends(get_admin_user)):
    """获取系统配置"""
    return config.dump_config()


@system.post("/config")
async def update_config_single(key=Body(...), value=Body(...), current_user: User = Depends(get_admin_user)) -> dict:
    """更新单个配置项"""
    config[key] = value
    config.save()
    return config.dump_config()


@system.post("/config/update")
async def update_config_batch(items: dict = Body(...), current_user: User = Depends(get_admin_user)) -> dict:
    """批量更新配置项"""
    config.update(items)
    config.save()
    return config.dump_config()


@system.get("/config/fingerprint")
def get_config_fingerprint(current_user: User = Depends(get_admin_user)):
    """配置与依赖指纹，用于识别环境漂移。"""
    config_dump = config.dump_config()
    config_bytes = str(sorted(config_dump.items())).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    requirements_path = Path("requirements.txt")
    req_hash = ""
    if requirements_path.exists():
        req_hash = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
    return {
        "config_hash": config_hash,
        "requirements_hash": req_hash,
        "python_env": {
            "STARTUP_PRECHECK_ENABLED": os.getenv("STARTUP_PRECHECK_ENABLED", "true"),
            "RETRIEVAL_CACHE_PROVIDER": os.getenv("RETRIEVAL_CACHE_PROVIDER", "local"),
            "RETRIEVAL_CACHE_TTL_SEC": os.getenv("RETRIEVAL_CACHE_TTL_SEC", "300"),
            "CHAT_STREAM_TOTAL_TIMEOUT_SEC": os.getenv("CHAT_STREAM_TOTAL_TIMEOUT_SEC", "300"),
            "CHAT_STREAM_IDLE_TIMEOUT_SEC": os.getenv("CHAT_STREAM_IDLE_TIMEOUT_SEC", "120"),
        },
    }


@system.post("/restart")
async def restart_system(current_user: User = Depends(get_superadmin_user)):
    """重启系统（仅超级管理员）"""
    graph_base.start()
    return {"message": "系统已重启"}


@system.get("/logs")
def get_system_logs(current_user: User = Depends(get_admin_user)):
    """获取系统日志"""
    try:
        from src.utils.logging_config import LOG_FILE

        with open(LOG_FILE) as f:
            last_lines = deque(f, maxlen=1000)

        log = "".join(last_lines)
        return {"log": log, "message": "success", "log_file": LOG_FILE}
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统日志失败: {str(e)}")


# =============================================================================
# === 信息管理分组 ===
# =============================================================================
def load_info_config():
    """加载信息配置文件"""
    try:
        # 配置文件路径
        brand_file_path = os.environ.get("YUXI_BRAND_FILE_PATH", "src/config/static/info.local.yaml")
        config_path = Path(brand_file_path)

        # 检查文件是否存在
        if not config_path.exists():
            logger.debug(f"The config file {config_path} does not exist, using default config")
            config_path = Path("src/config/static/info.template.yaml")

        # 读取配置文件
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return config

    except Exception as e:
        logger.error(f"Failed to load info config: {e}")
        return get_default_info_config()


def get_default_info_config():
    """获取默认信息配置"""
    return {

        "organization": {"name": "智析图谱", "logo": "/favicon.svg", "avatar": "/avatar.jpg"},
        "branding": {
            "name": "CompKG-QA",
            "title": "CompKG-QA",
            "subtitle": "大模型驱动的计算机知识图谱问答平台",
            "description": "融合向量检索与知识图谱，面向计算机领域提供可追溯问答",
        },
        "features": ["🧠 计算机知识图谱", "🕸️ 图谱+向量混合检索", "🤖 多模型切换", "🧩 可扩展工具链"],
        "footer": {"copyright": "©CS 智析图谱 2026 [WIP] v0.4.0"},
    }


@system.get("/info")
async def get_info_config():
    """获取系统信息配置（公开接口，无需认证）"""
    try:
        config = load_info_config()
        return {"success": True, "data": config}
    except Exception as e:
        logger.error(f"获取信息配置失败: {e}")
        raise HTTPException(status_code=500, detail="获取信息配置失败")


@system.post("/info/reload")
async def reload_info_config(current_user: User = Depends(get_admin_user)):
    """重新加载信息配置"""
    try:
        config = load_info_config()
        return {"success": True, "message": "配置重新加载成功", "data": config}
    except Exception as e:
        logger.error(f"重新加载信息配置失败: {e}")
        raise HTTPException(status_code=500, detail="重新加载信息配置失败")


# === OCR服务分组 ===
# =============================================================================


@system.get("/ocr/stats")
async def get_ocr_stats(current_user: User = Depends(get_admin_user)):
    """
    获取OCR服务使用统计信息
    返回各个OCR服务的处理统计和性能指标
    """
    try:
        from src.plugins._ocr import get_ocr_stats

        stats = get_ocr_stats()

        return {"status": "success", "stats": stats, "message": "OCR统计信息获取成功"}
    except Exception as e:
        logger.error(f"获取OCR统计信息失败: {str(e)}")
        return {"status": "error", "stats": {}, "message": f"获取OCR统计信息失败: {str(e)}"}


@system.get("/ocr/health")
async def check_ocr_services_health(current_user: User = Depends(get_admin_user)):
    """
    检查所有OCR服务的健康状态
    返回各个OCR服务的可用性信息
    """
    health_status = {
        "rapid_ocr": {"status": "unknown", "message": ""},
        "mineru_ocr": {"status": "unknown", "message": ""},
        "paddlex_ocr": {"status": "unknown", "message": ""},
    }

    # 检查 RapidOCR (ONNX) 模型
    try:
        from src.plugins._ocr import OCRPlugin

        model_dir_root = OCRPlugin().model_dir_root
        model_dir = os.path.join(model_dir_root, "SWHL", "RapidOCR")
        det_model_path = os.path.join(model_dir, "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx")
        rec_model_path = os.path.join(model_dir, "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx")
        checked_paths = {
            "model_dir_root": model_dir_root,
            "model_dir": model_dir,
            "det_model_path": det_model_path,
            "rec_model_path": rec_model_path,
        }

        if os.path.exists(model_dir) and os.path.exists(det_model_path) and os.path.exists(rec_model_path):
            # 尝试初始化RapidOCR
            from rapidocr_onnxruntime import RapidOCR

            test_ocr = RapidOCR(det_box_thresh=0.3, det_model_path=det_model_path, rec_model_path=rec_model_path)  # noqa: F841
            health_status["rapid_ocr"]["status"] = "healthy"
            health_status["rapid_ocr"]["message"] = "RapidOCR模型已加载"
            health_status["rapid_ocr"]["paths"] = checked_paths
        else:
            health_status["rapid_ocr"]["status"] = "unavailable"
            health_status["rapid_ocr"]["message"] = "模型文件不存在，请检查 MODEL_DIR 和目录结构"
            health_status["rapid_ocr"]["paths"] = checked_paths
    except Exception as e:
        health_status["rapid_ocr"]["status"] = "error"
        health_status["rapid_ocr"]["message"] = f"RapidOCR初始化失败: {str(e)}"

    # 检查 MinerU OCR 服务
    try:
        mineru_uri = os.getenv("MINERU_OCR_URI", "http://localhost:30000")
        health_url = f"{mineru_uri}/health"

        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            health_status["mineru_ocr"]["status"] = "healthy"
            health_status["mineru_ocr"]["message"] = f"MinerU服务运行正常 ({mineru_uri})"
        else:
            health_status["mineru_ocr"]["status"] = "unhealthy"
            health_status["mineru_ocr"]["message"] = f"MinerU服务响应异常({mineru_uri}): {response.status_code}"
    except requests.exceptions.ConnectionError:
        health_status["mineru_ocr"]["status"] = "unavailable"
        health_status["mineru_ocr"]["message"] = "MinerU服务无法连接，请检查服务是否启动"
    except requests.exceptions.Timeout:
        health_status["mineru_ocr"]["status"] = "timeout"
        health_status["mineru_ocr"]["message"] = "MinerU服务连接超时"
    except Exception as e:
        health_status["mineru_ocr"]["status"] = "error"
        health_status["mineru_ocr"]["message"] = f"MinerU服务检查失败: {str(e)}"

    # 检查 PaddleX OCR 服务
    try:
        paddlex_uri = os.getenv("PADDLEX_URI", "http://localhost:8080")
        health_url = f"{paddlex_uri}/health"

        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            health_status["paddlex_ocr"]["status"] = "healthy"
            health_status["paddlex_ocr"]["message"] = f"PaddleX服务运行正常({paddlex_uri})"
        else:
            health_status["paddlex_ocr"]["status"] = "unhealthy"
            health_status["paddlex_ocr"]["message"] = f"PaddleX服务响应异常({paddlex_uri}): {response.status_code}"
    except requests.exceptions.ConnectionError:
        health_status["paddlex_ocr"]["status"] = "unavailable"
        health_status["paddlex_ocr"]["message"] = f"PaddleX服务无法连接，请检查服务是否启动({paddlex_uri})"
    except requests.exceptions.Timeout:
        health_status["paddlex_ocr"]["status"] = "timeout"
        health_status["paddlex_ocr"]["message"] = f"PaddleX服务连接超时({paddlex_uri})"
    except Exception as e:
        health_status["paddlex_ocr"]["status"] = "error"
        health_status["paddlex_ocr"]["message"] = f"PaddleX服务检查失败: {str(e)}"

    # 计算整体健康状态
    overall_status = "healthy" if any(svc["status"] == "healthy" for svc in health_status.values()) else "unhealthy"

    return {"overall_status": overall_status, "services": health_status, "message": "OCR服务健康检查完成"}


@system.get("/observability/metrics")
async def get_observability_metrics(current_user: User = Depends(get_admin_user)):
    """关键路径观测指标（请求时延/失败率、缓存与熔断指标）。"""
    registry = get_observability_registry()
    return {
        "request_metrics": registry.get_metrics(),
        "breaker_metrics": get_global_breaker().get_metrics(),
        "retrieval_cache_metrics": get_retrieval_cache().get_metrics(),
    }


@system.get("/observability/failures")
async def get_observability_failures(
    limit: int = 50,
    current_user: User = Depends(get_admin_user),
):
    """失败样本回流（用于离线审计与回归集补充）。"""
    registry = get_observability_registry()
    return {"samples": registry.get_failed_samples(limit=limit)}


# =============================================================================
# === 聊天模型状态检查分组 ===
# =============================================================================


@system.get("/chat-models/status")
async def get_chat_model_status(provider: str, model_name: str, current_user: User = Depends(get_admin_user)):
    """获取指定聊天模型的状态"""
    logger.debug(f"Checking chat model status: {provider}/{model_name}")
    try:
        status = await test_chat_model_status(provider, model_name)
        return {"status": status, "message": "success"}
    except Exception as e:
        logger.error(f"获取聊天模型状态失败 {provider}/{model_name}: {e}")
        return {
            "message": f"获取聊天模型状态失败: {e}",
            "status": {"provider": provider, "model_name": model_name, "status": "error", "message": str(e)},
        }


@system.get("/chat-models/all/status")
async def get_all_chat_models_status(current_user: User = Depends(get_admin_user)):
    """获取所有聊天模型的状态"""
    logger.debug("Checking all chat models status")
    try:
        status = await test_all_chat_models_status()
        return {"status": status, "message": "success"}
    except Exception as e:
        logger.error(f"获取所有聊天模型状态失败: {e}")
        return {"message": f"获取所有聊天模型状态失败: {e}", "status": {"models": {}, "total": 0, "available": 0}}


# =============================================================================
# === 图片服务分组 ===
# =============================================================================


@system.get("/images/{filename}")
async def get_image(filename: str):
    """
    获取聊天图片（公开接口，无需认证）
    前端可以通过 http://localhost:8000/api/system/images/{filename} 访问图片
    例如：<img src="http://localhost:8000/api/system/images/773b2205d3f241e9a8e38765d77371ab.jpg" />
    """
    try:
        # 图片存储目录
        image_dir = Path("saves/chat_images")
        
        # 构建完整的图片路径
        image_path = image_dir / filename
        
        # 安全检查：确保文件在指定目录内
        if not image_path.resolve().is_relative_to(image_dir.resolve()):
            raise HTTPException(status_code=403, detail="访问路径非法")
        
        # 检查文件是否存在
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="图片不存在")
        
        # 检查文件是否为图片文件
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if image_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 读取图片文件并返回
        from fastapi.responses import FileResponse
        return FileResponse(
            path=image_path,
            media_type=f"image/{image_path.suffix[1:].lower()}",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取图片失败 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")
