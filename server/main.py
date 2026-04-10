import asyncio
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from server.routers import router
from server.utils.auth_utils import AuthUtils
from server.services.observability import get_observability_registry
from server.utils.lifespan import lifespan
from server.utils.auth_middleware import is_public_path
from server.utils.common_utils import setup_logging

# 设置日志配置
setup_logging()
logger = logging.getLogger(__name__)

RATE_LIMIT_MAX_ATTEMPTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_ENDPOINTS = {("/api/auth/token", "POST")}

# In-memory login attempt tracker to reduce brute-force exposure per worker
_login_attempts: defaultdict[str, deque[float]] = defaultdict(deque)
_attempt_lock = asyncio.Lock()

app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api")

# CORS 设置
cors_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
allow_all_origins = len(cors_origins) == 1 and cors_origins[0] == "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class LoginRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        normalized_path = request.url.path.rstrip("/") or "/"
        request_signature = (normalized_path, request.method.upper())

        if request_signature in RATE_LIMIT_ENDPOINTS:
            client_ip = _extract_client_ip(request)
            now = time.monotonic()

            async with _attempt_lock:
                attempt_history = _login_attempts[client_ip]

                while attempt_history and now - attempt_history[0] > RATE_LIMIT_WINDOW_SECONDS:
                    attempt_history.popleft()

                if len(attempt_history) >= RATE_LIMIT_MAX_ATTEMPTS:
                    retry_after = int(max(1, RATE_LIMIT_WINDOW_SECONDS - (now - attempt_history[0])))
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "登录尝试过于频繁，请稍后再试"},
                        headers={"Retry-After": str(retry_after)},
                    )

                attempt_history.append(now)

            response = await call_next(request)

            if response.status_code < 400:
                async with _attempt_lock:
                    _login_attempts.pop(client_ip, None)

            return response

        return await call_next(request)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        registry = get_observability_registry()
        start = time.monotonic()
        success = False
        try:
            response = await call_next(request)
            success = response.status_code < 500
            return response
        except Exception:
            success = False
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            route_key = f"{request.method.upper()} {request.url.path}"
            registry.record_route(route_key=route_key, latency_ms=latency_ms, success=success)


# 鉴权中间件
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 获取请求路径
        path = request.url.path

        # 检查是否为公开路径，公开路径无需身份验证
        if is_public_path(path):
            return await call_next(request)

        if not path.startswith("/api"):
            # 非API路径，可能是前端路由或静态资源
            return await call_next(request)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        # 提取并校验Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": f"请先登录。Path: {path}"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split("Bearer ", 1)[1].strip()
        try:
            payload = AuthUtils.verify_access_token(token)
            request.state.token = token
            request.state.user_id = payload.get("sub")
        except ValueError as e:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": str(e)},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 继续处理请求
        return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on path %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "服务器内部错误", "path": request.url.path},
    )
# 添加鉴权中间件
app.add_middleware(RequestMetricsMiddleware)
app.add_middleware(LoginRateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# 添加静态文件服务
# 创建静态文件目录
static_dir = Path("saves/chat_images")
static_dir.mkdir(parents=True, exist_ok=True)

# 挂载静态文件服务
app.mount("/static/chat_images", StaticFiles(directory=static_dir), name="chat_images")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
