"""Circuit breaker providers (local / redis) with lightweight metrics."""

from __future__ import annotations

import json
import os
from time import monotonic
from typing import Protocol

from src.utils.logging_config import logger


class BreakerProvider(Protocol):
    def allow(self, key: str) -> bool: ...
    def record_failure(self, key: str) -> None: ...
    def record_success(self, key: str) -> None: ...
    def get_metrics(self) -> dict: ...


class LocalBreakerProvider:
    def __init__(self, threshold: int = 3, cooldown_sec: int = 90):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.state: dict[str, dict[str, float | int]] = {}
        self.metrics = {"allow": 0, "blocked": 0, "failure": 0, "success": 0}

    def allow(self, key: str) -> bool:
        current = self.state.get(key, {})
        open_until = float(current.get("open_until", 0))
        allowed = monotonic() >= open_until
        self.metrics["allow" if allowed else "blocked"] += 1
        return allowed

    def record_failure(self, key: str) -> None:
        now = monotonic()
        current = self.state.setdefault(key, {"fail_count": 0, "open_until": 0.0})
        current["fail_count"] = int(current.get("fail_count", 0)) + 1
        if int(current["fail_count"]) >= self.threshold:
            current["open_until"] = now + self.cooldown_sec
            current["fail_count"] = 0
        self.metrics["failure"] += 1

    def record_success(self, key: str) -> None:
        self.state[key] = {"fail_count": 0, "open_until": 0.0}
        self.metrics["success"] += 1

    def get_metrics(self) -> dict:
        return {"provider": "local", **self.metrics, "keys": len(self.state)}


class RedisBreakerProvider:
    def __init__(self, redis_url: str, threshold: int = 3, cooldown_sec: int = 90):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.metrics = {"allow": 0, "blocked": 0, "failure": 0, "success": 0}
        self._client = None
        try:
            import redis  # type: ignore

            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Redis breaker provider unavailable, fallback required: {exc}")
            self._client = None

    def _k(self, key: str) -> str:
        return f"breaker:{key}"

    def allow(self, key: str) -> bool:
        if not self._client:
            self.metrics["allow"] += 1
            return True
        payload = self._client.get(self._k(key))
        if not payload:
            self.metrics["allow"] += 1
            return True
        state = json.loads(payload)
        allowed = monotonic() >= float(state.get("open_until", 0))
        self.metrics["allow" if allowed else "blocked"] += 1
        return allowed

    def record_failure(self, key: str) -> None:
        if not self._client:
            self.metrics["failure"] += 1
            return
        now = monotonic()
        raw = self._client.get(self._k(key))
        state = json.loads(raw) if raw else {"fail_count": 0, "open_until": 0.0}
        state["fail_count"] = int(state.get("fail_count", 0)) + 1
        if int(state["fail_count"]) >= self.threshold:
            state["open_until"] = now + self.cooldown_sec
            state["fail_count"] = 0
        self._client.set(self._k(key), json.dumps(state), ex=max(self.cooldown_sec * 3, 300))
        self.metrics["failure"] += 1

    def record_success(self, key: str) -> None:
        if self._client:
            self._client.delete(self._k(key))
        self.metrics["success"] += 1

    def get_metrics(self) -> dict:
        return {"provider": "redis", **self.metrics}


def get_breaker_provider() -> BreakerProvider:
    provider = os.getenv("BREAKER_PROVIDER", "local").lower()
    threshold = int(os.getenv("BREAKER_THRESHOLD", "3"))
    cooldown = int(os.getenv("BREAKER_COOLDOWN_SEC", "90"))
    if provider == "redis":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_provider = RedisBreakerProvider(redis_url, threshold=threshold, cooldown_sec=cooldown)
        if redis_provider._client:
            return redis_provider
    return LocalBreakerProvider(threshold=threshold, cooldown_sec=cooldown)


_GLOBAL_BREAKER: BreakerProvider | None = None


def get_global_breaker() -> BreakerProvider:
    global _GLOBAL_BREAKER
    if _GLOBAL_BREAKER is None:
        _GLOBAL_BREAKER = get_breaker_provider()
    return _GLOBAL_BREAKER

