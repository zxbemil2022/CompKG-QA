"""Retrieval cache provider with optional Redis backend and hit/miss metrics."""

from __future__ import annotations

import json
import os
from time import monotonic

from src.utils.logging_config import logger


class RetrievalCache:
    def __init__(self):
        self.provider = os.getenv("RETRIEVAL_CACHE_PROVIDER", "local").lower()
        self.ttl_sec = int(os.getenv("RETRIEVAL_CACHE_TTL_SEC", "300"))
        self.max_size = int(os.getenv("RETRIEVAL_CACHE_MAX_SIZE", "1024"))
        self.local_store: dict[str, dict] = {}
        self.metrics = {"hit": 0, "miss": 0, "set": 0}
        self.redis = None
        if self.provider == "redis":
            try:
                import redis  # type: ignore

                self.redis = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
                self.redis.ping()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Redis retrieval cache unavailable, fallback to local: {exc}")
                self.provider = "local"
                self.redis = None

    def _local_get(self, key: str):
        item = self.local_store.get(key)
        if not item:
            self.metrics["miss"] += 1
            return None
        if monotonic() - item["ts"] > self.ttl_sec:
            self.local_store.pop(key, None)
            self.metrics["miss"] += 1
            return None
        self.metrics["hit"] += 1
        return item["value"]

    def _local_set(self, key: str, value):
        self.local_store[key] = {"ts": monotonic(), "value": value}
        if len(self.local_store) > self.max_size:
            oldest_key = min(self.local_store.keys(), key=lambda k: self.local_store[k]["ts"])
            self.local_store.pop(oldest_key, None)
        self.metrics["set"] += 1

    def get(self, key: str):
        if self.provider == "redis" and self.redis:
            raw = self.redis.get(key)
            if raw is None:
                self.metrics["miss"] += 1
                return None
            self.metrics["hit"] += 1
            return json.loads(raw)
        return self._local_get(key)

    def set(self, key: str, value):
        if self.provider == "redis" and self.redis:
            self.redis.set(key, json.dumps(value, ensure_ascii=False), ex=self.ttl_sec)
            self.metrics["set"] += 1
            return
        self._local_set(key, value)

    def get_metrics(self):
        return {"provider": self.provider, **self.metrics, "size": len(self.local_store)}


_GLOBAL_RETRIEVAL_CACHE: RetrievalCache | None = None


def get_retrieval_cache() -> RetrievalCache:
    global _GLOBAL_RETRIEVAL_CACHE
    if _GLOBAL_RETRIEVAL_CACHE is None:
        _GLOBAL_RETRIEVAL_CACHE = RetrievalCache()
    return _GLOBAL_RETRIEVAL_CACHE


