from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any


@dataclass
class RouteStats:
    count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    p95_window: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def add(self, latency_ms: float, success: bool) -> None:
        self.count += 1
        if not success:
            self.error_count += 1
        self.total_latency_ms += latency_ms
        self.p95_window.append(latency_ms)

    def to_dict(self) -> dict[str, Any]:
        samples = sorted(self.p95_window)
        p95 = 0.0
        if samples:
            idx = max(0, min(len(samples) - 1, int(len(samples) * 0.95) - 1))
            p95 = round(samples[idx], 2)
        avg = round(self.total_latency_ms / self.count, 2) if self.count else 0.0
        error_rate = round(self.error_count / self.count, 4) if self.count else 0.0
        return {
            "count": self.count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_latency_ms": avg,
            "p95_latency_ms": p95,
        }


class ObservabilityRegistry:
    def __init__(self):
        self._lock = Lock()
        self._routes: dict[str, RouteStats] = {}
        self._failed_samples: deque[dict[str, Any]] = deque(maxlen=200)

    def record_route(self, route_key: str, latency_ms: float, success: bool) -> None:
        with self._lock:
            stats = self._routes.setdefault(route_key, RouteStats())
            stats.add(latency_ms=latency_ms, success=success)

    def record_failed_sample(self, sample: dict[str, Any]) -> None:
        payload = dict(sample)
        payload.setdefault("captured_at_monotonic", monotonic())
        with self._lock:
            self._failed_samples.appendleft(payload)

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "routes": {key: stats.to_dict() for key, stats in self._routes.items()},
                "failed_sample_count": len(self._failed_samples),
            }

    def get_failed_samples(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._failed_samples)[: max(1, min(limit, 200))]


_GLOBAL_REGISTRY: ObservabilityRegistry | None = None


def get_observability_registry() -> ObservabilityRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ObservabilityRegistry()
    return _GLOBAL_REGISTRY
