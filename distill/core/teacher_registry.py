"""
Dynamic Teacher Registry — manages per-teacher configs, health, and weights.

Loaded from config/teachers.yaml. Supports:
- Dynamic registration and enable/disable
- Health checks per endpoint
- Weighted contribution tracking
- Historical quality score tracking per teacher
- Automatic failover (skip unhealthy teachers)
"""

import os
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import httpx

logger = logging.getLogger("distill.teacher_registry")


@dataclass
class TeacherConfig:
    """Configuration for a single teacher model"""
    name: str                           # "Qwen 3.5 397B"
    provider: str = "openai"
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = ""
    max_tokens: int = 16000
    temperature: float = 0.3
    weight: float = 0.2                 # Contribution weight in merger (normalized across active)
    strengths: list[str] = field(default_factory=list)
    enabled: bool = True

    def resolve_api_key(self) -> str:
        """Resolve API key — supports ${ENV_VAR} substitution"""
        if self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            return os.environ.get(env_var, "")
        return self.api_key


@dataclass
class TeacherHealth:
    """Health status for a teacher endpoint"""
    is_healthy: bool = True
    last_check_time: float = 0.0
    last_response_time_ms: float = 0.0
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests


class TeacherRegistry:
    """
    Dynamic registry for all teacher models.

    Responsibilities:
    - Load teacher configs from YAML
    - Track health per endpoint
    - Provide list of active/healthy teachers
    - Dynamically adjust weights based on historical quality
    - Support runtime enable/disable
    """

    def __init__(self, config_path: Optional[str] = None):
        self._teachers: dict[str, TeacherConfig] = {}
        self._health: dict[str, TeacherHealth] = {}
        self._historical_scores: dict[str, list[float]] = {}
        self._config_path = config_path

        if config_path and Path(config_path).exists():
            self.load_from_yaml(config_path)

    # ── Loading ──────────────────────────────────────────────

    def load_from_yaml(self, path: str):
        """Load teacher configs from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Expand ${VAR} so endpoints stay out of version control.
        data = yaml.safe_load(os.path.expandvars(raw)) or {}

        teachers_data = data.get("teachers", {})
        for key, cfg in teachers_data.items():
            tc = TeacherConfig(
                name=cfg.get("name", key),
                provider=cfg.get("provider", "openai"),
                base_url=cfg.get("base_url", ""),
                model=cfg.get("model", key),
                api_key=cfg.get("api_key", ""),
                max_tokens=cfg.get("max_tokens", 16000),
                temperature=cfg.get("temperature", 0.3),
                weight=cfg.get("weight", 0.2),
                strengths=cfg.get("strengths", []),
                enabled=cfg.get("enabled", True),
            )
            self._teachers[key] = tc
            self._health[key] = TeacherHealth()

    def register(self, key: str, config: TeacherConfig):
        """Register a teacher at runtime"""
        self._teachers[key] = config
        self._health[key] = TeacherHealth()

    def unregister(self, key: str):
        """Remove a teacher"""
        self._teachers.pop(key, None)
        self._health.pop(key, None)
        self._historical_scores.pop(key, None)

    def set_enabled(self, key: str, enabled: bool):
        """Enable or disable a teacher at runtime"""
        if key in self._teachers:
            self._teachers[key].enabled = enabled

    # ── Queries ──────────────────────────────────────────────

    def get(self, key: str) -> Optional[TeacherConfig]:
        """Get teacher config by key"""
        return self._teachers.get(key)

    def get_all(self) -> dict[str, TeacherConfig]:
        """Get all registered teachers (including disabled)"""
        return dict(self._teachers)

    def get_active(self) -> list[tuple[str, TeacherConfig]]:
        """Get enabled + healthy + reachable teachers, sorted by weight descending.

        A teacher whose base_url is empty (unset ${VAR}) or still a ${...}
        placeholder is skipped: it cannot answer, and attempting it would cost a
        health-check timeout on every sample.
        """
        active = []
        for key, cfg in self._teachers.items():
            if not cfg.enabled:
                continue
            if not self._is_configured(cfg):
                logger.debug("Teacher %s skipped: no base_url configured", key)
                continue
            if self._health.get(key, TeacherHealth()).is_healthy:
                active.append((key, cfg))
        active.sort(key=lambda x: x[1].weight, reverse=True)
        return active

    @staticmethod
    def _is_configured(cfg: TeacherConfig) -> bool:
        """True when the endpoint URL resolved to something usable."""
        url = (cfg.base_url or "").strip()
        if not url or "$" in url:
            return False
        return not url.endswith("/v1") or len(url) > len("/v1")

    def get_active_count(self) -> int:
        """Number of active (enabled + healthy) teachers"""
        return len(self.get_active())

    def get_active_keys(self) -> list[str]:
        """List of active teacher keys"""
        return [k for k, _ in self.get_active()]

    def get_normalized_weights(self) -> dict[str, float]:
        """Get weights normalized to sum to 1.0 across active teachers"""
        active = self.get_active()
        if not active:
            return {}
        total = sum(cfg.weight for _, cfg in active)
        if total == 0:
            # Equal weights fallback
            n = len(active)
            return {k: 1.0 / n for k, _ in active}
        return {k: cfg.weight / total for k, cfg in active}

    # ── Health ───────────────────────────────────────────────

    def get_health(self, key: str) -> Optional[TeacherHealth]:
        """Get health status for a teacher"""
        return self._health.get(key)

    def record_success(self, key: str, response_time_ms: float):
        """Record a successful request"""
        if key in self._health:
            h = self._health[key]
            h.total_requests += 1
            h.last_response_time_ms = response_time_ms
            h.consecutive_failures = 0
            h.is_healthy = True

    def record_failure(self, key: str):
        """Record a failed request"""
        if key in self._health:
            h = self._health[key]
            h.total_requests += 1
            h.total_failures += 1
            h.consecutive_failures += 1
            # Mark unhealthy after 5 consecutive failures
            if h.consecutive_failures >= 5:
                h.is_healthy = False

    async def health_check(self, key: str, timeout: float = 10.0) -> bool:
        """Check if a teacher endpoint is reachable"""
        cfg = self._teachers.get(key)
        if not cfg:
            return False

        try:
            # trust_env=False: ignore HTTP_PROXY/HTTPS_PROXY env vars —
            # a broken proxy on the host would fail every endpoint here
            # even though direct access works (curl would still show 200).
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                start = time.monotonic()
                # Only send Authorization when a key is configured — an empty
                # key would produce "Bearer " which httpx rejects as illegal.
                headers = {}
                api_key = cfg.resolve_api_key()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                resp = await client.get(
                    f"{cfg.base_url.rstrip('/')}/models",
                    headers=headers
                )
                elapsed_ms = (time.monotonic() - start) * 1000

                if resp.status_code < 500:
                    self.record_success(key, elapsed_ms)
                    return True
                else:
                    logger.warning(
                        "Teacher health check failed: HTTP %s from %s (%s)",
                        resp.status_code, cfg.base_url, key,
                    )
                    self.record_failure(key)
                    return False
        except Exception as e:
            logger.warning(
                "Teacher health check error: %s: %s (%s @ %s)",
                type(e).__name__, e, key, cfg.base_url,
            )
            self.record_failure(key)
            return False

    async def health_check_all(self, timeout: float = 10.0) -> dict[str, bool]:
        """Run health check on all enabled teachers concurrently"""
        enabled = [(k, c) for k, c in self._teachers.items() if c.enabled]
        if not enabled:
            return {}

        results = await asyncio.gather(
            *[self.health_check(k, timeout) for k, _ in enabled],
            return_exceptions=True
        )

        return {
            k: (r if isinstance(r, bool) else False)
            for (k, _), r in zip(enabled, results)
        }

    # ── Quality tracking ─────────────────────────────────────

    def record_quality_score(self, key: str, score: float):
        """Record a quality score for weight adjustment"""
        if key not in self._historical_scores:
            self._historical_scores[key] = []
        self._historical_scores[key].append(score)
        # Keep last 1000 scores
        if len(self._historical_scores[key]) > 1000:
            self._historical_scores[key] = self._historical_scores[key][-1000:]

    def get_average_quality(self, key: str) -> Optional[float]:
        """Get historical average quality for a teacher"""
        scores = self._historical_scores.get(key, [])
        if not scores:
            return None
        return sum(scores) / len(scores)

    def recalculate_weights_from_quality(self):
        """Adjust teacher weights based on historical quality scores"""
        active = self.get_active()
        if len(active) < 2:
            return

        avg_scores = {}
        for key, _ in active:
            avg = self.get_average_quality(key)
            avg_scores[key] = avg if avg is not None else 0.7

        total = sum(avg_scores.values())
        if total == 0:
            return

        for key, cfg in active:
            cfg.weight = avg_scores[key] / total

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get comprehensive registry statistics"""
        active = self.get_active()
        return {
            "total_registered": len(self._teachers),
            "total_enabled": sum(1 for c in self._teachers.values() if c.enabled),
            "total_active": len(active),
            "total_healthy": sum(1 for h in self._health.values() if h.is_healthy),
            "active_teachers": [k for k, _ in active],
            "weights": {k: round(c.weight, 4) for k, c in active},
            "normalized_weights": {k: round(w, 4) for k, w in self.get_normalized_weights().items()},
            "health": {
                k: {
                    "is_healthy": h.is_healthy,
                    "failure_rate": round(h.failure_rate, 4),
                    "avg_response_ms": round(h.last_response_time_ms, 1),
                    "consecutive_failures": h.consecutive_failures,
                }
                for k, h in self._health.items()
            },
            "quality": {
                k: {
                    "avg": round(sum(v) / len(v), 4) if v else None,
                    "count": len(v),
                }
                for k, v in self._historical_scores.items()
            },
        }
