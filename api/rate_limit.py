from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket limiter."""

    capacity: float
    refill_per_sec: float
    tokens: float = 0.0
    updated_at: float = field(default_factory=lambda: time.time())

    def allow(self, cost: float = 1.0) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.time()
        dt = max(0.0, now - float(self.updated_at))
        self.updated_at = now

        if self.refill_per_sec > 0:
            self.tokens = min(self.capacity, float(self.tokens) + dt * float(self.refill_per_sec))
        else:
            self.tokens = min(self.capacity, float(self.tokens))

        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0

        missing = max(0.0, cost - float(self.tokens))
        retry_after = (missing / float(self.refill_per_sec)) if self.refill_per_sec > 0 else 3600.0
        return False, float(retry_after)


class RateLimiter:
    """In-memory rate limiter keyed by arbitrary strings."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}

    def allow(
        self,
        key: str,
        *,
        capacity: float,
        refill_per_sec: float,
        cost: float = 1.0,
    ) -> tuple[bool, float]:
        b = self._buckets.get(key)
        if b is None or b.capacity != capacity or b.refill_per_sec != refill_per_sec:
            b = TokenBucket(capacity=capacity, refill_per_sec=refill_per_sec, tokens=capacity)
            self._buckets[key] = b
        return b.allow(cost=cost)
