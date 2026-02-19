"""Менеджер кэша для ускорения повторяющихся вычислений."""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Кэш-менеджер с TTL и автоматической инвалидацией."""

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        logger.debug("Cache HIT: %s", key)
        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        cache_ttl = ttl or self._default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + cache_ttl,
            "created_at": time.time(),
        }
        logger.debug("Cache SET: %s (TTL: %ss)", key, cache_ttl)

    def invalidate(self, key: str):
        if key in self._cache:
            del self._cache[key]
            logger.debug("Cache INVALIDATE: %s", key)

    def invalidate_pattern(self, pattern: str):
        keys_to_delete = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
        logger.debug("Cache INVALIDATE pattern '%s': %s keys", pattern, len(keys_to_delete))

    def clear(self):
        count = len(self._cache)
        self._cache.clear()
        logger.info("Cache CLEAR: %s entries removed", count)

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self._cache),
            "total_requests": total,
        }


def cache_key_from_args(*args, **kwargs) -> str:
    data = {
        "args": str(args),
        "kwargs": json.dumps(kwargs, sort_keys=True, default=str),
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """Декоратор для кэширования результатов функции."""

    def decorator(func):
        cache = CacheManager(default_ttl=ttl)

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key_func(*args, **kwargs) if key_func else f"{func.__name__}:{cache_key_from_args(*args, **kwargs)}"

            result = cache.get(cache_key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        wrapper.cache = cache
        wrapper.invalidate_cache = lambda: cache.clear()
        return wrapper

    return decorator


global_cache = CacheManager(default_ttl=600)
