"""Система мониторинга и метрик."""

import logging
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Монитор производительности."""

    def __init__(self):
        self._metrics: Dict[str, list] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)

    def record_time(self, operation: str, duration: float):
        self._metrics[operation].append(duration)
        logger.debug("Performance: %s took %.2fms", operation, duration * 1000)

    def increment_counter(self, counter_name: str, value: int = 1):
        self._counters[counter_name] += value

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        for operation, times in self._metrics.items():
            if not times:
                continue
            stats[operation] = {
                "count": len(times),
                "avg_ms": round(sum(times) / len(times) * 1000, 2),
                "min_ms": round(min(times) * 1000, 2),
                "max_ms": round(max(times) * 1000, 2),
                "total_ms": round(sum(times) * 1000, 2),
            }

        stats["counters"] = dict(self._counters)
        return stats

    def reset(self):
        self._metrics.clear()
        self._counters.clear()


performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str):
    """Декоратор для мониторинга производительности функции."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_time(operation_name, duration)
                performance_monitor.increment_counter(f"{operation_name}_success")
                return result
            except Exception as exc:
                duration = time.time() - start_time
                performance_monitor.record_time(f"{operation_name}_error", duration)
                performance_monitor.increment_counter(f"{operation_name}_error")
                logger.error("Error in %s: %s", operation_name, str(exc))
                raise

        return wrapper

    return decorator


class RequestLogger:
    """Логирование HTTP запросов."""

    def __init__(self):
        self._request_log = []
        self._max_log_size = 1000

    def log_request(
        self,
        method: str,
        path: str,
        duration: float,
        status_code: int,
        session_id: str = None,
        error: str = None,
    ):
        log_entry = {
            "timestamp": time.time(),
            "method": method,
            "path": path,
            "duration_ms": round(duration * 1000, 2),
            "status_code": status_code,
            "session_id": session_id,
            "error": error,
        }

        self._request_log.append(log_entry)
        if len(self._request_log) > self._max_log_size:
            self._request_log = self._request_log[-self._max_log_size :]

        log_level = logging.INFO if status_code < 400 else logging.ERROR
        logger.log(
            log_level,
            "%s %s %s %.2fms %s",
            method,
            path,
            status_code,
            duration * 1000,
            f"session={session_id}" if session_id else "",
        )

    def get_recent_requests(self, limit: int = 100) -> list:
        return self._request_log[-limit:]

    def get_error_requests(self, limit: int = 50) -> list:
        errors = [record for record in self._request_log if record["status_code"] >= 400]
        return errors[-limit:]


request_logger = RequestLogger()
