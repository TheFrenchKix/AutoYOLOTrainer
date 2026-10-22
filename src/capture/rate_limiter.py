"""Sliding-window rate limiter (max N events per hour + min interval)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    WINDOW_SECONDS = 3600.0

    def __init__(
        self,
        max_per_hour: int,
        min_interval: float = 0.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_per_hour = max_per_hour
        self.min_interval = min_interval
        self.clock = clock
        self._events: deque[float] = deque()

    def allow(self) -> bool:
        """``True`` if an event may happen now (does not record it)."""
        now = self.clock()
        while self._events and now - self._events[0] > self.WINDOW_SECONDS:
            self._events.popleft()
        if self._events and now - self._events[-1] < self.min_interval:
            return False
        return len(self._events) < self.max_per_hour

    def record(self) -> None:
        self._events.append(self.clock())
