"""Exponential-moving-average rate tracking for progress logging."""

import math
import time


class EMARateMeter:
    """Tracks an EMA of iters/sec, averaged over roughly `window` iterations.

    Unlike a fixed-alpha EMA, the decay per `update()` call accounts for how
    many iterations elapsed since the previous call, so it gives consistent
    results whether you log every iteration or every N iterations.
    """

    def __init__(self, window: float = 200.0):
        self.window = window
        self._ema: float | None = None
        self._last_time: float | None = None
        self._last_iter: int | None = None

    def update(self, iter: int, now: float | None = None) -> float:
        """Record that `iter` was reached at time `now` (default: time.time()), return the current EMA rate."""
        if now is None:
            now = time.time()

        if self._last_time is None:
            self._last_time = now
            self._last_iter = iter
            return 0.0

        delta_iter = iter - self._last_iter
        dt = now - self._last_time
        if delta_iter > 0 and dt > 0:
            inst_rate = delta_iter / dt
            alpha = 1 - math.exp(-delta_iter / self.window)
            self._ema = (
                inst_rate
                if self._ema is None
                else alpha * inst_rate + (1 - alpha) * self._ema
            )
            self._last_time = now
            self._last_iter = iter

        return self._ema or 0.0
