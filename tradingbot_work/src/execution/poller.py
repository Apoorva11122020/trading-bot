from __future__ import annotations

import threading
import time
from typing import Any, Callable

from src.execution.order_manager import OrderManager


class OrderPoller:
    """Periodically pulls broker order state for non-terminal orders.

    Every result goes through `OrderManager.refresh_open_orders`, i.e. the
    same state machine as streamed updates. The poller only reads; it never
    submits, resubmits or cancels orders. Broker failures back off
    exponentially up to `max_backoff_seconds`; orders that stay open longer
    than `status_timeout_seconds` raise a single ORDER_STATUS_TIMEOUT event.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        interval_seconds: float = 2.0,
        max_backoff_seconds: float = 30.0,
        status_timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        on_cycle: Callable[[dict[str, Any]], None] | None = None,
    ):
        if interval_seconds <= 0 or max_backoff_seconds <= 0 or status_timeout_seconds <= 0:
            raise ValueError("poller intervals must be > 0")
        self.order_manager = order_manager
        self.interval_seconds = interval_seconds
        self.max_backoff_seconds = max(max_backoff_seconds, interval_seconds)
        self.status_timeout_seconds = status_timeout_seconds
        self.clock = clock
        self.on_cycle = on_cycle

        self.consecutive_failures = 0
        self.cycles = 0
        self.last_error: str | None = None
        self._first_seen: dict[str, float] = {}
        self._timed_out: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def next_delay(self) -> float:
        if self.consecutive_failures == 0:
            return self.interval_seconds
        return min(
            self.max_backoff_seconds,
            self.interval_seconds * (2 ** self.consecutive_failures),
        )

    def poll_once(self) -> dict[str, Any]:
        self.cycles += 1
        open_ids = self.order_manager.open_order_ids()
        for order_id in list(self._first_seen):
            if order_id not in open_ids:
                self._first_seen.pop(order_id, None)
                self._timed_out.discard(order_id)

        if not open_ids and not self.order_manager.in_doubt:
            self.consecutive_failures = 0
            result: dict[str, Any] = {"status": "idle", "refreshed": [], "missing": []}
            self._emit(result)
            return result

        try:
            refreshed = self.order_manager.refresh_open_orders()
        except Exception as exc:
            self.consecutive_failures += 1
            self.last_error = type(exc).__name__
            self.order_manager._event(
                "ORDER_POLL_FAILED", "WARNING",
                "Order status poll failed; backing off without resubmitting",
                error=str(exc)[:200],
                consecutive_failures=self.consecutive_failures,
                next_delay_seconds=self.next_delay(),
            )
            result = {"status": "error", "error": type(exc).__name__, "refreshed": [], "missing": []}
            self._emit(result)
            return result

        self.consecutive_failures = 0
        self.last_error = None

        now = self.clock()
        timed_out = []
        for order_id in self.order_manager.open_order_ids():
            first = self._first_seen.setdefault(order_id, now)
            if now - first >= self.status_timeout_seconds and order_id not in self._timed_out:
                self._timed_out.add(order_id)
                timed_out.append(order_id)
        if timed_out:
            self.order_manager._event(
                "ORDER_STATUS_TIMEOUT", "WARNING",
                "Orders remained open beyond the configured status timeout",
                order_ids=timed_out, timeout_seconds=self.status_timeout_seconds,
            )

        result = {"status": "ok", "timed_out": timed_out, **refreshed}
        self._emit(result)
        return result

    def _emit(self, result: dict[str, Any]) -> None:
        if self.on_cycle is not None:
            try:
                self.on_cycle(result)
            except Exception:
                pass

    # --------------------------------------------------
    # Background thread
    # --------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="order-poller", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                self.consecutive_failures += 1
            self._stop.wait(self.next_delay())

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
