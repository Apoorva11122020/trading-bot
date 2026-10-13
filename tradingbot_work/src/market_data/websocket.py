from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from src.market_data.health import MarketDataHealthMonitor
from src.market_data.interface import MarketDataFeed
from src.market_data.models import (
    MarketDataHealth,
    MarketDataStatus,
    Tick,
)


class WebSocketTransport(Protocol):
    """
    Provider-neutral WebSocket transport.

    Upstox V3 sends/receives binary protobuf messages, while other
    providers may use text/JSON messages. Therefore send() and recv()
    support both str and bytes.
    """

    def send(self, message: str | bytes) -> None:
        ...

    def recv(self) -> str | bytes:
        ...

    def close(self) -> None:
        ...


TransportFactory = Callable[[str], WebSocketTransport]

MessageParser = Callable[[str | bytes], list[Tick]]

SubscribeMessageBuilder = Callable[
    [list[str]],
    str | bytes,
]


def json_tick_parser(message: str | bytes) -> list[Tick]:
    """
    Parse a provider-neutral JSON WebSocket envelope.

    Accepted forms:

        {
            "symbol": "RELIANCE",
            "ltp": 100.2,
            "timestamp": "2026-10-03T10:00:00Z"
        }

    Or:

        {
            "ticks": [
                {
                    "symbol": "RELIANCE",
                    "ltp": 100.2,
                    "timestamp": "2026-10-03T10:00:00Z"
                }
            ]
        }

    Provider-specific payloads should be translated by the provider
    adapter rather than leaking provider fields into the trading engine.
    """

    # ------------------------------------------------------------
    # Convert bytes to text
    # ------------------------------------------------------------

    if isinstance(message, bytes):
        raw = message.decode("utf-8")
    else:
        raw = message

    # ------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------

    payload = json.loads(raw)

    # ------------------------------------------------------------
    # Normalize single tick / ticks list
    # ------------------------------------------------------------

    if isinstance(payload, dict):
        rows = payload.get("ticks", [payload])
    else:
        rows = payload

    if not isinstance(rows, list):
        raise ValueError(
            "WebSocket message must contain "
            "an object or ticks list"
        )

    ticks: list[Tick] = []

    # ------------------------------------------------------------
    # Convert every row to Tick
    # ------------------------------------------------------------

    for row in rows:
        tick = _parse_tick_row(row)
        if tick is not None:
            ticks.append(tick)

    return ticks


def _parse_tick_row(row) -> Tick | None:
    """Return a Tick, or None for malformed rows (missing fields, bad
    timestamp, non-finite or non-positive price). One bad row never
    discards the rest of the message."""
    if not isinstance(row, dict):
        return None
    if "symbol" not in row or "ltp" not in row:
        return None
    try:
        ltp = float(row["ltp"])
        volume = float(row.get("volume", 0.0) or 0.0)
        last_quantity = float(row.get("last_quantity", 0.0) or 0.0)
        if not math.isfinite(ltp) or ltp <= 0:
            return None
        if not math.isfinite(volume) or not math.isfinite(last_quantity):
            return None
        timestamp = _parse_tick_timestamp(row.get("timestamp"))
        return Tick(
            symbol=str(row["symbol"]),
            timestamp=timestamp,
            ltp=ltp,
            volume=volume,
            last_quantity=last_quantity,
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _parse_tick_timestamp(ts) -> datetime:
    # A missing exchange timestamp falls back to receive time.
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        value = float(ts)
        if value > 10_000_000_000:
            value /= 1000.0
        return datetime.fromtimestamp(value, tz=timezone.utc)
    timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp



class WebSocketMarketDataFeed(MarketDataFeed):
    """
    Provider-neutral threaded WebSocket market-data feed.

    Responsibilities:

    - WebSocket connection lifecycle
    - Subscription management
    - Tick parsing
    - Market-data health monitoring
    - Stale-feed detection through health monitor
    - Automatic reconnect
    - Exponential reconnect backoff
    - Tick handler callbacks

    Trading/order logic does NOT belong here.

    Provider-specific adapters supply:

    - WebSocket URL
    - Transport factory
    - Subscription message builder
    - Message parser
    """

    def __init__(
        self,
        url: str,
        *,
        transport_factory: TransportFactory,
        subscribe_message: SubscribeMessageBuilder,
        parser: MessageParser = json_tick_parser,
        stale_after_seconds: int = 15,
        reconnect_enabled: bool = True,
        initial_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        max_attempts: int = 10,
    ) -> None:

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        if not url:
            raise ValueError(
                "url is required"
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1"
            )

        if stale_after_seconds <= 0:
            raise ValueError(
                "stale_after_seconds must be > 0"
            )

        # --------------------------------------------------------
        # Configuration
        # --------------------------------------------------------

        self.url = url

        self._transport_factory = (
            transport_factory
        )

        self._subscribe_message = (
            subscribe_message
        )

        self._parser = parser

        # --------------------------------------------------------
        # Health monitor
        # --------------------------------------------------------

        self._monitor = MarketDataHealthMonitor(
            stale_after_seconds
        )

        # --------------------------------------------------------
        # Reconnect configuration
        # --------------------------------------------------------

        self._reconnect_enabled = (
            reconnect_enabled
        )

        self._initial_delay = max(
            0.0,
            initial_delay_seconds,
        )

        self._max_delay = max(
            self._initial_delay,
            max_delay_seconds,
        )

        self._max_attempts = max_attempts

        # --------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------

        self._handler: (
            Callable[[Tick], None] | None
        ) = None

        self._symbols: list[str] = []

        self._transport: (
            WebSocketTransport | None
        ) = None

        self._thread: (
            threading.Thread | None
        ) = None

        self._stop = threading.Event()

        self._lock = threading.RLock()

        self.malformed_messages = 0

        self.duplicate_ticks = 0

        self._last_tick_key: dict[str, tuple] = {}

        self.messages_received = 0
        self.ticks_parsed = 0
        self.ticks_delivered = 0
        self.ticks_filtered_symbol = 0
        self.subscription_messages_sent = 0
        self.connected_count = 0

    # ============================================================
    # CONNECT
    # ============================================================

    def connect(self) -> None:
        """
        Start the background WebSocket worker.

        The actual network connection is created by _run().
        """

        with self._lock:

            if (
                self._thread
                and self._thread.is_alive()
            ):
                return

            self._stop.clear()

            self._monitor.reconnecting()

            self._thread = threading.Thread(
                target=self._run,
                name="market-data-ws",
                daemon=True,
            )

            self._thread.start()

    # ============================================================
    # DISCONNECT
    # ============================================================

    def disconnect(self) -> None:
        """
        Stop the worker and close the current transport.
        """

        self._stop.set()

        with self._lock:
            transport = self._transport

        if transport is not None:

            try:
                transport.close()

            except Exception as exc:

                self._monitor.disconnected(
                    str(exc)
                )

        thread = self._thread

        if (
            thread
            and thread.is_alive()
            and thread
            is not threading.current_thread()
        ):

            thread.join(
                timeout=2
            )

        self._monitor.disconnected()

    # ============================================================
    # SUBSCRIBE
    # ============================================================

    def subscribe(
        self,
        symbols: list[str],
    ) -> None:
        """
        Subscribe to market-data symbols.

        If the WebSocket is already connected, the subscription is
        sent immediately.

        Otherwise symbols are stored and automatically subscribed
        after the next connection/reconnection.
        """
        self._set_symbols(list(dict.fromkeys(symbols)))
        self._send_subscription_if_connected()

    def subscribe_merge(
        self,
        symbols: list[str],
    ) -> None:
        """Add symbols to the subscription set without dropping existing ones."""
        self._set_symbols(list(dict.fromkeys([*self._symbols, *symbols])))
        self._send_subscription_if_connected()

    def _set_symbols(self, symbols: list[str]) -> None:
        self._symbols = symbols
        self._monitor.health.subscribed_symbols = list(self._symbols)

    def _send_subscription_if_connected(self) -> None:
        with self._lock:
            transport = self._transport
        if (
            transport is not None
            and self._symbols
            and self._monitor.health.status == MarketDataStatus.CONNECTED
        ):
            transport.send(self._subscribe_message(self._symbols))
            self.subscription_messages_sent += 1

    # ============================================================
    # UNSUBSCRIBE
    # ============================================================

    def unsubscribe(
        self,
        symbols: list[str],
    ) -> None:
        """
        Remove symbols from the local subscription set.

        Provider-specific unsubscribe behavior can be implemented
        by a provider adapter if required.
        """

        remove = set(symbols)

        self._symbols = [
            symbol
            for symbol in self._symbols
            if symbol not in remove
        ]

        self._monitor.health.subscribed_symbols = (
            list(self._symbols)
        )

    # ============================================================
    # TICK HANDLER
    # ============================================================

    def set_tick_handler(
        self,
        handler: Callable[[Tick], None],
    ) -> None:
        """
        Register a callback invoked for every accepted Tick.
        """

        self._handler = handler

    # ============================================================
    # HEALTH
    # ============================================================

    def health(self) -> MarketDataHealth:
        """
        Return a copy of the current market-data health state.
        """

        return self._monitor.health.model_copy(
            deep=True
        )

    # ============================================================
    # CAN TRADE
    # ============================================================

    def can_trade(
        self,
        now: datetime | None = None,
    ) -> bool:
        """
        Return whether market-data health allows trading.
        """

        return self._monitor.can_trade(
            now
        )

    # ============================================================
    # INTERNAL WORKER
    # ============================================================

    def _run(self) -> None:
        """
        Background WebSocket worker.

        Handles:

        1. Connection
        2. Subscription
        3. Receiving messages
        4. Parsing ticks
        5. Health updates
        6. Tick callbacks
        7. Disconnect detection
        8. Automatic reconnect
        """

        attempts = 0

        delay = self._initial_delay

        while not self._stop.is_set():

            transport: (
                WebSocketTransport | None
            ) = None

            try:

                # ------------------------------------------------
                # Connection attempt
                # ------------------------------------------------

                self._monitor.reconnecting()

                transport = (
                    self._transport_factory(
                        self.url
                    )
                )

                with self._lock:
                    self._transport = transport

                # Successful connection resets backoff.
                attempts = 0

                delay = self._initial_delay

                self._monitor.connected(
                    self._symbols
                )
                self.connected_count += 1

                # ------------------------------------------------
                # Initial subscription
                # ------------------------------------------------

                if self._symbols:

                    message = (
                        self._subscribe_message(
                            self._symbols
                        )
                    )

                    transport.send(message)
                    self.subscription_messages_sent += 1

                # ------------------------------------------------
                # Receive loop
                # ------------------------------------------------

                while not self._stop.is_set():

                    message = transport.recv()
                    self.messages_received += 1

                    # A malformed payload is dropped; it is not a
                    # connection failure and must not force a reconnect.
                    try:
                        ticks = self._parser(
                            message
                        )
                    except Exception:
                        self.malformed_messages += 1
                        continue

                    self.ticks_parsed += len(ticks)

                    for tick in ticks:

                        key = (tick.symbol, tick.timestamp, tick.ltp, tick.volume)
                        if self._last_tick_key.get(tick.symbol) == key:
                            self.duplicate_ticks += 1
                            continue
                        self._last_tick_key[tick.symbol] = key

                        # ----------------------------------------
                        # Ignore unexpected symbols
                        # ----------------------------------------

                        if (
                            tick.symbol
                            not in self._symbols
                        ):
                            self.ticks_filtered_symbol += 1
                            continue

                        # ----------------------------------------
                        # Record tick health
                        # ----------------------------------------

                        self._monitor.record_tick(
                            tick
                        )

                        # ----------------------------------------
                        # Send tick to consumer
                        # ----------------------------------------

                        if self._handler:
                            self.ticks_delivered += 1
                            self._handler(
                                tick
                            )

            # ====================================================
            # CONNECTION / PARSER ERROR
            # ====================================================

            except Exception as exc:

                self._monitor.disconnected(
                    str(exc)
                )

                attempts += 1

                # ----------------------------------------------
                # Stop reconnecting
                # ----------------------------------------------

                if (
                    not self._reconnect_enabled
                    or attempts >= self._max_attempts
                ):
                    break

                self._monitor.health.reconnect_attempts = (
                    attempts
                )

                # ----------------------------------------------
                # Backoff
                # ----------------------------------------------

                self._stop.wait(
                    delay
                )

                delay = min(
                    self._max_delay,
                    max(
                        self._initial_delay,
                        delay * 2
                        if delay
                        else 1.0,
                    ),
                )

            # ====================================================
            # CLEANUP
            # ====================================================

            finally:

                with self._lock:

                    current_transport = (
                        self._transport
                    )

                    self._transport = None

                # Prefer the current shared transport, but fall
                # back to the local transport created in this run.
                transport_to_close = (
                    current_transport
                    or transport
                )

                if (
                    transport_to_close
                    is not None
                ):

                    try:

                        transport_to_close.close()

                    except Exception:
                        pass

        # ========================================================
        # FINAL DISCONNECT
        # ========================================================

        if self._stop.is_set():

            self._monitor.disconnected()
