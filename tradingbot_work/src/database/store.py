from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    signal_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    average_fill_price REAL,
    requested_price REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_signal ON orders(signal_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    signal_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL,
    exit_price REAL,
    realized_pnl REAL,
    opened_at TEXT,
    closed_at TEXT,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    average_entry REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp);

CREATE TABLE IF NOT EXISTS order_status_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    average_fill_price REAL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_order ON order_status_history(order_id);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    signal_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    cumulative_filled INTEGER NOT NULL,
    reference_price REAL,
    slippage_abs REAL,
    slippage_pct REAL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_executions_order ON executions(order_id);

CREATE TABLE IF NOT EXISTS managed_positions (
    symbol TEXT PRIMARY KEY,
    closed INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_signal_outcomes (
    signal_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    underlying TEXT NOT NULL,
    direction TEXT NOT NULL,
    option_type TEXT,
    strike REAL,
    expiry TEXT,
    instrument_key TEXT,
    trading_symbol TEXT,
    signal_price REAL,
    entry_price REAL,
    stop_loss REAL,
    target REAL,
    quantity INTEGER,
    active_from TEXT,
    source TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    exit_price REAL,
    exit_time TEXT,
    pnl REAL,
    r_multiple REAL,
    resolved_at TEXT,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_signal_outcomes_day ON live_signal_outcomes(session_date, timestamp);

CREATE TRIGGER IF NOT EXISTS live_signal_record_immutable
BEFORE UPDATE OF signal_id, timestamp, session_date, strategy, underlying, direction, option_type,
    strike, expiry, instrument_key, trading_symbol, signal_price, entry_price, stop_loss, target,
    quantity, active_from, source, record_json, created_at
ON live_signal_outcomes
BEGIN
    SELECT RAISE(ABORT, 'live signal record is immutable');
END;

CREATE TABLE IF NOT EXISTS forward_validation_signals (
    signal_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    underlying TEXT NOT NULL,
    direction TEXT NOT NULL,
    instrument_key TEXT,
    expiry TEXT,
    dte INTEGER,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    pnl REAL,
    r_multiple REAL,
    resolved_at TEXT,
    UNIQUE (strategy, underlying, timestamp, direction)
);

CREATE INDEX IF NOT EXISTS idx_forward_validation_day ON forward_validation_signals(session_date, timestamp);

CREATE TRIGGER IF NOT EXISTS forward_validation_record_immutable
BEFORE UPDATE OF signal_id, timestamp, session_date, strategy, underlying, direction, instrument_key,
    expiry, dte, record_json, created_at
ON forward_validation_signals
BEGIN
    SELECT RAISE(ABORT, 'forward validation record is immutable');
END;

CREATE TRIGGER IF NOT EXISTS forward_validation_no_delete
BEFORE DELETE ON forward_validation_signals
BEGIN
    SELECT RAISE(ABORT, 'forward validation records cannot be deleted');
END;

CREATE TABLE IF NOT EXISTS shadow_strategy_signals (
    shadow_id TEXT PRIMARY KEY,
    candidate TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    session_date TEXT NOT NULL,
    underlying TEXT NOT NULL,
    direction TEXT NOT NULL,
    instrument_key TEXT,
    expiry TEXT,
    dte INTEGER,
    decision TEXT NOT NULL,
    skip_reason TEXT,
    status TEXT NOT NULL,
    entry_price REAL,
    stop_loss REAL,
    target REAL,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    pnl REAL,
    r_multiple REAL,
    market_regime TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(candidate, underlying, timestamp, direction)
);

CREATE INDEX IF NOT EXISTS idx_shadow_strategy_day ON shadow_strategy_signals(session_date, candidate, timestamp);

CREATE TRIGGER IF NOT EXISTS shadow_strategy_record_immutable
BEFORE UPDATE OF candidate, timestamp, session_date, underlying, direction, instrument_key, expiry, dte, decision,
    skip_reason, entry_price, stop_loss, target, market_regime, record_json, created_at
ON shadow_strategy_signals
BEGIN
    SELECT RAISE(ABORT, 'shadow strategy record is immutable');
END;

CREATE TRIGGER IF NOT EXISTS shadow_strategy_no_delete
BEFORE DELETE ON shadow_strategy_signals
BEGIN
    SELECT RAISE(ABORT, 'shadow strategy records cannot be deleted');
END;
"""

# Additive migrations: existing database files gain the columns on open.
MIGRATIONS: dict[str, dict[str, str]] = {
    "orders": {
        "signal_type": "TEXT",
        "tag": "TEXT",
        "rejection_reason": "TEXT",
        "slippage_abs": "REAL",
        "slippage_pct": "REAL",
        "realized_pnl": "REAL",
        "market_type": "TEXT",
        "underlying": "TEXT",
        "option_type": "TEXT",
        "strike": "REAL",
        "expiry": "TEXT",
        "lot_size": "INTEGER",
        "instrument_key": "TEXT",
    },
    "trades": {
        "exit_reason": "TEXT",
        "exit_quantity": "INTEGER NOT NULL DEFAULT 0",
        "last_exit_at": "TEXT",
        "entry_slippage_pct": "REAL",
        "exit_slippage_pct": "REAL",
    },
    "shadow_strategy_signals": {
        "gate_score": "INTEGER",
        "gate_max_score": "INTEGER",
        "gate_confidence": "REAL",
        "gate_version": "TEXT",
        "gate_reasons": "TEXT",
        "gate_failed_checks": "TEXT",
        "gate_features": "TEXT",
    },
}

SHADOW_GATE_COLUMNS = tuple(MIGRATIONS["shadow_strategy_signals"])
SHADOW_GATE_TRIGGER = f"""
CREATE TRIGGER IF NOT EXISTS shadow_strategy_gate_immutable
BEFORE UPDATE OF {', '.join(SHADOW_GATE_COLUMNS)}
ON shadow_strategy_signals
BEGIN
    SELECT RAISE(ABORT, 'shadow quality gate decision is immutable');
END;
"""

TERMINAL_ORDER_STATUSES = ("FILLED", "CANCELLED", "REJECTED", "FAILED")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


class Database:
    """Small repository-style SQLite persistence layer.

    The rest of the trading system talks to this class rather than directly
    to SQL, allowing a PostgreSQL repository to replace it later.

    All timestamps are stored as UTC ISO-8601 strings; trading-session
    decisions convert to Asia/Kolkata at the point of use.
    """

    def __init__(self, path: str | Path = "data/trading.db"):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._tx_depth = 0
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    @property
    def connection(self):
        return self._conn

    def close(self):
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        for table, columns in MIGRATIONS.items():
            existing = {
                row["name"]
                for row in self._conn.execute(f"PRAGMA table_info({table})")
            }
            for name, ddl in columns.items():
                if name not in existing:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"
                    )
        self._conn.executescript(SHADOW_GATE_TRIGGER)

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        """Group several writes atomically. Nested calls join the outer
        transaction; only the outermost commits or rolls back."""
        with self._lock:
            self._tx_depth += 1
            try:
                yield self
            except BaseException:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.commit()

    def ping(self) -> bool:
        with self._lock:
            self._conn.execute("SELECT 1").fetchone()
        return True

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def save_order(self, record: dict[str, Any]) -> None:
        now = iso(utc_now())
        order_id = str(record["order_id"])
        status = str(record["status"])
        filled = int(record.get("filled_quantity", 0))
        average = record.get("average_fill_price")

        with self._lock:
            previous = self._conn.execute(
                "SELECT status, filled_quantity FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()

            self._conn.execute(
                """
                INSERT INTO orders (
                    order_id, signal_id, symbol, side, order_type, quantity,
                    status, filled_quantity, average_fill_price, requested_price,
                    created_at, updated_at, reason, signal_type, tag,
                    rejection_reason, slippage_abs, slippage_pct, realized_pnl,
                    market_type, underlying, option_type, strike, expiry,
                    lot_size, instrument_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    market_type=CASE WHEN excluded.instrument_key IS NOT NULL
                        THEN excluded.market_type
                        ELSE COALESCE(orders.market_type, excluded.market_type) END,
                    underlying=COALESCE(excluded.underlying, orders.underlying),
                    option_type=COALESCE(excluded.option_type, orders.option_type),
                    strike=COALESCE(excluded.strike, orders.strike),
                    expiry=COALESCE(excluded.expiry, orders.expiry),
                    lot_size=COALESCE(excluded.lot_size, orders.lot_size),
                    instrument_key=COALESCE(excluded.instrument_key, orders.instrument_key),
                    signal_id=excluded.signal_id,
                    status=excluded.status,
                    filled_quantity=excluded.filled_quantity,
                    average_fill_price=excluded.average_fill_price,
                    requested_price=excluded.requested_price,
                    updated_at=excluded.updated_at,
                    reason=excluded.reason,
                    signal_type=COALESCE(excluded.signal_type, orders.signal_type),
                    tag=COALESCE(excluded.tag, orders.tag),
                    rejection_reason=COALESCE(excluded.rejection_reason, orders.rejection_reason),
                    slippage_abs=COALESCE(excluded.slippage_abs, orders.slippage_abs),
                    slippage_pct=COALESCE(excluded.slippage_pct, orders.slippage_pct),
                    realized_pnl=COALESCE(excluded.realized_pnl, orders.realized_pnl)
                """,
                (
                    order_id,
                    record.get("signal_id"),
                    str(record["symbol"]),
                    str(record["side"]),
                    str(record["order_type"]),
                    int(record["quantity"]),
                    status,
                    filled,
                    average,
                    record.get("requested_price"),
                    record.get("created_at") or now,
                    now,
                    record.get("reason"),
                    record.get("signal_type"),
                    record.get("tag"),
                    record.get("rejection_reason"),
                    record.get("slippage_abs"),
                    record.get("slippage_pct"),
                    record.get("realized_pnl"),
                    record.get("market_type") or "EQUITY",
                    record.get("underlying"),
                    record.get("option_type"),
                    record.get("strike"),
                    record.get("expiry"),
                    record.get("lot_size"),
                    record.get("instrument_key"),
                ),
            )

            if (
                previous is None
                or previous["status"] != status
                or int(previous["filled_quantity"]) != filled
            ):
                self._conn.execute(
                    """
                    INSERT INTO order_status_history (
                        order_id, status, filled_quantity,
                        average_fill_price, recorded_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (order_id, status, filled, average, now),
                )
            self._commit()

    def get_order(self, order_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()

    def get_orders(self, limit: int = 100):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM orders ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def get_all_orders(self):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM orders ORDER BY created_at ASC"
            ).fetchall()

    def get_open_orders(self):
        placeholders = ",".join("?" for _ in TERMINAL_ORDER_STATUSES)
        with self._lock:
            return self._conn.execute(
                f"SELECT * FROM orders WHERE status NOT IN ({placeholders}) "
                "ORDER BY created_at ASC",
                TERMINAL_ORDER_STATUSES,
            ).fetchall()

    def get_order_history(self, order_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM order_status_history WHERE order_id = ? "
                "ORDER BY history_id ASC",
                (order_id,),
            ).fetchall()

    # ------------------------------------------------------------------
    # Executions (fills)
    # ------------------------------------------------------------------

    def save_execution(self, record: dict[str, Any]) -> bool:
        """Insert one fill. Returns False if this fill was already stored,
        which makes repeated broker updates idempotent."""
        execution_id = f"{record['order_id']}:{int(record['cumulative_filled'])}"
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO executions (
                    execution_id, order_id, signal_id, symbol, side, quantity,
                    price, cumulative_filled, reference_price, slippage_abs,
                    slippage_pct, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    str(record["order_id"]),
                    record.get("signal_id"),
                    str(record["symbol"]),
                    str(record["side"]),
                    int(record["quantity"]),
                    float(record["price"]),
                    int(record["cumulative_filled"]),
                    record.get("reference_price"),
                    record.get("slippage_abs"),
                    record.get("slippage_pct"),
                    record.get("recorded_at") or iso(utc_now()),
                ),
            )
            self._commit()
            return cursor.rowcount == 1

    def get_executions(self, order_id: str | None = None):
        with self._lock:
            if order_id is None:
                return self._conn.execute(
                    "SELECT * FROM executions ORDER BY recorded_at ASC, cumulative_filled ASC"
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM executions WHERE order_id = ? "
                "ORDER BY cumulative_filled ASC",
                (order_id,),
            ).fetchall()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trades (
                    trade_id, order_id, signal_id, symbol, side, quantity,
                    entry_price, exit_price, realized_pnl, opened_at, closed_at,
                    status, exit_reason, exit_quantity, last_exit_at,
                    entry_slippage_pct, exit_slippage_pct
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    quantity=excluded.quantity,
                    entry_price=COALESCE(excluded.entry_price, trades.entry_price),
                    exit_price=excluded.exit_price,
                    realized_pnl=excluded.realized_pnl,
                    closed_at=excluded.closed_at,
                    status=excluded.status,
                    exit_reason=COALESCE(excluded.exit_reason, trades.exit_reason),
                    exit_quantity=excluded.exit_quantity,
                    last_exit_at=COALESCE(excluded.last_exit_at, trades.last_exit_at),
                    entry_slippage_pct=COALESCE(excluded.entry_slippage_pct, trades.entry_slippage_pct),
                    exit_slippage_pct=COALESCE(excluded.exit_slippage_pct, trades.exit_slippage_pct)
                """,
                (
                    str(record["trade_id"]),
                    str(record["order_id"]),
                    record.get("signal_id"),
                    str(record["symbol"]),
                    str(record["side"]),
                    int(record["quantity"]),
                    record.get("entry_price"),
                    record.get("exit_price"),
                    record.get("realized_pnl"),
                    record.get("opened_at"),
                    record.get("closed_at"),
                    str(record["status"]),
                    record.get("exit_reason"),
                    int(record.get("exit_quantity") or 0),
                    record.get("last_exit_at"),
                    record.get("entry_slippage_pct"),
                    record.get("exit_slippage_pct"),
                ),
            )
            self._commit()

    def get_trade(self, trade_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()

    def get_open_trade(self, symbol: str, side: str | None = None):
        query = "SELECT * FROM trades WHERE symbol = ? AND status = 'OPEN'"
        params: list[Any] = [symbol]
        if side is not None:
            query += " AND side = ?"
            params.append(side)
        query += " ORDER BY opened_at ASC LIMIT 1"
        with self._lock:
            return self._conn.execute(query, tuple(params)).fetchone()

    def get_trades(self, limit: int = 100):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM trades ORDER BY COALESCE(closed_at, opened_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def realized_pnl_since(self, since: datetime) -> float:
        """Realized P&L booked (fully or partially) at or after `since`."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM trades "
                "WHERE realized_pnl IS NOT NULL "
                "AND COALESCE(last_exit_at, closed_at) >= ?",
                (iso(since),),
            ).fetchone()
        return float(row["pnl"] or 0.0)

    def trades_opened_since(self, since: datetime) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM trades WHERE opened_at >= ?",
                (iso(since),),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def upsert_position(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO positions (
                    symbol, side, quantity, average_entry, current_price,
                    unrealized_pnl, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    side=excluded.side,
                    quantity=excluded.quantity,
                    average_entry=excluded.average_entry,
                    current_price=excluded.current_price,
                    unrealized_pnl=excluded.unrealized_pnl,
                    updated_at=excluded.updated_at
                """,
                (
                    str(record["symbol"]),
                    str(record["side"]),
                    int(record["quantity"]),
                    float(record["average_entry"]),
                    record.get("current_price"),
                    record.get("unrealized_pnl"),
                    record.get("updated_at") or iso(utc_now()),
                ),
            )
            self._commit()

    def delete_position(self, symbol: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            self._commit()

    def get_positions(self):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM positions ORDER BY symbol"
            ).fetchall()

    # ------------------------------------------------------------------
    # Managed position lifecycle state
    # ------------------------------------------------------------------

    def save_managed_position(self, symbol: str, payload: dict[str, Any], closed: bool) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO managed_positions (symbol, closed, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    closed=excluded.closed,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    symbol,
                    1 if closed else 0,
                    json.dumps(payload, sort_keys=True, default=str),
                    iso(utc_now()),
                ),
            )
            self._commit()

    def get_managed_positions(self, include_closed: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM managed_positions"
        if not include_closed:
            query += " WHERE closed = 0"
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY symbol").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    # ------------------------------------------------------------------
    # System state (kill switch, risk locks, runtime status)
    # ------------------------------------------------------------------

    def set_state(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO system_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, sort_keys=True, default=str), iso(utc_now())),
            )
            self._commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM system_state WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row["value_json"]) if row else default

    def get_states(self, prefix: str) -> list[Any]:
        """All state values whose key starts with ``prefix`` (newest first)."""
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            rows = self._conn.execute(
                "SELECT value_json FROM system_state WHERE key LIKE ? ESCAPE '\\' "
                "ORDER BY updated_at DESC",
                (escaped + "%",),
            ).fetchall()
        return [json.loads(row["value_json"]) for row in rows]

    # ------------------------------------------------------------------
    # Live signal accuracy (observation only; never linked to orders)
    # ------------------------------------------------------------------

    def save_live_signal(self, row: dict[str, Any]) -> bool:
        """Insert a signal record once; returns False when the signal_id already exists."""
        columns = (
            "signal_id", "timestamp", "session_date", "strategy", "underlying", "direction",
            "option_type", "strike", "expiry", "instrument_key", "trading_symbol", "signal_price",
            "entry_price", "stop_loss", "target", "quantity", "active_from", "source",
            "record_json", "created_at", "status", "reason",
        )
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO live_signal_outcomes ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(row.get(c) for c in columns),
            )
            self._commit()
        return cursor.rowcount == 1

    def resolve_live_signal(self, signal_id: str, outcome: dict[str, Any]) -> bool:
        """Record an outcome once; only PENDING signals can be resolved."""
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE live_signal_outcomes
                SET status=?, exit_price=?, exit_time=?, pnl=?, r_multiple=?, resolved_at=?, reason=?
                WHERE signal_id=? AND status='PENDING'
                """,
                (
                    outcome["status"], outcome.get("exit_price"), outcome.get("exit_time"),
                    outcome.get("pnl"), outcome.get("r_multiple"), outcome.get("resolved_at"),
                    outcome.get("reason"), signal_id,
                ),
            )
            self._commit()
        return cursor.rowcount == 1

    def note_pending_live_signal(self, signal_id: str, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE live_signal_outcomes SET reason=? WHERE signal_id=? AND status='PENDING'",
                (reason, signal_id),
            )
            self._commit()

    def get_live_signals(self, session_date: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM live_signal_outcomes"
        params: tuple = ()
        if session_date:
            query += " WHERE session_date = ?"
            params = (session_date,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY timestamp, signal_id", params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # S2 forward paper validation (observation only; never linked to orders)
    # ------------------------------------------------------------------

    def save_forward_signal(self, row: dict[str, Any]) -> bool:
        """Insert a forward-validation signal once; returns False for a duplicate."""
        columns = (
            "signal_id", "timestamp", "session_date", "strategy", "underlying", "direction",
            "instrument_key", "expiry", "dte", "record_json", "created_at", "status", "exit_reason",
            "resolved_at",
        )
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO forward_validation_signals ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(row.get(c) for c in columns),
            )
            self._commit()
        return cursor.rowcount == 1

    def resolve_forward_signal(self, signal_id: str, outcome: dict[str, Any]) -> bool:
        """Record an outcome once; only PENDING signals can be resolved."""
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE forward_validation_signals
                SET status=?, exit_price=?, exit_time=?, exit_reason=?, pnl=?, r_multiple=?, resolved_at=?
                WHERE signal_id=? AND status='PENDING'
                """,
                (
                    outcome["status"], outcome.get("exit_price"), outcome.get("exit_time"),
                    outcome.get("reason"), outcome.get("pnl"), outcome.get("r_multiple"),
                    outcome.get("resolved_at"), signal_id,
                ),
            )
            self._commit()
        return cursor.rowcount == 1

    def get_forward_signals(self, session_date: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM forward_validation_signals"
        params: tuple = ()
        if session_date:
            query += " WHERE session_date = ?"
            params = (session_date,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY timestamp, signal_id", params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Shadow strategy comparison (observation only; never linked to orders)
    # ------------------------------------------------------------------

    def save_shadow_signal(self, row: dict[str, Any]) -> bool:
        columns = (
            "shadow_id", "candidate", "timestamp", "session_date", "underlying", "direction",
            "instrument_key", "expiry", "dte", "decision", "skip_reason", "status", "entry_price",
            "stop_loss", "target", "exit_price", "exit_time", "exit_reason", "pnl", "r_multiple",
            "market_regime", "record_json", "created_at", *SHADOW_GATE_COLUMNS,
        )
        with self._lock:
            cur = self._conn.execute(
                f"INSERT OR IGNORE INTO shadow_strategy_signals ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(row.get(c) for c in columns),
            )
            self._commit()
        return cur.rowcount == 1

    def update_shadow_outcome(self, shadow_id: str, row: dict[str, Any]) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """UPDATE shadow_strategy_signals
                   SET status=?, exit_price=?, exit_time=?, exit_reason=?, pnl=?, r_multiple=?
                   WHERE shadow_id=? AND status IN ('PENDING', 'DATA_UNAVAILABLE')""",
                (row.get("status"), row.get("exit_price"), row.get("exit_time"), row.get("exit_reason"),
                 row.get("pnl"), row.get("r_multiple"), shadow_id),
            )
            self._commit()
        return cur.rowcount == 1

    def get_shadow_signals(self, session_date: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM shadow_strategy_signals"
        params: tuple = ()
        if session_date:
            query += " WHERE session_date = ?"
            params = (session_date,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY timestamp, candidate, shadow_id", params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        event_id = event_id or str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO system_events (
                    event_id, event_type, severity, message, timestamp,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    severity,
                    message,
                    iso(timestamp or utc_now()),
                    json.dumps(metadata or {}, sort_keys=True, default=str),
                ),
            )
            self._commit()
        return event_id

    def get_events(self, limit: int = 100, event_type: str | None = None):
        with self._lock:
            if event_type is None:
                return self._conn.execute(
                    "SELECT * FROM system_events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM system_events WHERE event_type = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()

    def counts(self) -> dict[str, int]:
        result = {}
        with self._lock:
            for table in ("orders", "trades", "positions", "system_events"):
                result[table] = self._conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
        return result

