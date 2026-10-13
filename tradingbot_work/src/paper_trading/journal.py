"""Paper trading journal: database events + session report files."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER_STATE_KEY = "paper_trading_status"


class PaperTradingJournal:
    def __init__(self, database, reports_dir: str | Path, session_id: str):
        self.database = database
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.session_file = self.reports_dir / f"session_{session_id}.jsonl"
        self._status: dict[str, Any] = {
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "events": 0,
        }

    def _write_line(self, payload: dict[str, Any]) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with self.session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
        self._status["events"] = int(self._status.get("events", 0)) + 1

    def event(self, event_type: str, severity: str, message: str, **metadata: Any) -> None:
        self.database.add_event(event_type, severity, message, metadata)
        self._write_line({"type": event_type, "severity": severity, "message": message, **metadata})

    def signal(self, payload: dict[str, Any]) -> None:
        self.event("PAPER_SIGNAL", "INFO", "S2 signal generated", signal=payload)
        self._status["latest_signal"] = payload

    def skipped(self, reason: str, **metadata: Any) -> None:
        self.event("PAPER_SIGNAL_SKIPPED", "WARNING", reason, reason=reason, **metadata)
        self._status["last_skip_reason"] = reason

    def risk_decision(self, allowed: bool, reasons: list[str], **metadata: Any) -> None:
        self.event(
            "PAPER_RISK_CHECK",
            "INFO" if allowed else "WARNING",
            "allowed" if allowed else "blocked",
            allowed=allowed,
            reasons=reasons,
            **metadata,
        )
        self._status["last_risk_decision"] = {"allowed": allowed, "reasons": reasons, **metadata}

    def contract(self, payload: dict[str, Any]) -> None:
        self.event("PAPER_CONTRACT", "INFO", "Option contract selected", contract=payload)
        self._status["current_option"] = payload

    def order_stage(self, stage: str, **metadata: Any) -> None:
        self.event(f"PAPER_{stage}", "INFO", stage, stage=stage, **metadata)

    def market_data(self, status: str, **metadata: Any) -> None:
        self.event("PAPER_MARKET_DATA", "WARNING" if status != "CONNECTED" else "INFO", status, status=status, **metadata)

    def risk_lock(self, **metadata: Any) -> None:
        self.event("PAPER_RISK_LOCK", "CRITICAL", "Daily loss lock active", **metadata)

    def update_status(self, patch: dict[str, Any]) -> None:
        self._status.update(patch)
        self.database.set_state(PAPER_STATE_KEY, self._status)
