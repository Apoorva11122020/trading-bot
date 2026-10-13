from __future__ import annotations
import os

class LiveTradingSafetyError(RuntimeError):
    pass


def live_trading_enabled() -> bool:
    return os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def trading_mode() -> str:
    return os.getenv("TRADING_MODE", "PAPER").strip().upper()


def assert_paper_trading_only() -> None:
    if trading_mode() != "PAPER" or live_trading_enabled():
        raise LiveTradingSafetyError(
            f"Paper trading refused: TRADING_MODE={trading_mode()} "
            f"LIVE_TRADING_ENABLED={live_trading_enabled()}"
        )


def assert_startup_safe(*, explicit_confirmation: bool = False) -> None:
    mode = trading_mode()
    enabled = live_trading_enabled()
    if mode != "LIVE":
        return
    if not enabled:
        raise LiveTradingSafetyError("LIVE_MODE_REQUIRES_LIVE_TRADING_ENABLED=true")
    if os.getenv("LIVE_TRADING_CONFIRMATION", "").strip() != "I_UNDERSTAND_LIVE_TRADING_RISK":
        raise LiveTradingSafetyError("LIVE_MODE_REQUIRES_EXPLICIT_CONFIRMATION")
    if not explicit_confirmation:
        raise LiveTradingSafetyError("LIVE_MODE_REQUIRES_RUNTIME_CONFIRMATION")


def startup_safety_snapshot() -> dict:
    return {
        "trading_mode": trading_mode(),
        "live_trading_enabled": live_trading_enabled(),
        "live_order_submission_allowed": trading_mode() == "LIVE" and live_trading_enabled() and os.getenv("LIVE_TRADING_CONFIRMATION", "") == "I_UNDERSTAND_LIVE_TRADING_RISK",
    }
