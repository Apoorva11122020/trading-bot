"""LIVE QUALITY GATE V1 - observation-only ACCEPT / SKIP layer above unchanged S2 signals.

PAPER ONLY. The gate never generates, modifies or places anything: it receives an S2 signal context that S2 has
already produced and returns a deterministic, auditable decision. Production S2 never imports this module; it is
used only by the shadow / forward-validation pipeline.

Causality: ``evaluate_quality`` reads only the whitelisted keys in ``CAUSAL_FEATURES``. ``gate_features_at`` derives
RSI14 / EMA20 / EMA50 / ATR14 / EMA20 slope / trend regime from underlying candles with timestamp <= the signal
candle, using the same definitions as the S2 diagnostic research (``strategy_improvement_lab.bar_features`` and
``s2_diagnostic_lab.classify_regime``). Outcome, exit, P&L and future premium fields are never read.

The V1 rules are research-derived quality filters, NOT a proven live edge and NOT a 70% guarantee.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from src.indicators.technical import atr, ema, rsi

GATE_VERSION = "LIVE_QUALITY_GATE_V1"
GATE_CANDIDATE = "LIVE_QUALITY_GATE_V1"
# The gate is never wired into production execution; only shadow / forward validation enables it.
PRODUCTION_EXECUTION_ENABLED = False
IST = ZoneInfo("Asia/Kolkata")

ACCEPT, SKIP = "ACCEPT", "SKIP"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
TOO_RESTRICTIVE = "QUALITY_GATE_TOO_RESTRICTIVE"
REJECTED = "QUALITY_GATE_REJECTED"
TREND_UP, TREND_DOWN, RANGE = "TREND_UP", "TREND_DOWN", "RANGE"

# Every key the gate may read. Anything else (outcome, exit, pnl, future premium ...) is invisible to it.
CAUSAL_FEATURES = ("direction", "signal_time", "spot", "breakout_level", "atr14", "break_atr", "time_bucket",
                   "rsi14", "ema20", "ema50", "ema_sep_atr", "ema20_slope_atr", "trend_regime", "premium_atr")
FORBIDDEN_FIELDS = ("outcome", "status", "pnl", "r_multiple", "exit_price", "exit_time", "exit_reason", "target",
                    "stop_loss", "premium", "future", "win", "loss", "accuracy")
REQUIRED = {"SELL": ("rsi14", "ema20", "ema50", "signal_time"),
            "BUY": ("ema_sep_atr", "trend_regime", "break_atr")}
SELL_TIME_WINDOW = (time(12, 0), time(13, 0))


@dataclass(frozen=True)
class QualityGateConfig:
    enabled: bool = True
    min_score_buy: int = 2
    min_score_sell: int = 2
    target_accuracy: float = 0.70
    min_live_samples: int = 30
    min_accept_rate: float = 0.05
    max_skip_ratio: float = 0.95
    allow_partial_features: bool = False
    sell_rsi_range: tuple[float, float] = (50.0, 60.0)
    buy_max_ema_sep_atr: float = -1.0
    buy_max_break_atr: float = 0.50

    def min_score(self, direction: str) -> int:
        return self.min_score_buy if direction == "BUY" else self.min_score_sell


@dataclass(frozen=True)
class QualityGateDecision:
    accepted: bool
    score: int
    max_score: int
    confidence: float
    direction: str
    reasons: list[str]
    failed_checks: list[str]
    features: dict[str, Any]
    missing_features: list[str] = field(default_factory=list)
    version: str = GATE_VERSION

    @property
    def decision(self) -> str:
        return ACCEPT if self.accepted else SKIP

    @property
    def skip_reason(self) -> str | None:
        return None if self.accepted else f"{REJECTED}:" + ",".join(self.failed_checks)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "decision": self.decision, "skip_reason": self.skip_reason}


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _ist(v: Any) -> datetime | None:
    if v is None:
        return None
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return (ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)).to_pydatetime()


def _jsonable(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _read(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelisted causal snapshot; missing keys stay None (never guessed)."""
    f: dict[str, Any] = {}
    for k in CAUSAL_FEATURES:
        v = ctx.get(k)
        if k == "signal_time":
            f[k] = _ist(v)
        elif k in ("direction", "time_bucket", "trend_regime"):
            f[k] = None if v in (None, "", "UNAVAILABLE") else str(v)
        else:
            f[k] = _num(v)
    if f["ema_sep_atr"] is None and None not in (f["ema20"], f["ema50"]) and f["atr14"]:
        f["ema_sep_atr"] = (f["ema20"] - f["ema50"]) / f["atr14"]
    return f


def evaluate_quality(ctx: Mapping[str, Any], config: QualityGateConfig = QualityGateConfig()) -> QualityGateDecision:
    """ACCEPT / SKIP for one S2 signal context. Pure, deterministic, reads only ``CAUSAL_FEATURES``."""
    f = _read(ctx)
    direction = (f["direction"] or "").upper()
    features = {k: _jsonable(v) for k, v in f.items()}
    if not config.enabled:
        return QualityGateDecision(True, 0, 0, 0.0, direction, ["GATE_DISABLED_PASS_THROUGH"], [], features)
    if direction not in REQUIRED:
        return QualityGateDecision(False, 0, 3, 0.0, direction, [], ["UNKNOWN_DIRECTION"], features)

    missing = [k for k in REQUIRED[direction] if f[k] is None]
    reasons: list[str] = []
    failed: list[str] = [f"{DATA_UNAVAILABLE}:{k}" for k in missing]

    def check(name: str, needs: tuple[str, ...], ok) -> None:
        if any(f[k] is None for k in needs):
            return
        (reasons if ok() else failed).append(name)

    if direction == "SELL":
        lo, hi = config.sell_rsi_range
        check(f"SELL_RSI_{lo:g}_{hi:g}", ("rsi14",), lambda: lo <= f["rsi14"] < hi)
        check("SELL_EMA20_GT_EMA50", ("ema20", "ema50"), lambda: f["ema20"] > f["ema50"])
        a, b = SELL_TIME_WINDOW
        check("SELL_TIME_12_13", ("signal_time",), lambda: a <= f["signal_time"].time() < b)
    else:
        check(f"BUY_EMA_SEP_ATR_LT_{config.buy_max_ema_sep_atr:g}", ("ema_sep_atr",),
              lambda: f["ema_sep_atr"] < config.buy_max_ema_sep_atr)
        check("BUY_TREND_DOWN", ("trend_regime",), lambda: f["trend_regime"] == TREND_DOWN)
        check(f"BUY_BREAK_ATR_LE_{config.buy_max_break_atr:.2f}", ("break_atr",),
              lambda: f["break_atr"] <= config.buy_max_break_atr)

    score, max_score = len(reasons), 3
    enough = score >= config.min_score(direction)
    accepted = enough and (not missing or config.allow_partial_features)
    if not enough:
        failed.append(f"SCORE_{score}_BELOW_MIN_{config.min_score(direction)}")
    elif missing and not config.allow_partial_features:
        failed.append(f"{DATA_UNAVAILABLE}_POLICY_REJECT")
    return QualityGateDecision(accepted, score, max_score, round(score / max_score, 4), direction, reasons, failed,
                               features, missing)


def restrictiveness(generated: int, accepted: int, config: QualityGateConfig = QualityGateConfig()) -> dict[str, Any]:
    """Overfilter guard: an almost-empty accepted sample is flagged, never reported as accuracy."""
    if generated <= 0:
        return {"status": "NO_SIGNALS", "accept_rate": None, "skip_ratio": None}
    rate = accepted / generated
    skip = 1 - rate
    status = TOO_RESTRICTIVE if rate < config.min_accept_rate or skip > config.max_skip_ratio else "OK"
    return {"status": status, "accept_rate": round(rate, 4), "skip_ratio": round(skip, 4),
            "min_accept_rate": config.min_accept_rate, "max_skip_ratio": config.max_skip_ratio}


# ------------------------------------------------------------------ causal features from candles
def _bars(spot: pd.DataFrame) -> pd.DataFrame:
    f = spot[["timestamp", "open", "high", "low", "close"]].copy()
    ts = pd.to_datetime(f["timestamp"])
    f["timestamp"] = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    f = f.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    f["atr14"] = atr(f, 14)
    f["ema20"], f["ema50"], f["rsi14"] = ema(f["close"], 20), ema(f["close"], 50), rsi(f["close"], 14)
    f["ema20_slope_atr"] = (f["ema20"] - f["ema20"].shift(3)) / f["atr14"]
    return f


def trend_regime(close: float | None, ema20: float | None, ema50: float | None, slope: float | None) -> str | None:
    """Same rule as s2_diagnostic_lab.classify_regime (trend axis)."""
    if None in (close, ema20, ema50, slope):
        return None
    if close > ema20 > ema50 and slope > 0:
        return TREND_UP
    if close < ema20 < ema50 and slope < 0:
        return TREND_DOWN
    return RANGE


def gate_features_at(spot: pd.DataFrame, signal_time: Any) -> dict[str, Any]:
    """Indicator values at the signal candle using only candles with timestamp <= signal_time."""
    ts = pd.Timestamp(_ist(signal_time))
    if spot is None or spot.empty:
        return {}
    stamps = pd.to_datetime(spot["timestamp"])
    stamps = stamps.dt.tz_localize(IST) if stamps.dt.tz is None else stamps.dt.tz_convert(IST)
    bars = _bars(spot[(stamps <= ts).to_numpy()].reset_index(drop=True))
    if bars.empty or bars["timestamp"].iloc[-1] != ts:
        return {}
    b = bars.iloc[-1]
    close, a, e20, e50, sl = (_num(b[k]) for k in ("close", "atr14", "ema20", "ema50", "ema20_slope_atr"))
    return {"close": close, "atr14": a, "ema20": e20, "ema50": e50, "rsi14": _num(b["rsi14"]),
            "ema20_slope_atr": sl, "ema_sep_atr": None if None in (e20, e50) or not a else (e20 - e50) / a,
            "trend_regime": trend_regime(close, e20, e50, sl)}


def gate_context(ctx: Mapping[str, Any], spot: pd.DataFrame | None) -> dict[str, Any]:
    """New dict: causal S2 context fields plus candle-derived features. ``ctx`` itself is never modified."""
    out = {k: ctx.get(k) for k in CAUSAL_FEATURES if k in ctx}
    if spot is not None and ctx.get("signal_time") is not None:
        for k, v in gate_features_at(spot, ctx["signal_time"]).items():
            if k == "atr14" and out.get("atr14") is not None:
                continue
            out[k] = v
    return out
