"""Option backtests: underlying signals, P&L from historical option premiums.

UNDERLYING backtests (BacktestEngine) trade the underlying itself. OPTION
backtests take the same StrategyEngine signal on the underlying, map it to
BUY CE / BUY PE on the contract listed at that time, and value the trade
only from that contract's historical premium candles. Premiums are never
estimated: without premium history the result is
OPTION_HISTORICAL_DATA_UNAVAILABLE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterable, Mapping

import pandas as pd

from src.models.positions import Position
from src.models.risk import AccountState
from src.options.instrument_resolver import OptionInstrumentResolver, StaticContractSource
from src.options.models import OptionContract, OptionResolutionError
from src.options.strategy import OptionSignalEngine, OptionSignalError, option_exit_config, resolve_option_contract

from .engine import BacktestEngine, BacktestResult

OK = "OK"
DATA_UNAVAILABLE = "OPTION_HISTORICAL_DATA_UNAVAILABLE"
NO_SIGNALS = "NO_SIGNALS"
DATA_UNAVAILABLE_MESSAGE = "Option historical data unavailable"


@dataclass
class OptionBacktestResult:
    status: str
    message: str
    backtest_type: str = "OPTION"
    underlying: str | None = None
    signal_source: str = "UNDERLYING"
    signals: int = 0
    skipped: list[dict] = field(default_factory=list)
    result: BacktestResult | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["result"] = self.result.as_dict() if self.result else None
        return data


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value)


class OptionBacktestEngine:
    def __init__(
        self,
        config,
        strategy_engine,
        risk_manager,
        session_manager,
        contracts: Iterable[OptionContract],
        premium_history: Mapping[str, pd.DataFrame] | None,
        initial_equity: float = 1_000_000.0,
        slippage_bps: float = 0.0,
        commission_per_order: float = 0.0,
    ) -> None:
        self.config = config
        self.strategy_engine = strategy_engine
        self.session_manager = session_manager
        self.signals = OptionSignalEngine(config, strategy_engine, risk_manager, session_manager)
        self._as_of: date | None = None
        self.resolver = OptionInstrumentResolver(StaticContractSource(contracts), today=lambda: self._as_of)
        self.premium_history = {
            key: frame.sort_values("timestamp").reset_index(drop=True)
            for key, frame in (premium_history or {}).items()
            if frame is not None and not frame.empty
        }
        self.engine = BacktestEngine(strategy_engine, initial_equity, slippage_bps, commission_per_order)
        self.initial_equity = float(initial_equity)

    def run(
        self,
        underlying_candles: pd.DataFrame,
        underlying: str,
        *,
        option_type: str = "AUTO",
        expiry_mode: str = "nearest",
        strike_mode: str = "ATM",
        strike_offset: int = 0,
    ) -> OptionBacktestResult:
        name = underlying.upper()
        if not self.premium_history:
            return OptionBacktestResult(DATA_UNAVAILABLE, DATA_UNAVAILABLE_MESSAGE, underlying=name)
        if underlying_candles.empty:
            return OptionBacktestResult(NO_SIGNALS, "No underlying candles", underlying=name)

        from src.position.monitor import PositionMonitor

        candles = underlying_candles.sort_values("timestamp").reset_index(drop=True)
        account = AccountState(equity=self.initial_equity)
        trades = []
        skipped: list[dict] = []
        signals = 0
        resume_after: pd.Timestamp | None = None

        for i in range(len(candles)):
            ts = _ts(candles.iloc[i]["timestamp"])
            if resume_after is not None and ts <= resume_after:
                continue
            decision = self.signals.underlying_decision(candles.iloc[: i + 1], option_type)
            if not decision.tradeable:
                continue
            signals += 1
            self._as_of = ts.date()
            try:
                resolved = resolve_option_contract(
                    self.resolver, underlying=name, option_type=decision.option_type, spot=decision.spot,
                    expiry_mode=expiry_mode, strike_mode=strike_mode, strike_offset=strike_offset)
            except OptionResolutionError as exc:
                skipped.append({"time": str(ts), "reason": exc.code})
                continue
            contract = resolved.contract
            premium = self.premium_history.get(contract.instrument_key)
            if premium is None:
                skipped.append({"time": str(ts), "reason": DATA_UNAVAILABLE,
                                "instrument_key": contract.instrument_key})
                continue
            history = premium[premium["timestamp"].map(_ts) <= ts]
            future = premium[premium["timestamp"].map(_ts) > ts]
            if history.empty or _ts(history.iloc[-1]["timestamp"]) != ts:
                skipped.append({"time": str(ts), "reason": DATA_UNAVAILABLE,
                                "instrument_key": contract.instrument_key})
                continue
            try:
                signal = self.signals.build_signal(
                    contract=contract, decision=decision, premium_candles=history, account=account)
            except OptionSignalError as exc:
                skipped.append({"time": str(ts), "reason": exc.code})
                continue
            if signal.blocked or signal.quantity <= 0:
                skipped.append({"time": str(ts), "reason": signal.block_reason or "QUANTITY_BELOW_ONE_LOT"})
                continue

            monitor = PositionMonitor(option_exit_config(self.config), self.session_manager)
            entry = self.engine._fill_price(signal.entry_price, "BUY", True)
            position = Position(
                symbol=contract.instrument_key, side="LONG", quantity=signal.quantity,
                average_entry=signal.entry_price, current_price=signal.entry_price,
                stop_loss=signal.stop_loss, target=signal.target, entry_time=ts.to_pydatetime())
            monitor.register(position)
            open_pos = {
                "symbol": contract.trading_symbol or contract.instrument_key, "side": "BUY",
                "quantity": signal.quantity, "entry_price": entry, "stop_loss": signal.stop_loss,
                "target": signal.target, "entry_time": ts, "position": position,
            }
            atr = self.strategy_engine.indicator_frame(premium)["atr"]
            last_atr = float(signal.details.get("premium_atr") or 0.0)
            exit_info = None
            exit_ts = ts
            for j in future.index:
                row = premium.loc[j]
                bar_ts = _ts(row["timestamp"])
                if pd.notna(atr.loc[j]) and atr.loc[j] > 0:
                    last_atr = float(atr.loc[j])
                exit_info = self.engine._replay_bar(monitor, open_pos, row, last_atr, bar_ts.to_pydatetime())
                exit_ts = bar_ts
                if exit_info is not None:
                    break
            if exit_info is None:
                last = premium.iloc[-1]
                exit_ts = _ts(last["timestamp"])
                reason = "EXPIRY_DATA_END" if exit_ts.date() >= contract.expiry else "DATA_END"
                exit_info = (reason, float(last["close"]))
            trade, pnl = self.engine._close(open_pos, exit_info[1], exit_ts, exit_info[0])
            trades.append(trade)
            account = account.model_copy(update={
                "equity": account.equity + pnl, "daily_pnl": account.daily_pnl + pnl})
            resume_after = exit_ts

        if not trades:
            unavailable = skipped and all(s["reason"] == DATA_UNAVAILABLE for s in skipped)
            return OptionBacktestResult(
                DATA_UNAVAILABLE if unavailable else NO_SIGNALS,
                DATA_UNAVAILABLE_MESSAGE if unavailable else "No option trades were taken",
                underlying=name, signals=signals, skipped=skipped)

        result = self.engine._result(trades, exit_engine="position_monitor")
        result.backtest_type = "OPTION"
        result.trades_closed_at_data_end = sum(
            1 for t in trades if t.exit_reason in ("DATA_END", "EXPIRY_DATA_END"))
        return OptionBacktestResult(OK, "Option backtest completed from historical premium data",
                                    underlying=name, signals=signals, skipped=skipped, result=result)
