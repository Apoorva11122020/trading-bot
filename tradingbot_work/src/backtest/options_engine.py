"""Option backtest engine for strategy research.

Underlying 5-minute candles generate the signal; option premium candles
determine P&L. BUY -> BUY CE, SELL -> BUY PE. No naked selling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from types import SimpleNamespace
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from src.models.positions import Position
from src.models.risk import AccountState
from src.options.instrument_resolver import OptionInstrumentResolver, StaticContractSource
from src.options.models import OptionResolutionError, OptionType
from src.options.selection import atm_strike, select_expiry
from src.options.strategy import option_exit_config, option_stop_loss, option_target, round_to_tick
from src.position.monitor import PositionMonitor
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager

from .research_strategies import ResearchSignal, generate_signals

IST = ZoneInfo("Asia/Kolkata")
DATA_INVALID = "DATA_INVALID"
SKIPPED = "SKIPPED"
DAILY_LOSS = "DAILY_LOSS_LIMIT_HIT"
REGIME_FILTERED = "REGIME_FILTERED"

# Returns the reasons a signal must be blocked; an empty list allows it.
SignalFilter = Callable[[ResearchSignal], list[str]]


@dataclass
class ResearchTrade:
    strategy_id: str
    underlying: str
    direction: str
    option_type: str
    instrument_key: str
    trading_symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    entry_time: str
    exit_time: str
    exit_reason: str
    gross_pnl: float
    costs: float
    pnl: float
    r_multiple: float
    signal_time: str
    conditions: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchMetrics:
    total_signals: int = 0
    resolved_trades: int = 0
    wins: int = 0
    losses: int = 0
    eod_exits: int = 0
    other_exits: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    average_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    average_trade_pnl: float = 0.0
    max_consecutive_losses: int = 0
    average_trades_per_day: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    median_r: float = 0.0
    unresolved_signals: int = 0
    trading_days: int = 0
    data_invalid: int = 0
    skipped: int = 0
    daily_loss_hits: int = 0
    costs_label: str = "Costs excluded"
    disclaimer: str = (
        "Historical simulation only. Results depend on downloaded Upstox data "
        "and configured assumptions. Costs may be excluded."
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptionsResearchResult:
    status: str
    message: str
    run_id: str
    underlying: str
    strategy_id: str
    split: str
    from_date: str
    to_date: str
    metrics: ResearchMetrics
    trades: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "run_id": self.run_id,
            "underlying": self.underlying,
            "strategy_id": self.strategy_id,
            "split": self.split,
            "from_date": self.from_date,
            "to_date": self.to_date,
            "metrics": self.metrics.as_dict(),
            "trades": self.trades,
            "signals": self.signals,
            "skipped": self.skipped,
        }


def _ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


class OptionsResearchEngine:
    def __init__(
        self,
        research_config,
        strategy_config,
        contracts,
        premium_history: dict[str, pd.DataFrame],
        *,
        run_id: str,
        split: str = "FULL",
        initial_equity: float | None = None,
        master_resolver: OptionInstrumentResolver | None = None,
        signal_filter: SignalFilter | None = None,
    ) -> None:
        self.signal_filter = signal_filter
        self.research = research_config
        self.strategy_config = strategy_config
        self.run_id = run_id
        self.split = split
        self.premium_history = {
            key: frame.sort_values("timestamp").reset_index(drop=True)
            for key, frame in premium_history.items()
            if frame is not None and not frame.empty
        }
        self._as_of: date | None = None
        self._master_resolver = master_resolver
        self.resolver = OptionInstrumentResolver(StaticContractSource(contracts), today=lambda: self._as_of)
        self.risk_manager = RiskManager(self._research_risk_config())
        self.session_manager = SessionManager(strategy_config.session)
        self.initial_equity = float(initial_equity or research_config.risk.initial_equity)
        self.entry_timing = str(research_config.option_mapping.get("entry_timing", "next_open"))
        self.stop_entries = _parse_time(research_config.session.stop_new_entries)
        self.square_off = _parse_time(research_config.session.square_off)
        self.max_daily_loss = float(research_config.risk.max_daily_loss_inr)

    def _research_risk_config(self):
        """Research INR daily cap + strategy gap buffer; does not alter S2 signals."""
        r = self.research.risk
        s = self.strategy_config.risk
        return SimpleNamespace(
            risk_per_trade_percent=r.risk_per_trade_percent,
            max_daily_loss_inr=r.max_daily_loss_inr,
            max_daily_loss_percent=s.max_daily_loss_percent,
            max_open_positions=r.max_open_positions,
            max_trades_per_day=r.max_trades_per_day,
            max_portfolio_exposure_percent=r.max_portfolio_exposure_percent,
            max_loss_gap_buffer_percent=getattr(s, "max_loss_gap_buffer_percent", 10.0),
        )

    def _entry_allowed(self, ts: datetime) -> bool:
        local = ts.astimezone(IST) if ts.tzinfo else ts.replace(tzinfo=IST)
        return self.session_manager.is_entry_allowed(local) and local.time() < self.stop_entries

    def _resolve_contract(self, signal: ResearchSignal):
        resolver = self._master_resolver or self.resolver
        as_of = signal.timestamp.date()
        if self._master_resolver is not None:
            self._master_resolver.today = lambda: as_of
        else:
            self._as_of = as_of
        option_type = OptionType.CE if signal.direction == "BUY" else OptionType.PE
        expiry = select_expiry(resolver.expiries(signal.underlying),
                               self.research.option_mapping.get("expiry_mode", "nearest"))
        strike = atm_strike(resolver.strikes(signal.underlying, expiry, option_type), signal.spot)
        offset = int(self.research.option_mapping.get("strike_offset", 0))
        if offset:
            from src.options.selection import select_strike
            strike = select_strike(resolver.strikes(signal.underlying, expiry, option_type),
                                   option_type=option_type, mode=self.research.option_mapping.get("strike_mode", "ATM"),
                                   spot=signal.spot, offset=offset)
        return resolver.resolve(signal.underlying, expiry, strike, option_type)

    def _trade_cost(self, premium: float, quantity: int) -> float:
        costs = self.research.costs
        turnover = premium * quantity
        brokerage = costs.brokerage_per_order * 2
        stt = turnover * costs.stt_percent / 100
        exchange = turnover * costs.exchange_charges_percent / 100
        gst_base = brokerage + exchange
        gst = gst_base * costs.gst_percent / 100
        stamp = turnover * costs.stamp_duty_percent / 100
        return brokerage + stt + exchange + gst + stamp

    def _slippage(self, price: float) -> float:
        return price * self.research.costs.slippage_bps / 10_000.0

    def _metrics(self, trades: list[ResearchTrade], signals: int, skipped: list, invalid: int, daily_hits: int,
                 trading_days: int) -> ResearchMetrics:
        metrics = ResearchMetrics(
            total_signals=signals, data_invalid=invalid, skipped=len(skipped),
            daily_loss_hits=daily_hits, costs_label=self.research.costs.label,
            trading_days=trading_days, unresolved_signals=max(0, signals),
        )
        resolved = [t for t in trades if t.exit_reason not in (DATA_INVALID, SKIPPED)]
        metrics.resolved_trades = len(resolved)
        metrics.unresolved_signals = max(0, signals - len(resolved))
        if not resolved:
            return metrics
        wins = [t for t in resolved if t.pnl > 0]
        losses = [t for t in resolved if t.pnl < 0]
        metrics.wins, metrics.losses = len(wins), len(losses)
        decided = metrics.wins + metrics.losses
        metrics.win_rate = metrics.wins / decided if decided else 0.0
        metrics.loss_rate = metrics.losses / decided if decided else 0.0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        metrics.gross_profit = gross_profit
        metrics.gross_loss = gross_loss
        metrics.profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        metrics.total_pnl = sum(t.pnl for t in resolved)
        metrics.average_trade_pnl = metrics.total_pnl / len(resolved)
        r_values = [t.r_multiple for t in resolved]
        metrics.average_r = sum(r_values) / len(resolved)
        metrics.median_r = float(pd.Series(r_values).median())
        metrics.eod_exits = sum(1 for t in resolved if t.exit_reason in ("FORCE_EXIT", "EOD_EXIT", "SQUARE_OFF"))
        metrics.other_exits = len(resolved) - metrics.wins - metrics.losses - metrics.eod_exits
        equity = self.initial_equity
        peak = equity
        max_dd = 0.0
        streak = max_streak = 0
        for trade in resolved:
            equity += trade.pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            streak = streak + 1 if trade.pnl < 0 else 0
            max_streak = max(max_streak, streak)
        metrics.max_drawdown = max_dd
        metrics.max_consecutive_losses = max_streak
        metrics.average_trades_per_day = len(resolved) / trading_days if trading_days else 0.0
        return metrics

    def run(self, underlying_candles: pd.DataFrame, underlying: str, strategy_id: str,
            from_date: date, to_date: date) -> OptionsResearchResult:
        name = underlying.upper()
        if underlying_candles is None or underlying_candles.empty:
            return OptionsResearchResult("NO_DATA", "No underlying candles in range", self.run_id, name, strategy_id,
                                         self.split, from_date.isoformat(), to_date.isoformat(), ResearchMetrics())
        frame = underlying_candles.sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            return OptionsResearchResult("NO_DATA", "No underlying candles in range", self.run_id, name, strategy_id,
                                         self.split, from_date.isoformat(), to_date.isoformat(), ResearchMetrics())
        if "symbol" not in frame:
            frame["symbol"] = name

        raw_signals = generate_signals(strategy_id, frame, name, self.research)
        signals = [s for s in raw_signals if from_date <= s.timestamp.date() <= to_date and self._entry_allowed(s.timestamp)]
        trades: list[ResearchTrade] = []
        skipped: list[dict] = []
        invalid = 0
        daily_hits = 0
        account = AccountState(equity=self.initial_equity)
        daily_pnl = 0.0
        current_day: date | None = None
        trades_today = 0
        open_positions = 0
        resume_after: pd.Timestamp | None = None
        trading_days = len(set(frame["timestamp"].dt.date))

        for signal in signals:
            ts = _ts(signal.timestamp)
            if resume_after is not None and ts <= resume_after:
                continue
            if self.signal_filter is not None:
                blocked = self.signal_filter(signal)
                if blocked:
                    skipped.append({"time": ts.isoformat(), "reason": REGIME_FILTERED, "regimes": list(blocked)})
                    continue
            day = ts.date()
            if current_day != day:
                current_day, daily_pnl, trades_today = day, 0.0, 0
            account = account.model_copy(update={"daily_pnl": daily_pnl, "trades_today": trades_today})
            if self.risk_manager.remaining_daily_loss_budget(account) <= 0:
                daily_hits += 1
                skipped.append({"time": ts.isoformat(), "reason": DAILY_LOSS})
                continue

            try:
                contract = self._resolve_contract(signal)
            except OptionResolutionError as exc:
                skipped.append({"time": ts.isoformat(), "reason": exc.code, "strategy": strategy_id})
                continue

            premium = self.premium_history.get(contract.instrument_key)
            if premium is None or premium.empty:
                invalid += 1
                skipped.append({"time": ts.isoformat(), "reason": DATA_INVALID,
                                "instrument_key": contract.instrument_key})
                continue

            idx = premium[premium["timestamp"].map(_ts) <= ts].index
            if len(idx) == 0:
                invalid += 1
                skipped.append({"time": ts.isoformat(), "reason": DATA_INVALID,
                                "instrument_key": contract.instrument_key})
                continue
            signal_idx = idx[-1]
            future = premium[premium.index > signal_idx]
            if future.empty:
                invalid += 1
                skipped.append({"time": ts.isoformat(), "reason": DATA_INVALID,
                                "instrument_key": contract.instrument_key})
                continue

            entry_row = future.iloc[0] if self.entry_timing == "next_open" else premium.loc[signal_idx]
            entry_ts = _ts(entry_row["timestamp"])
            entry = round_to_tick(float(entry_row["open"] if self.entry_timing == "next_open" else entry_row["close"]),
                                  contract.tick_size, "nearest")
            # Premium ATR may only use candles completed by the signal bar; for next_open entries the
            # entry candle's high/low/close are unknown at its open.
            history = premium.loc[: signal_idx]
            from src.indicators.technical import atr as calc_atr
            prem_frame = history.copy()
            prem_frame["atr"] = calc_atr(prem_frame, self.strategy_config.indicators.atr_period)
            if prem_frame.empty or pd.isna(prem_frame.iloc[-1].get("atr")):
                invalid += 1
                skipped.append({"time": ts.isoformat(), "reason": "INSUFFICIENT_ATR_HISTORY",
                                "instrument_key": contract.instrument_key})
                continue
            premium_atr = float(prem_frame.iloc[-1]["atr"])
            if premium_atr <= 0:
                invalid += 1
                skipped.append({"time": ts.isoformat(), "reason": "PREMIUM_ATR_ZERO",
                                "instrument_key": contract.instrument_key})
                continue
            stop = option_stop_loss(entry, stop_mode="premium_atr", multiplier=self.strategy_config.options.risk.stop_atr_multiplier,
                                    premium_atr=premium_atr, tick_size=contract.tick_size)
            target = option_target(entry, stop, self.strategy_config.options.risk.target_r, contract.tick_size)
            sizing = self.risk_manager.calculate_option_quantity(
                account, entry, stop, contract.lot_size, contract.freeze_quantity,
                self.research.risk.max_premium_exposure_pct,
                open_position_max_loss=0.0,
            )
            if sizing.quantity <= 0:
                skipped.append({"time": ts.isoformat(), "reason": SKIPPED, "detail": "QUANTITY_BELOW_ONE_LOT"})
                continue
            check = self.risk_manager.validate_option_entry(
                account, entry, stop, target, sizing.quantity, contract.lot_size,
                session_allowed=True, freeze_quantity=contract.freeze_quantity,
                max_premium_exposure_pct=self.research.risk.max_premium_exposure_pct,
                open_option_positions=open_positions, max_option_positions=self.research.risk.max_option_positions)
            if not check.allowed or trades_today >= self.research.risk.max_trades_per_day:
                skipped.append({"time": ts.isoformat(), "reason": SKIPPED, "detail": check.reasons})
                continue

            monitor = PositionMonitor(option_exit_config(self.strategy_config), self.session_manager)
            position = Position(symbol=contract.instrument_key, side="LONG", quantity=sizing.quantity,
                                average_entry=entry, current_price=entry, stop_loss=stop, target=target,
                                entry_time=entry_ts.to_pydatetime())
            monitor.register(position)
            atr_series = calc_atr(premium, self.strategy_config.indicators.atr_period)
            last_atr = premium_atr
            exit_info = None
            exit_ts = entry_ts
            start_idx = future.index[0] if self.entry_timing == "next_open" else signal_idx
            for j in premium.loc[start_idx:].index:
                row = premium.loc[j]
                bar_ts = _ts(row["timestamp"])
                if bar_ts.time() >= self.square_off:
                    exit_info = ("SQUARE_OFF", float(row["close"]))
                    exit_ts = bar_ts
                    break
                if pd.notna(atr_series.loc[j]) and atr_series.loc[j] > 0:
                    last_atr = float(atr_series.loc[j])
                if pd.isna(last_atr) or last_atr <= 0:
                    invalid += 1
                    skipped.append({"time": ts.isoformat(), "reason": "INSUFFICIENT_ATR_HISTORY"})
                    exit_info = None
                    break
                exit_info = self._replay_bar(monitor, position, row, last_atr, bar_ts.to_pydatetime())
                exit_ts = bar_ts
                if exit_info is not None:
                    break
            if exit_info is None and not any(s.get("time") == ts.isoformat() and s.get("reason") == "INSUFFICIENT_ATR_HISTORY"
                                             for s in skipped[-1:]):
                last = premium.iloc[-1]
                exit_ts = _ts(last["timestamp"])
                reason = "EXPIRY_DATA_END" if exit_ts.date() >= contract.expiry else "DATA_END"
                exit_info = (reason, float(last["close"]))

            if exit_info is None:
                continue

            exit_price = exit_info[1] + self._slippage(exit_info[1])
            entry_fill = entry + self._slippage(entry)
            gross = (exit_price - entry_fill) * sizing.quantity
            costs = self._trade_cost(entry_fill, sizing.quantity)
            pnl = gross - costs
            risk = max(entry_fill - stop, 1e-9)
            trade = ResearchTrade(strategy_id=strategy_id, underlying=name, direction=signal.direction,
                                  option_type=contract.option_type.value, instrument_key=contract.instrument_key,
                                  trading_symbol=contract.trading_symbol, quantity=sizing.quantity,
                                  entry_price=entry_fill, exit_price=exit_price, stop_loss=stop, target=target,
                                  entry_time=entry_ts.isoformat(), exit_time=exit_ts.isoformat(),
                                  exit_reason=exit_info[0], gross_pnl=gross, costs=costs, pnl=pnl,
                                  r_multiple=pnl / (risk * sizing.quantity), signal_time=ts.isoformat(),
                                  conditions=signal.conditions)
            trades.append(trade)
            account = account.model_copy(update={"equity": account.equity + pnl, "daily_pnl": account.daily_pnl + pnl})
            daily_pnl += pnl
            trades_today += 1
            resume_after = exit_ts

        metrics = self._metrics(trades, len(signals), skipped, invalid, daily_hits, trading_days)
        status = "OK" if metrics.resolved_trades else ("NO_TRADES" if signals else "NO_SIGNALS")
        return OptionsResearchResult(status, "Research backtest completed", self.run_id, name, strategy_id, self.split,
                                     from_date.isoformat(), to_date.isoformat(), metrics,
                                     trades=[t.as_dict() for t in trades],
                                     signals=[s.as_dict() for s in signals],
                                     skipped=skipped)

    @staticmethod
    def _replay_bar(monitor, open_pos: Position, row, atr_value: float, timestamp: datetime):
        from src.backtest.engine import BacktestEngine
        engine = BacktestEngine.__new__(BacktestEngine)
        return engine._replay_bar(monitor, {"position": open_pos, "entry_price": open_pos.average_entry,
                                            "stop_loss": open_pos.stop_loss, "target": open_pos.target},
                                  row, atr_value, timestamp)
