from __future__ import annotations
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.models.positions import Position
from src.models.risk import AccountState


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    gross_pnl: float
    costs: float
    pnl: float
    exit_reason: str
    stop_loss: float | None = None
    target: float | None = None


@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    max_drawdown: float
    average_trade: float
    profit_factor: float
    total_costs: float
    ending_equity: float
    trades_closed_at_data_end: int
    trades: list[dict]
    equity_curve: list[dict] = field(default_factory=list)
    max_drawdown_percent: float = 0.0
    exit_engine: str = "legacy_ohlc"
    backtest_type: str = "UNDERLYING"
    disclaimer: str = (
        "Historical simulation only. Results depend on the supplied data and "
        "assumptions and do not indicate future profitability."
    )

    def as_dict(self):
        return asdict(self)


TRADE_COLUMNS = [
    "symbol", "side", "quantity", "entry_price", "exit_price", "entry_time",
    "exit_time", "gross_pnl", "costs", "pnl", "exit_reason", "stop_loss", "target",
]


def export_trade_journal(result: BacktestResult, path: str | Path) -> Path:
    """Write the backtest trade journal as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRADE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for trade in result.trades:
            writer.writerow(trade)
    return path


class BacktestEngine:
    """Offline validation using the same StrategyEngine used by paper/live flows.

    Entries always come from StrategyEngine.evaluate_entry(). When the
    strategy carries a full config (the real StrategyEngine), exits are
    produced by the same PositionMonitor used at runtime, so stop-loss,
    target, breakeven, trailing stop and EOD force exit behave identically.
    Intrabar order is conservative: a long bar is replayed open, low, high,
    close (short: open, high, low, close), so an adverse touch is seen first.

    Strategies without that config fall back to plain OHLC stop/target exits.
    """

    def __init__(self, strategy_engine, initial_equity: float = 100_000.0,
                 slippage_bps: float = 0.0, commission_per_order: float = 0.0):
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if slippage_bps < 0 or commission_per_order < 0:
            raise ValueError("slippage_bps and commission_per_order cannot be negative")
        self.strategy_engine = strategy_engine
        self.initial_equity = float(initial_equity)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_order = float(commission_per_order)

    def _uses_monitor(self) -> bool:
        config = getattr(self.strategy_engine, "config", None)
        return (
            config is not None
            and hasattr(config, "exit")
            and hasattr(self.strategy_engine, "_calculate_indicators")
        )

    def _fill_price(self, price: float, side: str, is_entry: bool) -> float:
        # BUY pays up, SELL receives less. This is intentionally adverse.
        slip = price * self.slippage_bps / 10_000.0
        if side == "BUY":
            return price + slip
        return price - slip

    def _trade_cost(self) -> float:
        return self.commission_per_order * 2.0  # entry + exit

    def run(self, candles: pd.DataFrame) -> BacktestResult:
        if candles.empty:
            return self._result([])
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        missing = required - set(candles.columns)
        if missing:
            raise ValueError(f"Missing candle columns: {sorted(missing)}")
        candles = candles.sort_values("timestamp").reset_index(drop=True)
        if self._uses_monitor():
            return self._run_with_monitor(candles)
        return self._run_legacy(candles)

    # ------------------------------------------------------------------
    # Runtime-identical exits (PositionMonitor)
    # ------------------------------------------------------------------

    def _run_with_monitor(self, candles: pd.DataFrame) -> BacktestResult:
        from src.position.monitor import PositionMonitor

        engine = self.strategy_engine
        monitor = PositionMonitor(engine.config, engine.session_manager)
        # Indicators are causal (rolling/ewm over past bars), so one pass
        # over the full frame gives the same ATR each bar would have seen.
        atr = engine._calculate_indicators(candles)["atr"]

        account = AccountState(equity=self.initial_equity)
        open_pos: dict | None = None
        trades: list[BacktestTrade] = []
        current_day = None

        for i in range(len(candles)):
            row = candles.iloc[i]
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            day = ts.date()
            if current_day is None:
                current_day = day
            elif day != current_day and open_pos is None:
                account = account.model_copy(update={"daily_pnl": 0.0, "trades_today": 0})
                current_day = day

            if open_pos is not None:
                atr_value = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) and atr.iloc[i] > 0 else open_pos["last_atr"]
                open_pos["last_atr"] = atr_value
                exit_info = self._replay_bar(monitor, open_pos, row, atr_value, ts)
                if exit_info is not None:
                    reason, raw_exit = exit_info
                    trade, pnl = self._close(open_pos, raw_exit, ts, reason)
                    trades.append(trade)
                    monitor.mark_exit_completed(open_pos["symbol"])
                    account = account.model_copy(update={
                        "equity": account.equity + pnl,
                        "daily_pnl": account.daily_pnl + pnl,
                        "open_positions": 0,
                        "current_exposure": 0.0,
                    })
                    open_pos = None
                continue

            signal = engine.evaluate_entry(candles.iloc[: i + 1], account, ts)
            if signal and not signal.blocked and signal.quantity > 0 and pd.notna(atr.iloc[i]):
                entry_price = self._fill_price(float(signal.entry_price), signal.side.value, True)
                position = Position(
                    symbol=signal.symbol,
                    side="LONG" if signal.side.value == "BUY" else "SHORT",
                    quantity=signal.quantity,
                    average_entry=float(signal.entry_price),
                    current_price=float(signal.entry_price),
                    stop_loss=float(signal.stop_loss),
                    target=float(signal.target),
                    entry_time=ts,
                )
                monitor.register(position)
                open_pos = {
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "quantity": signal.quantity,
                    "entry_price": entry_price,
                    "stop_loss": float(signal.stop_loss),
                    "target": float(signal.target),
                    "entry_time": ts,
                    "position": position,
                    "last_atr": float(atr.iloc[i]),
                }
                account = account.model_copy(update={
                    "open_positions": 1,
                    "trades_today": account.trades_today + 1,
                    "current_exposure": entry_price * signal.quantity,
                })

        if open_pos is not None:
            row = candles.iloc[-1]
            trade, _ = self._close(open_pos, float(row["close"]), row["timestamp"], "DATA_END")
            trades.append(trade)

        return self._result(trades, exit_engine="position_monitor")

    def _replay_bar(self, monitor, open_pos: dict, row, atr_value: float, ts) -> tuple[str, float] | None:
        position = open_pos["position"]
        o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        path = (o, l, h, c) if position.side == "LONG" else (o, h, l, c)
        for step, price in enumerate(path):
            state_before = monitor.state(position.symbol)
            stop_before = state_before.active_stop if state_before else position.stop_loss
            signal = monitor.evaluate(position, price, atr_value, ts)
            if signal is None:
                continue
            reason = signal.reason
            if reason == "FORCE_EXIT_EOD":
                return "EOD_FORCE_EXIT", price
            if reason == "TARGET":
                target = monitor.state(position.symbol).target
                gapped = step == 0
                return reason, price if gapped else target
            # Stop family: fill at the stop unless the bar opened beyond it.
            if step == 0:
                return reason, price
            return reason, stop_before
        return None

    def _close(self, open_pos: dict, raw_exit: float, ts, reason: str):
        side = open_pos["side"]
        exit_side = "SELL" if side == "BUY" else "BUY"
        exit_price = self._fill_price(float(raw_exit), exit_side, False)
        gross = (exit_price - open_pos["entry_price"]) * open_pos["quantity"] * (1 if side == "BUY" else -1)
        costs = self._trade_cost()
        pnl = gross - costs
        trade = BacktestTrade(
            open_pos["symbol"], side, open_pos["quantity"], open_pos["entry_price"],
            exit_price, str(open_pos["entry_time"]), str(ts), gross, costs, pnl, reason,
            open_pos.get("stop_loss"), open_pos.get("target"),
        )
        return trade, pnl

    # ------------------------------------------------------------------
    # Legacy OHLC exits (strategies without exit config)
    # ------------------------------------------------------------------

    def _run_legacy(self, candles: pd.DataFrame) -> BacktestResult:
        account = AccountState(
            equity=self.initial_equity,
            daily_pnl=0.0,
            open_positions=0,
            trades_today=0,
            current_exposure=0.0,
        )
        open_pos = None
        trades: list[BacktestTrade] = []
        current_day = None

        for i in range(len(candles)):
            history = candles.iloc[: i + 1]
            row = candles.iloc[i]
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            day = ts.date() if hasattr(ts, "date") else None
            if current_day is None:
                current_day = day
            elif day != current_day and open_pos is None:
                account = account.model_copy(update={"daily_pnl": 0.0, "trades_today": 0})
                current_day = day

            price = float(row["close"])
            if open_pos is not None:
                side = open_pos["side"]
                stop = open_pos["stop_loss"]
                target = open_pos["target"]
                reason = None
                raw_exit = price

                force_exit = self.strategy_engine.session_manager.should_force_exit(ts)
                if side == "BUY":
                    if float(row["low"]) <= stop:
                        reason, raw_exit = "STOP_LOSS", stop
                    elif float(row["high"]) >= target:
                        reason, raw_exit = "TARGET", target
                    elif force_exit:
                        reason, raw_exit = "FORCE_EXIT", price
                else:
                    if float(row["high"]) >= stop:
                        reason, raw_exit = "STOP_LOSS", stop
                    elif float(row["low"]) <= target:
                        reason, raw_exit = "TARGET", target
                    elif force_exit:
                        reason, raw_exit = "FORCE_EXIT", price

                if reason:
                    trade, pnl = self._close(open_pos, raw_exit, ts, reason)
                    trades.append(trade)
                    account = account.model_copy(update={
                        "equity": account.equity + pnl,
                        "daily_pnl": account.daily_pnl + pnl,
                        "open_positions": 0,
                        "current_exposure": 0.0,
                    })
                    open_pos = None
                    continue

            if open_pos is None:
                signal = self.strategy_engine.evaluate_entry(history, account, ts)
                if signal and not signal.blocked and signal.quantity > 0:
                    entry_price = self._fill_price(float(signal.entry_price), signal.side.value, True)
                    open_pos = {
                        "symbol": signal.symbol,
                        "side": signal.side.value,
                        "quantity": signal.quantity,
                        "entry_price": entry_price,
                        "stop_loss": signal.stop_loss,
                        "target": signal.target,
                        "entry_time": ts,
                    }
                    account = account.model_copy(update={
                        "open_positions": 1,
                        "trades_today": account.trades_today + 1,
                        "current_exposure": entry_price * signal.quantity,
                    })

        # Close an otherwise valid open position at the last available close so
        # metrics are fully realized. This is explicitly labelled DATA_END.
        if open_pos is not None:
            row = candles.iloc[-1]
            trade, _ = self._close(open_pos, float(row["close"]), row["timestamp"], "DATA_END")
            trades.append(trade)

        return self._result(trades)

    def _result(self, trades, exit_engine: str = "legacy_ohlc"):
        pnls = [t.pnl for t in trades]
        equity = self.initial_equity
        peak = equity
        max_dd = 0.0
        max_dd_pct = 0.0
        curve = [{"time": None, "equity": equity}]
        for trade in trades:
            equity += trade.pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100 if peak else 0.0)
            curve.append({"time": trade.exit_time, "equity": equity})
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        costs = sum(t.costs for t in trades)
        data_end = sum(1 for t in trades if t.exit_reason == "DATA_END")
        return BacktestResult(
            total_trades=len(trades), winning_trades=len(wins), losing_trades=len(losses),
            win_rate=(len(wins) / len(trades) * 100) if trades else 0.0,
            gross_profit=gross_profit, gross_loss=gross_loss, net_pnl=sum(pnls),
            max_drawdown=max_dd, average_trade=(sum(pnls) / len(pnls)) if pnls else 0.0,
            profit_factor=(gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit else 0.0),
            total_costs=costs, ending_equity=self.initial_equity + sum(pnls),
            trades_closed_at_data_end=data_end,
            trades=[asdict(t) for t in trades],
            equity_curve=curve,
            max_drawdown_percent=max_dd_pct,
            exit_engine=exit_engine,
        )
