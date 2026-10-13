from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import uuid4

from src.models.positions import Position
from src.models.signals import Signal, SignalSide, SignalType


@dataclass
class ExitState:
    symbol: str
    side: str
    initial_stop: float
    target: float
    risk_per_share: float
    highest_price: float
    lowest_price: float
    active_stop: float
    breakeven_activated: bool = False
    exit_emitted: bool = False


class PositionMonitor:
    """Continuously evaluates an open position for automatic exit conditions.

    It does not place orders itself. It emits one standardized EXIT signal;
    OrderManager remains the only component allowed to submit the exit order.
    """

    def __init__(self, strategy_config, session_manager):
        self.config = strategy_config
        self.session_manager = session_manager
        self._states: dict[str, ExitState] = {}

    def register(self, position: Position) -> None:
        if position.side not in {"LONG", "SHORT"}:
            raise ValueError("position.side must be LONG or SHORT")

        risk_per_share = abs(position.average_entry - position.stop_loss)
        if risk_per_share <= 0:
            raise ValueError("position stop_loss must differ from average_entry")

        self._states[position.symbol] = ExitState(
            symbol=position.symbol,
            side=position.side,
            initial_stop=position.stop_loss,
            target=position.target,
            risk_per_share=risk_per_share,
            highest_price=position.average_entry,
            lowest_price=position.average_entry,
            active_stop=position.stop_loss,
        )

    def unregister(self, symbol: str) -> None:
        self._states.pop(symbol, None)

    def state(self, symbol: str) -> ExitState | None:
        return self._states.get(symbol)

    def _update_dynamic_stop(self, state: ExitState, current_price: float, atr_value: float) -> None:
        if atr_value <= 0:
            raise ValueError("atr_value must be > 0")

        if state.side == "LONG":
            state.highest_price = max(state.highest_price, current_price)
            entry = state.initial_stop + state.risk_per_share
            profit = current_price - entry

            stop = state.initial_stop
            if self.config.exit.breakeven.enabled:
                activation = self.config.exit.breakeven.activation_r
                if profit >= state.risk_per_share * activation:
                    state.breakeven_activated = True
            if state.breakeven_activated:
                stop = max(stop, entry)

            if self.config.exit.trailing_stop.enabled and state.breakeven_activated:
                trailing = state.highest_price - (
                    atr_value * self.config.exit.trailing_stop.atr_multiplier
                )
                stop = max(stop, trailing)

            state.active_stop = stop

        else:
            state.lowest_price = min(state.lowest_price, current_price)
            entry = state.initial_stop - state.risk_per_share
            profit = entry - current_price

            stop = state.initial_stop
            if self.config.exit.breakeven.enabled:
                activation = self.config.exit.breakeven.activation_r
                if profit >= state.risk_per_share * activation:
                    state.breakeven_activated = True
            if state.breakeven_activated:
                stop = min(stop, entry)

            if self.config.exit.trailing_stop.enabled and state.breakeven_activated:
                trailing = state.lowest_price + (
                    atr_value * self.config.exit.trailing_stop.atr_multiplier
                )
                stop = min(stop, trailing)

            state.active_stop = stop

    def evaluate(
        self,
        position: Position,
        current_price: float,
        atr_value: float,
        timestamp: datetime,
    ) -> Signal | None:
        if current_price <= 0:
            raise ValueError("current_price must be > 0")

        state = self._states.get(position.symbol)
        if state is None:
            self.register(position)
            state = self._states[position.symbol]

        if state.exit_emitted:
            return None

        self._update_dynamic_stop(state, current_price, atr_value)

        reason = None
        if self.session_manager.is_force_exit_window(timestamp):
            reason = "FORCE_EXIT_EOD"
        elif position.side == "LONG":
            breakeven_price = state.initial_stop + state.risk_per_share
            if current_price <= state.active_stop:
                if state.active_stop > breakeven_price:
                    reason = "TRAILING_STOP"
                elif state.breakeven_activated:
                    reason = "BREAKEVEN_STOP"
                else:
                    reason = "STOP_LOSS"
            elif current_price >= state.target:
                reason = "TARGET"
        else:
            breakeven_price = state.initial_stop - state.risk_per_share
            if current_price >= state.active_stop:
                if state.active_stop < breakeven_price:
                    reason = "TRAILING_STOP"
                elif state.breakeven_activated:
                    reason = "BREAKEVEN_STOP"
                else:
                    reason = "STOP_LOSS"
            elif current_price <= state.target:
                reason = "TARGET"

        if reason is None:
            return None

        state.exit_emitted = True
        exit_side = SignalSide.SELL if position.side == "LONG" else SignalSide.BUY
        return Signal(
            signal_id=str(uuid4()),
            timestamp=timestamp,
            symbol=position.symbol,
            side=exit_side,
            signal_type=SignalType.EXIT,
            entry_price=current_price,
            quantity=position.quantity,
            reason=reason,
            risk_status="ALLOWED",
        )

    def mark_exit_completed(self, symbol: str) -> None:
        self.unregister(symbol)

    def export_state(self, symbol: str) -> dict | None:
        state = self._states.get(symbol)
        return asdict(state) if state else None

    def restore_state(self, data: dict) -> ExitState:
        """Reinstall a persisted exit state (stops, extremes, breakeven)
        so a restart never loosens an already-tightened stop."""
        state = ExitState(**data)
        self._states[state.symbol] = state
        return state
