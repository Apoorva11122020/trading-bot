from __future__ import annotations

from uuid import uuid4

import pandas as pd

from src.models.risk import AccountState
from src.models.signals import (
    Signal,
    SignalSide,
    SignalType,
)
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager


class StrategyEngine:
    def __init__(
        self,
        config,
        risk_manager: RiskManager,
        session_manager: SessionManager,
    ):
        self.config = config
        self.risk_manager = risk_manager
        self.session_manager = session_manager
        self.history: dict[str, list] = {}

    def evaluate_entry(
        self,
        candles_or_symbol,
        account: AccountState,
        timestamp,
    ) -> Signal | None:

        if isinstance(
            candles_or_symbol,
            str,
        ):
            symbol = candles_or_symbol
            candles = self.history.get(
                symbol,
                [],
            )

        else:
            candles = candles_or_symbol

            if candles is None:
                return None

            if isinstance(
                candles,
                pd.DataFrame,
            ):
                if candles.empty:
                    return None

                symbol = str(
                    candles.iloc[-1]["symbol"]
                )

            else:
                candles = list(candles)

                if not candles:
                    return None

                symbol = str(
                    candles[-1]["symbol"]
                )

        dataframe = pd.DataFrame(
            candles
        )

        if dataframe.empty:
            return None

        dataframe = (
            self._calculate_indicators(
                dataframe
            )
        )

        if len(dataframe) < 2:
            return None

        row = dataframe.iloc[-1]

        if not self.session_manager.is_entry_allowed(
            timestamp
        ):
            return None

        side, conditions = self.entry_direction(row)

        if side is None:
            return None

        return self._build_entry(
            symbol=symbol,
            side=side,
            row=row,
            account=account,
            timestamp=timestamp,
            conditions=conditions,
        )

    def indicator_frame(
        self,
        candles,
    ) -> pd.DataFrame:
        """Indicator columns for a candle list/DataFrame (no session or
        risk logic). Shared by the equity and option signal paths."""

        if isinstance(candles, (list, tuple)) and candles and hasattr(candles[0], "model_dump"):
            candles = [c.model_dump() for c in candles]
        dataframe = pd.DataFrame(candles)

        if dataframe.empty:
            return dataframe

        return self._calculate_indicators(dataframe)

    def entry_conditions(
        self,
        row,
    ) -> dict[str, dict[str, bool] | None]:
        """Raw long/short entry conditions for one indicator row.
        A disabled side is reported as None."""

        long_conditions = None
        short_conditions = None

        if self.config.entry["long"].enabled:

            long_conditions = {
                "ema_trend": (
                    row["ema_fast"]
                    > row["ema_slow"]
                ),
                "vwap": (
                    row["close"]
                    > row["vwap"]
                ),
                "rsi": (
                    row["rsi"]
                    >= self.config.entry[
                        "long"
                    ].rsi_min
                ),
                "breakout": (
                    row["close"]
                    > row["previous_high"]
                ),
                "volume": (
                    row["volume"]
                    >= (
                        row["volume_sma"]
                        * self.config.entry[
                            "long"
                        ].volume_multiplier
                    )
                ),
            }

        if self.config.entry["short"].enabled:

            short_conditions = {
                "ema_trend": (
                    row["ema_fast"]
                    < row["ema_slow"]
                ),
                "vwap": (
                    row["close"]
                    < row["vwap"]
                ),
                "rsi": (
                    row["rsi"]
                    <= self.config.entry[
                        "short"
                    ].rsi_max
                ),
                "breakout": (
                    row["close"]
                    < row["previous_low"]
                ),
                "volume": (
                    row["volume"]
                    >= (
                        row["volume_sma"]
                        * self.config.entry[
                            "short"
                        ].volume_multiplier
                    )
                ),
            }

        return {
            "long": long_conditions,
            "short": short_conditions,
        }

    def entry_direction(
        self,
        row,
    ) -> tuple[SignalSide | None, dict[str, bool]]:
        """BUY when every long condition holds, else SELL when every short
        condition holds (long is checked first), else no direction."""

        conditions = self.entry_conditions(row)

        if conditions["long"] is not None and all(
            conditions["long"].values()
        ):
            return SignalSide.BUY, conditions["long"]

        if conditions["short"] is not None and all(
            conditions["short"].values()
        ):
            return SignalSide.SELL, conditions["short"]

        return None, {}

    def add_candle(
        self,
        candle,
    ) -> None:

        symbol = candle.symbol

        if symbol not in self.history:
            self.history[symbol] = []

        self.history[symbol].append(
            {
                "timestamp": candle.timestamp,
                "symbol": candle.symbol,
                "open": float(
                    candle.open
                ),
                "high": float(
                    candle.high
                ),
                "low": float(
                    candle.low
                ),
                "close": float(
                    candle.close
                ),
                "volume": float(
                    candle.volume
                ),
            }
        )

        max_history = 500

        if len(
            self.history[symbol]
        ) > max_history:
            self.history[symbol] = (
                self.history[symbol][
                    -max_history:
                ]
            )

    def _calculate_indicators(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()
        ind = self.config.indicators

        # EMA fast
        df["ema_fast"] = (
            df["close"]
            .ewm(
                span=ind.ema_fast,
                adjust=False,
            )
            .mean()
        )

        # EMA slow
        df["ema_slow"] = (
            df["close"]
            .ewm(
                span=ind.ema_slow,
                adjust=False,
            )
            .mean()
        )

        # VWAP
        typical_price = (
            df["high"]
            + df["low"]
            + df["close"]
        ) / 3.0

        cumulative_volume = (
            df["volume"].cumsum()
        )

        cumulative_value = (
            typical_price
            * df["volume"]
        ).cumsum()

        df["vwap"] = (
            cumulative_value
            / cumulative_volume.replace(
                0,
                float("nan"),
            )
        )

        # RSI
        delta = df["close"].diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        average_gain = (
            gain.rolling(ind.rsi_period)
            .mean()
        )

        average_loss = (
            loss.rolling(ind.rsi_period)
            .mean()
        )

        rs = (
            average_gain
            / average_loss.replace(
                0,
                float("nan"),
            )
        )

        df["rsi"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

        df.loc[
            (average_loss == 0)
            & (average_gain > 0),
            "rsi",
        ] = 100.0

        # ATR
        previous_close = (
            df["close"].shift(1)
        )

        true_range = pd.concat(
            [
                df["high"]
                - df["low"],

                (
                    df["high"]
                    - previous_close
                ).abs(),

                (
                    df["low"]
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["atr"] = (
            true_range
            .rolling(ind.atr_period)
            .mean()
        )

        # Volume SMA
        df["volume_sma"] = (
            df["volume"]
            .rolling(ind.volume_sma_period)
            .mean()
        )

        # Previous breakout high
        df["previous_high"] = (
            df["high"]
            .rolling(ind.breakout_period)
            .max()
            .shift(1)
        )

        # Previous breakout low
        df["previous_low"] = (
            df["low"]
            .rolling(ind.breakout_period)
            .min()
            .shift(1)
        )

        return df

    def _build_entry(
        self,
        symbol,
        side,
        row,
        account,
        timestamp,
        conditions,
    ):

        entry = float(
            row["close"]
        )

        atr_value = float(
            row["atr"]
        )

        if atr_value <= 0:
            return None

        multiplier = float(
            self.config.exit.stop_loss.multiplier
        )

        if side == SignalSide.BUY:
            stop = (
                entry
                - multiplier * atr_value
            )
        else:
            stop = (
                entry
                + multiplier * atr_value
            )

        risk_distance = abs(
            entry - stop
        )

        if risk_distance <= 0:
            return None

        risk_reward = float(
            self.config.exit.target.risk_reward
        )

        target_distance = (
            risk_distance
            * risk_reward
        )

        if side == SignalSide.BUY:
            target = (
                entry
                + target_distance
            )
        else:
            target = (
                entry
                - target_distance
            )

        quantity, _ = (
            self.risk_manager.calculate_quantity(
                account=account,
                entry_price=entry,
                stop_loss=stop,
            )
        )

        # IMPORTANT:
        # Exposure is based on actual order quantity.
        proposed_exposure = (
            entry * quantity
        )

        risk_result = (
            self.risk_manager.validate_entry(
                account=account,
                entry_price=entry,
                stop_loss=stop,
                proposed_exposure=(
                    proposed_exposure
                ),
                session_allowed=(
                    self.session_manager
                    .is_entry_allowed(
                        timestamp
                    )
                ),
                quantity=quantity,
            )
        )

        return Signal(
            signal_id=str(
                uuid4()
            ),
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            signal_type=SignalType.ENTRY,
            entry_price=entry,
            quantity=risk_result.quantity,
            stop_loss=stop,
            target=target,
            trailing_stop=None,
            reason=(
                "ALL_ENTRY_CONDITIONS_MET"
                if risk_result.allowed
                else "ENTRY_BLOCKED_BY_RISK"
            ),
            confidence_metadata={
                name: bool(value)
                for name, value in conditions.items()
            },
            risk_status=(
                "ALLOWED"
                if risk_result.allowed
                else "BLOCKED"
            ),
            blocked=(
                not risk_result.allowed
            ),
            block_reason=(
                ";".join(
                    risk_result.reasons
                )
                if risk_result.reasons
                else None
            ),
        )

    def evaluate_exit(
        self,
        position,
        current_price: float,
        atr_value: float,
        timestamp,
    ) -> Signal | None:

        if self.session_manager.is_force_exit_window(
            timestamp
        ):
            return Signal(
                signal_id=str(
                    uuid4()
                ),
                timestamp=timestamp,
                symbol=position.symbol,
                side=(
                    SignalSide.SELL
                    if position.side == "LONG"
                    else SignalSide.BUY
                ),
                signal_type=SignalType.EXIT,
                entry_price=current_price,
                quantity=position.quantity,
                reason="FORCE_EXIT",
                risk_status="ALLOWED",
            )

        if position.side == "LONG":

            if (
                current_price
                <= position.stop_loss
            ):
                reason = "STOP_LOSS"

            elif (
                current_price
                >= position.target
            ):
                reason = "TARGET"

            else:
                return None

            exit_side = SignalSide.SELL

        else:

            if (
                current_price
                >= position.stop_loss
            ):
                reason = "STOP_LOSS"

            elif (
                current_price
                <= position.target
            ):
                reason = "TARGET"

            else:
                return None

            exit_side = SignalSide.BUY

        return Signal(
            signal_id=str(
                uuid4()
            ),
            timestamp=timestamp,
            symbol=position.symbol,
            side=exit_side,
            signal_type=SignalType.EXIT,
            entry_price=current_price,
            quantity=position.quantity,
            reason=reason,
            risk_status="ALLOWED",
        )