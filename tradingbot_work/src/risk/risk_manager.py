from __future__ import annotations

import math

from src.models.risk import AccountState, OptionRiskResult, OptionSizing, RiskValidationResult
from src.risk.position_sizer import calculate_position_size


class RiskManager:
    def __init__(self, config):
        self.config = config

    def daily_loss_limit(self, account: AccountState) -> float:
        """Absolute INR cap when configured; otherwise equity percent fallback."""
        inr = getattr(self.config, "max_daily_loss_inr", None)
        if inr is not None and float(inr) > 0:
            return float(inr)
        return account.equity * self.config.max_daily_loss_percent / 100

    def remaining_daily_loss_budget(self, account: AccountState) -> float:
        """Realized P&L headroom before the daily loss cap is reached."""
        return max(0.0, self.daily_loss_limit(account) + account.daily_pnl)

    def gap_loss_buffer_multiplier(self) -> float:
        """Conservative sizing multiplier for gap/slippage beyond the stop."""
        pct = float(getattr(self.config, "max_loss_gap_buffer_percent", 0.0) or 0.0)
        return 1.0 + pct / 100.0

    def calculate_quantity(
        self,
        account: AccountState,
        entry_price: float,
        stop_loss: float,
    ) -> tuple[int, float]:
        quantity, risk_amount = calculate_position_size(
            account.equity,
            self.config.risk_per_trade_percent,
            entry_price,
            stop_loss,
        )

        exposure_headroom = (
            account.equity
            * self.config.max_portfolio_exposure_percent
            / 100
            - account.current_exposure
        )

        max_quantity = max(
            0,
            math.floor(exposure_headroom / entry_price),
        )

        return min(quantity, max_quantity), risk_amount

    def validate_entry(
        self,
        account: AccountState,
        entry_price: float,
        stop_loss: float,
        proposed_exposure: float | None,
        session_allowed: bool,
        quantity: int | None = None,
    ) -> RiskValidationResult:
        reasons: list[str] = []

        if entry_price <= 0:
            reasons.append("INVALID_ENTRY_PRICE")

        if stop_loss <= 0:
            reasons.append("INVALID_STOP_LOSS")

        if entry_price <= 0 or stop_loss <= 0:
            return RiskValidationResult(
                allowed=False,
                reasons=reasons,
                risk_amount=0.0,
                estimated_exposure=0.0,
                quantity=0,
            )

        calculated_quantity, risk_amount = (
            self.calculate_quantity(
                account=account,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )
        )

        final_quantity = (
            calculated_quantity
            if quantity is None
            else quantity
        )

        if final_quantity <= 0:
            reasons.append(
                "CALCULATED_QUANTITY_IS_ZERO"
            )

        actual_exposure = (
            entry_price * final_quantity
            if final_quantity > 0
            else 0.0
        )

        if (
            proposed_exposure is not None
            and final_quantity > 0
        ):
            tolerance = max(
                0.01,
                actual_exposure * 0.000001,
            )

            if abs(
                proposed_exposure - actual_exposure
            ) > tolerance:
                reasons.append(
                    "EXPOSURE_CALCULATION_MISMATCH"
                )

        if not session_allowed:
            reasons.append(
                "ENTRY_OUTSIDE_TRADING_SESSION"
            )

        if (
            account.open_positions
            >= self.config.max_open_positions
        ):
            reasons.append(
                "MAX_OPEN_POSITIONS_REACHED"
            )

        if (
            account.trades_today
            >= self.config.max_trades_per_day
        ):
            reasons.append(
                "MAX_TRADES_PER_DAY_REACHED"
            )

        daily_loss_limit = self.daily_loss_limit(account)

        if account.daily_pnl <= -daily_loss_limit:
            reasons.append(
                "DAILY_LOSS_LIMIT_REACHED"
            )

        exposure_limit = (
            account.equity
            * self.config.max_portfolio_exposure_percent
            / 100
        )

        total_exposure = (
            account.current_exposure
            + actual_exposure
        )

        if total_exposure > exposure_limit:
            reasons.append(
                "MAX_PORTFOLIO_EXPOSURE_REACHED"
            )

        return RiskValidationResult(
            allowed=not reasons,
            reasons=reasons,
            risk_amount=risk_amount,
            estimated_exposure=actual_exposure,
            quantity=final_quantity,
        )

    # ------------------------------------------------------------------
    # Options (BUY CE / BUY PE). Equity sizing above is unchanged.
    # ------------------------------------------------------------------

    def calculate_option_quantity(
        self,
        account: AccountState,
        entry_premium: float,
        stop_premium: float,
        lot_size: int,
        freeze_quantity: int | None = None,
        max_premium_exposure_pct: float | None = None,
        *,
        open_position_max_loss: float = 0.0,
    ) -> OptionSizing:
        """Risk budget / (entry - stop) units, rounded DOWN to whole lots and
        capped by premium exposure, portfolio exposure, freeze quantity and the
        remaining daily INR loss budget."""
        if lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if entry_premium <= 0 or stop_premium <= 0:
            raise ValueError("option premiums must be > 0")
        risk_per_unit = entry_premium - stop_premium
        if risk_per_unit <= 0:
            raise ValueError("BUY option stop must be below the entry premium")

        per_trade_budget = account.equity * self.config.risk_per_trade_percent / 100
        remaining_daily = max(
            0.0,
            self.remaining_daily_loss_budget(account) - max(0.0, open_position_max_loss),
        )
        risk_budget = min(per_trade_budget, remaining_daily)
        gap_mult = self.gap_loss_buffer_multiplier()
        effective_risk_per_unit = risk_per_unit * gap_mult
        raw_quantity = max(0, math.floor(risk_budget / effective_risk_per_unit + 1e-9))

        caps: dict[str, int] = {"RISK": raw_quantity}
        by_premium = None
        if max_premium_exposure_pct is not None:
            by_premium = max(0, math.floor(account.equity * max_premium_exposure_pct / 100 / entry_premium + 1e-9))
            caps["PREMIUM_EXPOSURE"] = by_premium
        headroom = account.equity * self.config.max_portfolio_exposure_percent / 100 - account.current_exposure
        by_exposure = max(0, math.floor(headroom / entry_premium + 1e-9))
        caps["PORTFOLIO_EXPOSURE"] = by_exposure
        by_freeze = None
        if freeze_quantity:
            by_freeze = int(freeze_quantity)
            caps["FREEZE_QUANTITY"] = by_freeze

        limiting = min(caps, key=lambda k: (caps[k], k != "RISK"))
        lots = caps[limiting] // lot_size
        quantity = lots * lot_size
        return OptionSizing(
            risk_budget=risk_budget,
            risk_per_unit=risk_per_unit,
            raw_quantity=raw_quantity,
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
            freeze_quantity=freeze_quantity,
            max_quantity_by_risk=raw_quantity,
            max_quantity_by_premium=by_premium,
            max_quantity_by_exposure=by_exposure,
            max_quantity_by_freeze=by_freeze,
            limiting_factor=limiting if lots > 0 else "LOT_SIZE",
            max_loss=quantity * risk_per_unit,
            premium_exposure=quantity * entry_premium,
        )

    def validate_option_entry(
        self,
        account: AccountState,
        entry_premium: float,
        stop_premium: float,
        target_premium: float,
        quantity: int,
        lot_size: int,
        session_allowed: bool,
        freeze_quantity: int | None = None,
        max_premium_exposure_pct: float | None = None,
        open_option_positions: int = 0,
        max_option_positions: int | None = None,
        *,
        open_position_max_loss: float = 0.0,
    ) -> OptionRiskResult:
        """Account-level checks from validate_entry plus option rules:
        lot multiple, freeze quantity, premium exposure, option position
        count and the daily-loss impact of a full stop-out."""
        reasons: list[str] = []
        if entry_premium <= 0 or stop_premium <= 0 or stop_premium >= entry_premium:
            reasons.append("INVALID_OPTION_STOP")
        if target_premium <= entry_premium:
            reasons.append("INVALID_OPTION_TARGET")
        if lot_size <= 0:
            reasons.append("INVALID_LOT_SIZE")
        if quantity <= 0:
            reasons.append("QUANTITY_BELOW_ONE_LOT")
        elif lot_size > 0 and quantity % lot_size:
            reasons.append("QUANTITY_NOT_LOT_MULTIPLE")
        if freeze_quantity and quantity > freeze_quantity:
            reasons.append("FREEZE_QUANTITY_EXCEEDED")

        premium_exposure = max(0.0, entry_premium) * max(0, quantity)
        if max_premium_exposure_pct is not None and premium_exposure > account.equity * max_premium_exposure_pct / 100:
            reasons.append("MAX_PREMIUM_EXPOSURE_REACHED")
        if max_option_positions is not None and open_option_positions >= max_option_positions:
            reasons.append("MAX_OPTION_POSITIONS_REACHED")

        risk_per_unit = max(0.0, entry_premium - stop_premium)
        max_loss = risk_per_unit * max(0, quantity)
        worst_case_loss = max_loss * self.gap_loss_buffer_multiplier()
        daily_limit = self.daily_loss_limit(account)
        after_stop = account.daily_pnl - max(0.0, open_position_max_loss) - worst_case_loss
        if quantity > 0 and account.daily_pnl > -daily_limit and after_stop < -daily_limit:
            reasons.append("DAILY_LOSS_LIMIT_WOULD_BE_EXCEEDED")
        if account.daily_pnl <= -daily_limit:
            reasons.append("DAILY_LOSS_LIMIT_REACHED")

        if 0 < stop_premium < entry_premium and quantity > 0:
            base = self.validate_entry(
                account=account,
                entry_price=entry_premium,
                stop_loss=stop_premium,
                proposed_exposure=premium_exposure,
                session_allowed=session_allowed,
                quantity=quantity,
            )
            reasons.extend(r for r in base.reasons if r not in reasons)
        elif not session_allowed:
            reasons.append("ENTRY_OUTSIDE_TRADING_SESSION")

        return OptionRiskResult(
            allowed=not reasons,
            reasons=reasons,
            quantity=quantity,
            lots=quantity // lot_size if lot_size > 0 else 0,
            lot_size=lot_size,
            risk_per_unit=risk_per_unit,
            max_loss=max_loss,
            premium_exposure=premium_exposure,
            position_notional=premium_exposure,
            daily_loss_limit=daily_limit,
            daily_loss_after_stop=after_stop,
            open_option_positions=open_option_positions,
            max_option_positions=max_option_positions,
        )
