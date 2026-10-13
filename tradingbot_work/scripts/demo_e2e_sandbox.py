from __future__ import annotations

import os
import sys

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


load_dotenv(
    PROJECT_ROOT / ".env"
)


from src.broker.upstox_sandbox import (
    UpstoxSandboxBroker,
    UpstoxSandboxError,
)

from src.database import Database

from src.demo import sandbox_demo
from src.demo.sandbox_demo import (
    DemoStageError,
    execute_sandbox_order,
)

from src.models.market import Candle


DEMO_SYMBOL = os.getenv(
    "DEMO_STRATEGY_SYMBOL",
    "RELIANCE",
)


UPSTOX_INSTRUMENT = os.getenv(
    "UPSTOX_SANDBOX_INSTRUMENT",
    "NSE_EQ|INE669E01016",
)


DEMO_EQUITY = float(
    os.getenv(
        "DEMO_EQUITY",
        "100000",
    )
)


RUN_SANDBOX_ORDER = (
    os.getenv(
        "UPSTOX_RUN_SANDBOX_ORDER_TEST",
        "false",
    ).lower()
    == "true"
)


def build_demo_candles(
    count: int = 55,
) -> list[Candle]:
    return sandbox_demo.build_demo_candles(
        DEMO_SYMBOL,
        count,
    )


def connected_market_health():
    return sandbox_demo.connected_market_health(
        DEMO_SYMBOL
    )


def create_strategy_runtime():
    return sandbox_demo.create_strategy_runtime(
        PROJECT_ROOT
        / "config"
        / "strategy.yaml"
    )


def run_actual_strategy():

    print()
    print("=" * 70)
    print(
        "ACTUAL STRATEGY ENGINE"
    )
    print("=" * 70)

    print()

    print(
        f"Strategy symbol : "
        f"{DEMO_SYMBOL}"
    )

    print(
        f"Demo equity     : "
        f"{DEMO_EQUITY:.2f}"
    )

    print(
        "Timeframe       : 5m"
    )

    print(
        "Market data     : PAPER / DEMO"
    )

    candles = (
        build_demo_candles()
    )

    print(
        f"Demo candles    : "
        f"{len(candles)}"
    )

    print()

    print(
        "Feeding candles into StrategyRuntime..."
    )

    def on_signal(candle, _signal):

        print()

        print(
            f"["
            f"{candle.timestamp.strftime('%H:%M')}"
            f"] "
            "STRATEGY SIGNAL GENERATED"
        )

    outcome = sandbox_demo.evaluate_strategy(
        candles,
        equity=DEMO_EQUITY,
        config_path=(
            PROJECT_ROOT
            / "config"
            / "strategy.yaml"
        ),
        on_signal=on_signal,
    )

    signal = outcome.signal
    config = outcome.config
    risk_manager = outcome.risk_manager
    session_manager = outcome.session_manager

    if signal is None:

        print()

        print("-" * 70)

        print(
            "ACTUAL SIGNAL"
        )

        print("-" * 70)

        print()

        print(
            "SIGNAL: NONE"
        )

        return (
            None,
            config,
            risk_manager,
            session_manager,
        )

    print()

    print("-" * 70)

    print(
        "ACTUAL SIGNAL"
    )

    print("-" * 70)

    print(
        f"Signal ID       : "
        f"{signal.signal_id}"
    )

    print(
        f"Symbol          : "
        f"{signal.symbol}"
    )

    print(
        f"Signal type     : "
        f"{signal.signal_type.value}"
    )

    print(
        f"Side            : "
        f"{signal.side.value}"
    )

    print(
        f"Entry price     : "
        f"{signal.entry_price:.4f}"
    )

    print(
        f"Quantity        : "
        f"{signal.quantity}"
    )

    print(
        f"Stop loss       : "
        f"{signal.stop_loss}"
    )

    print(
        f"Target          : "
        f"{signal.target}"
    )

    print(
        f"Risk status     : "
        f"{signal.risk_status}"
    )

    print(
        f"Blocked         : "
        f"{signal.blocked}"
    )

    print(
        f"Reason          : "
        f"{signal.reason}"
    )

    if signal.block_reason:

        print(
            f"Block reason    : "
            f"{signal.block_reason}"
        )

    print()

    print(
        "CONFIDENCE / CONDITIONS"
    )

    for (
        name,
        value,
    ) in (
        signal.confidence_metadata
        or {}
    ).items():

        print(
            f"  {name:<15}: "
            f"{value}"
        )

    return (
        signal,
        config,
        risk_manager,
        session_manager,
    )


def _print_stage(signal):
    """Console output for each stage of the shared Sandbox execution flow."""

    def on_stage(stage, **data):

        if stage == "BROKER_CONNECTING":

            print()

            print(
                "Connecting to Upstox Sandbox..."
            )

        elif stage == "BROKER_CONNECTED":

            print(
                "Sandbox connection: OK"
            )

        elif stage == "ORDER_SUBMITTING":

            print()

            print("-" * 70)

            print(
                "FRESH RISK VALIDATION + "
                "ORDER SUBMISSION"
            )

            print("-" * 70)

            print()

            print(
                f"Broker instrument : "
                f"{UPSTOX_INSTRUMENT}"
            )

            print(
                f"Strategy symbol   : "
                f"{signal.symbol}"
            )

            print(
                f"Side              : "
                f"{signal.side.value}"
            )

            print(
                f"Quantity          : "
                f"{signal.quantity}"
            )

            print(
                f"Entry             : "
                f"{signal.entry_price:.4f}"
            )

            print(
                f"Stop loss         : "
                f"{signal.stop_loss}"
            )

            print(
                f"Target            : "
                f"{signal.target}"
            )

            print()

            print(
                "Submitting through "
                "OrderManager..."
            )

        elif stage == "ORDER_SUBMITTED":

            record = data["record"]

            print()

            print(
                "ORDER SUBMITTED AND VERIFIED"
            )

            print(
                f"Order ID         : "
                f"{record.order_id}"
            )

            print(
                f"Status           : "
                f"{record.status.value}"
            )

            print(
                f"Requested qty    : "
                f"{record.requested_quantity}"
            )

            print(
                f"Filled qty       : "
                f"{record.filled_quantity}"
            )

            print(
                f"Average fill     : "
                f"{record.average_fill_price}"
            )

        elif stage == "ORDER_VERIFYING":

            print()

            print(
                "Fetching broker order details..."
            )

        elif stage == "ORDER_VERIFIED":

            details = data["details"]

            print()

            print(
                "BROKER ORDER DETAILS"
            )

            print(
                f"Status           : "
                f"{details.status.value}"
            )

            print(
                f"Transaction      : "
                f"{details.side.value}"
            )

            print(
                f"Order type       : "
                f"{details.order_type.value}"
            )

            print(
                f"Quantity         : "
                f"{details.quantity}"
            )

            print(
                f"Price            : "
                f"{details.price}"
            )

        elif stage == "ORDER_CANCELLING":

            print()

            print(
                "Cancelling controlled "
                "Sandbox order..."
            )

        elif stage == "ORDER_CANCELLED":

            print()

            print(
                "ORDER CANCELLED SUCCESSFULLY"
            )

    return on_stage


def execute_with_order_manager(
    signal,
    config,
    risk_manager,
    session_manager,
):

    if signal is None:
        return None

    print()

    print("=" * 70)

    print(
        "ORDER MANAGER"
    )

    print("=" * 70)

    print()

    print(
        "Signal came from ACTUAL StrategyRuntime."
    )

    print(
        "Signal will pass through ACTUAL OrderManager."
    )

    if signal.blocked:

        print()

        print(
            "ORDER BLOCKED BY STRATEGY/RISK."
        )

        return None

    if (
        signal.signal_type.value
        != "ENTRY"
    ):

        print()

        print(
            "Signal is not an ENTRY."
        )

        return None

    if not RUN_SANDBOX_ORDER:

        print()

        print(
            "SANDBOX EXECUTION: DISABLED"
        )

        print()

        print(
            "Set this in .env when you want "
            "the controlled Sandbox test:"
        )

        print()

        print(
            "UPSTOX_RUN_SANDBOX_ORDER_TEST=true"
        )

        return None

    broker = UpstoxSandboxBroker()

    database = Database(
        os.getenv(
            "DATABASE_PATH",
            "data/trading.db",
        )
    )

    try:

        outcome = execute_sandbox_order(
            signal,
            broker=broker,
            risk_manager=risk_manager,
            session_manager=session_manager,
            database=database,
            instrument=UPSTOX_INSTRUMENT,
            equity=DEMO_EQUITY,
            max_slippage_percent=0.10,
            on_stage=_print_stage(signal),
        )

        return outcome["record"]

    except DemoStageError as exc:

        cause = exc.cause

        print()

        if isinstance(
            cause,
            UpstoxSandboxError,
        ):

            print(
                "UPSTOX SANDBOX ERROR:"
            )

            print(cause)

        else:

            print(
                "ORDER PIPELINE ERROR:"
            )

            print(
                f"{exc.code}: "
                f"{exc.message}"
            )

        if exc.cancel_required:

            print()

            print(
                "ORDER CANCELLATION REQUIRES ATTENTION"
            )

        return None

    except Exception as exc:

        print()

        print(
            "ORDER PIPELINE ERROR:"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None


def main():

    print("=" * 70)

    print(
        "AUTOMATED TRADING BOT - "
        "ACTUAL STRATEGY E2E DEMO"
    )

    print("=" * 70)

    print()

    print(
        "MODE: PAPER MARKET DATA + "
        "UPSTOX SANDBOX"
    )

    print(
        "LIVE TRADING: DISABLED"
    )

    (
        signal,
        config,
        risk_manager,
        session_manager,
    ) = run_actual_strategy()

    if signal is None:

        print()

        print("=" * 70)

        print(
            "DEMO FINISHED - NO SIGNAL"
        )

        print("=" * 70)

        return

    if signal.blocked:

        print()

        print("=" * 70)

        print(
            "DEMO STOPPED - SIGNAL BLOCKED"
        )

        print("=" * 70)

        return

    record = (
        execute_with_order_manager(
            signal=signal,
            config=config,
            risk_manager=risk_manager,
            session_manager=session_manager,
        )
    )

    print()

    print("=" * 70)

    print(
        "DEMO E2E COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Strategy signal : "
        f"{signal.side.value}"
    )

    print(
        f"Risk status     : "
        f"{signal.risk_status}"
    )

    if record is not None:

        print(
            f"Sandbox order   : "
            f"{record.order_id}"
        )

        print(
            f"Final status    : "
            f"{record.status.value}"
        )

        print()

        print(
            "Execution       : SUCCESS"
        )

    else:

        print(
            "Sandbox order   : "
            "NOT EXECUTED"
        )

    print()

    print(
        "Live trading was NOT used."
    )

    print(
        "This demo does NOT establish "
        "profitability or accuracy."
    )


if __name__ == "__main__":
    main()
