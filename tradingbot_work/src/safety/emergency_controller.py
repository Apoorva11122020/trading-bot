from dataclasses import dataclass


@dataclass
class EmergencyResult:
    activated: bool
    cancelled_order_ids: list[str]
    closed_symbols: list[str]
    errors: list[str]


class EmergencyController:
    """Coordinates kill-switch actions without owning broker logic."""

    def __init__(self, kill_switch, broker):
        self.kill_switch = kill_switch
        self.broker = broker

    def activate(
        self,
        reason: str,
        cancel_pending: bool = True,
        close_positions: bool = False,
    ) -> EmergencyResult:
        self.kill_switch.activate(reason)

        cancelled = []
        closed = []
        errors = []

        if cancel_pending:
            try:
                for raw in self.broker.get_orders():
                    status = raw["status"]
                    status_value = getattr(status, "value", status)
                    if status_value in {"OPEN", "PENDING", "MODIFIED", "PARTIALLY_FILLED"}:
                        try:
                            self.broker.cancel_order(str(raw["order_id"]))
                            cancelled.append(str(raw["order_id"]))
                        except Exception as exc:
                            errors.append(f"cancel:{raw['order_id']}:{exc}")
            except Exception as exc:
                errors.append(f"get_orders:{exc}")

        if close_positions:
            try:
                positions = self.broker.get_positions()
                for position in positions:
                    if int(position["quantity"]) == 0:
                        continue
                    symbol = str(position["symbol"])
                    try:
                        self.broker.close_position(symbol)
                        closed.append(symbol)
                    except Exception as exc:
                        errors.append(f"close:{symbol}:{exc}")
            except Exception as exc:
                errors.append(f"get_positions:{exc}")

        return EmergencyResult(
            activated=True,
            cancelled_order_ids=cancelled,
            closed_symbols=closed,
            errors=errors,
        )
