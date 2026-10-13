from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json
import math
from datetime import date
from pathlib import Path

from scripts.run_shadow_historical_validation import prepare_replay_inputs
from src.backtest import live_quality_gate as qg
from src.backtest import s2_direction_research as dr
from src.paper_trading import shadow_validation as sv
from src.safety.live_gate import assert_paper_trading_only

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live_quality_gate" / "historical_replay"
OUT.mkdir(parents=True, exist_ok=True)

FROM_DATE = date(2026, 4, 1)
TO_DATE = date(2026, 10, 1)

def pf(rows):
    gp = sum(float(r["pnl"]) for r in rows if float(r["pnl"]) > 0)
    gl = -sum(float(r["pnl"]) for r in rows if float(r["pnl"]) < 0)
    if gl > 0:
        return gp / gl
    return math.inf if gp > 0 else None

def pnl(row):
    if row.get("pnl") is not None:
        return float(row["pnl"])
    if row.get("exit_price") is not None and row.get("entry") is not None:
        return (float(row["exit_price"]) - float(row["entry"])) * 65
    if row.get("outcome") == "AMBIGUOUS" and row.get("stop_loss") is not None:
        return (float(row["stop_loss"]) - float(row["entry"])) * 65
    return 0.0

def metrics(rows):
    accepted = [r for r in rows if r.get("decision") == "ACCEPTED"]
    skipped = [r for r in rows if r.get("decision") != "ACCEPTED"]

    evaluated = [
        r for r in accepted
        if r.get("outcome") in ("WIN", "LOSS", "AMBIGUOUS", "SCRATCH")
    ]

    wins = sum(r.get("outcome") == "WIN" for r in evaluated)
    losses = sum(r.get("outcome") == "LOSS" for r in evaluated)
    pnls = [pnl(r) for r in evaluated]

    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)

    return {
        "signals": len(rows),
        "accepted": len(accepted),
        "skipped": len(skipped),
        "evaluated": len(evaluated),
        "wins": wins,
        "losses": losses,
        "accuracy": wins / len(evaluated) if evaluated else None,
        "pf": pf([dict(r, pnl=p) for r, p in zip(evaluated, pnls)]) if evaluated else None,
        "pnl": round(sum(pnls), 2),
        "max_dd": round(dd, 2),
    }

def main():
    assert_paper_trading_only()

    print("=" * 78)
    print("LIVE QUALITY GATE V1 | HISTORICAL REPLAY")
    print("PAPER ONLY | READ ONLY | NO ORDERS")
    print(f"Range: {FROM_DATE} .. {TO_DATE}")
    print("=" * 78)

    x = prepare_replay_inputs(
        from_date=FROM_DATE,
        to_date=TO_DATE,
        days=None,
        expired_expiries=None,
        underlying="NIFTY",
    )

    pipeline = __import__(
        "src.backtest.s2_1dte_improvement",
        fromlist=["_pipeline_factory"]
    )._pipeline_factory(
        x.signal_fn,
        x.resolve,
        x.params,
        x.s2["range_start"],
        x.s2["range_end"],
    )

    contexts, feats = pipeline(
        x.spot,
        x.premium_for,
        set(x.selected),
    )

    cache = {}

    baseline_rows = dr.run_candidate(
        contexts,
        feats,
        dr.BASELINE,
        square_off=x.rc.session.square_off,
        lot_size=65,
        cache=cache,
    )

    baseline_rows = [dict(r) for r in baseline_rows]

    config = qg.QualityGateConfig(
        enabled=True,
        min_score_buy=2,
        min_score_sell=2,
        target_accuracy=0.70,
        min_live_samples=30,
        min_accept_rate=0.05,
        max_skip_ratio=0.95,
        allow_partial_features=False,
        sell_rsi_range=(50.0, 60.0),
        buy_max_ema_sep_atr=-1.0,
        buy_max_break_atr=0.50,
    )

    gate_rows = sv.quality_gate_rows(
        contexts,
        baseline_rows,
        x.spot,
        config,
    )

    m = metrics(gate_rows)

    baseline_decided = [
        r for r in baseline_rows
        if r.get("outcome") in ("WIN", "LOSS", "AMBIGUOUS", "SCRATCH")
    ]

    gate_accepted = [
        r for r in gate_rows
        if r.get("decision") == "ACCEPTED"
    ]

    gate_skipped = [
        r for r in gate_rows
        if r.get("decision") != "ACCEPTED"
    ]

    accepted_decided = [
        r for r in gate_accepted
        if r.get("outcome") in ("WIN", "LOSS", "AMBIGUOUS", "SCRATCH")
    ]

    skipped_baseline_decided = []
    for base, gate in zip(baseline_rows, gate_rows):
        if gate.get("decision") != "ACCEPTED" and base.get("outcome") in (
            "WIN", "LOSS", "AMBIGUOUS", "SCRATCH"
        ):
            skipped_baseline_decided.append(base)

    # Counterfactual baseline result at gate-skipped times.
    skipped_pnls = [pnl(r) for r in skipped_baseline_decided]
    skipped_wins = sum(r.get("outcome") == "WIN" for r in skipped_baseline_decided)

    # Score breakdown.
    score_rows = {}
    for r in gate_rows:
        score = r.get("quality_gate", {}).get("score")
        score_rows.setdefault(score, []).append(r)

    print()
    print("HISTORICAL QUALITY GATE RESULT")
    print(f"Signals generated : {m['signals']}")
    print(f"Accepted          : {m['accepted']}")
    print(f"Skipped           : {m['skipped']}")
    print(f"Resolved accepted : {m['evaluated']}")
    print(f"Wins              : {m['wins']}")
    print(f"Losses            : {m['losses']}")
    print(
        f"Accuracy          : "
        f"{m['accuracy']:.1%}" if m["accuracy"] is not None else "Accuracy          : N/A"
    )
    print(
        f"Profit Factor     : "
        f"{m['pf']:.2f}" if m["pf"] not in (None, math.inf)
        else ("Profit Factor     : INF" if m["pf"] == math.inf else "Profit Factor     : N/A")
    )
    print(f"P&L               : Rs {m['pnl']:,.2f}")
    print(f"Max DD            : Rs {m['max_dd']:,.2f}")

    print()
    print("COUNTERFACTUAL CHECK")
    print(f"Skipped signals    : {len(gate_skipped)}")
    print(f"Skipped resolved   : {len(skipped_baseline_decided)}")
    if skipped_baseline_decided:
        print(f"Skipped wins       : {skipped_wins}")
        print(
            f"Skipped accuracy  : "
            f"{skipped_wins / len(skipped_baseline_decided):.1%}"
        )
        print(f"Skipped baseline P&L: Rs {sum(skipped_pnls):,.2f}")
    else:
        print("Skipped baseline outcome: N/A")

    print()
    print("SCORE BREAKDOWN")
    for score in sorted(score_rows, key=lambda x: (-1 if x is None else x)):
        rows = score_rows[score]
        accepted = sum(r.get("decision") == "ACCEPTED" for r in rows)
        print(
            f"Score {score}: generated={len(rows)} "
            f"accepted={accepted} skipped={len(rows)-accepted}"
        )

    # Day-by-day gate result.
    days = sorted({str(r["session_date"])[:10] for r in gate_rows})
    daily = []

    for d in days:
        drs = [r for r in gate_rows if str(r["session_date"])[:10] == d]
        dm = metrics(drs)
        daily.append({
            "date": d,
            **dm,
        })

    # Save detailed rows.
    csv_path = OUT / "historical_quality_gate_signals.csv"
    fields = [
        "session_date", "signal_time", "direction", "decision",
        "skip_reason", "gate_score", "gate_max_score",
        "gate_confidence", "gate_reasons", "gate_failed_checks",
        "outcome", "entry", "stop_loss", "target",
        "exit_price", "exit_time", "pnl", "r_multiple",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in gate_rows:
            g = r.get("quality_gate") or {}
            w.writerow({
                "session_date": r.get("session_date"),
                "signal_time": r.get("signal_time"),
                "direction": r.get("direction"),
                "decision": r.get("decision"),
                "skip_reason": r.get("skip_reason"),
                "gate_score": g.get("score"),
                "gate_max_score": g.get("max_score"),
                "gate_confidence": g.get("confidence"),
                "gate_reasons": json.dumps(g.get("reasons", [])),
                "gate_failed_checks": json.dumps(g.get("failed_checks", [])),
                "outcome": r.get("outcome"),
                "entry": r.get("entry"),
                "stop_loss": r.get("stop_loss"),
                "target": r.get("target"),
                "exit_price": r.get("exit_price"),
                "exit_time": r.get("exit_time"),
                "pnl": pnl(r) if r.get("outcome") in (
                    "WIN", "LOSS", "AMBIGUOUS", "SCRATCH"
                ) else None,
                "r_multiple": r.get("r_multiple"),
            })

    summary = {
        "mode": "PAPER",
        "read_only": True,
        "production_s2_changed": False,
        "live_orders": 0,
        "historical_range": f"{FROM_DATE}..{TO_DATE}",
        "gate_config": {
            "min_score_buy": 2,
            "min_score_sell": 2,
            "target_accuracy": 0.70,
            "min_live_samples": 30,
            "min_accept_rate": 0.05,
            "max_skip_ratio": 0.95,
            "sell_rsi_range": [50.0, 60.0],
            "buy_max_ema_sep_atr": -1.0,
            "buy_max_break_atr": 0.50,
        },
        "result": m,
        "counterfactual_skipped": {
            "signals": len(gate_skipped),
            "resolved": len(skipped_baseline_decided),
            "wins": skipped_wins,
            "accuracy": (
                skipped_wins / len(skipped_baseline_decided)
                if skipped_baseline_decided else None
            ),
            "baseline_pnl": round(sum(skipped_pnls), 2),
        },
        "daily": daily,
    }

    json_path = OUT / "historical_quality_gate_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    md_path = OUT / "historical_quality_gate_summary.md"
    lines = [
        "# LIVE QUALITY GATE V1 - HISTORICAL REPLAY",
        "",
        "PAPER ONLY / READ ONLY / NO ORDERS",
        "",
        f"- Range: {FROM_DATE} .. {TO_DATE}",
        "- Production S2 changed: NO",
        "- Live orders: 0",
        "",
        "## Result",
        "",
        f"- Signals generated: {m['signals']}",
        f"- Accepted: {m['accepted']}",
        f"- Skipped: {m['skipped']}",
        f"- Resolved accepted: {m['evaluated']}",
        f"- Wins: {m['wins']}",
        f"- Losses: {m['losses']}",
        f"- Accuracy: {m['accuracy']:.1%}" if m["accuracy"] is not None else "- Accuracy: N/A",
        f"- Profit factor: {m['pf']:.2f}" if m["pf"] not in (None, math.inf)
            else ("- Profit factor: INF" if m["pf"] == math.inf else "- Profit factor: N/A"),
        f"- P&L: Rs {m['pnl']:,.2f}",
        f"- Max DD: Rs {m['max_dd']:,.2f}",
        "",
        "## Counterfactual skipped signals",
        "",
        f"- Skipped signals: {len(gate_skipped)}",
        f"- Skipped resolved: {len(skipped_baseline_decided)}",
        f"- Skipped baseline wins: {skipped_wins}",
        f"- Skipped baseline accuracy: "
        f"{skipped_wins / len(skipped_baseline_decided):.1%}"
        if skipped_baseline_decided else "- Skipped baseline accuracy: N/A",
        f"- Skipped baseline P&L: Rs {sum(skipped_pnls):,.2f}",
        "",
        "## Day by day",
        "",
        "| Date | Signals | Accepted | Resolved | Wins | Losses | Accuracy | P&L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for d in daily:
        acc = "N/A" if d["accuracy"] is None else f"{d['accuracy']:.1%}"
        lines.append(
            f"| {d['date']} | {d['signals']} | {d['accepted']} | "
            f"{d['evaluated']} | {d['wins']} | {d['losses']} | "
            f"{acc} | Rs {d['pnl']:,.2f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This is retrospective historical validation only.",
        "It is not fresh forward/OOS evidence and does not prove a 70% live accuracy guarantee.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 78)
    print("REPORTS")
    print(csv_path)
    print(json_path)
    print(md_path)
    print("=" * 78)

if __name__ == "__main__":
    main()



