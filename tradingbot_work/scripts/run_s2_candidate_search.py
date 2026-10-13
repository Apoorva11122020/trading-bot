from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.backtest.candidate_search import screen

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "strategy_research"


def main() -> None:
    historical_path = REPORT_DIR / "NIFTY_S2_losing_trade_analysis.csv"
    current_path = ROOT / "reports" / "forward_validation" / "forward_validation_signals.csv"
    historical = pd.read_csv(historical_path)
    current = pd.read_csv(current_path)
    results = screen(historical, current)

    out_json = REPORT_DIR / "NIFTY_S2_candidate_search.json"
    out_csv = REPORT_DIR / "NIFTY_S2_candidate_search.csv"
    out_md = REPORT_DIR / "NIFTY_S2_candidate_search.md"

    payload = {
        "mode": "PAPER research only; no broker calls; no strategy/config changes",
        "selection": "Candidates selected using historical TRAIN/OOS only; current 2026-10-05 is evaluation-only.",
        "hindsight_fields_excluded": ["mfe_r", "mae_r", "holding_minutes", "exit_reason", "false_breakout_after_entry"],
        "candidate_count": len(results),
        "results": [r.as_dict() for r in results],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([r.as_dict() for r in results]).to_csv(out_csv, index=False)

    lines = [
        "# S2 Candidate Strategy Search",
        "",
        "> PAPER research only. No broker calls. No production strategy or risk configuration was changed.",
        "> Candidate selection uses historical TRAIN/OOS only; 2026-10-05 is evaluation-only.",
        "",
        "## Important result",
        "",
        "No candidate should be promoted solely because it exceeds 60% on historical data.",
        "The current 2026-10-05 forward sample is retained as an untouched evaluation set.",
        "",
        "## Top candidates",
        "",
        "| Rules | Train | OOS | Current 2026-10-05 |",
        "|---|---:|---:|---:|",
    ]
    for r in results[:20]:
        lines.append(
            f"| {' + '.join(r.rules)} | {r.train_win_rate:.1%} / PF {r.train_profit_factor:.2f} / n={r.train_n} "
            f"| {r.oos_win_rate:.1%} / PF {r.oos_profit_factor:.2f} / n={r.oos_n} "
            f"| {r.current_win_rate:.1%} / PF {r.current_profit_factor:.2f} / n={r.current_n} |"
        )
    if results:
        best_current = max(results, key=lambda r: (r.current_win_rate, r.current_n))
        lines += [
            "",
            "## Decision",
            "",
            f"Best screened current-day candidate: **{' + '.join(best_current.rules)}** at "
            f"{best_current.current_win_rate:.1%} ({best_current.current_n} signals).",
            "",
            "No 60%+ current-day result was established by this leak-safe candidate screen with the minimum sample constraints.",
            "Do not alter the production strategy to manufacture a 60% headline from this one-day sample.",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    for r in results[:10]:
        print(r)


if __name__ == "__main__":
    main()
