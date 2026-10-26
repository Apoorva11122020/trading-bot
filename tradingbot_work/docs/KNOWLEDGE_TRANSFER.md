# Knowledge Transfer — Doc Pivot Strategy, Backtest UI & Live Paper

Last updated: 2026-10-26  
Repo: https://github.com/Apoorva11122020/trading-bot  
Work directory: `tradingbot_work/`

---

## 1. What this project does

Paper-only trading research for **NIFTY / BANKNIFTY** using:

1. **Classic pivot strategy** (from the client strategy document)
2. **NIFTY PE recovery rule** (optional leg: morning −200 pts → cross open−70 → buy PE)
3. **Web backtest UI** (historical runs, filter ranking, strategy comparison)
4. **Live paper tracker** (real-time Upstox ticks, no broker orders)

**Nothing places live orders.** `TRADING_MODE=PAPER` and `LIVE_TRADING_ENABLED=false` must stay that way until a formal go-live process exists.

---

## 2. Doc pure vs Doc optimized (critical)

| | **Doc pure** (`doc_pure`) | **Doc optimized** (`doc_optimized`) |
|---|---------------------------|-------------------------------------|
| Purpose | Show client the **original document** rules | **Research-tuned** version for paper / demo |
| Pivot rules | Classic pivot + EMA + RSI (guide baseline) | Same engine |
| Filters | None | NIFTY short **11:00–13:00**, **1 trade/day** |
| RR / ATR | 1:1.5, ATR 0.10 (doc default) | 1:2, ATR **0.08** |
| Symbols | NIFTY + BANKNIFTY | **NIFTY only** |
| 6-month backtest (Apr–Oct 2026) | Large **loss** (~−5000+ pts) | Small **profit** (~**+27 pts**, ~22 trades) |
| 3-month only (Jul–Oct 2026) | Loss | Often still **negative** |

**Why two versions?**  
The document describes a valid **framework** (pivot levels, trend, RSI, stop/target). Run **as-written** on spot data with realistic costs → net loss in backtest. **Optimized** adds filters found by research on the same rules — not a different strategy.

**Client messaging:**  
> “We implemented your document exactly (`doc pure`). Backtest showed it needs refinement. We tuned filters on the same rules (`doc optimized`). We are now forward-testing on live paper.”

---

## 3. Best strategy to use

**Default for demo, live paper, and paper tracking:**

- Strategy pack: **`doc_optimized`**
- Symbol: **NIFTY**
- Filter: `short_nifty_11_13+one_per_day`
- RR: **2.0**, ATR buffer: **0.08**
- Backtest range for client demo: **6 months** (not 3 — recent 3 months can show all strategies red)

Registry: `scripts/strategy_catalog.py`  
Ranking: `scripts/multi_strategy_runner.py`, UI **Compare all strategies**

---

## 4. Running the backtest UI

```powershell
cd tradingbot_work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\run_backtest_ui.py --force
```

Open: **http://127.0.0.1:8765** → hard refresh **Ctrl+Shift+R**

### If UI hangs or APIs timeout

Port open but frozen is common. Always:

```powershell
python scripts\run_backtest_ui.py --force
```

### UI features

| Feature | What it does |
|---------|----------------|
| Strategy pack dropdown | Select `doc_pure`, `doc_optimized`, alt strategies, etc. |
| Run backtest | Full report + CSV under `reports/ui_runs/` |
| Compare all strategies | Ranks all packs (slow: 5–15 min first time) |
| Compare doc filters | Ranks 15 guide filters on pivot only |
| Live paper panel | Start/stop today’s paper tracker from browser |

### API endpoints (v1.4)

- `GET /api/health`
- `GET /api/strategies`
- `GET /api/recommended-strategy`
- `POST /api/backtest`
- `POST /api/compare-strategies`
- `POST /api/compare-filters`
- `GET /api/live/status`, `GET /api/live/events`
- `POST /api/live/start`, `POST /api/live/stop`

---

## 5. Live paper tracker

### From UI (recommended for client screen-share)

1. Select strategy **`doc_optimized`**
2. Click **Start live paper** (Mon–Fri **09:00–15:15 IST** only)
3. Events appear when `PAPER_ENTRY` / `PAPER_EXIT` / `PE_ENTRY` fire — **not every tick**

### From terminal

```powershell
cd tradingbot_work
python scripts\run_daily_pivot_paper.py --symbol NIFTY --strategy-id doc_optimized
```

### Why live shows “nothing” often

- **Normal.** Optimized allows at most **one NIFTY short** between **11:00–13:00** when pivot + EMA + RSI align.
- Ticks and `feed=CONNECTED` in `reports/daily_pivot_paper/live_runner.log` mean it is working.
- **Zero trades today ≠ broken.**

### Output files

```
reports/daily_pivot_paper/
  YYYY-MM-DD_NIFTY_paper.csv    # today's events
  daily_summary.csv             # per-day summary
  live_runner.log               # subprocess log when started from UI
```

Forward review after ~20 sessions:

```powershell
python scripts\summarize_pivot_paper_forward.py
```

---

## 6. Client demo script (15 minutes)

1. **6-month range** → `doc_pure` → Run backtest → show loss (honest baseline)
2. Same range → `doc_optimized` → Run → show small positive net + exit breakdown
3. **Compare all strategies** (6 month) → only optimized profitable among doc packs
4. Optional: **Start live paper** → “real market, no orders, selective signals”

**Do not promise:** daily trades, guaranteed profit, or that 3-month backtest will be green.

**Do say:** paper only, rules are transparent, forward validation in progress.

---

## 7. Environment & secrets

Copy `tradingbot_work/.env` manually — **never commit** (in `.gitignore`).

Minimum for backtest + live paper:

```env
TRADING_MODE=PAPER
LIVE_TRADING_ENABLED=false
UPSTOX_HISTORICAL_ACCESS_TOKEN=<from Upstox>
UPSTOX_MARKET_DATA_ACCESS_TOKEN=<from Upstox>
```

Template: `tradingbot_work/.env.example`

Without tokens: historical backtest and live WebSocket will fail or return no data.

---

## 8. New machine setup

```powershell
git clone https://github.com/Apoorva11122020/trading-bot.git
cd trading-bot\tradingbot_work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Copy .env from old machine
python scripts\run_backtest_ui.py --force
```

Updates:

```powershell
cd trading-bot
git pull origin main
cd tradingbot_work
pip install -r requirements.txt
```

---

## 9. Key files map

| Path | Role |
|------|------|
| `backtest_ui/server.py` | FastAPI UI server |
| `backtest_ui/static/index.html` | Web UI |
| `scripts/run_backtest_ui.py` | UI launcher (`--force`) |
| `scripts/strategy_catalog.py` | All strategy packs |
| `scripts/doc_guide_filters.py` | Doc guide filters + combos |
| `scripts/combined_backtest_service.py` | Shared backtest runner |
| `scripts/multi_strategy_runner.py` | Strategy ranking |
| `scripts/guide_filter_ranking.py` | Filter ranking |
| `scripts/run_daily_pivot_paper.py` | Live paper (CLI + UI subprocess) |
| `scripts/live_paper_service.py` | UI live paper control |
| `scripts/classic_pivot_historical_backtest.py` | Doc guide pivot engine |
| `scripts/nifty_pe_recovery_rule.py` | PE recovery logic |
| `docs/DAILY_PIVOT_PAPER_TRACKER.md` | Paper tracker details |
| `tradingview/classic_pivot_alerts.pine` | TradingView alerts (visual only) |

Strategy document (Word): repo root `Nifty_BankNifty_Pivot_Point_Algo_Guide (3).docx`

---

## 10. CLI backtest (without UI)

```powershell
cd tradingbot_work
python scripts\run_combined_last7_report.py ^
  --guide-filter short_nifty_11_13+one_per_day ^
  --rr 2 --atr-buffer 0.08 ^
  --symbols NIFTY ^
  --start 2026-04-25 --end 2026-10-25
```

Reports: `reports/combined_last7_guide_pe/REPORT.md`

---

## 11. What is NOT in git

| Ignored | Why |
|---------|-----|
| `.env` | Secrets |
| `reports/` | Regenerated by scripts |
| `data/` | Downloaded market data |
| `__pycache__/` | Python cache |

---

## 12. Troubleshooting

| Problem | Fix |
|---------|-----|
| UI 404 on `/api/strategies` | Old server — `run_backtest_ui.py --force` |
| API timeout | Hung server — `--force` restart |
| Backtest empty / errors | Check `UPSTOX_HISTORICAL_ACCESS_TOKEN` |
| Live paper no ticks | Check `UPSTOX_MARKET_DATA_ACCESS_TOKEN`, market hours |
| Live paper no trades | Normal — wait 11:00–13:00 or no setup today |
| All strategies red (3 month) | Use **6-month** range for demo |
| `git push` SSH fail | `git remote set-url origin https://github.com/Apoorva11122020/trading-bot.git` |

---

## 13. Git workflow

```powershell
cd D:\trading-bot-shadow-comparison
git status
git add <files>
git commit -m "message"
git push origin main
```

Remote: `https://github.com/Apoorva11122020/trading-bot.git` (branch `main`)

---

## 14. Honest limitations (tell the client)

1. Backtest edge is **small** (+27 pts over 6 months ≈ ~22 trades on NIFTY spot points).
2. **Past performance ≠ future results.**
3. Spot index backtest ≠ futures/options rupee P&amp;L (PE leg is separate, INR).
4. Live paper must accumulate **~20 sessions** before any forward claim (`summarize_pivot_paper_forward.py`).
5. **Doc pure** remains the reference implementation of the written guide — not the recommended trading config.

---

## 15. Contacts & handoff checklist

- [ ] `.env` copied to new machine (tokens valid)
- [ ] UI starts on 8765
- [ ] 6-month `doc_optimized` backtest runs
- [ ] `doc_pure` backtest runs (shows loss — expected)
- [ ] Live paper connects during market hours
- [ ] Client demo script reviewed (Section 6)
- [ ] Git pull works on second machine

For code questions, start from `strategy_catalog.py` and `backtest_ui/server.py`.
