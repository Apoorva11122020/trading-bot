"""S13.2 - option-chain structure strategy research (PAPER | RESEARCH ONLY | NO ORDERS | rules only, no ML).

Separate strategy family. Asks whether point-in-time option-chain structure (ATM and ATM +/-1 only - 100% CE/PE
coverage in S13.1) carries a measurable edge. Six PRE-DECLARED rule families, one fixed parameter set each; nothing
is searched or tuned. Clean window only (2025-09-26 -> 2026-06-30): TRAIN 2025-09-26..2026-03-31 (family selection),
VALIDATION April 2026 (frozen), OOS May-June 2026 (frozen, only candidates that passed April). Files that could hold
candles after 2026-06-30 are never opened; October is never loaded. ATM +/-3 / +/-5 are never used; ATM +/-2 is
descriptive only (presence share) and never required.

Chain at signal candle t (completed): nearest listed expiry >= session date; ATM = strike-grid point nearest to the
NIFTY close of candle t (ties lower); ATM +/-1 = ATM +/- one strike step. Every chain value is read from the stored
candle of that exact contract at t, t-5, t-10, t-15 min; a missing candle makes the feature unavailable (never filled).
Entry = NEXT option candle OPEN; R = 1.5 x option ATR14 (completed candles <= t); STOP entry - 1R; TARGET entry + 1.5R;
6 candles; same-candle stop+target = AMBIGUOUS; neither = TIMEOUT (S10/S11 framework, unchanged). Gross, no costs.
Historical results are not live accuracy; no profit is guaranteed.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.historical_data import nifty_futures as nf
from src.research import s6_clean_strategy_lab as s6
from src.research import s10_ml_data as sd
from src.research import s12_2_constrained_ml as s122
from src.research import s13_chain_audit as s13a

HEADING = "S13.2 OPTION-CHAIN STRATEGY COMPLETE"
STRATEGY = "S13_CHAIN_STRATEGY"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | rule families only (no ML) | historical, not live accuracy"
IST = sd.IST
STEP = sd.STEP
TARGET_R, HORIZON = 1.5, sd.HORIZON
LABEL = {"entry": "next option candle OPEN", "risk": "R = 1.5 x option ATR14 (completed candles <= signal)",
         "stop": "entry - 1R", "target": "entry + 1.5R", "horizon": "6 option candles",
         "same_candle": "AMBIGUOUS (booked -1R)", "neither": "TIMEOUT (exit at horizon close)"}

CLEAN_START = date(2025, 9, 26)
TRAIN_END = date(2026, 3, 31)
VALIDATION = (date(2026, 4, 1), date(2026, 4, 30))
OOS = (date(2026, 5, 1), date(2026, 6, 30))
MAX_LOAD = date(2026, 6, 30)
SIDES = ("CE", "PE")

# Pre-declared rule parameters (fixed before any S13.2 result; never tuned).
RULES = {"MOM_MIN": 0.5,          # premium move over 3 bars in option-ATR units
         "VOL_IMB_MIN": 0.10,     # (CE vol - PE vol) / (CE vol + PE vol)
         "OI_CHG_IMB_MIN": 0.20,  # (CE OI chg - PE OI chg) / (|CE OI chg| + |PE OI chg|)
         "PCR_NEUTRAL": 1.0}
MIN_RESOLVED = 30
MAX_DD_R = 10.0
DEGRADE_SHARE = 0.5               # April expectancy below half of TRAIN expectancy -> DEGRADED
PASS, DEGRADED, REJECTED, INSUFF, NOT_RUN = "PASS", "DEGRADED", "REJECTED", "INSUFFICIENT_DATA", "NOT_RUN"
PROMISING = "PROMISING"

FUT_COLS = ("fut_ret_1", "fut_body_range", "fut_natr", "fut_close_ema20", "fut_close_ema50", "fut_ema_sep",
            "fut_vwap_dist", "fut_volume_ratio", "minutes_since_open")
FEATURES = ("ce_ret1", "ce_ret2", "ce_ret3", "pe_ret1", "pe_ret2", "pe_ret3", "ce_mom", "pe_mom", "ret_diff",
            "mom_diff", "strength_diff", "ce_vol", "pe_vol", "total_vol", "vol_ratio", "vol_imb", "ce_oi", "pe_oi",
            "oi_ratio", "oi_imb", "ce_oi_chg", "pe_oi_chg", "oi_chg_imb", "bull_breadth", "bear_breadth",
            "call_breadth", "put_breadth", "call_oi_conc", "put_oi_conc", "band_vol_imb", "band_oi_imb", "pcr_oi",
            "pcr_vol", "straddle", "straddle_ret1", "straddle_mom", "straddle_expanding", *FUT_COLS)


# ------------------------------------------------------------------ small numeric helpers (NaN when undefined)
def _f(v) -> float:
    return float(v) if v is not None and np.isfinite(v) else np.nan


def ratio(a: float, b: float) -> float:
    return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def imb(a: float, b: float) -> float:
    s = a + b
    return (a - b) / s if np.isfinite(s) and s > 0 else np.nan


def ret(now: float, past: float) -> float:
    return now / past - 1.0 if np.isfinite(now) and np.isfinite(past) and past > 0 else np.nan


# ------------------------------------------------------------------ point-in-time chain panel
class Contract:
    """Exact-timestamp lookups on ONE stored contract (no forward / backward filling)."""

    def __init__(self, premium: pd.DataFrame | None):
        p = sd._clean(premium)
        self.ok = p is not None
        if not self.ok:
            return
        t = list(p["timestamp"])
        self.close = dict(zip(t, p["close"].astype(float)))
        self.vol = dict(zip(t, p["volume"].astype(float))) if "volume" in p else {}
        self.oi = dict(zip(t, p["open_interest"].astype(float))) if "open_interest" in p else {}
        self.series = sd.option_series(p)
        self.atr = dict(zip(self.series.t, self.series.feat["atr14"].astype(float)))

    def c(self, ts) -> float:
        return _f(self.close.get(ts))

    def v(self, ts) -> float:
        x = _f(self.vol.get(ts))
        return x if x >= 0 else np.nan

    def o(self, ts) -> float:
        x = _f(self.oi.get(ts))
        return x if x >= 0 else np.nan


def strike_step(market: sd.MarketData) -> float | None:
    diffs: Counter = Counter()
    for strikes in market.chains.values():
        s = sorted(strikes)
        diffs.update(round(b - a, 6) for a, b in zip(s, s[1:]))
    return float(diffs.most_common(1)[0][0]) if diffs else None


def _chain_features(L: dict, ts) -> dict[str, float]:
    """L[(kind, k)] -> Contract for k in -1, 0, 1. Values at t, t-5, t-10, t-15 only."""
    t1, t2, t3 = ts - STEP, ts - 2 * STEP, ts - 3 * STEP
    ce, pe = L[("CE", 0)], L[("PE", 0)]
    out: dict[str, float] = {}
    for kind, x in (("ce", ce), ("pe", pe)):
        c0 = x.c(ts)
        out[f"{kind}_ret1"], out[f"{kind}_ret2"], out[f"{kind}_ret3"] = ret(c0, x.c(t1)), ret(c0, x.c(t2)), ret(c0, x.c(t3))
        a = _f(x.atr.get(ts))
        out[f"{kind}_mom"] = (c0 - x.c(t3)) / a if np.isfinite(a) and a > 0 and np.isfinite(x.c(t3)) else np.nan
        out[f"{kind}_vol"], out[f"{kind}_oi"] = x.v(ts), x.o(ts)
        prev_oi = x.o(t1)
        out[f"{kind}_oi_chg"] = out[f"{kind}_oi"] - prev_oi if np.isfinite(prev_oi) else np.nan
    out["ret_diff"] = out["ce_ret3"] - out["pe_ret3"]
    out["mom_diff"] = out["ce_mom"] - out["pe_mom"]
    out["strength_diff"] = math.log((1 + out["ce_ret3"]) / (1 + out["pe_ret3"])) \
        if np.isfinite(out["ret_diff"]) and out["ce_ret3"] > -1 and out["pe_ret3"] > -1 else np.nan
    out["total_vol"] = out["ce_vol"] + out["pe_vol"]
    out["vol_ratio"], out["vol_imb"] = ratio(out["ce_vol"], out["pe_vol"]), imb(out["ce_vol"], out["pe_vol"])
    out["oi_ratio"], out["oi_imb"] = ratio(out["ce_oi"], out["pe_oi"]), imb(out["ce_oi"], out["pe_oi"])
    a, b = out["ce_oi_chg"], out["pe_oi_chg"]
    den = abs(a) + abs(b)
    out["oi_chg_imb"] = (a - b) / den if np.isfinite(den) and den > 0 else np.nan
    r3 = {(kind, k): ret(L[(kind, k)].c(ts), L[(kind, k)].c(t3)) for kind in SIDES for k in (-1, 0, 1)}
    if all(np.isfinite(v) for v in r3.values()):
        up_ce = sum(r3[("CE", k)] > 0 for k in (-1, 0, 1))
        up_pe = sum(r3[("PE", k)] > 0 for k in (-1, 0, 1))
        out["bull_breadth"] = (up_ce + (3 - up_pe)) / 6.0
        out["bear_breadth"] = (up_pe + (3 - up_ce)) / 6.0
        out["call_breadth"], out["put_breadth"] = up_ce / 3.0, up_pe / 3.0
    else:
        out["bull_breadth"] = out["bear_breadth"] = out["call_breadth"] = out["put_breadth"] = np.nan
    vols = {kind: [L[(kind, k)].v(ts) for k in (-1, 0, 1)] for kind in SIDES}
    ois = {kind: [L[(kind, k)].o(ts) for k in (-1, 0, 1)] for kind in SIDES}
    sv = {kind: sum(v) if all(np.isfinite(v)) else np.nan for kind, v in vols.items()}
    so = {kind: sum(v) if all(np.isfinite(v)) else np.nan for kind, v in ois.items()}
    out["call_oi_conc"] = ratio(ois["CE"][1], so["CE"])
    out["put_oi_conc"] = ratio(ois["PE"][1], so["PE"])
    out["band_vol_imb"], out["band_oi_imb"] = imb(sv["CE"], sv["PE"]), imb(so["CE"], so["PE"])
    out["pcr_oi"], out["pcr_vol"] = ratio(so["PE"], so["CE"]), ratio(sv["PE"], sv["CE"])
    s0 = ce.c(ts) + pe.c(ts)
    s1, s3 = ce.c(t1) + pe.c(t1), ce.c(t3) + pe.c(t3)
    out["straddle"] = s0
    out["straddle_ret1"], out["straddle_mom"] = ret(s0, s1), ret(s0, s3)
    out["straddle_expanding"] = float(out["straddle_mom"] > 0) if np.isfinite(out["straddle_mom"]) else np.nan
    return out


def build_panel(market: sd.MarketData, days: Sequence[date], step: float,
                caches: dict | None = None) -> tuple[pd.DataFrame, Counter]:
    """One row per signal candle (09:45 .. last candle whose 6-candle horizon ends by 15:15) on ``days``."""
    caches = {"opt": {}, "fut": {}} if caches is None else caches
    want = [d for d in sorted(set(days)) if d in set(market.sessions)]
    sel = {date.fromisoformat(r["session_date"]): r for r in nf.select_contracts(market.contracts, market.frames, want)}
    spot, bars = market.spot_close(), market.day_bars()
    last = sd.last_signal(HORIZON)
    funnel: Counter = Counter()
    rows = []

    def contract(c) -> Contract:
        if c.instrument_key not in caches["opt"]:
            caches["opt"][c.instrument_key] = Contract(market.premium(c.instrument_key))
        return caches["opt"][c.instrument_key]

    for d in want:
        sts = [ts for ts in bars.get(d, []) if sd.SIGNAL_FIRST <= ts.time() <= last]
        fr = sel.get(d) or {}
        ff = None
        if fr.get("status") == nf.SELECTED:
            key = fr["instrument_key"]
            if key not in caches["fut"]:
                caches["fut"][key] = sd.futures_feature_frame(market.frames[key])
            ff = caches["fut"][key]
        expiry = sd.nearest_expiry(market.calendar, d)
        for ts in sts:
            rec: dict[str, Any] = {"signal_time": ts, "session_date": d.isoformat(), "eligible": False, "reason": None,
                                   "expiry": expiry.isoformat() if expiry else None, "spot": spot.get(ts)}
            rows.append(rec)
            if ff is None or ts not in ff.index:
                rec["reason"] = "FUTURES_CANDLE_UNAVAILABLE"
                continue
            rec.update({k: float(ff.at[ts, k]) for k in FUT_COLS})
            if expiry is None:
                rec["reason"] = "NO_LISTED_EXPIRY"
                continue
            atm = s13a.atm_of(float(spot[ts]), step)
            rec["atm"] = atm
            L, why = {}, None
            for kind in SIDES:
                chain = market.chains.get((d, expiry, kind)) or {}
                for k in (-1, 0, 1):
                    c = chain.get(atm + k * step)
                    if c is None:
                        why = "ATM_PM1_CONTRACT_NOT_STORED"
                        break
                    x = contract(c)
                    if not x.ok or not np.isfinite(x.c(ts)):
                        why = "OPTION_CANDLE_MISSING_AT_SIGNAL"
                        break
                    L[(kind, k)] = x
                    if k == 0:
                        rec[f"{kind}_key"], rec[f"{kind}_strike"] = c.instrument_key, float(c.strike)
                        rec[f"{kind}_contract"] = c.trading_symbol
                if why:
                    break
            pm2 = all(((market.chains.get((d, expiry, kind)) or {}).get(atm + k * step) is not None)
                      for kind in SIDES for k in (-2, 2))
            rec["pm2_listed_descriptive"] = float(pm2)
            if why:
                rec["reason"] = why
                continue
            rec.update(_chain_features(L, ts))
            atrs = {kind: _f(L[(kind, 0)].atr.get(ts)) for kind in SIDES}
            if not all(np.isfinite(a) and a > 0 for a in atrs.values()):
                rec["reason"] = "OPTION_ATR_UNAVAILABLE"
                continue
            if not (np.isfinite(rec["ce_mom"]) and np.isfinite(rec["pe_mom"])):
                rec["reason"] = "ATM_HISTORY_MISSING"
                continue
            for kind in SIDES:
                rec[f"{kind}_atr"] = atrs[kind]
                rec[f"lab_{kind}"] = sd.label_trade(L[(kind, 0)].series, ts, atrs[kind], HORIZON, target_r=TARGET_R)
            rec["eligible"] = True
    for r in rows:
        funnel["ELIGIBLE" if r["eligible"] else r["reason"]] += 1
    P = pd.DataFrame(rows)
    for c in (*FEATURES, "atm", "CE_key", "PE_key", "CE_strike", "PE_strike", "CE_atr", "PE_atr", "lab_CE", "lab_PE",
              "pm2_listed_descriptive"):
        if c not in P.columns:
            P[c] = np.nan if c not in ("lab_CE", "lab_PE", "CE_key", "PE_key") else None
    return (P.sort_values("signal_time", kind="mergesort").reset_index(drop=True) if len(P) else P), funnel


# ------------------------------------------------------------------ pre-declared rule families
def _m(P: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(P[c], errors="coerce")


def fam_a(P):
    M = RULES["MOM_MIN"]
    return ((_m(P, "ce_mom") >= M) & (_m(P, "mom_diff") >= M) & (_m(P, "strength_diff") > 0),
            (_m(P, "pe_mom") >= M) & (_m(P, "mom_diff") <= -M) & (_m(P, "strength_diff") < 0))


def fam_b(P):
    M, V = RULES["MOM_MIN"], RULES["VOL_IMB_MIN"]
    return (_m(P, "ce_mom") >= M) & (_m(P, "vol_imb") >= V), (_m(P, "pe_mom") >= M) & (_m(P, "vol_imb") <= -V)


def fam_c(P):
    # rising CE premium while put OI builds faster than call OI (writers' side) = bullish; mirror for PE
    M, O = RULES["MOM_MIN"], RULES["OI_CHG_IMB_MIN"]
    return ((_m(P, "ce_mom") >= M) & (_m(P, "oi_chg_imb") <= -O),
            (_m(P, "pe_mom") >= M) & (_m(P, "oi_chg_imb") >= O))


def fam_d(P):
    M, N = RULES["MOM_MIN"], RULES["PCR_NEUTRAL"]
    return ((_m(P, "ce_mom") >= M) & (_m(P, "pcr_oi") >= N) & (_m(P, "pcr_vol") <= N),
            (_m(P, "pe_mom") >= M) & (_m(P, "pcr_oi") <= N) & (_m(P, "pcr_vol") >= N))


def fam_e(P):
    M = RULES["MOM_MIN"]
    exp = _m(P, "straddle_mom") > 0
    return ((_m(P, "ce_mom") >= M) & (_m(P, "mom_diff") > 0) & exp,
            (_m(P, "pe_mom") >= M) & (_m(P, "mom_diff") < 0) & exp)


def fam_f(P):
    M = RULES["MOM_MIN"]
    up = (_m(P, "fut_ret_1") > 0) & (_m(P, "fut_close_ema20") > 0) & (_m(P, "fut_vwap_dist") > 0)
    dn = (_m(P, "fut_ret_1") < 0) & (_m(P, "fut_close_ema20") < 0) & (_m(P, "fut_vwap_dist") < 0)
    return ((_m(P, "ce_mom") >= M) & (_m(P, "mom_diff") > 0) & up,
            (_m(P, "pe_mom") >= M) & (_m(P, "mom_diff") < 0) & dn)


FAMILIES = {"A_CHAIN_PRICE_MOMENTUM": (fam_a, "ATM CE/PE premium momentum + relative CE/PE strength"),
            "B_CHAIN_VOLUME": (fam_b, "premium momentum + ATM CE/PE volume imbalance"),
            "C_CHAIN_OI": (fam_c, "premium momentum + ATM CE/PE OI-change imbalance"),
            "D_PCR_CONFIRMATION": (fam_d, "premium momentum + ATM+/-1 PCR_OI / PCR_VOLUME confirmation"),
            "E_STRADDLE_CONTEXT": (fam_e, "premium direction + ATM straddle expansion"),
            "F_CHAIN_PLUS_FUTURES": (fam_f, "chain direction + NIFTY futures direction agreement")}


def signals(P: pd.DataFrame, family: str) -> tuple[pd.Series, Counter]:
    """'CE' / 'PE' / None per row. Both sides true at once -> NO TRADE (never both)."""
    if P.empty:
        return pd.Series([], dtype=object), Counter()
    ce, pe = FAMILIES[family][0](P)
    el = P["eligible"].astype(bool)
    ce, pe = (ce.fillna(False) & el), (pe.fillna(False) & el)
    both = ce & pe
    side = pd.Series(np.where(ce & ~both, "CE", np.where(pe & ~both, "PE", None)), index=P.index, dtype=object)
    return side, Counter({"CE": int((ce & ~both).sum()), "PE": int((pe & ~both).sum()), "CONFLICT_NO_TRADE": int(both.sum())})


# ------------------------------------------------------------------ simulation / metrics
TRADE_COLS = ("candidate", "family", "signal_time", "session_date", "side", "instrument_key", "contract", "strike",
              "expiry", "atr", "entry_time", "entry", "stop_loss", "target", "risk", "outcome", "exit_time",
              "exit_price", "exit_reason", "r_multiple", "pnl", "holding_minutes")


def simulate(P: pd.DataFrame, side: pd.Series, name: str, family: str) -> tuple[list[dict], Counter]:
    """One active trade at a time; a new entry only at or after the previous exit."""
    trades, dec, busy = [], Counter(), None
    for i in side.index[side.notna()]:
        r, s = P.loc[i], side[i]
        lab = r[f"lab_{s}"]
        entry_time = r["signal_time"] + STEP
        if busy is not None and entry_time < busy:
            dec["ACTIVE_TRADE_OPEN"] += 1
            continue
        t = {"candidate": name, "family": family, "signal_time": r["signal_time"], "session_date": r["session_date"],
             "side": s, "instrument_key": r[f"{s}_key"], "contract": r.get(f"{s}_contract"), "strike": r[f"{s}_strike"],
             "expiry": r["expiry"], "atr": r[f"{s}_atr"], **{k: lab.get(k) for k in (
                 "entry_time", "entry", "stop_loss", "target", "risk", "outcome", "exit_time", "exit_price",
                 "exit_reason", "r_multiple", "pnl", "holding_minutes")}}
        trades.append(t)
        if t["outcome"] in sd.EXECUTED:
            busy = t["exit_time"]
    return trades, dec


def sample_label(resolved: int) -> str:
    return "INSUFFICIENT_DATA" if resolved < 30 else "LOW_SAMPLE" if resolved < 50 else \
        "MODERATE_SAMPLE" if resolved < 100 else "STRONGER_SAMPLE"


def metrics(trades: Sequence[dict], opportunities: int) -> dict[str, Any]:
    trades = sorted(trades, key=lambda t: t["signal_time"])
    ex = [t for t in trades if t["outcome"] in sd.EXECUTED]
    n = Counter(t["outcome"] for t in trades)
    wins, losses = n[sd.WIN], n[sd.LOSS]
    pnl = [float(t["pnl"]) for t in ex]
    rs = [float(t["r_multiple"]) for t in ex]
    gp, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    eq = peak = dd = eq_r = peak_r = dd_r = 0.0
    streak = worst = 0
    for p, r in zip(pnl, rs):
        eq += p
        peak, dd = max(peak, eq), max(dd, peak - eq)
        eq_r += r
        peak_r, dd_r = max(peak_r, eq_r), max(dd_r, peak_r - eq_r)
        streak = streak + 1 if r < 0 else 0
        worst = max(worst, streak)
    resolved = wins + losses
    pf = (gp / gl if gl > 0 else (math.inf if gp > 0 else None)) if ex else None
    return {"opportunities": int(opportunities), "signals_taken": len(trades), "trades": len(ex), "wins": wins,
            "losses": losses, "timeout": n[sd.TIMEOUT], "ambiguous": n[sd.AMBIGUOUS],
            "no_entry": n[sd.NO_ENTRY] + n[sd.INVALID_LEVELS], "resolved": resolved,
            "accuracy": wins / resolved if resolved else None, "profit_factor": pf, "gross_profit": round(gp, 2),
            "gross_loss": round(gl, 2), "net_pnl": round(gp - gl, 2),
            "expectancy_r": round(sum(rs) / len(rs), 4) if rs else None, "total_r": round(sum(rs), 4),
            "max_drawdown": round(dd, 2), "max_drawdown_r": round(dd_r, 4), "max_losing_streak": worst,
            "coverage": len(ex) / opportunities if opportunities else None, "sample": sample_label(resolved)}


def candidates() -> list[tuple[str, str, str]]:
    """(candidate name, family, side) - 6 families x 2 independent sides."""
    return [(f"{fam}__BUY_{s}", fam, s) for fam in FAMILIES for s in SIDES]


def evaluate(P: pd.DataFrame, names: Sequence[str] | None = None) -> dict[str, dict]:
    """Every (or the listed) candidate stream plus each family's CE+PE combined stream on panel P."""
    opp = int(P["eligible"].sum()) if len(P) else 0
    out: dict[str, dict] = {}
    for fam in FAMILIES:
        side, counts = signals(P, fam)
        for s in SIDES:
            name = f"{fam}__BUY_{s}"
            if names is not None and name not in names:
                continue
            only = side.where(side == s)
            tr, dec = simulate(P, only, name, fam)
            out[name] = {"family": fam, "side": s, "trades": tr, "metrics": metrics(tr, opp), "decisions": dict(dec),
                         "signals": counts.get(s, 0), "conflicts": counts.get("CONFLICT_NO_TRADE", 0)}
        if names is None:
            name = f"{fam}__COMBINED"
            tr, dec = simulate(P, side, name, fam)
            out[name] = {"family": fam, "side": "COMBINED", "trades": tr, "metrics": metrics(tr, opp),
                         "decisions": dict(dec), "signals": counts.get("CE", 0) + counts.get("PE", 0),
                         "conflicts": counts.get("CONFLICT_NO_TRADE", 0)}
    return out


def combine(P: pd.DataFrame, picks: dict[str, str | None]) -> tuple[list[dict], dict]:
    """Chosen CE candidate + chosen PE candidate, one active trade; both firing at one timestamp -> NO TRADE."""
    if P.empty or not any(picks.values()):
        return [], metrics([], int(P["eligible"].sum()) if len(P) else 0)
    fired = {}
    for s, name in picks.items():
        if name:
            fam = name.split("__")[0]
            sig, _ = signals(P, fam)
            fired[s] = sig == s
    ce = fired.get("CE", pd.Series(False, index=P.index))
    pe = fired.get("PE", pd.Series(False, index=P.index))
    side = pd.Series(np.where(ce & ~pe, "CE", np.where(pe & ~ce, "PE", None)), index=P.index, dtype=object)
    tr, _ = simulate(P, side, "PRIMARY_COMBINED", "+".join(n.split("__")[0] for n in picks.values() if n))
    return tr, metrics(tr, int(P["eligible"].sum()))


# ------------------------------------------------------------------ selection / status (pre-declared)
def train_qualified(m: dict) -> bool:
    pf, ex = m["profit_factor"], m["expectancy_r"]
    return m["resolved"] >= MIN_RESOLVED and ex is not None and ex > 0 and pf is not None and pf > 1 \
        and m["max_drawdown_r"] <= MAX_DD_R


def rank_key(m: dict) -> tuple:
    return (m["expectancy_r"] if m["expectancy_r"] is not None else -9e9, m["resolved"])


def stage_status(m: dict | None, train_m: dict | None) -> str:
    if not m or m["resolved"] < MIN_RESOLVED:
        return INSUFF
    pf, ex = m["profit_factor"], m["expectancy_r"]
    if ex is None or ex <= 0 or pf is None or pf <= 1:
        return REJECTED
    t_ex = (train_m or {}).get("expectancy_r")
    return DEGRADED if t_ex and ex < DEGRADE_SHARE * t_ex else PASS


def best(names: Sequence[str], train: dict[str, dict], side: str) -> str | None:
    pool = [n for n in names if train[n]["side"] == side]
    return max(pool, key=lambda n: (*rank_key(train[n]["metrics"]), n)) if pool else None


def final_decision(*, checks_ok: bool, qualified: Sequence[str], april_pass: Sequence[str],
                   april_status: dict[str, str], oos_m: dict | None) -> tuple[str, str]:
    if not checks_ok:
        return REJECTED, "lookahead / data-quality check failed"
    if not qualified:
        return REJECTED, (f"no family reached TRAIN expectancy > 0, PF > 1, DD <= {MAX_DD_R}R and >= {MIN_RESOLVED} "
                          "resolved trades")
    if not april_pass:
        if any(april_status.get(n) == INSUFF for n in qualified):
            return INSUFF, f"April: fewer than {MIN_RESOLVED} resolved trades for the TRAIN-qualified candidates"
        return REJECTED, "every TRAIN-qualified candidate degraded or failed in April (not rescued by tuning)"
    if not oos_m or oos_m["resolved"] < MIN_RESOLVED:
        return INSUFF, f"May-June OOS: fewer than {MIN_RESOLVED} resolved trades"
    pf, ex = oos_m["profit_factor"], oos_m["expectancy_r"]
    if ex is not None and ex > 0 and pf is not None and pf > 1 and oos_m["max_drawdown_r"] <= MAX_DD_R:
        return PROMISING, "positive frozen May-June expectancy with PF > 1 - candidate for fresh-paper validation only"
    return REJECTED, "frozen May-June OOS expectancy <= 0, PF <= 1 or drawdown too large"


# ------------------------------------------------------------------ data-quality gate
def s131_gate(summary: dict | None, lo: date = CLEAN_START, hi: date = MAX_LOAD) -> dict[str, Any]:
    P = ((summary or {}).get("periods") or {}).get("CLEAN_USABLE") or {}
    if not P.get("ran"):
        return {"passed": False, "checks": {}, "reason": "S13.1 summary missing (run scripts/run_s13_chain_audit.py)"}
    c = P.get("chain") or {}
    checks = {"period_matches": P.get("range") == [lo.isoformat(), hi.isoformat()],
              "data_quality_pass": bool((P.get("data_quality") or {}).get("passed")),
              "point_in_time_pass": bool((P.get("point_in_time") or {}).get("passed")),
              "reconstruction_pass": bool((P.get("reconstruction") or {}).get("passed")),
              "atm_ce_pe_coverage": min(c.get("atm_ce") or 0, c.get("atm_pe") or 0) >= s13a.COVERAGE_STRONG,
              "atm_pm1_ce_pe_coverage": min(c.get("band_pm1_ce") or 0, c.get("band_pm1_pe") or 0)
              >= s13a.COVERAGE_STRONG}
    return {"passed": all(checks.values()), "checks": checks,
            "measured": {k: c.get(k) for k in ("atm_ce", "atm_pe", "band_pm1_ce", "band_pm1_pe", "band_pm2_ce",
                                               "band_pm2_pe", "band_pm3_ce", "band_pm3_pe")}}


# ------------------------------------------------------------------ anti-lookahead checks
def _feat_equal(a: pd.DataFrame, b: pd.DataFrame, cols: Sequence[str]) -> bool:
    if len(a) != len(b):
        return False
    if not len(a):
        return True
    x = a[list(cols)].reset_index(drop=True)
    y = b[list(cols)].reset_index(drop=True)
    for c in cols:
        u, v = x[c], y[c]
        if u.dtype == object or v.dtype == object:
            if not u.where(u.notna(), "").astype(str).equals(v.where(v.notna(), "").astype(str)):
                return False
        elif not np.allclose(u.to_numpy(dtype=float), v.to_numpy(dtype=float), equal_nan=True, rtol=1e-12, atol=1e-12):
            return False
    return True


CHECK_COLS = (*FEATURES, "eligible", "atm", "CE_key", "PE_key", "expiry")


def mutation_check(market: sd.MarketData, P: pd.DataFrame, days: Sequence[str], step: float, seed: int,
                   **kw) -> dict[str, Any]:
    """Candles after 12:00 mutated in every source: chain / futures features and contracts at signals <= 12:00
    must be identical."""
    bad = []
    for d in days:
        day = date.fromisoformat(d)
        cut = pd.Timestamp(datetime.combine(day, time(12, 0)), tz=IST)
        mm = sd.mutated_market(market, cut, seed=seed, **kw)
        Q, _ = build_panel(mm, [day], step)
        a = P[(P["session_date"] == d) & (P["signal_time"] <= cut)]
        b = Q[Q["signal_time"] <= cut] if len(Q) else Q
        if not _feat_equal(a, b, CHECK_COLS):
            bad.append(d)
    return {"days": list(days), "mismatches": bad, "passed": bool(days) and not bad}


def truncation_check(market: sd.MarketData, P: pd.DataFrame, step: float, sample: int) -> dict[str, Any]:
    """Each sampled signal rebuilt from data with every candle after t removed must give identical features."""
    E = P[P["eligible"]]
    rows = E.iloc[np.unique(np.linspace(0, len(E) - 1, min(sample, len(E))).astype(int))] if len(E) else E
    bad = []
    for _, r in rows.iterrows():
        t = r["signal_time"]
        mm = sd.mutated_market(market, t, drop=True)
        Q, _ = build_panel(mm, [t.date()], step)
        q = Q[Q["signal_time"] == t]
        if not _feat_equal(P[P["signal_time"] == t], q, CHECK_COLS):
            bad.append(t.isoformat())
    return {"checked": int(len(rows)), "mismatches": bad[:20], "passed": len(rows) > 0 and not bad}


def contract_check(market: sd.MarketData, P: pd.DataFrame, step: float) -> dict[str, Any]:
    bad = []
    for r in P[P["eligible"]].itertuples(index=False):
        d = date.fromisoformat(r.session_date)
        e = sd.nearest_expiry(market.calendar, d)
        atm = s13a.atm_of(float(r.spot), step)
        if e is None or r.expiry != e.isoformat() or e < d or r.atm != atm or r.CE_strike != atm or r.PE_strike != atm:
            bad.append(r.signal_time.isoformat())
    return {"checked": int(P["eligible"].sum()), "violations": bad[:20], "passed": not bad,
            "rule": "nearest listed expiry >= session date; ATM from NIFTY close at t on the strike grid; ATM+/-1 "
                    "one strike step away; no other contract substituted"}


def trade_checks(streams: dict[str, list[dict]]) -> dict[str, dict]:
    bad_entry, bad_dup = [], []
    for name, tr in streams.items():
        ts = [t["signal_time"] for t in tr]
        if len(ts) != len(set(ts)):
            bad_dup.append(name)
        ex = [t for t in tr if t["outcome"] in sd.EXECUTED]
        for t in tr:
            if t["entry_time"] != t["signal_time"] + STEP:
                bad_entry.append(f"{name}|{t['signal_time']}")
        for a, b in zip(ex, ex[1:]):
            if b["entry_time"] < a["exit_time"]:
                bad_dup.append(f"{name}|overlap|{b['signal_time']}")
    return {"next_candle_entry": {"violations": bad_entry[:20], "passed": not bad_entry},
            "duplicate_trade_prevention": {"violations": bad_dup[:20], "passed": not bad_dup}}


def _lab_key(P: pd.DataFrame) -> list[str]:
    return [json.dumps({s: (r[f"lab_{s}"] or {}).get("outcome") if isinstance(r[f"lab_{s}"], dict) else None
                        for s in SIDES} | {"t": str(r["signal_time"])}, sort_keys=True) for _, r in P.iterrows()]


def replay_check(market: sd.MarketData, P: pd.DataFrame, days: Sequence[str], step: float) -> dict[str, Any]:
    Q, _ = build_panel(market, [date.fromisoformat(d) for d in days], step)
    a = P[P["session_date"].isin(set(days))]
    same_f = _feat_equal(a, Q, CHECK_COLS)
    same_l = _lab_key(a) == _lab_key(Q)
    r1, r2 = evaluate(a), evaluate(Q)
    same_t = all([json.dumps(x, default=str, sort_keys=True) for x in r1[k]["trades"]]
                 == [json.dumps(x, default=str, sort_keys=True) for x in r2[k]["trades"]] for k in r1)
    return {"days": list(days), "features_identical": same_f, "labels_identical": same_l, "trades_identical": same_t,
            "passed": bool(days) and same_f and same_l and same_t}


# ------------------------------------------------------------------ comparison (saved reports only)
def _read_csv(p: Path | None) -> list[dict]:
    if p is None or not Path(p).is_file():
        return []
    with Path(p).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def oos_comparison(sources: dict[str, Path], s13_m: dict | None) -> list[dict]:
    """S2 / S9 May-June 2026 from their saved monthly reports (previously researched, different labels / exits)."""
    out = []
    s2 = [r for r in _read_csv(sources.get("S2")) if r.get("scope") == "COMBINED" and r.get("month") in ("2026-05",
                                                                                                       "2026-06")]
    if s2:
        w, l = sum(_num(r["wins"]) or 0 for r in s2), sum(_num(r["losses"]) or 0 for r in s2)
        gp, gl = sum(_num(r.get("gross_profit")) or 0 for r in s2), sum(_num(r.get("gross_loss")) or 0 for r in s2)
        out.append({"system": "S2", "trades": sum(_num(r["total_trades"]) or 0 for r in s2), "resolved": w + l,
                    "accuracy": w / (w + l) if w + l else None, "profit_factor": gp / abs(gl) if gl else None,
                    "net_pnl": sum(_num(r["net_pnl"]) or 0 for r in s2), "status": "AVAILABLE",
                    "note": "saved clean_historical_backtest report; frozen S2 rules, period previously researched"})
    else:
        out.append({"system": "S2", "status": "NOT_AVAILABLE"})
    s9 = [r for r in _read_csv(sources.get("S9")) if r.get("scope") == "COMBINED" and r.get("month") in ("MAY", "JUNE")]
    if s9:
        w, l = sum(_num(r["wins"]) or 0 for r in s9), sum(_num(r["losses"]) or 0 for r in s9)
        pfs = {r["month"]: _num(r["profit_factor"]) for r in s9}
        out.append({"system": "S9_FUTURES_VOLUME", "trades": sum(_num(r["trades"]) or 0 for r in s9),
                    "resolved": w + l, "accuracy": w / (w + l) if w + l else None, "profit_factor": None,
                    "net_pnl": sum(_num(r["net_pnl"]) or 0 for r in s9), "status": "AVAILABLE",
                    "note": f"saved S9 report (REJECTED); combined PF not derivable (no gross columns), per-month PF "
                            f"{pfs}"})
    else:
        out.append({"system": "S9_FUTURES_VOLUME", "status": "NOT_AVAILABLE"})
    out.append({"system": STRATEGY, **({k: s13_m.get(k) for k in ("trades", "resolved", "accuracy", "profit_factor",
                                                                    "net_pnl")} if s13_m else {}),
                "status": "AVAILABLE" if s13_m else NOT_RUN, "note": "frozen May-June OOS, gross, no costs"})
    return out


# ------------------------------------------------------------------ the study
def _days(market: sd.MarketData, lo: date, hi: date) -> list[date]:
    return [d for d in market.sessions if lo <= d <= hi]


def feature_stats(panels: dict[str, pd.DataFrame]) -> list[dict]:
    out = []
    for split, P in panels.items():
        E = P[P["eligible"]] if len(P) else P
        for c in FEATURES:
            x = pd.to_numeric(E[c], errors="coerce").dropna() if len(E) else pd.Series(dtype=float)
            out.append({"split": split, "feature": c, "eligible_rows": int(len(E)), "non_null": int(len(x)),
                        "non_null_share": len(x) / len(E) if len(E) else None,
                        **({"mean": x.mean(), "std": x.std(), "p10": x.quantile(0.1), "p50": x.median(),
                            "p90": x.quantile(0.9)} if len(x) else {})})
    return out


def coverage_rows(panels: dict[str, pd.DataFrame], funnels: dict[str, Counter]) -> list[dict]:
    out = []
    for split, P in panels.items():
        E = P[P["eligible"]] if len(P) else P
        row = {"split": split, "signal_timestamps": int(len(P)), "eligible_atm_pm1": int(len(E)),
               "eligible_share": len(E) / len(P) if len(P) else None, **dict(funnels.get(split, {}))}
        for name, cols in (("oi_atm_both", ["ce_oi", "pe_oi"]), ("volume_atm_both", ["ce_vol", "pe_vol"]),
                           ("pcr_oi", ["pcr_oi"]), ("pcr_volume", ["pcr_vol"]), ("oi_change_both", ["ce_oi_chg",
                                                                                                   "pe_oi_chg"]),
                           ("futures_context", ["fut_ret_1", "fut_close_ema20", "fut_vwap_dist"]),
                           ("straddle_mom", ["straddle_mom"])):
            row[f"{name}_share"] = float(E[cols].notna().all(axis=1).mean()) if len(E) else None
        row["atm_pm2_listed_share_descriptive"] = float(pd.to_numeric(P["pm2_listed_descriptive"],
                                                                      errors="coerce").fillna(0).mean()) if len(P) else None
        for fam in FAMILIES:
            _, cnt = signals(P, fam)
            row[f"{fam}_ce_signals"], row[f"{fam}_pe_signals"] = cnt.get("CE", 0), cnt.get("PE", 0)
            row[f"{fam}_conflicts"] = cnt.get("CONFLICT_NO_TRADE", 0)
        out.append(row)
    return out


def robustness_rows(P: pd.DataFrame, res: dict[str, dict]) -> list[dict]:
    """TRAIN month-by-month view of every candidate (reported only; selection does not use it)."""
    out = []
    months = sorted({d[:7] for d in P["session_date"]}) if len(P) else []
    for name, r in res.items():
        pos = 0
        for m in months:
            tr = [t for t in r["trades"] if str(t["session_date"]).startswith(m)]
            mm = metrics(tr, 0)
            pos += int((mm["expectancy_r"] or 0) > 0)
            out.append({"candidate": name, "month": m, "trades": mm["trades"], "resolved": mm["resolved"],
                        "accuracy": mm["accuracy"], "profit_factor": mm["profit_factor"],
                        "expectancy_r": mm["expectancy_r"], "net_pnl": mm["net_pnl"]})
        out.append({"candidate": name, "month": "ALL_TRAIN_MONTHS", "positive_expectancy_months": pos,
                    "months": len(months)})
    return out


def not_run(base: dict, reason: str) -> dict[str, Any]:
    return {**base, "ran": False, "reason": reason, "decision": INSUFF, "decision_reason": reason,
            "forward_paper": "NOT SAFE", "target_70": INSUFF, "lookahead_ok": None, "results": {}}


def build_study(*, market: sd.MarketData, read_log: list[dict], data_quality: dict,
                comparison_sources: dict[str, Path] | None = None, mutation_days: int = 4, check_sample: int = 20,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    stages: list[str] = []
    base = {"heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "label": LABEL, "rules": RULES,
            "families": {k: v[1] for k, v in FAMILIES.items()}, "stages": stages,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
            "splits": {"TRAIN": [CLEAN_START.isoformat(), TRAIN_END.isoformat()],
                       "VALIDATION": [x.isoformat() for x in VALIDATION], "OOS": [x.isoformat() for x in OOS]},
            "universe": "ATM and ATM+/-1 only (ATM+/-2 descriptive only; ATM+/-3 / +/-5 never used)",
            "data_quality": data_quality, "october": "NOT LOADED", "july_plus": "NOT LOADED",
            "production_unchanged": None, "live_orders": 0, "trading_mode": "PAPER", "live_trading_enabled": False,
            "ml_trained": False, "read_summary": s122.read_summary(read_log)}
    if not data_quality.get("passed"):
        return not_run(base, f"DATA_QUALITY_GATE_FAILED: {data_quality}")
    step = strike_step(market)
    if step is None:
        return not_run(base, "no stored option chain")
    base["strike_step"] = step

    say("TRAIN panel from a view truncated at 2026-03-31 (April+ invisible) ...")
    stages.append("train_panel")
    sel = s122.phase_view(market, TRAIN_END)
    P_tr, f_tr = build_panel(sel, _days(sel, CLEAN_START, TRAIN_END), step)
    if P_tr.empty or not P_tr["eligible"].any():
        return not_run(base, "no eligible TRAIN signal candles")
    stages.append("train_selection")
    train = evaluate(P_tr)
    cand = [c[0] for c in candidates()]
    qualified = [n for n in cand if train_qualified(train[n]["metrics"])]
    picks_train = {s: best(qualified, train, s) for s in SIDES}
    best_train = {s: best(cand, train, s) for s in SIDES}

    say("freezing; April 2026 panel (view truncated at 2026-04-30) ...")
    stages.append("freeze")
    frozen = {"qualified": qualified, "picks": picks_train, "rules": RULES,
              "frozen_at": datetime.now(IST).isoformat(timespec="seconds")}
    stages.append("validation_panel")
    vview = s122.phase_view(market, VALIDATION[1])
    P_va, f_va = build_panel(vview, _days(vview, *VALIDATION), step)
    stages.append("validation_april")
    val = evaluate(P_va)
    april_status = {n: (stage_status(val[n]["metrics"], train[n]["metrics"]) if n in qualified
                        else "NOT_SELECTED_IN_TRAIN") for n in cand}
    april_pass = [n for n in qualified if april_status[n] == PASS]

    oos, P_oo, f_oo, oos_status = {}, pd.DataFrame(), Counter(), {n: NOT_RUN for n in cand}
    picks_oos = {s: None for s in SIDES}
    comb_tr, comb_m = [], None
    if april_pass:
        say("May-June 2026 OOS (frozen; April-passing candidates only) ...")
        stages.append("oos_panel")
        P_oo, f_oo = build_panel(market, _days(market, *OOS), step)
        stages.append("oos_may_june")
        oos = evaluate(P_oo, names=april_pass)
        oos_status.update({n: stage_status(oos[n]["metrics"], None) for n in april_pass})
        picks_oos = {s: best(april_pass, train, s) for s in SIDES}
        comb_tr, comb_m = combine(P_oo, picks_oos)

    say("anti-lookahead checks ...")
    mdays = sorted({d for d in P_tr["session_date"]})
    mdays = [mdays[int(i)] for i in np.unique(np.linspace(0, len(mdays) - 1, min(mutation_days, len(mdays))))] \
        if mutation_days and mdays else []
    lookahead = {
        "chain_point_in_time_mutation": mutation_check(sel, P_tr, mdays, step, seed=13, price=True, volume=True,
                                                       oi=True),
        "feature_truncation": truncation_check(sel, P_tr, step, check_sample),
        "contract_selection_causal": contract_check(sel, P_tr, step),
        **trade_checks({**{f"TRAIN|{k}": v["trades"] for k, v in train.items()},
                        **{f"VAL|{k}": v["trades"] for k, v in val.items()},
                        **{f"OOS|{k}": v["trades"] for k, v in oos.items()}, "OOS|PRIMARY_COMBINED": comb_tr}),
        "deterministic_replay": replay_check(sel, P_tr, mdays[:2], step),
        "max_loaded_timestamp": s122.max_loaded_check(market, read_log),
        "selection_isolation": {"selection_input_last_day": str(P_tr["session_date"].max()),
                                "stage_order": stages,
                                "passed": str(P_tr["session_date"].max()) <= TRAIN_END.isoformat()
                                and stages.index("freeze") < stages.index("validation_april")
                                and ("oos_panel" not in stages or stages.index("validation_april")
                                     < stages.index("oos_panel"))},
    }
    lookahead_ok = all(v["passed"] for v in lookahead.values())
    data_ok = bool(data_quality.get("passed")) and lookahead["contract_selection_causal"]["passed"]
    checks_ok = lookahead_ok and data_ok
    decision, why = final_decision(checks_ok=checks_ok, qualified=qualified, april_pass=april_pass,
                                   april_status=april_status, oos_m=comb_m)
    if comb_m and comb_m["resolved"] >= MIN_RESOLVED:
        target = "ACHIEVED" if (comb_m["accuracy"] or 0) >= 0.70 else "NOT ACHIEVED"
    else:
        target = INSUFF
    forward = "SAFE" if decision == PROMISING and checks_ok else "NOT SAFE"
    panels = {"TRAIN": P_tr, "VALIDATION": P_va, **({"OOS": P_oo} if len(P_oo) else {})}
    return {**base, "ran": True, "train": train, "validation": val, "oos": oos, "qualified": qualified,
            "frozen": frozen, "picks_train": picks_train, "best_train_any": best_train, "picks_oos": picks_oos,
            "april_status": april_status, "april_pass": april_pass, "oos_status": oos_status,
            "combined_oos": comb_m, "combined_oos_trades": comb_tr,
            "comparison": oos_comparison(comparison_sources or {}, comb_m), "lookahead": lookahead,
            "lookahead_ok": lookahead_ok, "data_ok": data_ok, "decision": decision, "decision_reason": why,
            "target_70": target, "forward_paper": forward,
            "funnels": {"TRAIN": dict(f_tr), "VALIDATION": dict(f_va), "OOS": dict(f_oo)},
            "coverage": coverage_rows(panels, {"TRAIN": f_tr, "VALIDATION": f_va, "OOS": f_oo}),
            "feature_stats": feature_stats(panels), "robustness": robustness_rows(P_tr, train),
            "_panels": panels, "_market": market}


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s13_chain_strategy_report.md", "candidate_results.csv", "train_results.csv",
                "validation_results.csv", "oos_results.csv", "buy_ce_results.csv", "buy_pe_results.csv",
                "combined_results.csv", "chain_feature_stats.csv", "coverage.csv", "robustness.csv", "trade_log.csv",
                "data_quality.json")
MKEYS = ("trades", "resolved", "wins", "losses", "timeout", "ambiguous", "accuracy", "profit_factor", "expectancy_r",
         "net_pnl", "max_drawdown", "max_drawdown_r", "max_losing_streak", "coverage", "sample")


def _res_rows(a: dict, split: str) -> list[dict]:
    res = a.get({"TRAIN": "train", "VALIDATION": "validation", "OOS": "oos"}[split]) or {}
    status = {"TRAIN": {n: ("QUALIFIED" if n in a.get("qualified", []) else REJECTED) for n in res},
              "VALIDATION": a.get("april_status", {}), "OOS": a.get("oos_status", {})}[split]
    return [{"split": split, "candidate": n, "family": r["family"], "side": r["side"],
             "status": status.get(n, "-"), "signals": r["signals"], "conflicts_no_trade": r["conflicts"],
             **{k: r["metrics"][k] for k in MKEYS}} for n, r in res.items()]


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not str(k).startswith("_") and k not in ("trades",
                                                                                                "combined_oos_trades")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    js = lambda path, obj: path.write_text(json.dumps(s6._jsonable(obj), indent=2, default=str),  # noqa: E731
                                           encoding="utf-8")
    summ = {k: v for k, v in a.items() if k not in ("train", "validation", "oos", "coverage", "feature_stats",
                                                    "robustness")}
    summ["results"] = {split: {r["candidate"]: {k: r[k] for k in ("status", *MKEYS)} for r in _res_rows(a, split)}
                       for split in ("TRAIN", "VALIDATION", "OOS")} if a.get("ran") else {}
    js(p["summary.json"], _strip(summ))
    js(p["data_quality.json"], {"gate": a.get("data_quality"), "funnels": a.get("funnels"),
                                "contract_selection": (a.get("lookahead") or {}).get("contract_selection_causal"),
                                "max_loaded": (a.get("lookahead") or {}).get("max_loaded_timestamp"),
                                "data_ok": a.get("data_ok")})
    rows = {s: _res_rows(a, s) if a.get("ran") else [] for s in ("TRAIN", "VALIDATION", "OOS")}
    allr = rows["TRAIN"] + rows["VALIDATION"] + rows["OOS"]
    s6._csv(p["candidate_results.csv"], allr)
    s6._csv(p["train_results.csv"], rows["TRAIN"])
    s6._csv(p["validation_results.csv"], rows["VALIDATION"])
    s6._csv(p["oos_results.csv"], rows["OOS"] or [{"split": "OOS", "status": NOT_RUN,
                                                   "reason": "no candidate passed April" if a.get("ran") else a.get("reason")}])
    s6._csv(p["buy_ce_results.csv"], [r for r in allr if r["side"] == "CE"])
    s6._csv(p["buy_pe_results.csv"], [r for r in allr if r["side"] == "PE"])
    comb = [r for r in allr if r["side"] == "COMBINED"]
    if a.get("combined_oos"):
        comb.append({"split": "OOS", "candidate": "PRIMARY_COMBINED", "family": "+".join(
            n.split("__")[0] for n in (a.get("picks_oos") or {}).values() if n), "side": "COMBINED",
            "status": a.get("decision"), **{k: a["combined_oos"][k] for k in MKEYS}})
    s6._csv(p["combined_results.csv"], comb)
    s6._csv(p["chain_feature_stats.csv"], a.get("feature_stats", []))
    s6._csv(p["coverage.csv"], a.get("coverage", []))
    s6._csv(p["robustness.csv"], a.get("robustness", []))
    trades = []
    for split, key in (("TRAIN", "train"), ("VALIDATION", "validation"), ("OOS", "oos")):
        for r in (a.get(key) or {}).values():
            trades += [{"split": split, **{c: t.get(c) for c in TRADE_COLS}} for t in r["trades"]]
    trades += [{"split": "OOS", **{c: t.get(c) for c in TRADE_COLS}} for t in a.get("combined_oos_trades") or []]
    s6._csv(p["trade_log.csv"], trades)
    p["s13_chain_strategy_report.md"].write_text(markdown(a), encoding="utf-8")
    return p


# ------------------------------------------------------------------ text
def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _x(v, nd: int = 2) -> str:
    return "N/A" if v is None else ("inf" if v == math.inf else f"{v:.{nd}f}")


def _block(name: str | None, m: dict | None, money: bool = False) -> list[str]:
    if not name or not m:
        return ["name: NONE", "trades: 0", "accuracy: N/A", "PF: N/A", "expectancy: N/A"] + \
            (["P&L: N/A", "DD: N/A"] if money else [])
    out = [f"name: {name}", f"trades: {m['trades']} (resolved {m['resolved']}, {m['sample']})",
           f"accuracy: {_pct(m['accuracy'])}", f"PF: {_x(m['profit_factor'])}",
           f"expectancy: {_x(m['expectancy_r'], 3)}R"]
    if money:
        out += [f"P&L: Rs {m['net_pnl']:,.2f}", f"DD: Rs {m['max_drawdown']:,.2f} ({_x(m['max_drawdown_r'])}R)"]
    return out


def terminal_text(a: dict) -> str:
    out = [HEADING, f"({BANNER})", ""]
    if not a.get("ran"):
        out += [f"S13.2 NOT RUN: {a.get('reason')}", ""]
    else:
        tr, va, oo = a["train"], a["validation"], a["oos"]
        pt = {s: a["picks_train"][s] or a["best_train_any"][s] for s in SIDES}
        out.append("TRAIN:")
        for s in SIDES:
            tag = "" if a["picks_train"][s] else " [no candidate qualified; best by expectancy shown]"
            out += ["", f"Best {s}:{tag}", *_block(pt[s], tr[pt[s]]["metrics"] if pt[s] else None)]
        out += ["", "VALIDATION APRIL:"]
        for s in SIDES:
            n = a["picks_train"][s]
            st = f" [{a['april_status'][n]}]" if n else " [no TRAIN-qualified candidate]"
            out += ["", f"Best {s}:{st}", *_block(n, va[n]["metrics"] if n else None)]
        out += ["", "OOS MAY-JUNE:" + ("" if oo else " NOT_RUN (no candidate passed April)")]
        for s in SIDES:
            n = a["picks_oos"][s]
            out += ["", f"Best {s}:" + (f" [{a['oos_status'][n]}]" if n else ""),
                    *_block(n, oo[n]["metrics"] if n else None, money=True)]
        c = a.get("combined_oos")
        out += ["", "COMBINED:", *(_block("PRIMARY_COMBINED", c, money=True)[1:] if c else
                                    ["trades: 0", "accuracy: N/A", "PF: N/A", "expectancy: N/A", "P&L: N/A", "DD: N/A"])]
        out += ["", "S2 vs S9 vs S13 OOS:"]
        for r in a["comparison"]:
            label = {"S2": "S2", "S9_FUTURES_VOLUME": "S9", STRATEGY: "S13"}[r["system"]]
            if r["status"] != "AVAILABLE":
                out += ["", f"{label}:", f"accuracy: {r['status']}", f"PF: {r['status']}", f"P&L: {r['status']}"]
                continue
            out += ["", f"{label}:", f"accuracy: {_pct(r['accuracy'])}",
                    f"PF: {_x(r['profit_factor']) if r['profit_factor'] is not None else 'N/A (see note)'}",
                    f"P&L: Rs {r['net_pnl']:,.2f}"]
        out.append("(S2 / S9 from saved reports: different labels / exits, periods previously researched - "
                   "descriptive only)")
    out += ["", "70%:", a.get("target_70", INSUFF), "", "LOOKAHEAD:",
            "N/A (not run)" if a.get("lookahead_ok") is None else ("PASS" if a["lookahead_ok"] else "FAIL"), "",
            "DATA QUALITY:", "PASS" if (a.get("data_ok") if a.get("ran") else (a.get("data_quality") or {}).get(
                "passed")) else "FAIL", "", "OCTOBER LOADED:", "NO", "", "PRODUCTION CHANGED:",
            "NO" if a.get("production_unchanged") is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER:", a.get("forward_paper", "NOT SAFE"), "", "FINAL:", a.get("decision", INSUFF),
            f"({a.get('decision_reason')})", "",
            "Historical results on a 9.1-month window are not live accuracy; no profitability is guaranteed."]
    return "\n".join(out)


def markdown(a: dict) -> str:
    md = ["# S13.2 option-chain structure strategy research", "", f"**{BANNER}**", "",
          f"## Final: **{a.get('decision')}** - {a.get('decision_reason')}", "",
          f"Forward PAPER: **{a.get('forward_paper')}** | 70%: {a.get('target_70')} | Lookahead "
          f"{a.get('lookahead_ok')} | Data quality {a.get('data_ok')} | October / July+: NOT LOADED | ML trained: no | "
          "Live orders: 0", "", f"Universe: {a.get('universe')}", "",
          f"Splits: {a.get('splits')}", "", "## Exit (fixed)", ""] + [f"- {k}: {v}" for k, v in LABEL.items()]
    md += ["", "## Families (pre-declared, one parameter set each)", ""]
    md += [f"- **{k}**: {v}" for k, v in (a.get("families") or {}).items()]
    md += ["", f"Parameters: {RULES}. Selection: TRAIN expectancy > 0, PF > 1, DD <= {MAX_DD_R}R, >= {MIN_RESOLVED} "
           "resolved; best = highest TRAIN expectancy per side. April / OOS never change a rule.", ""]
    if not a.get("ran"):
        return "\n".join(md + [f"**Not run:** {a.get('reason')}"]) + "\n"
    md += ["## Results", "", "| Split | Candidate | Status | Trades | Resolved | Win rate | PF | Expectancy R | P&L | "
           "Max DD R |", "|---|---|---|---|---|---|---|---|---|---|"]
    for split in ("TRAIN", "VALIDATION", "OOS"):
        for r in _res_rows(a, split):
            md.append(f"| {split} | {r['candidate']} | {r['status']} | {r['trades']} | {r['resolved']} | "
                      f"{_pct(r['accuracy'])} | {_x(r['profit_factor'])} | {_x(r['expectancy_r'], 3)} | "
                      f"{r['net_pnl']} | {_x(r['max_drawdown_r'])} |")
    md += ["", "## Terminal summary", "", "```", terminal_text(a), "```", "", "## Checks", ""]
    md += [f"- `{k}`: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += ["", "Gross P&L, no costs. Historical results are not live accuracy; no profit is guaranteed."]
    return "\n".join(md) + "\n"
