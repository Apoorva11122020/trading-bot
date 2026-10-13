"""Read-only FastAPI route for the dashboard's Live Quality Gate V1 panel.

Import `router` into the existing API app. This module only SELECTs from
`data/trading.db`; it never writes to the DB, places orders, or changes S2.
"""
from __future__ import annotations
import json, sqlite3
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()
DB = Path("data/trading.db")
CANDIDATE = "LIVE_QUALITY_GATE_V1"

def _json(v):
    try: return json.loads(v or "{}")
    except Exception: return {}

def _rows():
    con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
    rows=con.execute("""
      SELECT timestamp,direction,decision,status,pnl,r_multiple,entry,stop,target,
             exit_price,exit_time,record_json,skip_reason
      FROM shadow_strategy_signals WHERE candidate=?
      ORDER BY timestamp DESC
    """,(CANDIDATE,)).fetchall(); con.close(); return rows

def _dd(values):
    peak=0.0; cur=0.0; dd=0.0
    for v in values:
        cur+=float(v or 0); peak=max(peak,cur); dd=max(dd,peak-cur)
    return dd

@router.get("/api/live-quality-gate")
def live_quality_gate():
    rows=_rows(); today=datetime.now().astimezone().date().isoformat()
    rows=[r for r in rows if str(r["timestamp"]).startswith(today)]
    accepted=[r for r in rows if r["decision"]=="ACCEPTED"]
    resolved=[r for r in accepted if r["status"] in ("WIN","LOSS","AMBIGUOUS")]
    wins=[r for r in resolved if r["status"]=="WIN"]; losses=[r for r in resolved if r["status"]=="LOSS"]
    pending=[r for r in accepted if r["status"]=="PENDING"]
    pnl=sum(float(r["pnl"] or 0) for r in resolved)
    gp=sum(max(float(r["pnl"] or 0),0) for r in resolved); gl=abs(sum(min(float(r["pnl"] or 0),0) for r in resolved))
    acc=(len(wins)/len(resolved)*100) if resolved else None; pf=(gp/gl) if gl else None
    rate=(len(accepted)/len(rows)*100) if rows else 0
    target="TARGET ACHIEVED" if len(resolved)>=30 and acc is not None and acc>=70 and (pf is None or pf>=1) else "INSUFFICIENT SAMPLE" if len(resolved)<30 else "TARGET NOT MET"
    signals=[]
    for r in rows[:100]:
        j=_json(r["record_json"]); features=j.get("features") or {}
        signals.append({"timestamp":r["timestamp"],"direction":r["direction"],"decision":r["decision"],"status":r["status"],"pnl":r["pnl"],"r_multiple":r["r_multiple"],"entry":r["entry"],"stop":r["stop"],"target":r["target"],"exit":r["exit_price"],"contract":j.get("option") or j.get("option_symbol") or j.get("trading_symbol"),"gate_score":j.get("gate_score"),"gate_max_score":j.get("gate_max_score",3)})
    return {"summary":{"generated":len(rows),"accepted":len(accepted),"skipped":len(rows)-len(accepted),"resolved":len(resolved),"wins":len(wins),"losses":len(losses),"pending":len(pending),"accuracy":acc,"profit_factor":pf,"pnl":pnl,"max_drawdown":_dd([r["pnl"] for r in reversed(resolved)]),"acceptance_rate":rate,"target_status":target,"target_detail":"Needs ≥30 resolved accepted trades" if len(resolved)<30 else "Evaluated on current resolved sample","updated_at":datetime.now().astimezone().isoformat(),"live_orders":0},"signals":signals}
