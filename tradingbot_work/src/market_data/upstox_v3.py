from __future__ import annotations
import json, os, time
from datetime import datetime, timezone
from typing import Callable
import requests
from google.protobuf.message import DecodeError
from src.market_data.models import Tick
from src.market_data.upstox_v3_proto import FeedResponse

AUTHORIZE_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"

MARKET_DATA_TOKEN_ENV_NAMES = ("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "ACCESS_TOKEN")

class UpstoxV3MarketDataError(RuntimeError): pass


def resolve_market_data_access_token(access_token: str | None = None) -> tuple[str, str]:
    """Return (token, env_var_name) for production market-data APIs. Never uses sandbox tokens."""
    if access_token and access_token.strip():
        return access_token.strip(), "access_token_argument"
    for name in MARKET_DATA_TOKEN_ENV_NAMES:
        value = (os.getenv(name) or "").strip()
        if value:
            return value, name
    if (os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN") or "").strip():
        raise UpstoxV3MarketDataError(
            "UPSTOX_SANDBOX_ACCESS_TOKEN cannot be used for market data. "
            "Set UPSTOX_MARKET_DATA_ACCESS_TOKEN or ACCESS_TOKEN."
        )
    raise UpstoxV3MarketDataError(
        "Upstox market-data access token is required "
        "(UPSTOX_MARKET_DATA_ACCESS_TOKEN or ACCESS_TOKEN)."
    )


def authorize_v3(access_token: str | None = None, timeout: float = 10.0) -> str:
    token, _ = resolve_market_data_access_token(access_token)
    r=requests.get(AUTHORIZE_URL, headers={"Accept":"application/json","Authorization":f"Bearer {token}"}, timeout=timeout)
    if r.status_code >= 400: raise UpstoxV3MarketDataError(f"authorize HTTP {r.status_code}: {r.text[:500]}")
    body=r.json(); uri=body.get("data",{}).get("authorized_redirect_uri")
    if not uri or not uri.startswith("wss://"): raise UpstoxV3MarketDataError("Invalid authorized_redirect_uri")
    return uri

def subscription_payload(instrument_keys:list[str], mode:str="ltpc", guid:str="trading-bot") -> bytes:
    if mode not in {"ltpc","full","option_greeks","full_d30"}: raise ValueError("invalid Upstox V3 mode")
    return json.dumps({"guid":guid,"method":"sub","data":{"mode":mode,"instrumentKeys":instrument_keys}},separators=(",",":")).encode()

def decode_ticks(buffer:bytes, *, now:datetime|None=None) -> list[Tick]:
    try: response=FeedResponse.FromString(buffer)
    except DecodeError as exc: raise UpstoxV3MarketDataError("Invalid Upstox V3 protobuf payload") from exc
    ts_now=datetime.fromtimestamp(response.currentTs/1000, tz=timezone.utc) if response.currentTs else (now or datetime.now(timezone.utc))
    out=[]
    for symbol,feed in response.feeds.items():
        cumulative_volume = 0.0
        ltpc = feed.ltpc if feed.WhichOneof("FeedUnion")=="ltpc" else None
        if feed.WhichOneof("FeedUnion")=="fullFeed":
            full=feed.fullFeed
            if full.WhichOneof("FullFeedUnion")=="marketFF":
                ltpc=full.marketFF.ltpc
                cumulative_volume=float(full.marketFF.vtt)
            else:
                ltpc=full.indexFF.ltpc
        elif feed.WhichOneof("FeedUnion")=="firstLevelWithGreeks": ltpc=feed.firstLevelWithGreeks.ltpc
        if ltpc is None or ltpc.ltp <= 0: continue
        ts=datetime.fromtimestamp(ltpc.ltt/1000, tz=timezone.utc) if ltpc.ltt else ts_now
        out.append(Tick(symbol=symbol,timestamp=ts,ltp=float(ltpc.ltp),last_quantity=float(ltpc.ltq),volume=cumulative_volume))
    return out

class UpstoxV3MarketDataAdapter:
    """Upstox V3 binary WebSocket adapter; transport lifecycle is injected for tests."""
    def __init__(self, transport_factory:Callable[[str],object], access_token:str|None=None, timeout:float=10.0):
        self.transport_factory=transport_factory; self.access_token=access_token; self.timeout=timeout; self.transport=None; self.symbols=[]
    def connect(self):
        self.transport=self.transport_factory(authorize_v3(self.access_token,self.timeout))
    def subscribe(self, instrument_keys:list[str], mode:str="ltpc"):
        if self.transport is None: raise UpstoxV3MarketDataError("not connected")
        self.symbols=list(dict.fromkeys(instrument_keys)); self.transport.send(subscription_payload(self.symbols,mode))
    def recv_ticks(self):
        if self.transport is None: raise UpstoxV3MarketDataError("not connected")
        return decode_ticks(self.transport.recv())
    def close(self):
        if self.transport: self.transport.close(); self.transport=None
