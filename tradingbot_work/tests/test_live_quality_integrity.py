from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from src.backtest import live_quality_gate as qg
from src.paper_trading import live_quality_monitor, live_quality_report
from src.paper_trading import shadow_validation as sv

ROOT = Path(__file__).resolve().parents[1]
GATE_FILES = [Path(qg.__file__), Path(live_quality_monitor.__file__), Path(live_quality_report.__file__),
              ROOT / "scripts" / "run_live_quality_gate.py", ROOT / "scripts" / "run_live_quality_report.py"]
FORBIDDEN_IMPORTS = ("src.broker", "src.execution", "src.engine", "upstox_sandbox", "order_manager")
FORBIDDEN_CALLS = ("place_order", "submit_signal", "submit_entry", "modify_order", "cancel_order")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def test_strategy_yaml_is_unmodified_by_gate_and_paper_mode():
    cfg = (ROOT / "config" / "strategy.yaml").read_text(encoding="utf-8")
    assert "quality_gate" not in cfg.lower() and "LIVE_QUALITY_GATE" not in cfg
    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    assert settings["trading"]["mode"] == "PAPER"
    assert settings["trading"]["live_trading_enabled"] is False
    for path in GATE_FILES:
        assert "strategy.yaml" not in path.read_text(encoding="utf-8")


def test_no_live_broker_or_order_imports_in_quality_gate():
    for path in GATE_FILES:
        for mod in _imports(path):
            assert not any(f in mod for f in FORBIDDEN_IMPORTS), f"{path.name} imports {mod}"


def test_no_order_placement_calls():
    for path in GATE_FILES:
        src = path.read_text(encoding="utf-8")
        for call in FORBIDDEN_CALLS:
            assert f"{call}(" not in src, f"{path.name} calls {call}"


def test_gate_decision_reads_only_causal_whitelist():
    src = Path(qg.__file__).read_text(encoding="utf-8")
    body = src[src.index("def evaluate_quality"):src.index("def restrictiveness")]
    reader = src[src.index("def _read"):src.index("def evaluate_quality")]
    assert "ctx[" not in body and "ctx.get" not in body
    assert "ctx.get(k)" in reader and "for k in CAUSAL_FEATURES" in reader
    for field in ("outcome", "pnl", "exit_price", "exit_time", "exit_reason", "r_multiple"):
        assert field not in qg.CAUSAL_FEATURES
        assert not re.search(rf"[\"']{field}[\"']", body + reader)


def test_feature_builder_uses_only_candles_up_to_signal():
    src = Path(qg.__file__).read_text(encoding="utf-8")
    fn = src[src.index("def gate_features_at"):src.index("def gate_context")]
    assert "stamps <= ts" in fn
    assert "shift(-" not in src and ".iloc[-1]" in fn


def test_s2_baseline_candidate_unchanged():
    assert [c.name for c in sv.CANDIDATES] == ["BASELINE_S2_BUY_SELL", "SELL_ONLY", "SELL_CONFIRMED"]
    base = sv.CANDIDATES[0]
    assert base.directions == ("BUY", "SELL") and base.confirm is False


def test_buy_breakout_confirm_is_not_production():
    assert "BUY_BREAKOUT_CONFIRM" not in [c.name for c in sv.CANDIDATES]
    assert "BUY_BREAKOUT_CONFIRM" not in sv.OBSERVATION_CANDIDATES
    assert "BUY_BREAKOUT_CONFIRM" not in Path(qg.__file__).read_text(encoding="utf-8")
    assert sv.report_candidates([{"candidate": "BUY_BREAKOUT_CONFIRM"}]) == [c.name for c in sv.CANDIDATES]


def test_gate_version_is_explicit_and_disabled_for_production():
    assert qg.GATE_VERSION == "LIVE_QUALITY_GATE_V1" == qg.GATE_CANDIDATE
    assert qg.PRODUCTION_EXECUTION_ENABLED is False
    assert qg.evaluate_quality({"direction": "SELL"}).version == "LIVE_QUALITY_GATE_V1"


def test_production_code_never_imports_the_gate():
    allowed = {Path(qg.__file__).resolve(), Path(sv.__file__).resolve(), Path(live_quality_report.__file__).resolve()}
    for path in (ROOT / "src").rglob("*.py"):
        if path.resolve() in allowed:
            continue
        assert "live_quality_gate" not in path.read_text(encoding="utf-8"), path
