import pandas as pd

from scripts.frozen_pivot_config import (
    CONFIG_VERSION,
    V3_HYBRID_CONFIG,
    V4_SPLIT_VERSION,
    V5_SHORT_ONLY_VERSION,
    FrozenPivotConfig,
    V1_SHORT_ONLY_CONFIG,
    apply_frozen_filters,
    apply_side_entry_time_filter,
    config_for_symbol,
    mark_side_taken,
    max_risk_for,
    side_quota_available,
    spot_paper_entry_window_ok,
)


def test_config_version_is_v5_short_only_default():
    assert CONFIG_VERSION == V5_SHORT_ONLY_VERSION
    assert config_for_symbol("NIFTY") == V1_SHORT_ONLY_CONFIG
    assert config_for_symbol("BANKNIFTY") == V1_SHORT_ONLY_CONFIG


def test_spot_paper_entry_windows_by_side():
    ts_long = pd.Timestamp("2026-10-15 09:55:00+05:30")
    ts_short = pd.Timestamp("2026-10-15 11:55:00+05:30")
    assert spot_paper_entry_window_ok("NIFTY", ts_long, "LONG")
    assert spot_paper_entry_window_ok("NIFTY", ts_short, "SHORT")
    assert not spot_paper_entry_window_ok("NIFTY", ts_short, "LONG")


def test_frozen_filters_keep_long_and_short_with_vwap():
    frame = pd.DataFrame({
        "ts": pd.to_datetime(
            ["2026-10-15 09:55:00+05:30", "2026-10-15 11:55:00+05:30"],
            utc=True,
        ).tz_convert("Asia/Kolkata"),
        "session": [pd.Timestamp("2026-10-15").date()] * 2,
        "long_signal": [True, False],
        "short_signal": [False, True],
        "close_fut": [102.0, 98.0],
        "high_fut": [103.0, 99.0],
        "low_fut": [101.0, 97.0],
        "volume_fut": [1000.0, 1000.0],
        "close": [100.0, 100.0],
        "high": [101.0, 101.0],
        "low": [99.0, 98.0],
    })
    cfg = FrozenPivotConfig(require_vwap_above_for_long=False, require_vwap_below_for_short=False)
    out = apply_frozen_filters(frame, "NIFTY", cfg, execution_basis="futures")
    assert out.long_signal.iloc[0]
    assert out.short_signal.iloc[1]
    assert not out.long_signal.iloc[1]
    assert not out.short_signal.iloc[0]


def test_side_time_filter_blocks_banknifty_morning_short():
    frame = pd.DataFrame({
        "ts": pd.to_datetime(["2026-10-16 09:25:00+05:30"], utc=True).tz_convert("Asia/Kolkata"),
        "long_signal": [False],
        "short_signal": [True],
    })
    out = apply_side_entry_time_filter(frame, "BANKNIFTY")
    assert not out.short_signal.iloc[0]


def test_max_risk_defaults():
    assert max_risk_for("NIFTY") == 40.0
    assert max_risk_for("BANKNIFTY") == 100.0


def test_v4_split_uses_v1_short_only_on_nifty():
    nifty = config_for_symbol("NIFTY", V4_SPLIT_VERSION)
    bn = config_for_symbol("BANKNIFTY", V4_SPLIT_VERSION)
    assert nifty == V1_SHORT_ONLY_CONFIG
    assert bn == V3_HYBRID_CONFIG


def test_v5_short_only_on_both_symbols():
    assert config_for_symbol("NIFTY", V5_SHORT_ONLY_VERSION) == V1_SHORT_ONLY_CONFIG
    assert config_for_symbol("BANKNIFTY", V5_SHORT_ONLY_VERSION) == V1_SHORT_ONLY_CONFIG


def test_side_quota_helpers():
    cfg = V1_SHORT_ONLY_CONFIG
    assert side_quota_available(cfg, "SHORT", long_taken=False, short_taken=False, trade_taken=False)
    assert not side_quota_available(cfg, "LONG", long_taken=False, short_taken=False, trade_taken=False)
    long_taken, short_taken, trade_taken = mark_side_taken(
        cfg, "SHORT", long_taken=False, short_taken=False, trade_taken=False,
    )
    assert trade_taken
    assert not side_quota_available(
        cfg, "SHORT", long_taken=long_taken, short_taken=short_taken, trade_taken=trade_taken,
    )
