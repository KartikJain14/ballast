import pandas as pd
import pytest

from engine.risk.pathmax import night_id, compute_nightly_path_max, is_weekend_night


def _hours(start, n, freq="h"):
    return pd.date_range(start, periods=n, freq=freq)


def test_night_id_boundaries():
    ts = pd.Series(pd.to_datetime([
        "2026-01-05 16:00",  # daytime, not in any night
        "2026-01-05 17:00",  # start of Mon night
        "2026-01-05 23:00",  # still Mon night
        "2026-01-06 07:00",  # still Mon night (last hour, s+14h)
        "2026-01-06 08:00",  # mature hour -- excluded, belongs to no night
        "2026-01-06 16:00",  # daytime
    ]))
    nid = night_id(ts)
    assert pd.isna(nid.iloc[0])
    assert nid.iloc[1] == pd.Timestamp("2026-01-05")
    assert nid.iloc[2] == pd.Timestamp("2026-01-05")
    assert nid.iloc[3] == pd.Timestamp("2026-01-05")
    assert pd.isna(nid.iloc[4])
    assert pd.isna(nid.iloc[5])


def test_compute_nightly_path_max_single_night_known_values():
    # Build exactly one complete night: 17:00 -> next 07:00 (15 hours), plus
    # padding hours before/after that must NOT be included.
    ts = list(_hours("2026-01-05 10:00", 7))      # 10:00..16:00 (daytime, excluded)
    ts += list(_hours("2026-01-05 17:00", 15))     # the night itself
    ts += list(_hours("2026-01-06 08:00", 5))      # next day daytime, excluded
    n = len(ts)
    net_flow = [0.0] * 7 + [-1_000_000.0] * 5 + [2_000_000.0] * 10 + [0.0] * 5
    # within the night: 5 hours of -1M outflow (cum outflow climbs to +5M),
    # then 10 hours of +2M inflow (cum outflow drops back down)
    df = pd.DataFrame({"timestamp": ts, "net_flow_usd": net_flow[:n]})
    nights = compute_nightly_path_max(df)
    assert len(nights) == 1
    row = nights.iloc[0]
    assert row["night_start"] == pd.Timestamp("2026-01-05 17:00")
    assert row["M"] == pytest.approx(5_000_000.0)
    assert row["n_hours"] == 15


def test_incomplete_night_excluded_by_default():
    ts = list(_hours("2026-01-05 17:00", 10))  # only 10 of 15 hours present
    df = pd.DataFrame({"timestamp": ts, "net_flow_usd": [0.0] * 10})
    nights = compute_nightly_path_max(df)
    assert len(nights) == 0


def test_is_weekend_night():
    assert is_weekend_night(pd.Timestamp("2026-01-02 17:00"))  # Friday
    assert is_weekend_night(pd.Timestamp("2026-01-03 17:00"))  # Saturday
    assert not is_weekend_night(pd.Timestamp("2026-01-04 17:00"))  # Sunday
    assert not is_weekend_night(pd.Timestamp("2026-01-05 17:00"))  # Monday


def test_matches_full_dataset_audit_numbers():
    from engine.data import load_raw, DEFAULT_CSV_PATH
    df = load_raw(DEFAULT_CSV_PATH)
    nights = compute_nightly_path_max(df)
    assert len(nights) == 364
    assert nights["M"].mean() == pytest.approx(12_983_555.32, rel=1e-6)
    assert nights["M"].max() == pytest.approx(46_660_691.28, rel=1e-6)
