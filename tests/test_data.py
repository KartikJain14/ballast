from hypothesis import given, strategies as st
import pandas as pd
import pytest

from engine.data import load_raw, validate, HistoryView, DEFAULT_CSV_PATH


def test_dataset_loads():
    df = load_raw(DEFAULT_CSV_PATH)
    assert len(df) == 8760
    assert df["timestamp"].is_monotonic_increasing


def test_dataset_is_continuous_hourly():
    df = load_raw(DEFAULT_CSV_PATH)
    report = validate(df)
    assert report.n_duplicate_timestamps == 0
    assert report.is_continuous_hourly


def test_net_flow_identity():
    df = load_raw(DEFAULT_CSV_PATH)
    report = validate(df)
    assert report.max_net_flow_mismatch <= 0.01


def test_history_view_cannot_see_future():
    df = load_raw(DEFAULT_CSV_PATH)
    hv = HistoryView(df)
    view = hv.as_of(100)
    assert len(view) == 101
    assert view["timestamp"].max() == df["timestamp"].iloc[100]
    # the row immediately after must not be visible
    assert df["timestamp"].iloc[101] not in set(view["timestamp"])


def test_history_view_as_of_minus_one_is_empty():
    df = load_raw(DEFAULT_CSV_PATH)
    hv = HistoryView(df)
    assert len(hv.as_of(-1)) == 0


@given(st.integers(min_value=0, max_value=8759))
def test_history_view_never_exceeds_requested_index(t_idx):
    df = load_raw(DEFAULT_CSV_PATH)
    hv = HistoryView(df)
    view = hv.as_of(t_idx)
    assert len(view) == t_idx + 1
    assert view.index.max() == t_idx


def test_history_view_rejects_out_of_range():
    df = load_raw(DEFAULT_CSV_PATH)
    hv = HistoryView(df)
    with pytest.raises(IndexError):
        hv.as_of(-2)
