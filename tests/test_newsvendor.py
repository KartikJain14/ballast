import numpy as np
import pytest

from engine.risk.newsvendor import critical_fractile, effective_quantile, empirical_quantile_buffer


def test_critical_fractile_matches_brief_range_at_dataset_mean_rate():
    # brief Section 2.3: "q* ~= 97.7%-98.6%" for a 0.5% break fee and ~4.3% repo
    q = critical_fractile(break_fee_pct=0.5, repo_rate_apy_pct=4.055)
    assert 0.977 <= q <= 0.987


def test_critical_fractile_increases_with_break_fee():
    q_small_fee = critical_fractile(0.1, 4.0)
    q_large_fee = critical_fractile(2.0, 4.0)
    assert q_large_fee > q_small_fee


def test_policy_overlay_takes_the_stricter_quantile():
    q_star = critical_fractile(0.5, 4.055)  # ~0.986
    assert effective_quantile(0.5, 4.055, q_policy=0.999) == pytest.approx(0.999)
    assert effective_quantile(0.5, 4.055, q_policy=0.50) == pytest.approx(q_star)


def test_empirical_quantile_buffer_cold_start_fallback():
    assert empirical_quantile_buffer(np.array([1.0, 2.0]), 0.99, min_samples=8, fallback=42.0) == 42.0


def test_empirical_quantile_buffer_with_enough_samples():
    samples = np.arange(1, 101, dtype=float)  # 1..100
    b = empirical_quantile_buffer(samples, 0.95, min_samples=8)
    assert b == pytest.approx(95.0, abs=1.0)
