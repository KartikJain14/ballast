import pandas as pd
import pytest

from engine.config import EngineConfig
from engine.data import load_raw, DEFAULT_CSV_PATH
from engine.simulator.engine import Simulator
from engine.policy.oracle import OraclePolicy
from engine.risk.pathmax import compute_nightly_path_max


@pytest.fixture(scope="module")
def df():
    return load_raw(DEFAULT_CSV_PATH)


def test_oracle_never_breaches_when_capital_is_sufficient(df):
    """The central correctness property of OraclePolicy: whenever cash
    available at 17:00 is >= the night's realized path-max M, the oracle's
    exact buffer (cash - M) must produce ZERO overnight overdraft/breaks --
    by construction, M is the worst cumulative outflow point, so buffer=M
    leaves the trajectory exactly non-negative throughout.

    On this dataset there ARE nights where cash < M (the book is genuinely
    capital-constrained by Q4 due to the -$2.53B structural annual decline --
    docs/DATA_AUDIT.md Section 14); no policy, oracle included, can avoid
    overdraft on those specific nights, and this test correctly excludes them
    rather than asserting an impossible global zero-breach guarantee.
    """
    cfg = EngineConfig.full_4tier()
    sim = Simulator(df, cfg)
    res = sim.run(OraclePolicy(df))
    trace = res.trace.set_index("timestamp")

    nights = compute_nightly_path_max(df)
    n_checked = 0
    for night_start, M in zip(nights["night_start"], nights["M"]):
        prior_hour = night_start - pd.Timedelta(hours=1)
        if prior_hour not in trace.index:
            continue
        cash_before = trace.loc[prior_hour, "tier0"]
        if cash_before < M:
            continue  # genuinely capital-constrained; no policy can help here
        window = trace.loc[night_start: night_start + pd.Timedelta(hours=14)]
        assert not (window["waterfall_overdraft"] > 1e-6).any(), (
            f"Oracle breached on {night_start} despite sufficient capital "
            f"(cash_before={cash_before:,.0f} >= M={M:,.0f}) -- this would be a real bug."
        )
        n_checked += 1
    assert n_checked > 250  # most of the year should have sufficient capital
