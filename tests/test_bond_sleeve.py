from engine.config import EngineConfig
from engine.data import load_raw, DEFAULT_CSV_PATH
from engine.simulator.engine import Simulator
from engine.policy.ballast_policy import BallastPolicy


def test_bond_sleeve_defaults_off():
    cfg = EngineConfig.full_4tier()
    assert cfg.risk.bond_dv01_cap_frac == 0.0


def test_bond_sleeve_off_means_zero_tier2_balance():
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    sim = Simulator(df, cfg)
    res = sim.run(BallastPolicy())
    assert (res.trace["tier2"] == 0).all()


def test_bond_sleeve_mechanism_works_when_opted_in():
    """The sleeve is a real, functional, documented-off-by-default capability
    (D9) -- confirm it actually allocates to Tier2 when a treasurer opts in,
    even though the default policy doesn't use it."""
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    cfg.risk.bond_dv01_cap_frac = 0.10
    sim = Simulator(df, cfg)
    res = sim.run(BallastPolicy())
    assert res.trace["tier2"].mean() > 0
