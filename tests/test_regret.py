from engine.config import EngineConfig
from engine.data import load_raw, DEFAULT_CSV_PATH
from engine.simulator.engine import Simulator
from engine.policy.ballast_policy import BallastPolicy
from engine.policy.oracle import OraclePolicy
from engine.risk.regret import decompose_regret, summarize_regret


def test_regret_buckets_sum_to_total():
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    sim = Simulator(df, cfg)
    res = sim.run(BallastPolicy())
    rdf = decompose_regret(df, cfg, res.trace)
    s = summarize_regret(rdf)
    assert abs(s["buckets_sum_check"] - s["total_regret"]) < 1e-6


def test_oracle_has_near_zero_regret_against_itself():
    """Should be exactly zero in theory; in practice the 0.1% max_single_sweep_frac
    safety floor (engine/config.py) still binds on ~14 exceptionally calm nights
    where realized M is near zero, leaving a small residual (~$625, down from
    ~$239K before raising the cap from 0.98->0.999 -- docs/DECISIONS.md D10).
    This residual is the deliberate cost of keeping a nonzero safety backstop;
    chasing it to literal zero would mean removing the backstop entirely."""
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    sim = Simulator(df, cfg)
    res = sim.run(OraclePolicy(df))
    rdf = decompose_regret(df, cfg, res.trace)
    s = summarize_regret(rdf)
    assert abs(s["total_regret"]) < 1000.0


def test_capital_constrained_count_matches_known_finding():
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    sim = Simulator(df, cfg)
    res = sim.run(BallastPolicy())
    rdf = decompose_regret(df, cfg, res.trace)
    s = summarize_regret(rdf)
    assert s["n_capital_constrained_nights"] == 66  # D8 finding
