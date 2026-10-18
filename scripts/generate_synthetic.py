"""
make synthetic -> python scripts/generate_synthetic.py

Generates a few variant datasets (same 16-column schema as
treasury_4tier_dataset.csv) for exploring how Ballast behaves on different
kinds of books. These are DERIVED from the real sample dataset's own
hour-of-day/day-of-week shape and rate/haircut columns -- only the deposit/
withdrawal trend or volatility is deliberately altered, and each variant says
exactly how in its own docstring below. Never presented as real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data import load_raw, DEFAULT_CSV_PATH  # noqa: E402
from engine.stress.scenarios import svb_calibrated_run  # noqa: E402

OUT_DIR = ROOT / "data"
RNG = np.random.default_rng(42)


def _recompute_net(df: pd.DataFrame) -> pd.DataFrame:
    df["net_flow_usd"] = df["deposits_usd"] - df["withdrawals_usd"]
    return df


def make_growing_bank(base: pd.DataFrame) -> pd.DataFrame:
    """A healthy, growing deposit franchise: deposits drift up ~+40%/year
    relative to withdrawals (the base dataset drifts down -- this is the
    mirror image), same intraday/weekly shape and same rate environment."""
    df = base.copy()
    n = len(df)
    growth = 1.0 + 0.40 * (np.arange(n) / n)  # deposits scale up to +40% by year-end
    df["deposits_usd"] = df["deposits_usd"] * growth
    return _recompute_net(df)


def make_volatile_bank(base: pd.DataFrame) -> pd.DataFrame:
    """Same average flow levels, but hour-to-hour noise is ~2.5x larger --
    a book with a much choppier, less predictable customer base (e.g. a
    platform with concentrated institutional flows rather than diversified
    retail deposits)."""
    df = base.copy()
    for col in ["deposits_usd", "withdrawals_usd"]:
        df["hod"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek
        seasonal_mean = df.groupby(["dow", "hod"])[col].transform("mean")
        deviation = df[col] - seasonal_mean
        df[col] = (seasonal_mean + deviation * 2.5).clip(lower=0)
    df = df.drop(columns=["hod", "dow"])
    return _recompute_net(df)


def make_bank_run_aftermath(base: pd.DataFrame) -> pd.DataFrame:
    """An SVB-calibrated run (25% of deposit base requested in 24h, doubled
    queue the next day -- same shock engine/stress/scenarios.py uses live in
    the Stress Lab) baked permanently into the dataset around the midpoint,
    so the back half of the year shows a book recovering from an acute,
    already-happened crisis rather than a live simulated one."""
    df = base.copy()
    deposit_base_estimate = df["deposits_usd"].mean() * 24 * 90  # ~90 days of gross deposits
    shock_start = df["timestamp"].iloc[int(len(df) * 0.5)].replace(hour=9, minute=0)
    df = svb_calibrated_run(df, shock_start, deposit_base_estimate, day1_frac=0.25, day2_queue_multiplier=2.0)
    return df


def main():
    base = load_raw(DEFAULT_CSV_PATH)
    variants = {
        "synthetic_growing_bank.csv": make_growing_bank(base),
        "synthetic_volatile_bank.csv": make_volatile_bank(base),
        "synthetic_bank_run_aftermath.csv": make_bank_run_aftermath(base),
    }
    for name, df in variants.items():
        out_path = OUT_DIR / name
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
