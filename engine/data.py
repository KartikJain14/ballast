"""
Ingest and validate data/treasury_4tier_dataset.csv.

Design goal (OPERATING RULE #4 -- "no look-ahead, ever"): the only way code
downstream of this module can get at the data is through `HistoryView`, whose
`as_of(t)` method physically cannot return rows with timestamp > t. There is no
"just don't peek" discipline required of callers -- the accessor makes peeking
impossible by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "timestamp", "deposits_usd", "withdrawals_usd", "net_flow_usd",
    "repo_sweep_window_open", "repo_sweep_mature", "tier1_repo_rate_apy",
    "tier2_bond_yield_pct", "tier2_bond_price_index", "tier2_bond_mod_duration",
    "tier3_stock_index_price", "tier3_stock_volatility_pct",
    "tier1_early_redemption_haircut_pct", "tier2_bond_liquidation_haircut_pct",
    "tier3_stock_liquidation_haircut_pct", "emergency_overdraft_penalty_apr_pct",
]

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "treasury_4tier_dataset.csv"


class DataValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    n_rows: int
    n_expected: int
    is_continuous_hourly: bool
    n_duplicate_timestamps: int
    n_missing_hours: int
    net_flow_matches_deposits_minus_withdrawals: bool
    max_net_flow_mismatch: float
    rate_unit_findings: dict
    warnings: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def load_raw(csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place treasury_4tier_dataset.csv there before continuing."
        )
    df = pd.read_csv(csv_path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ("repo_sweep_window_open", "repo_sweep_mature"):
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> ValidationReport:
    warnings: list[str] = []
    errors: list[str] = []

    n_rows = len(df)
    start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    full_range = pd.date_range(start, end, freq="h")
    n_expected = len(full_range)
    n_dupes = int(df["timestamp"].duplicated().sum())
    missing_hours = sorted(set(full_range) - set(df["timestamp"]))
    is_continuous = (n_dupes == 0) and (len(missing_hours) == 0) and (n_rows == n_expected)

    if n_dupes:
        errors.append(f"{n_dupes} duplicate timestamps found.")
    if missing_hours:
        warnings.append(f"{len(missing_hours)} missing hourly timestamps, e.g. {missing_hours[:3]}.")

    # net_flow_usd == deposits_usd - withdrawals_usd ?
    implied = df["deposits_usd"] - df["withdrawals_usd"]
    diff = (implied - df["net_flow_usd"]).abs().round(2)
    max_mismatch = float(diff.max())
    net_flow_ok = max_mismatch < 1e-2
    if not net_flow_ok:
        warnings.append(
            f"net_flow_usd != deposits-withdrawals for some rows (max abs mismatch ${max_mismatch:,.2f})."
        )

    # Unit sanity on rate/pct columns: APY/pct columns should look like "4.3" not "0.043"
    rate_cols = [
        "tier1_repo_rate_apy", "tier2_bond_yield_pct", "tier3_stock_volatility_pct",
        "tier1_early_redemption_haircut_pct", "tier2_bond_liquidation_haircut_pct",
        "tier3_stock_liquidation_haircut_pct", "emergency_overdraft_penalty_apr_pct",
    ]
    rate_unit_findings = {}
    for c in rate_cols:
        lo, hi, mean = df[c].min(), df[c].max(), df[c].mean()
        rate_unit_findings[c] = {"min": float(lo), "max": float(hi), "mean": float(mean)}
        if hi < 1.0 and mean < 1.0:
            warnings.append(
                f"Column {c} has max={hi:.4f} -- looks like a fraction (0-1), not a percent-points "
                f"value as its name implies. Verify before use."
            )
        if hi > 100:
            warnings.append(f"Column {c} has max={hi:.2f} -- unexpectedly large for a percent column.")

    if n_rows < 100:
        errors.append("Suspiciously few rows; refusing to proceed.")

    return ValidationReport(
        n_rows=n_rows,
        n_expected=n_expected,
        is_continuous_hourly=is_continuous,
        n_duplicate_timestamps=n_dupes,
        n_missing_hours=len(missing_hours),
        net_flow_matches_deposits_minus_withdrawals=net_flow_ok,
        max_net_flow_mismatch=max_mismatch,
        rate_unit_findings=rate_unit_findings,
        warnings=warnings,
        errors=errors,
    )


def load_and_validate(csv_path: Path | str = DEFAULT_CSV_PATH) -> tuple[pd.DataFrame, ValidationReport]:
    df = load_raw(csv_path)
    report = validate(df)
    if not report.ok:
        raise DataValidationError(f"Validation failed: {report.errors}")
    return df, report


class HistoryView:
    """A no-look-ahead accessor over the hourly dataset.

    `as_of(t_idx)` returns only rows with integer index <= t_idx (i.e. the row
    for hour t_idx and everything strictly before it never leaks a later row).
    A decision being made *for* hour t_idx must call `as_of(t_idx - 1)` to get
    "information available at the start of hour t_idx" -- see
    `engine.config.DataAssumptions.event_ordering`.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)
        self._n = len(self._df)

    def __len__(self) -> int:
        return self._n

    def as_of(self, t_idx: int) -> pd.DataFrame:
        if t_idx < -1:
            raise IndexError("t_idx must be >= -1 (use -1 for 'nothing known yet')")
        return self._df.iloc[: t_idx + 1]

    def row_at(self, t_idx: int) -> pd.Series:
        """The realized row for hour t_idx. Only to be used by the simulator
        when *settling* hour t_idx, never by a forecaster/policy deciding hour t_idx."""
        return self._df.iloc[t_idx]

    def full_unsafe(self) -> pd.DataFrame:
        """Escape hatch for audit/backtesting scripts that intentionally need
        the full series (e.g. computing realized M_d for historical scoring).
        Never call this from a policy or forecaster."""
        return self._df
