#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_create_dataset.py
===============================================================================
Purpose:
    Build a merged *complete* dataset for the Polymarket Corporate Earnings study
    and write BOTH:
      (1) LONG dataset: one row per market x brier snapshot row
      (2) WIDE dataset: one row per market, brier variables pivoted by horizon

Inputs (relative to project root):
    data/markets/markets.jsonl
    data/corporate_info/corporate_info_by_market.jsonl
    data/brier_scores/brier_scores_market_horizon.jsonl

Outputs (relative to project root):
    data/complete_dataset_long.csv
    data/complete_dataset_long.jsonl
    data/complete_dataset_wide.csv
    data/complete_dataset_wide.jsonl

Design choices:
    - The LONG dataset uses all rows from brier_scores_market_horizon.jsonl
      (repeats market/corporate fields for each snapshot).
    - The WIDE dataset uses ONE row per (slug, horizon). If the brier file has
      multiple rows per (slug, horizon), we pick a canonical row:
          * minimize abs(seconds_before_close - horizon_seconds)
          * tie-breaker: latest snapshot_dt_utc
    - Wide columns look like: p_polymarket_yes__12h, loss_polymarket__1w, ...

Usage (from project root):
    python python/07_create_dataset.py
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column specs (as requested)
# ---------------------------------------------------------------------------

MARKETS_COLS = [
    "id",
    "ticker",
    "slug",
    "resolvedOutcome",
    "umaEndDate",
    "acceptingOrdersTimestamp",
    "volumeNum",
    "val_eikon_eps_stddev_estimate",
    "val_surprise",
]

CORPORATE_COLS = [
    "slug",  # join key
    "ric",
    "earnings_release_datetime",
    "asof_date",
    "market_cap_usd_asof",
    "analysts_covering_asof",
    "gics_sector",
    "turnover_6m_sum_volume",
    "turnover_6m_avg_daily_volume",
    "volatility_6m",
]

BRIER_COLS = [
    "slug",  # join key
    "horizon",
    "horizon_seconds",
    "p_hist_leakage_safe",
    "snapshot_dt_utc",
    "p_polymarket_yes",
    "p_dice_0p5",
    "loss_polymarket",
    "loss_dice",
    "loss_hist",
    "seconds_before_event",
    "status"
]

# In the WIDE dataset, we pivot only these brier variables across horizons.
# (We do NOT pivot `horizon` itself; it's the column key.)
BRIER_PIVOT_VARS = [
    "p_polymarket_yes",
    "p_dice_0p5",
    "p_hist_leakage_safe",
    "loss_polymarket",
    "loss_dice",
    "loss_hist",
    "seconds_before_event",
    "snapshot_dt_utc",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_root_from_this_file() -> Path:
    """
    Infer project root from this script's location.
    Expected structure: <root>/python/07_create_dataset.py
    """
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> pd.DataFrame:
    """Read JSONL to DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_json(path, lines=True)


def require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    """Fail fast if required columns are missing."""
    required = list(required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def coerce_slug(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Clean join key."""
    if "slug" not in df.columns:
        raise KeyError(f"{df_name} has no 'slug' column.")
    out = df.copy()
    out["slug"] = out["slug"].astype(str).str.strip()
    out.loc[out["slug"].isin(["nan", "None", ""]), "slug"] = pd.NA
    return out


def dedupe_one_row_per_slug(
    df: pd.DataFrame, df_name: str, date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Defensive: enforce <=1 row/slug for market-level tables.
    If duplicates exist:
      - keep latest date_col if provided
      - else keep first row
    """
    if "slug" not in df.columns:
        raise KeyError(f"{df_name} has no 'slug' column.")

    out = df.copy()
    before = len(out)

    if out["slug"].duplicated().any():
        if date_col and date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=True)
            out = out.sort_values(["slug", date_col], kind="mergesort").drop_duplicates(
                ["slug"], keep="last"
            )
        else:
            out = out.drop_duplicates(["slug"], keep="first")

        after = len(out)
        print(
            f"[WARN] {df_name}: duplicate slugs detected; deduplicated {before} -> {after} rows.",
            file=sys.stderr,
        )
    return out


def write_outputs(df: pd.DataFrame, out_csv: Path, out_jsonl: Path) -> None:
    """Save CSV + JSONL."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_json(out_jsonl, orient="records", lines=True, date_format="iso")
    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_jsonl}")


def canonical_brier_per_slug_horizon(brier: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce brier to exactly 1 row per (slug, horizon) for the WIDE dataset.

    Rule:
      1) minimize abs(seconds_before_event - horizon_seconds)
      2) tie-breaker: latest snapshot_dt_utc
    """
    b = brier.copy()

    # Ensure types
    b["horizon"] = b["horizon"].astype(str).str.strip()
    b.loc[b["horizon"].isin(["nan", "None", ""]), "horizon"] = pd.NA

    b["snapshot_dt_utc"] = pd.to_datetime(b["snapshot_dt_utc"], errors="coerce", utc=True)
    b["horizon_seconds"] = pd.to_numeric(b["horizon_seconds"], errors="coerce")
    b["seconds_before_event"] = pd.to_numeric(b["seconds_before_event"], errors="coerce")

    # Compute abs difference; if missing, treat as very large so it loses unless all are missing
    b["abs_diff"] = (b["seconds_before_event"] - b["horizon_seconds"]).abs()
    b["abs_diff"] = b["abs_diff"].fillna(float("inf"))

    # Sort so "best" row is first per group
    # (smallest abs_diff, then latest snapshot_dt_utc)
    b = b.sort_values(
        ["slug", "horizon", "abs_diff", "snapshot_dt_utc"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )

    # Drop to one row per (slug, horizon)
    before = len(b)
    b = b.dropna(subset=["slug", "horizon"])
    b = b.drop_duplicates(["slug", "horizon"], keep="first")
    after = len(b)

    dropped = before - after
    if dropped > 0:
        print(
            f"[INFO] WIDE selection: kept 1 row per (slug,horizon); dropped {dropped:,} extra brier rows.",
            file=sys.stderr,
        )

    return b


def horizon_order(brier: pd.DataFrame) -> List[str]:
    """
    Build a stable horizon ordering (small -> large) using median horizon_seconds per horizon.
    """
    tmp = brier.copy()
    tmp["horizon"] = tmp["horizon"].astype(str)
    tmp["horizon_seconds"] = pd.to_numeric(tmp["horizon_seconds"], errors="coerce")
    grp = (
        tmp.dropna(subset=["horizon", "horizon_seconds"])
        .groupby("horizon", as_index=False)["horizon_seconds"]
        .median()
        .sort_values("horizon_seconds")
    )
    ordered = grp["horizon"].tolist()

    # Fallback: include any horizons with missing seconds at the end
    all_h = sorted([h for h in tmp["horizon"].dropna().unique().tolist()])
    for h in all_h:
        if h not in ordered:
            ordered.append(h)
    return ordered


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetPaths:
    markets: Path
    corporate: Path
    brier: Path
    outdir: Path


def build_datasets(paths: DatasetPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_df, wide_df)."""
    # --- Load
    markets = read_jsonl(paths.markets)
    corporate = read_jsonl(paths.corporate)
    brier = read_jsonl(paths.brier)

    # --- Validate required columns
    require_columns(markets, MARKETS_COLS, "markets.jsonl")
    require_columns(corporate, CORPORATE_COLS, "corporate_info_by_market.jsonl")
    require_columns(brier, BRIER_COLS, "brier_scores_market_horizon.jsonl")

    # --- Select columns
    markets = markets[MARKETS_COLS].copy()
    corporate = corporate[CORPORATE_COLS].copy()
    brier = brier[BRIER_COLS].copy()

    # --- Clean join keys
    markets = coerce_slug(markets, "markets.jsonl")
    corporate = coerce_slug(corporate, "corporate_info_by_market.jsonl")
    brier = coerce_slug(brier, "brier_scores_market_horizon.jsonl")

    # Defensive dedupe for 1-row-per-market tables
    markets = dedupe_one_row_per_slug(markets, "markets.jsonl")
    corporate = dedupe_one_row_per_slug(corporate, "corporate_info_by_market.jsonl", date_col="asof_date")

    # Parse datetime on brier for sorting/selection
    brier["snapshot_dt_utc"] = pd.to_datetime(brier["snapshot_dt_utc"], errors="coerce", utc=True)

    # --- Base dataset (1 row per slug)
    base = markets.merge(
        corporate,
        on="slug",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------------
    # LONG dataset: keep ALL brier rows, repeat base info across them
    # -----------------------------------------------------------------------
    long_df = base.merge(
        brier,
        on="slug",
        how="left",
        validate="one_to_many",
    )

    # -----------------------------------------------------------------------
    # WIDE dataset: 1 row per slug; pivot brier vars across HORIZON
    # -----------------------------------------------------------------------
    brier_one = canonical_brier_per_slug_horizon(brier)

    # Build horizon ordering for nicer column ordering
    h_order = horizon_order(brier_one)

    # Pivot each requested brier variable by horizon into columns var__horizon
    wide_parts = []
    for var in BRIER_PIVOT_VARS:
        if var not in brier_one.columns:
            continue

        tmp = brier_one.pivot_table(
            index="slug",
            columns="horizon",
            values=var,
            aggfunc="first",
        )

        # Reindex columns to stable horizon order (then any leftover)
        cols_present = [h for h in h_order if h in tmp.columns]
        tmp = tmp.reindex(columns=cols_present)

        tmp.columns = [f"{var}__{h}" for h in tmp.columns.astype(str)]
        wide_parts.append(tmp)

    if wide_parts:
        brier_wide = pd.concat(wide_parts, axis=1).reset_index()
    else:
        brier_wide = pd.DataFrame({"slug": base["slug"].unique()})

    wide_df = base.merge(
        brier_wide,
        on="slug",
        how="left",
        validate="one_to_one",
    )

    # Small diagnostic prints (helps catch unexpected explosion)
    n_horizons = brier_one["horizon"].nunique(dropna=True)
    print(f"[INFO] Unique horizons used in WIDE: {n_horizons:,}")
    print(f"[INFO] WIDE shape: {wide_df.shape[0]:,} rows x {wide_df.shape[1]:,} cols")
    print(f"[INFO] LONG shape: {long_df.shape[0]:,} rows x {long_df.shape[1]:,} cols")

    # If horizons are unexpectedly huge, warn loudly
    if n_horizons > 100:
        print(
            "[WARN] You have >100 unique `horizon` values. "
            "That will still create many wide columns. "
            "This suggests `horizon` may not be the intended snapshot label.",
            file=sys.stderr,
        )

    return long_df, wide_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(project_root: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create merged LONG and WIDE datasets for the Polymarket Earnings study."
    )
    p.add_argument(
        "--markets",
        type=str,
        default=str(project_root / "data" / "markets" / "markets.jsonl"),
        help="Path to markets.jsonl",
    )
    p.add_argument(
        "--corporate",
        type=str,
        default=str(project_root / "data" / "corporate_info" / "corporate_info_by_market.jsonl"),
        help="Path to corporate_info_by_market.jsonl",
    )
    p.add_argument(
        "--brier",
        type=str,
        default=str(project_root / "data" / "brier_scores" / "brier_scores_market_horizon.jsonl"),
        help="Path to brier_scores_market_horizon.jsonl",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(project_root / "data"),
        help="Output directory (default: <root>/data)",
    )
    return p.parse_args()


def main() -> int:
    root = project_root_from_this_file()
    args = parse_args(root)

    paths = DatasetPaths(
        markets=Path(args.markets),
        corporate=Path(args.corporate),
        brier=Path(args.brier),
        outdir=Path(args.outdir),
    )

    long_df, wide_df = build_datasets(paths)

    # Output paths
    out_long_csv = paths.outdir / "complete_dataset_long.csv"
    out_long_jsonl = paths.outdir / "complete_dataset_long.jsonl"
    out_wide_csv = paths.outdir / "complete_dataset_wide.csv"
    out_wide_jsonl = paths.outdir / "complete_dataset_wide.jsonl"

    write_outputs(long_df, out_long_csv, out_long_jsonl)
    write_outputs(wide_df, out_wide_csv, out_wide_jsonl)

    print("\n[SUMMARY]")
    print(f"  Long rows (market x snapshot rows): {long_df.shape[0]:,}")
    print(f"  Wide rows (markets): {wide_df.shape[0]:,}")
    print(f"  Wide cols: {wide_df.shape[1]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
