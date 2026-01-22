#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
05_descriptive_statistics.py

Polymarket Corporate Earnings — Descriptive Statistics + Plots (+ Corporate coverage)

UPDATES (per your latest feedback)
----------------------------------
1) Exchange labels in the *plot* are abbreviated (e.g., NYSE, NASDAQ-GS, NASDAQ-CM, ...).
   - We keep the RAW exchange distribution table, and also save an "abbrev" table.
2) Removed plots:
   - 10_top_tickers.png  (table is still saved)
   - 13_distance_to_certainty_hist_leq_1d.png (table is still saved)
3) Replaced the old histogram 14_markets_per_ticker_hist.png with an integer-x bar plot:
   - New name: 14_firm_repeat_counts_distribution.png
   - New title focuses on how many firms appear more than once (repeat count distribution).
   - Also saves a summary table: firms seen >=2 times.
4) All bar-plot x tick labels are rotated straight down (90 degrees) to reduce overlap.
   (Outcome plot stays unrotated where it’s already clean.)

NEW (this request)
------------------
5) Timeline plot: stacked dots by day for when markets ended (UTC date).
   - Uses observed end timestamps from prices if available; falls back to markets end fields.
   - Outputs:
     - 02b_market_end_dates_counts.csv
     - 02b_market_end_dates_timeline.png
6) “All variables” summary table (numeric columns) with:
   min, mean, max, p0.05, p0.95, stdev, IQR (+ n)
   - Outputs:
     - 11b_all_variables_summary_stats.csv

OUTPUT
------
- Tables are CSV only.
- Plots are PNG only.
- Output directory is overwritten each run.

DEFAULT PATHS (relative to project root containing "data/")
----------------------------------------------------------
- data/markets/markets.jsonl
- data/prices/historical_prices.jsonl
- data/validation/correct.jsonl
- data/corporate_info/corporate_info.jsonl
- statistics/descriptive_statistics/

Run
---
python 04_descriptive_statistics.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Snapshot configuration
# -----------------------------
SNAPSHOT_ORDER = ["4w", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h"]
SNAPSHOT_ORDER_SET = set(SNAPSHOT_ORDER)

# -----------------------------
# Project-root + default paths
# -----------------------------
def _this_file_fallback() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve()
    return Path.cwd() / "04_descriptive_statistics.py"


def find_project_root(start: Path) -> Path:
    """
    Find the project root by walking upwards until we find a folder containing 'data/'.
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data").is_dir():
            return p
    return start


PROJECT_ROOT = find_project_root(_this_file_fallback().parent)

DEFAULT_MARKETS_PATH = PROJECT_ROOT / "data" / "markets" / "markets.jsonl"
DEFAULT_PRICES_PATH = PROJECT_ROOT / "data" / "poly_prices" / "poly_prices.jsonl"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "correct.jsonl"
DEFAULT_CORPORATE_INFO_PATH = PROJECT_ROOT / "data" / "corporate_info" / "corporate_info.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "statistics" / "descriptive_statistics"


# -----------------------------
# I/O + directory management
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def ensure_clean_dir(out_dir: Path) -> None:
    """
    Overwrite behavior: delete output directory entirely, then recreate.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_table_csv(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / f"{stem}.csv", index=False)


def save_plot_png(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "plots" / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Snapshot ordering helpers
# -----------------------------
def snapshot_sort_key(snapshot: str) -> Tuple[int, str]:
    if snapshot in SNAPSHOT_ORDER_SET:
        return (SNAPSHOT_ORDER.index(snapshot), snapshot)
    return (10_000, snapshot)


def sort_snapshots(values: List[str]) -> List[str]:
    return sorted(values, key=lambda s: snapshot_sort_key(str(s)))


def parse_snapshot_to_hours(snapshot: str) -> Optional[float]:
    s = str(snapshot).strip().lower()
    try:
        if s.endswith("h"):
            return float(s[:-1])
        if s.endswith("d"):
            return float(s[:-1]) * 24.0
        if s.endswith("w"):
            return float(s[:-1]) * 7.0 * 24.0
    except Exception:
        return None
    return None


def snapshot_leq_1d(snapshot: str) -> bool:
    h = parse_snapshot_to_hours(snapshot)
    return (h is not None) and (h <= 24.0)


# -----------------------------
# Descriptive stats helpers
# -----------------------------
def describe_numeric(series: pd.Series, name: str) -> pd.DataFrame:
    s = safe_numeric(series).dropna()
    if s.empty:
        return pd.DataFrame([{
            "variable": name, "n": 0, "mean": np.nan, "std": np.nan,
            "min": np.nan, "p01": np.nan, "p05": np.nan, "p10": np.nan,
            "p25": np.nan, "median": np.nan, "p75": np.nan, "p90": np.nan,
            "p95": np.nan, "p99": np.nan, "max": np.nan
        }])

    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    std = float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0
    return pd.DataFrame([{
        "variable": name,
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": std,
        "min": float(s.min()),
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p10": float(q.get(0.10, np.nan)),
        "p25": float(q.get(0.25, np.nan)),
        "median": float(q.get(0.50, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "p90": float(q.get(0.90, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "max": float(s.max()),
    }])


def summarize_numeric_minmax_p05_p95_iqr(series: pd.Series, dataset: str, variable: str) -> Optional[Dict[str, Any]]:
    """
    Minimal numeric summary for the “all variables” table:
    min, mean, max, p0.05, p0.95, stdev, IQR (p75-p25), and n.
    Returns None if no numeric data exists.
    """
    s = safe_numeric(series).dropna()
    if s.empty:
        return None

    p05 = float(s.quantile(0.05))
    p25 = float(s.quantile(0.25))
    p75 = float(s.quantile(0.75))
    p95 = float(s.quantile(0.95))
    iqr = p75 - p25
    std = float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0

    return {
        "dataset": dataset,
        "variable": variable,
        "n": int(s.shape[0]),
        "min": float(s.min()),
        "mean": float(s.mean()),
        "max": float(s.max()),
        "p0_05": p05,
        "p0_95": p95,
        "stdev": std,
        "iqr": float(iqr),
    }


def plot_hist(series: pd.Series, title: str, xlabel: str, out_dir: Path, stem: str, bins: int = 40) -> None:
    s = safe_numeric(series).dropna()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if s.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_title(title)
        save_plot_png(fig, out_dir, stem)
        return

    ax.hist(s.values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    save_plot_png(fig, out_dir, stem)


def plot_log_hist(series: pd.Series, title: str, xlabel: str, out_dir: Path, stem: str, bins: int = 40) -> None:
    s = safe_numeric(series).dropna()
    s = s[s > 0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if s.empty:
        ax.text(0.5, 0.5, "No positive values available", ha="center", va="center")
        ax.set_title(title)
        save_plot_png(fig, out_dir, stem)
        return

    vmin, vmax = float(s.min()), float(s.max())
    if vmin <= 0 or vmax <= 0 or vmin == vmax:
        ax.hist(s.values, bins=bins)
        ax.set_title(title + " (fallback linear bins)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        save_plot_png(fig, out_dir, stem)
        return

    edges = np.logspace(np.log10(vmin), np.log10(vmax), bins)
    ax.hist(s.values, bins=edges)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel + " (log axis)")
    ax.set_ylabel("Count")
    save_plot_png(fig, out_dir, stem)


def plot_bar(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    rotate: int = 90,  # default straight down
    order: Optional[List[str]] = None,
    top_n: Optional[int] = None,
) -> None:
    """
    Bar plot helper:
    - defaults to rotate=90 for x tick labels (straight down).
    - uses a wider figure when there are many categories.
    """
    s = counts.copy()

    if order is not None:
        idx = [x for x in order if x in s.index]
        rest = [x for x in s.index if x not in idx]
        s = pd.concat([s.loc[idx], s.loc[sorted(rest)]]) if len(rest) else s.loc[idx]

    if top_n is not None and s.shape[0] > top_n:
        s = s.sort_values(ascending=False).head(top_n)

    fig_w = max(8.0, 0.35 * max(1, len(s)))
    fig = plt.figure(figsize=(fig_w, 5.0))
    ax = fig.add_subplot(111)
    ax.bar(s.index.astype(str), s.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    save_plot_png(fig, out_dir, stem)


def frequency_table(series: pd.Series, name: str) -> pd.DataFrame:
    s = series.copy().astype("object")
    missing_n = int(s.isna().sum())
    s2 = s.fillna("(missing)").astype(str).str.strip()
    counts = s2.value_counts(dropna=False)
    tbl = counts.reset_index()
    tbl.columns = [name, "count"]
    denom = float(tbl["count"].sum()) if tbl.shape[0] else 0.0
    tbl["pct"] = (tbl["count"] / denom * 100.0) if denom else np.nan
    tbl.attrs["missing_n"] = missing_n
    return tbl


# -----------------------------
# End-date / timeline helpers
# -----------------------------
def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _coerce_datetime_utc(series: pd.Series) -> pd.Series:
    """
    Best-effort conversion of a column into UTC timestamps (pandas datetime64[ns, UTC]).
    Handles:
      - numeric unix seconds or milliseconds
      - ISO strings (with or without timezone)
    """
    if series is None:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    s = series.copy()

    # If it's already datetime-like:
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            # ensure UTC (if naive, pandas may treat as naive; we coerce with to_datetime(utc=True))
            return pd.to_datetime(s, utc=True, errors="coerce")
        except Exception:
            return pd.to_datetime(s.astype(str), utc=True, errors="coerce")

    # Try numeric first
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() > 0:
        # infer seconds vs milliseconds by magnitude
        med = float(num.dropna().median())
        unit = "ms" if med > 1e12 else "s"
        return pd.to_datetime(num, unit=unit, utc=True, errors="coerce")

    # Fallback parse as string datetime
    return pd.to_datetime(s.astype(str), utc=True, errors="coerce")


def build_market_end_dates(markets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table with market_id/slug and a best-effort end datetime (UTC) and end_date (UTC date).

    Priority:
      1) observed end timestamp from prices (if present)
      2) end timestamp/date fields from markets (if present)
    """
    base_cols = ["market_id"] + (["slug"] if "slug" in markets.columns else [])
    base = markets[base_cols].copy() if all(c in markets.columns for c in ["market_id"]) else pd.DataFrame()

    if base.empty and "market_id" in prices.columns:
        base_cols = ["market_id"] + (["slug"] if "slug" in prices.columns else [])
        base = prices[base_cols].copy()

    if base.empty:
        return pd.DataFrame(columns=["market_id", "slug", "end_dt_utc", "end_date_utc", "end_source"])

    # Candidate columns
    price_end_candidates = [
        "observed_end_ts", "observed_end_timestamp", "observed_end_ts_s", "observed_end_ts_sec",
        "observed_end_ts_seconds", "observed_end_ts_ms", "observed_end_timestamp_ms",
        "observed_end_time", "observed_end_datetime", "observed_end_iso", "observed_end",
    ]
    market_end_candidates = [
        "end_date", "endDate", "endDateIso", "endDateISO", "endDateTime", "end_time", "endTime",
        "endTimestamp", "end_ts", "endTs", "endTsMs", "end_ts_ms", "end_ts_s", "endTimestampMs",
        "closeTime", "resolutionTime", "resolvedTime",
    ]

    # Build end_dt from prices if possible
    end_dt_prices = None
    price_col = _first_existing_column(prices, price_end_candidates)
    if price_col is not None:
        tmp = prices[["market_id"] + (["slug"] if "slug" in prices.columns else []) + [price_col]].copy()
        tmp["end_dt_prices"] = _coerce_datetime_utc(tmp[price_col])
        join_cols = ["market_id"] + (["slug"] if ("slug" in tmp.columns and "slug" in base.columns) else [])
        end_dt_prices = tmp[join_cols + ["end_dt_prices"]].drop_duplicates(subset=join_cols, keep="first")

    # Build end_dt from markets if possible
    end_dt_markets = None
    mkt_col = _first_existing_column(markets, market_end_candidates)
    if mkt_col is not None:
        tmp = markets[["market_id"] + (["slug"] if "slug" in markets.columns else []) + [mkt_col]].copy()
        tmp["end_dt_markets"] = _coerce_datetime_utc(tmp[mkt_col])
        join_cols = ["market_id"] + (["slug"] if ("slug" in tmp.columns and "slug" in base.columns) else [])
        end_dt_markets = tmp[join_cols + ["end_dt_markets"]].drop_duplicates(subset=join_cols, keep="first")

    out = base.copy()
    join_cols = ["market_id"] + (["slug"] if "slug" in out.columns else [])

    if end_dt_prices is not None:
        out = out.merge(end_dt_prices, on=join_cols, how="left")
    else:
        out["end_dt_prices"] = pd.NaT

    if end_dt_markets is not None:
        out = out.merge(end_dt_markets, on=join_cols, how="left")
    else:
        out["end_dt_markets"] = pd.NaT

    out["end_dt_utc"] = out["end_dt_prices"].where(out["end_dt_prices"].notna(), out["end_dt_markets"])
    out["end_source"] = np.where(out["end_dt_prices"].notna(), "prices", np.where(out["end_dt_markets"].notna(), "markets", "missing"))

    out["end_date_utc"] = out["end_dt_utc"].dt.date
    out = out.drop(columns=["end_dt_prices", "end_dt_markets"], errors="ignore")

    return out


def plot_market_end_timeline(end_dates: pd.DataFrame, out_dir: Path, stem: str) -> None:
    """
    Timeline plot: one dot per market end date; multiple dots on same date are stacked vertically.
    """
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(111)

    if end_dates.empty or end_dates["end_date_utc"].isna().all():
        ax.text(0.5, 0.5, "No end-date data available to plot", ha="center", va="center")
        ax.set_title("Market end dates timeline (stacked dots)")
        save_plot_png(fig, out_dir, stem)
        return

    # Count markets per date
    counts = (
        end_dates.dropna(subset=["end_date_utc"])
        .groupby("end_date_utc", as_index=True)
        .size()
        .sort_index()
    )

    xs: List[pd.Timestamp] = []
    ys: List[int] = []

    for d, c in counts.items():
        dt = pd.to_datetime(str(d), utc=True)
        for j in range(int(c)):
            xs.append(dt)
            ys.append(j + 1)

    ax.scatter(xs, ys)
    ax.set_title("Market end dates timeline (stacked dots per day, UTC)")
    ax.set_xlabel("End date (UTC)")
    ax.set_ylabel("Stack index (markets ended on that day)")

    # Make date labels readable
    fig.autofmt_xdate(rotation=45, ha="right")
    save_plot_png(fig, out_dir, stem)


# -----------------------------
# Data transforms
# -----------------------------
def build_prices_long(prices_df: pd.DataFrame, complement_tolerance_default: float) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for _, row in prices_df.iterrows():
        prices_yes = row.get("prices_yes") or {}
        prices_no = row.get("prices_no") or {}

        labels = set(prices_yes.keys()) | set(prices_no.keys())
        for snap in labels:
            records.append({
                "market_id": str(row.get("market_id")),
                "slug": row.get("slug"),
                "snapshot": snap,
                "yes_price": prices_yes.get(snap, None),
                "no_price": prices_no.get(snap, None),
                "complement_tolerance": row.get("complement_tolerance", complement_tolerance_default),
            })

    long_df = pd.DataFrame(records)
    long_df["yes_price"] = safe_numeric(long_df["yes_price"])
    long_df["no_price"] = safe_numeric(long_df["no_price"])

    long_df["sum_price"] = long_df["yes_price"] + long_df["no_price"]
    long_df["abs_sum_minus_1"] = (long_df["sum_price"] - 1.0).abs()

    tol = safe_numeric(long_df["complement_tolerance"]).fillna(complement_tolerance_default)
    long_df["violates_tolerance"] = (
        long_df["yes_price"].notna()
        & long_df["no_price"].notna()
        & (long_df["abs_sum_minus_1"] > tol)
    )

    return long_df


def flatten_tags(tags_val: Any) -> List[str]:
    if tags_val is None:
        return []
    if isinstance(tags_val, list):
        return [str(t).strip().lower() for t in tags_val if str(t).strip()]
    if isinstance(tags_val, str):
        parts = [p.strip().lower() for p in tags_val.split(",")]
        return [p for p in parts if p]
    return []


# -----------------------------
# Corporate info helpers
# -----------------------------
def normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype("object").astype(str).str.strip().str.upper().replace({"NAN": np.nan, "NONE": np.nan})


def dedupe_by_best_nonnull(df: pd.DataFrame, key: str, prefer_cols: List[str]) -> pd.DataFrame:
    if key not in df.columns:
        return df

    cols = [c for c in prefer_cols if c in df.columns]
    if not cols:
        return df.drop_duplicates(subset=[key], keep="first")

    tmp = df.copy()
    tmp["_nonnull_score"] = tmp[cols].notna().sum(axis=1)
    tmp = tmp.sort_values(by=[key, "_nonnull_score"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=[key], keep="first").drop(columns=["_nonnull_score"])
    return tmp


def abbreviate_exchange(raw: Any) -> str:
    """
    Abbreviate exchange strings so the plot is readable.

    Examples:
    - "New York Stock Exchange" -> "NYSE"
    - "New York Consolidated" -> "NYSE"
    - "... Nasdaq Global Select Market" -> "NASDAQ-GS"
    - "... Nasdaq Capital Market" -> "NASDAQ-CM"
    - "NASDAQ National Market System" -> "NASDAQ"
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "(missing)"
    s = str(raw).strip()
    if not s:
        return "(missing)"
    u = s.upper()

    if "NEW YORK STOCK EXCHANGE" in u or "NYSE" in u or "NEW YORK CONSOLIDATED" in u:
        if "NYSE AMERICAN" in u or "AMEX" in u:
            return "NYSEA"
        return "NYSE"

    if "NASDAQ" in u:
        if "GLOBAL SELECT" in u:
            return "NASDAQ-GS"
        if "GLOBAL MARKET" in u:
            return "NASDAQ-GM"
        if "CAPITAL" in u:
            return "NASDAQ-CM"
        if "NATIONAL MARKET SYSTEM" in u or " NMS" in u:
            return "NASDAQ"
        return "NASDAQ"

    if "BOSTON STOCK EXCHANGE" in u or "BSE" in u:
        return "BSE"

    if "LONDON STOCK EXCHANGE" in u or u == "LSE" or " LSE" in u:
        return "LSE"

    if "TORONTO" in u or "TSX" in u:
        return "TSX"

    if "FRANKFURT" in u or "XETRA" in u:
        return "FRA/XETRA"

    stop = {"THE", "OF", "AND", "ON", "LISTED", "CONSOLIDATED", "ISSUE", "MARKET", "SYSTEM", "STOCK", "EXCHANGE", "INC"}
    parts = [p for p in u.replace(",", " ").replace(".", " ").split() if p and p not in stop]
    if not parts:
        return s[:10]
    acr = "".join([p[0] for p in parts[:6]])
    return acr if len(acr) >= 2 else s[:10]


def load_and_prepare_corporate_info(corporate_info_path: Path) -> pd.DataFrame:
    corp = pd.DataFrame(read_jsonl(corporate_info_path))
    if corp.empty:
        return corp

    ticker_col = _first_existing_column(corp, ["ticker", "Ticker", "symbol", "Symbol"])
    if ticker_col is None:
        return corp

    corp = corp.copy()
    corp["ticker"] = normalize_ticker(corp[ticker_col])

    rename_map: Dict[str, str] = {}

    mcap_col = _first_existing_column(corp, ["market_cap_usd", "market_cap", "marketCap", "mkt_cap", "mktCap", "MarketCap"])
    if mcap_col and mcap_col != "market_cap":
        rename_map[mcap_col] = "market_cap"

    hq_col = _first_existing_column(corp, ["hq_country", "hqCountry", "country", "Country", "hq_country_code"])
    if hq_col and hq_col != "hq_country":
        rename_map[hq_col] = "hq_country"

    exch_col = _first_existing_column(corp, ["main_exchange", "mainExchange", "exchange", "Exchange", "primary_exchange"])
    if exch_col and exch_col != "main_exchange":
        rename_map[exch_col] = "main_exchange"

    gics_s_col = _first_existing_column(corp, ["gics_sector", "gicsSector", "sector", "Sector"])
    if gics_s_col and gics_s_col != "gics_sector":
        rename_map[gics_s_col] = "gics_sector"

    gics_i_col = _first_existing_column(corp, ["gics_industry", "gicsIndustry", "industry", "Industry"])
    if gics_i_col and gics_i_col != "gics_industry":
        rename_map[gics_i_col] = "gics_industry"
    
    analyst_col = _first_existing_column(
        corp,
        [
            "analysts_covering_sample_median", "analystCoverage",
            "analyst_coverage_count", "analystCoverageCount",
            "num_analysts", "numAnalysts",
            "n_analysts", "nAnalysts",
            "analyst_count", "analystCount",
        ],
    )
    if analyst_col and analyst_col != "analyst_coverage":
        rename_map[analyst_col] = "analyst_coverage"

    turn_col = _first_existing_column(
        corp,
        ["turnover_6m_sum_volume_mean", "turnover6m_sum_volume_mean", "turnover_6m", "turnover6m", "avg_turnover_6m"],
    )
    if turn_col and turn_col != "turnover_6m_sum_volume_mean":
        rename_map[turn_col] = "turnover_6m_sum_volume_mean"

    if rename_map:
        corp = corp.rename(columns=rename_map)

    prefer = ["market_cap", "hq_country", "main_exchange", "gics_sector", "gics_industry", "turnover_6m_sum_volume_mean", "analyst_coverage",]
    corp = dedupe_by_best_nonnull(corp, key="ticker", prefer_cols=prefer)

    if "market_cap" in corp.columns:
        s = corp["market_cap"].astype("object")
        s = s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        corp["market_cap"] = pd.to_numeric(s, errors="coerce")
    if "turnover_6m_sum_volume_mean" in corp.columns:
        corp["turnover_6m_sum_volume_mean"] = safe_numeric(corp["turnover_6m_sum_volume_mean"])
    if "analyst_coverage" in corp.columns:
        corp["analyst_coverage"] = safe_numeric(corp["analyst_coverage"])

    for c in ["hq_country", "main_exchange", "gics_sector", "gics_industry"]:
        if c in corp.columns:
            corp[c] = corp[c].astype("object").where(corp[c].notna(), np.nan)
            corp[c] = corp[c].astype("object").astype(str).str.strip()
            corp.loc[corp[c].str.upper().isin(["NAN", "NONE", ""]), c] = np.nan

    if "main_exchange" in corp.columns:
        corp["main_exchange_abbrev"] = corp["main_exchange"].map(abbreviate_exchange)

    return corp


# -----------------------------
# Main analysis
# -----------------------------
def run_descriptive_statistics(
    markets_path: Path,
    prices_path: Path,
    validation_path: Path,
    corporate_info_path: Path,
    out_dir: Path,
    complement_tolerance_default: float = 0.05,
    top_n_tags: int = 30,
    top_n_tickers: int = 25,  # still used for the ticker TABLE, but the plot is removed
    top_n_countries: int = 30,
    top_n_exchanges: int = 30,
    top_n_industries: int = 40,
) -> None:
    ensure_clean_dir(out_dir)

    markets = pd.DataFrame(read_jsonl(markets_path))
    prices = pd.DataFrame(read_jsonl(prices_path))
    validation = pd.DataFrame(read_jsonl(validation_path))

    corp = pd.DataFrame()
    if corporate_info_path.exists():
        corp = load_and_prepare_corporate_info(corporate_info_path)

    # Normalize IDs
    if "id" in markets.columns:
        markets["market_id"] = markets["id"].astype(str)
    elif "market_id" in markets.columns:
        markets["market_id"] = markets["market_id"].astype(str)

    if "market_id" in prices.columns:
        prices["market_id"] = prices["market_id"].astype(str)
    if "market_id" in validation.columns:
        validation["market_id"] = validation["market_id"].astype(str)

    # Normalize tickers in markets for joining with corporate_info
    if "ticker" in markets.columns:
        markets["ticker_norm"] = normalize_ticker(markets["ticker"])
    else:
        markets["ticker_norm"] = np.nan

    # -----------------------------
    # 00) Dataset sizes + missingness
    # -----------------------------
    def missingness_table(df: pd.DataFrame, name: str, key_cols: List[str]) -> pd.DataFrame:
        n = df.shape[0]
        rows = [{"dataset": name, "n_rows": n, "n_cols": df.shape[1], "field": "__ALL__",
                 "missing_n": int(df.isna().sum().sum()), "missing_pct": np.nan}]
        for c in key_cols:
            if c in df.columns:
                rows.append({
                    "dataset": name,
                    "n_rows": n,
                    "n_cols": df.shape[1],
                    "field": c,
                    "missing_n": int(df[c].isna().sum()),
                    "missing_pct": float(df[c].isna().mean()) * 100.0,
                })
            else:
                rows.append({
                    "dataset": name,
                    "n_rows": n,
                    "n_cols": df.shape[1],
                    "field": c,
                    "missing_n": n,
                    "missing_pct": 100.0,
                })
        return pd.DataFrame(rows)

    miss_parts = [
        missingness_table(markets, "markets", ["market_id", "slug", "ticker", "resolvedOutcome", "volumeNum", "tags"]),
        missingness_table(prices, "prices", ["market_id", "slug", "observed_span_hours", "prices_yes", "prices_no"]),
        missingness_table(
            validation,
            "validation",
            ["market_id", "slug", "polymarket_estimate", "eikon_eps_mean_estimate", "eikon_actual_eps", "surprise", "label"],
        ),
    ]
    if not corp.empty:
        miss_parts.append(
            missingness_table(
                corp,
                "corporate_info",
                ["ticker", "market_cap", "hq_country", "main_exchange", "gics_sector",
                 "gics_industry", "turnover_6m_sum_volume_mean", "analyst_coverage"],
            )
        )

    save_table_csv(pd.concat(miss_parts, ignore_index=True), out_dir, "00_dataset_sizes_and_missingness")

    # -----------------------------
    # Build long prices
    # -----------------------------
    prices_long = build_prices_long(prices, complement_tolerance_default)

    # -----------------------------
    # 01) Observations per snapshot
    # -----------------------------
    obs_counts = (
        prices_long.assign(
            has_yes=prices_long["yes_price"].notna(),
            has_no=prices_long["no_price"].notna(),
            has_both=prices_long["yes_price"].notna() & prices_long["no_price"].notna(),
        )
        .groupby("snapshot", as_index=False)
        .agg(n_yes=("has_yes", "sum"), n_no=("has_no", "sum"), n_both=("has_both", "sum"))
    )
    obs_counts = obs_counts.sort_values(by="snapshot", key=lambda s: s.map(lambda x: snapshot_sort_key(str(x))[0]))
    save_table_csv(obs_counts, out_dir, "01_observations_per_snapshot")

    plot_bar(
        counts=obs_counts.set_index("snapshot")["n_both"],
        title="Number of markets with BOTH YES and NO prices by snapshot",
        xlabel="Snapshot label (ordered)",
        ylabel="Count (both YES & NO available)",
        out_dir=out_dir,
        stem="01_observations_per_snapshot_n_both",
        rotate=90,
        order=SNAPSHOT_ORDER,
    )

    # -----------------------------
    # 02) Distribution of observed_span_hours
    # -----------------------------
    if "observed_span_hours" in prices.columns:
        prices["observed_span_hours"] = safe_numeric(prices["observed_span_hours"])
        save_table_csv(describe_numeric(prices["observed_span_hours"], "observed_span_hours"),
                       out_dir, "02_observed_span_hours_describe")
        plot_hist(prices["observed_span_hours"],
                  "Distribution of observed market active span (hours)",
                  "observed_span_hours",
                  out_dir,
                  "02_observed_span_hours_hist",
                  bins=40)

    # -----------------------------
    # 02b) Market end-date timeline (stacked dots) + counts table
    # -----------------------------
    try:
        end_dates = build_market_end_dates(markets=markets, prices=prices)
        # Counts table
        if (not end_dates.empty) and end_dates["end_date_utc"].notna().any():
            counts_tbl = (
                end_dates.dropna(subset=["end_date_utc"])
                .groupby(["end_date_utc", "end_source"], as_index=False)
                .size()
                .rename(columns={"size": "n_markets"})
                .sort_values(["end_date_utc", "end_source"])
            )
            save_table_csv(counts_tbl, out_dir, "02b_market_end_dates_counts")
        else:
            note = [
                "Market end-date timeline was skipped because no end-date columns were found/parsable.",
                "Tried end-date candidates in prices (observed_end_*) and markets (endDate/endTimestamp/etc).",
            ]
            (out_dir / "logs" / "MARKET_END_DATES_NOTE.txt").write_text("\n".join(note), encoding="utf-8")

        # Plot
        plot_market_end_timeline(end_dates, out_dir, "02b_market_end_dates_timeline")
    except Exception as e:
        (out_dir / "logs" / "MARKET_END_DATES_ERROR.txt").write_text(
            f"Failed to build/plot market end dates.\nError: {repr(e)}",
            encoding="utf-8",
        )

    # -----------------------------
    # 03) Distribution of outcomes
    # -----------------------------
    if "resolvedOutcome" in markets.columns:
        outcomes_market = markets["resolvedOutcome"].astype(str).replace({"nan": np.nan}).dropna()
        outcomes_market = outcomes_market.str.strip().str.upper()
        outcome_counts_mkt = outcomes_market.value_counts()
        outcome_table_mkt = outcome_counts_mkt.reset_index()
        outcome_table_mkt.columns = ["resolvedOutcome_marketsjsonl", "count"]
        outcome_table_mkt["pct"] = outcome_table_mkt["count"] / outcome_table_mkt["count"].sum() * 100.0
        save_table_csv(outcome_table_mkt, out_dir, "03_outcomes_distribution_markets_table_only")

    if "polymarket_resolved_outcome" in validation.columns:
        outcomes_val = validation["polymarket_resolved_outcome"].astype(str).replace({"nan": np.nan}).dropna()
        outcomes_val = outcomes_val.str.strip().str.upper()
        outcome_counts_val = outcomes_val.value_counts()
        outcome_table_val = outcome_counts_val.reset_index()
        outcome_table_val.columns = ["polymarket_resolved_outcome_validation", "count"]
        outcome_table_val["pct"] = outcome_table_val["count"] / outcome_table_val["count"].sum() * 100.0
        save_table_csv(outcome_table_val, out_dir, "03_outcomes_distribution_validation")

        plot_bar(
            counts=outcome_counts_val,
            title="Resolved outcomes distribution (correct.jsonl)",
            xlabel="Outcome",
            ylabel="Count",
            out_dir=out_dir,
            stem="03_outcomes_distribution_validation",
            rotate=0,
        )

    # -----------------------------
    # 04) Complement tolerance violations
    # -----------------------------
    viol_by_snap = (
        prices_long.groupby("snapshot", as_index=False)
        .agg(
            n_both=("snapshot", lambda s: int((prices_long.loc[s.index, "yes_price"].notna()
                                              & prices_long.loc[s.index, "no_price"].notna()).sum())),
            n_violations=("violates_tolerance", "sum"),
        )
    )
    viol_by_snap["violation_rate_pct"] = np.where(
        viol_by_snap["n_both"] > 0,
        viol_by_snap["n_violations"] / viol_by_snap["n_both"] * 100.0,
        np.nan,
    )
    viol_by_snap = viol_by_snap.sort_values(by="snapshot", key=lambda s: s.map(lambda x: snapshot_sort_key(str(x))[0]))
    save_table_csv(viol_by_snap, out_dir, "04_complement_violations_by_snapshot")

    plot_bar(
        counts=viol_by_snap.set_index("snapshot")["violation_rate_pct"],
        title="Complement violation rate by snapshot (|YES+NO-1| > tolerance)",
        xlabel="Snapshot label (ordered)",
        ylabel="Violation rate (%)",
        out_dir=out_dir,
        stem="04_complement_violations_rate_by_snapshot",
        rotate=90,
        order=SNAPSHOT_ORDER,
    )

    overall_n_both = int((prices_long["yes_price"].notna() & prices_long["no_price"].notna()).sum())
    overall_n_viol = int(prices_long["violates_tolerance"].sum())
    save_table_csv(pd.DataFrame([{
        "overall_n_rows_prices_long": int(prices_long.shape[0]),
        "overall_n_both_prices": overall_n_both,
        "overall_n_violations": overall_n_viol,
        "overall_violation_rate_pct": (overall_n_viol / overall_n_both * 100.0) if overall_n_both else np.nan,
        "tolerance_default_used_when_missing": complement_tolerance_default,
    }]), out_dir, "04_complement_violations_overall")

    # -----------------------------
    # 05) Tags excluding "earnings"
    # -----------------------------
    if "tags" in markets.columns:
        tags_all: List[str] = []
        for v in markets["tags"].tolist():
            tags_all.extend(flatten_tags(v))
        tags_ser = pd.Series(tags_all, dtype="object")
        tags_ser = tags_ser[tags_ser != "earnings"]
        tag_counts = tags_ser.value_counts()

        tag_table = tag_counts.reset_index()
        tag_table.columns = ["tag", "count"]
        denom = tag_table["count"].sum()
        tag_table["pct_of_all_non_earnings_tags"] = (tag_table["count"] / denom * 100.0) if denom else np.nan
        save_table_csv(tag_table, out_dir, "05_tags_distribution_excluding_earnings")

        plot_bar(
            counts=tag_counts,
            title='Tag frequency (excluding "earnings")',
            xlabel="Tag",
            ylabel="Count",
            out_dir=out_dir,
            stem="05_tags_top",
            rotate=90,
            top_n=top_n_tags,
        )

    # -----------------------------
    # 06) Volume distribution
    # -----------------------------
    if "volumeNum" in markets.columns:
        markets["volumeNum"] = safe_numeric(markets["volumeNum"])
        save_table_csv(describe_numeric(markets["volumeNum"], "volumeNum"), out_dir, "06_volume_describe")

        plot_hist(markets["volumeNum"],
                  "Distribution of market volume (volumeNum)",
                  "volumeNum",
                  out_dir,
                  "06_volume_hist_linear",
                  bins=40)

        plot_log_hist(markets["volumeNum"],
                      "Distribution of market volume (volumeNum) — log axis + log-spaced bins",
                      "volumeNum",
                      out_dir,
                      "06_volume_hist_logbins",
                      bins=40)

        vol = markets["volumeNum"].dropna()
        vol = vol[vol > 0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if vol.empty:
            ax.text(0.5, 0.5, "No positive volume values available", ha="center", va="center")
        else:
            ax.hist(np.log10(vol.values), bins=40)
        ax.set_title("Distribution of log10(volumeNum)")
        ax.set_xlabel("log10(volumeNum)")
        ax.set_ylabel("Count")
        save_plot_png(fig, out_dir, "06_volume_hist_log10")

    # -----------------------------
    # 07-09) Validation numeric distributions
    # -----------------------------
    val = validation.copy()
    for c in ["polymarket_estimate", "eikon_eps_mean_estimate", "eikon_actual_eps", "surprise"]:
        if c in val.columns:
            val[c] = safe_numeric(val[c])

    if {"eikon_eps_mean_estimate", "polymarket_estimate"}.issubset(val.columns):
        val["diff_eikon_mean_minus_poly_est"] = val["eikon_eps_mean_estimate"] - val["polymarket_estimate"]
        save_table_csv(describe_numeric(val["diff_eikon_mean_minus_poly_est"], "diff_eikon_mean_minus_poly_est"),
                       out_dir, "07_diff_eikon_mean_minus_polymarket_estimate_describe")
        plot_hist(val["diff_eikon_mean_minus_poly_est"],
                  "Eikon mean EPS estimate minus Polymarket estimate",
                  "(eikon_eps_mean_estimate - polymarket_estimate)",
                  out_dir,
                  "07_diff_eikon_mean_minus_poly_est_hist",
                  bins=40)

    if {"eikon_actual_eps", "polymarket_estimate"}.issubset(val.columns):
        val["diff_eikon_actual_minus_poly_est"] = val["eikon_actual_eps"] - val["polymarket_estimate"]
        save_table_csv(describe_numeric(val["diff_eikon_actual_minus_poly_est"], "diff_eikon_actual_minus_poly_est"),
                       out_dir, "08_diff_eikon_actual_minus_polymarket_estimate_describe")
        plot_hist(val["diff_eikon_actual_minus_poly_est"],
                  "Eikon actual EPS minus Polymarket estimate",
                  "(eikon_actual_eps - polymarket_estimate)",
                  out_dir,
                  "08_diff_eikon_actual_minus_poly_est_hist",
                  bins=40)

    if "surprise" in val.columns:
        save_table_csv(describe_numeric(val["surprise"], "surprise"), out_dir, "09_surprise_describe")
        plot_hist(val["surprise"],
                  "Distribution of earnings surprise",
                  "surprise",
                  out_dir,
                  "09_surprise_hist",
                  bins=40)

    # -----------------------------
    # 10) Top tickers by market count (TABLE ONLY; plot removed)
    # -----------------------------
    if "ticker" in markets.columns:
        tick_counts = markets["ticker"].astype(str).replace({"nan": np.nan}).dropna().value_counts()
        tick_table = tick_counts.reset_index()
        tick_table.columns = ["ticker", "n_markets"]
        save_table_csv(tick_table, out_dir, "10_ticker_counts_all")

    # -----------------------------
    # 11) Numeric panel ("Table 1 style")
    # -----------------------------
    numeric_panels: List[pd.DataFrame] = []
    if "volumeNum" in markets.columns:
        numeric_panels.append(describe_numeric(markets["volumeNum"], "volumeNum"))
    if "observed_span_hours" in prices.columns:
        numeric_panels.append(describe_numeric(prices["observed_span_hours"], "observed_span_hours"))
    if "diff_eikon_mean_minus_poly_est" in val.columns:
        numeric_panels.append(describe_numeric(val["diff_eikon_mean_minus_poly_est"], "diff_eikon_mean_minus_poly_est"))
    if "diff_eikon_actual_minus_poly_est" in val.columns:
        numeric_panels.append(describe_numeric(val["diff_eikon_actual_minus_poly_est"], "diff_eikon_actual_minus_poly_est"))
    if "surprise" in val.columns:
        numeric_panels.append(describe_numeric(val["surprise"], "surprise"))

    if numeric_panels:
        save_table_csv(pd.concat(numeric_panels, ignore_index=True), out_dir, "11_numeric_descriptive_panel")

    # -----------------------------
    # 12) Span vs volume scatter (log-x) + slope line
    # -----------------------------
    if ("volumeNum" in markets.columns) and ("observed_span_hours" in prices.columns):
        if "slug" in markets.columns and "slug" in prices.columns:
            mp = markets[["market_id", "slug", "volumeNum"]].merge(
                prices[["market_id", "slug", "observed_span_hours"]],
                on=["market_id", "slug"],
                how="inner",
            )
        else:
            mp = markets[["market_id", "volumeNum"]].merge(
                prices[["market_id", "observed_span_hours"]],
                on=["market_id"],
                how="inner",
            )

        mp["volumeNum"] = safe_numeric(mp["volumeNum"])
        mp["observed_span_hours"] = safe_numeric(mp["observed_span_hours"])
        mp = mp.dropna(subset=["volumeNum", "observed_span_hours"])
        mp = mp[(mp["volumeNum"] > 0) & (mp["observed_span_hours"] >= 0)]

        if not mp.empty:
            x = mp["volumeNum"].values.astype(float)
            y = mp["observed_span_hours"].values.astype(float)
            lx = np.log10(x)

            b, a = np.polyfit(lx, y, 1)
            x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            y_line = a + b * np.log10(x_line)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x, y)
            ax.plot(x_line, y_line)
            ax.set_xscale("log")
            ax.set_title(
                "Market activity span vs volume (log volume) with slope line\n"
                f"Fit: span = {a:.3f} + {b:.3f}*log10(volume)"
            )
            ax.set_xlabel("volumeNum (log axis)")
            ax.set_ylabel("observed_span_hours")
            save_plot_png(fig, out_dir, "12_span_vs_volume_scatter_logx_with_slope")

            save_table_csv(pd.DataFrame([{"a_intercept": a, "b_slope_log10_volume": b, "n_obs": int(mp.shape[0])}]),
                           out_dir, "12_span_vs_volume_slope_coefficients")

    # -----------------------------
    # 13) Distance to certainty for snapshots <= 1d (TABLE ONLY; plot removed)
    # -----------------------------
    lt = prices_long.copy()
    lt = lt[lt["snapshot"].map(lambda s: snapshot_leq_1d(str(s)))]
    lt = lt[lt["yes_price"].notna() & lt["no_price"].notna()]

    if not lt.empty:
        lt["max_outcome_price"] = np.maximum(lt["yes_price"], lt["no_price"])
        lt["distance_to_1_from_max_outcome"] = 1.0 - lt["max_outcome_price"]

        rows = []
        for snap in sort_snapshots(list(lt["snapshot"].unique())):
            sub = lt.loc[lt["snapshot"] == snap, "distance_to_1_from_max_outcome"]
            d = describe_numeric(sub, f"distance_to_1_from_max_outcome__{snap}")
            r = d.iloc[0].to_dict()
            r["snapshot"] = snap
            rows.append(r)

        dist_tbl = pd.DataFrame(rows)
        cols = ["snapshot", "n", "mean", "std", "min", "p25", "median", "p75", "p90", "p95", "p99", "max"]
        dist_tbl = dist_tbl[[c for c in cols if c in dist_tbl.columns] + [c for c in dist_tbl.columns if c not in cols]]
        save_table_csv(dist_tbl, out_dir, "13_distance_to_certainty_by_snapshot_leq_1d")

    # ============================================================
    # Corporate descriptive statistics
    # ============================================================
    corp_in_dataset = pd.DataFrame()  # keep for “all variables” summary later

    if not corp.empty and "ticker" in corp.columns:
        tickers_in_markets = (
            markets["ticker_norm"].dropna().astype("object").astype(str).str.strip().str.upper().unique().tolist()
        )
        tickers_in_markets = [t for t in tickers_in_markets if t and t not in ["NAN", "NONE"]]

        corp_join = corp.copy()
        corp_join["ticker"] = normalize_ticker(corp_join["ticker"])
        corp_join = corp_join.dropna(subset=["ticker"])

        corp_in_dataset = corp_join[corp_join["ticker"].isin(set(tickers_in_markets))].copy()

        n_tickers = int(len(set(tickers_in_markets)))
        n_corp_matched = int(corp_in_dataset["ticker"].nunique()) if not corp_in_dataset.empty else 0
        matched_pct = (n_corp_matched / n_tickers * 100.0) if n_tickers else np.nan

        save_table_csv(pd.DataFrame([{
            "n_unique_tickers_in_markets": n_tickers,
            "n_unique_tickers_matched_in_corporate_info": n_corp_matched,
            "pct_tickers_matched": matched_pct,
            "n_rows_corporate_info_file": int(corp.shape[0]),
            "n_rows_corporate_info_used_after_join": int(corp_in_dataset.shape[0]) if not corp_in_dataset.empty else 0,
        }]), out_dir, "14_corporate_coverage_summary")

        unmatched = sorted(list(set(tickers_in_markets) - set(corp_in_dataset["ticker"].unique().tolist())))
        save_table_csv(pd.DataFrame({"ticker_unmatched_in_corporate_info": unmatched}),
                       out_dir, "14_corporate_unmatched_tickers")

        # 14) Firm repeat distribution (REPLACED)
        mpt = (
            markets.dropna(subset=["ticker_norm"])
            .groupby("ticker_norm", as_index=False)
            .agg(n_markets=("market_id", "count"))
            .rename(columns={"ticker_norm": "ticker"})
        )

        save_table_csv(describe_numeric(mpt["n_markets"], "n_markets_per_ticker"),
                       out_dir, "14_markets_per_ticker_describe")

        n_firms_total = int(mpt.shape[0])
        n_firms_repeat = int((mpt["n_markets"] >= 2).sum())
        save_table_csv(pd.DataFrame([{
            "n_firms_total": n_firms_total,
            "n_firms_seen_more_than_once": n_firms_repeat,
            "pct_firms_seen_more_than_once": (n_firms_repeat / n_firms_total * 100.0) if n_firms_total else np.nan,
            "max_markets_for_single_firm": int(mpt["n_markets"].max()) if n_firms_total else np.nan,
        }]), out_dir, "14_firms_seen_more_than_once_summary")

        dist = mpt["n_markets"].value_counts().sort_index()
        dist_tbl = dist.reset_index()
        dist_tbl.columns = ["n_markets_for_firm", "n_firms"]
        dist_tbl["pct_of_firms"] = dist_tbl["n_firms"] / dist_tbl["n_firms"].sum() * 100.0
        save_table_csv(dist_tbl, out_dir, "14_firm_repeat_counts_distribution")

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(dist_tbl["n_markets_for_firm"].astype(int).astype(str), dist_tbl["n_firms"].values)
        ax.set_title("Firm repeat frequency in the dataset\n(Number of markets per firm/ticker)")
        ax.set_xlabel("Number of markets for a firm (integer)")
        ax.set_ylabel("Number of firms")
        ax.tick_params(axis="x", rotation=0)
        save_plot_png(fig, out_dir, "14_firm_repeat_counts_distribution")

        # 15) Market cap
        if (not corp_in_dataset.empty) and ("market_cap" in corp_in_dataset.columns):
            save_table_csv(describe_numeric(corp_in_dataset["market_cap"], "market_cap"), out_dir, "15_market_cap_describe")

            plot_hist(
                corp_in_dataset["market_cap"],
                "Distribution of corporate market cap (USD) — linear x-axis",
                "market_cap (USD)",
                out_dir,
                "15_market_cap_hist_linear",
                bins=40,
            )

            mc = safe_numeric(corp_in_dataset["market_cap"]).dropna()
            mc = mc[mc > 0]

            fig = plt.figure()
            ax = fig.add_subplot(111)

            if mc.empty:
                ax.text(0.5, 0.5, "No positive market cap values available", ha="center", va="center")
            else:
                edges = np.logspace(np.log10(float(mc.min())), np.log10(float(mc.max())), 40)
                ax.hist(mc.values, bins=edges)
                ax.set_xscale("log")

                def fmt_usd(x, pos):
                    if x >= 1e12: return f"${x/1e12:.0f}T"
                    if x >= 1e9:  return f"${x/1e9:.0f}B"
                    if x >= 1e6:  return f"${x/1e6:.0f}M"
                    if x >= 1e3:  return f"${x/1e3:.0f}K"
                    return f"${x:.0f}"

                ax.xaxis.set_major_formatter(FuncFormatter(fmt_usd))

            ax.set_title("Distribution of corporate market cap (USD) — log x-axis")
            ax.set_xlabel("Market cap (USD, log scale)")
            ax.set_ylabel("Count")
            save_plot_png(fig, out_dir, "15_market_cap_hist_logx")

        # 16) HQ country
        if (not corp_in_dataset.empty) and ("hq_country" in corp_in_dataset.columns):
            tbl = frequency_table(corp_in_dataset["hq_country"], "hq_country")
            save_table_csv(tbl, out_dir, "16_hq_country_distribution")
            plot_bar(tbl.set_index("hq_country")["count"],
                     "Distribution of HQ country (top shown in plot)",
                     "hq_country", "Count",
                     out_dir, "16_hq_country_distribution_top",
                     rotate=90, top_n=top_n_countries)

        # 17) Main exchange (RAW table + ABBREV table; plot uses ABBREV labels)
        if (not corp_in_dataset.empty) and ("main_exchange" in corp_in_dataset.columns):
            raw_tbl = frequency_table(corp_in_dataset["main_exchange"], "main_exchange")
            save_table_csv(raw_tbl, out_dir, "17_main_exchange_distribution_raw")

            if "main_exchange_abbrev" in corp_in_dataset.columns:
                ab_tbl = frequency_table(corp_in_dataset["main_exchange_abbrev"], "main_exchange_abbrev")
                save_table_csv(ab_tbl, out_dir, "17_main_exchange_distribution_abbrev")

                plot_bar(ab_tbl.set_index("main_exchange_abbrev")["count"],
                         "Distribution of main exchange (abbreviated; top shown in plot)",
                         "main_exchange", "Count",
                         out_dir, "17_main_exchange_distribution_top",
                         rotate=90, top_n=top_n_exchanges)
            else:
                plot_bar(raw_tbl.set_index("main_exchange")["count"],
                         "Distribution of main exchange (top shown in plot)",
                         "main_exchange", "Count",
                         out_dir, "17_main_exchange_distribution_top",
                         rotate=90, top_n=top_n_exchanges)

        # 18) GICS sector
        if (not corp_in_dataset.empty) and ("gics_sector" in corp_in_dataset.columns):
            tbl = frequency_table(corp_in_dataset["gics_sector"], "gics_sector")
            save_table_csv(tbl, out_dir, "18_gics_sector_distribution")
            plot_bar(tbl.set_index("gics_sector")["count"],
                     "Distribution of GICS sector",
                     "gics_sector", "Count",
                     out_dir, "18_gics_sector_distribution",
                     rotate=90)

        # 19) GICS industry
        if (not corp_in_dataset.empty) and ("gics_industry" in corp_in_dataset.columns):
            tbl = frequency_table(corp_in_dataset["gics_industry"], "gics_industry")
            save_table_csv(tbl, out_dir, "19_gics_industry_distribution")
            plot_bar(tbl.set_index("gics_industry")["count"],
                     "Distribution of GICS industry (top shown in plot)",
                     "gics_industry", "Count",
                     out_dir, "19_gics_industry_distribution_top",
                     rotate=90, top_n=top_n_industries)

        # 20) Turnover
        if (not corp_in_dataset.empty) and ("turnover_6m_sum_volume_mean" in corp_in_dataset.columns):
            save_table_csv(describe_numeric(corp_in_dataset["turnover_6m_sum_volume_mean"], "turnover_6m_sum_volume_mean"),
                           out_dir, "20_turnover_6m_sum_volume_mean_describe")
            plot_hist(corp_in_dataset["turnover_6m_sum_volume_mean"],
                      "Distribution of turnover_6m_sum_volume_mean (linear)",
                      "turnover_6m_sum_volume_mean",
                      out_dir, "20_turnover_6m_sum_volume_mean_hist_linear", bins=40)
            plot_log_hist(corp_in_dataset["turnover_6m_sum_volume_mean"],
                          "Distribution of turnover_6m_sum_volume_mean — log axis + log-spaced bins",
                          "turnover_6m_sum_volume_mean",
                          out_dir, "20_turnover_6m_sum_volume_mean_hist_logbins", bins=40)

            tv = corp_in_dataset["turnover_6m_sum_volume_mean"].dropna()
            tv = tv[tv > 0]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if tv.empty:
                ax.text(0.5, 0.5, "No positive turnover values available", ha="center", va="center")
            else:
                ax.hist(np.log10(tv.values), bins=40)
            ax.set_title("Distribution of log10(turnover_6m_sum_volume_mean)")
            ax.set_xlabel("log10(turnover_6m_sum_volume_mean)")
            ax.set_ylabel("Count")
            save_plot_png(fig, out_dir, "20_turnover_6m_sum_volume_mean_hist_log10")

        # 21) Analyst coverage
        if (not corp_in_dataset.empty) and ("analyst_coverage" in corp_in_dataset.columns):
            cov = safe_numeric(corp_in_dataset["analyst_coverage"]).dropna()
            cov = cov[cov >= 0]

            if cov.empty:
                (out_dir / "logs" / "ANALYST_COVERAGE_NOTE.txt").write_text(
                    "analyst_coverage column existed but had no usable non-negative numeric values after cleaning.",
                    encoding="utf-8",
                )
            else:
                # Table: numeric summary
                save_table_csv(
                    describe_numeric(cov, "analyst_coverage"),
                    out_dir,
                    "21_analyst_coverage_describe",
                )

                 # Table + plot: decade bins (0-9, 10-19, 20-29, ...)
                bin_width = 10
                cov_int = cov.round().astype(int)

                max_val = int(cov_int.max())
                # Need the top edge to be > max_val for right=False bins [a,b)
                upper = ((max_val // bin_width) + 1) * bin_width
                if max_val % bin_width == 0:
                    upper = max_val + bin_width

                edges = list(range(0, upper + 1, bin_width))
                labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)]

                cov_bins = pd.cut(
                    cov_int,
                    bins=edges,
                    right=False,          # [a, b)
                    labels=labels,
                    include_lowest=True,
                )

                counts = cov_bins.value_counts().reindex(labels, fill_value=0)

                dist_tbl = counts.reset_index()
                dist_tbl.columns = ["analyst_coverage_bin", "n_firms"]
                dist_tbl["pct_of_firms"] = dist_tbl["n_firms"] / dist_tbl["n_firms"].sum() * 100.0

                save_table_csv(dist_tbl, out_dir, "21_analyst_coverage_distribution")

                plot_bar(
                    counts=counts,
                    title="Distribution of analyst coverage per firm (decade bins)",
                    xlabel="Number of analysts covering firm (binned)",
                    ylabel="Number of firms",
                    out_dir=out_dir,
                    stem="21_analyst_coverage_distribution",
                    rotate=0,
                )

    else:
        note = [
            "Corporate stats were skipped because corporate_info.jsonl was missing, empty,",
            "or did not contain a recognizable ticker column.",
            f"Attempted path: {str(corporate_info_path)}",
        ]
        (out_dir / "logs" / "CORPORATE_INFO_NOTE.txt").write_text("\n".join(note), encoding="utf-8")

    # -----------------------------
    # 11b) All-variables summary table (numeric columns)
    # -----------------------------
    # We include numeric-looking columns from:
    # - markets
    # - prices
    # - validation (with derived diffs)
    # - prices_long (YES/NO, sum, abs deviation, etc.)
    # - corporate subset used in dataset (if available)
    try:
        datasets_for_allvars: List[Tuple[str, pd.DataFrame]] = [
            ("markets", markets),
            ("prices", prices),
            ("validation", val),
            ("prices_long", prices_long),
        ]
        if (not corp_in_dataset.empty):
            datasets_for_allvars.append(("corporate_info_matched", corp_in_dataset))

        rows_all: List[Dict[str, Any]] = []
        for dname, df in datasets_for_allvars:
            if df is None or df.empty:
                continue
            for col in df.columns:
                # Keep booleans too (they are numeric-like), but avoid exploding on dict/list columns
                # (safe_numeric will yield NaN for non-scalar types).
                try:
                    row = summarize_numeric_minmax_p05_p95_iqr(df[col], dname, col)
                    if row is not None:
                        rows_all.append(row)
                except Exception:
                    continue

        allvars_tbl = pd.DataFrame(rows_all)
        if not allvars_tbl.empty:
            allvars_tbl = allvars_tbl.sort_values(["dataset", "variable"]).reset_index(drop=True)
            save_table_csv(allvars_tbl, out_dir, "11b_all_variables_summary_stats")
        else:
            (out_dir / "logs" / "ALL_VARIABLES_SUMMARY_NOTE.txt").write_text(
                "No numeric variables found for 11b_all_variables_summary_stats (after numeric coercion).",
                encoding="utf-8",
            )
    except Exception as e:
        (out_dir / "logs" / "ALL_VARIABLES_SUMMARY_ERROR.txt").write_text(
            f"Failed to build all-variables summary.\nError: {repr(e)}",
            encoding="utf-8",
        )

    # -----------------------------
    # Manifest
    # -----------------------------
    manifest_lines = [
        "Descriptive statistics outputs written to:",
        str(out_dir),
        "",
        "Snapshot order enforced:",
        ", ".join(SNAPSHOT_ORDER),
        "",
        "",
        "Key outputs:",
        "- 01_observations_per_snapshot (CSV + PNG plot)",
        "- 02_observed_span_hours (CSV + PNG hist)",
        "- 02b_market_end_dates_counts (CSV) + 02b_market_end_dates_timeline (PNG)",
        "- 03_outcomes_distribution_validation (CSV + PNG plot) and markets table-only",
        "- 04_complement_violations (CSV + PNG rate plot)",
        "- 05_tags_distribution (CSV + PNG plot)",
        "- 06_volume distributions (CSV + PNG plots)",
        "- 07/08 estimate diffs (CSV + PNG hists)",
        "- 09_surprise (CSV + PNG hist)",
        "- 10_ticker_counts_all (CSV only)",
        "- 11_numeric_descriptive_panel (CSV)",
        "- 11b_all_variables_summary_stats (CSV)",
        "- 12_span_vs_volume scatter + coefficients (PNG + CSV)",
        "- 13_distance_to_certainty_by_snapshot_leq_1d (CSV only)",
        "",
        "Corporate outputs:",
        "- 14_firms_seen_more_than_once_summary (CSV)",
        "- 14_firm_repeat_counts_distribution (CSV + PNG bar plot; integer x-axis)",
        "- 17_main_exchange_distribution_raw (CSV)",
        "- 17_main_exchange_distribution_abbrev (CSV) + plot uses abbreviations",
        "- 21_analyst_coverage_distribution (CSV + PNG bar plot)",
    ]
    (out_dir / "logs" / "MANIFEST.txt").write_text("\n".join(manifest_lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create descriptive statistics tables and plots for Polymarket Corporate Earnings dataset."
    )
    p.add_argument("--markets", type=str, default=str(DEFAULT_MARKETS_PATH), help="Path to markets.jsonl")
    p.add_argument("--prices", type=str, default=str(DEFAULT_PRICES_PATH), help="Path to historical_prices.jsonl")
    p.add_argument("--validation", type=str, default=str(DEFAULT_VALIDATION_PATH), help="Path to correct.jsonl")
    p.add_argument("--corporate-info", type=str, default=str(DEFAULT_CORPORATE_INFO_PATH), help="Path to corporate_info.jsonl")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory (will be overwritten)")
    p.add_argument("--tolerance", type=float, default=0.05, help="Default complement tolerance if missing in price rows")
    p.add_argument("--top-tags", type=int, default=30, help="Top N tags (excluding earnings) to plot")
    p.add_argument("--top-tickers", type=int, default=25, help="Top N tickers (table only; plot removed)")
    p.add_argument("--top-countries", type=int, default=30, help="Top N HQ countries to plot")
    p.add_argument("--top-exchanges", type=int, default=30, help="Top N exchanges to plot")
    p.add_argument("--top-industries", type=int, default=40, help="Top N GICS industries to plot")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    run_descriptive_statistics(
        markets_path=Path(args.markets),
        prices_path=Path(args.prices),
        validation_path=Path(args.validation),
        corporate_info_path=Path(args.corporate_info),
        out_dir=Path(args.out),
        complement_tolerance_default=float(args.tolerance),
        top_n_tags=int(args.top_tags),
        top_n_tickers=int(args.top_tickers),
        top_n_countries=int(args.top_countries),
        top_n_exchanges=int(args.top_exchanges),
        top_n_industries=int(args.top_industries),
    )

    print(f"[OK] Wrote descriptive statistics outputs to: {args.out}")


if __name__ == "__main__":
    main()
