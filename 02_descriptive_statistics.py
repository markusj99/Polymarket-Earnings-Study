#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_descriptive_statistics.py

Polymarket Corporate Earnings — Descriptive Statistics + Plots

CHANGES (per your feedback)
---------------------------
1) NO PDFs are produced (plots saved only as PNG).
2) Snapshots are ordered as:
   4w, 3w, 2w, 1w, 6d, 5d, 4d, 3d, 2d, 1d, 12h, 6h
   - We also DROP "7d" to avoid duplicate-with-1w confusion.
3) Removed 03_outcomes_distribution_markets.png (kept the table; kept validation outcomes plot).
4) Fixed "weird" volume histogram:
   - Uses log-spaced bins when x-axis is log scale (correct way).
   - Also provides a log10(volume) histogram as an alternative view.
5) Updated 12_span_vs_volume_scatter_logx.png to include a fitted slope line:
   - Fit: observed_span_hours ~ a + b * log10(volumeNum)
6) Tables are saved ONLY as CSV (no XLSX, no LaTeX).
7) Added:
   - Plot + table: distribution of (1 - max(YES,NO)) for snapshots <= 1d (i.e., 1d, 12h, 6h).

OUTPUT DIRECTORY (overwritten each run)
---------------------------------------
C:\\Users\\lasts\\Desktop\\Polymarket\\Corporate_Earnings\\statistics\\descriptive_statistics

INPUT FILES
-----------
- markets.jsonl:
  C:\\Users\\lasts\\Desktop\\Polymarket\\Corporate_Earnings\\data\\markets\\markets.jsonl
- historical_prices.jsonl:
  C:\\Users\\lasts\\Desktop\\Polymarket\\Corporate_Earnings\\data\\prices\\historical_prices.jsonl
- correct.jsonl:
  C:\\Users\\lasts\\Desktop\\Polymarket\\Corporate_Earnings\\data\\validation\\correct.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Defaults (your Windows paths)
# -----------------------------
DEFAULT_MARKETS_PATH = r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\markets\markets.jsonl"
DEFAULT_PRICES_PATH = r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\prices\historical_prices.jsonl"
DEFAULT_VALIDATION_PATH = r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\validation\correct.jsonl"
DEFAULT_OUT_DIR = r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\statistics\descriptive_statistics"

# Your required snapshot order (closest-to-end at the bottom)
SNAPSHOT_ORDER = ["4w", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h"]
SNAPSHOT_ORDER_SET = set(SNAPSHOT_ORDER)

# We drop 7d because it duplicates 1w in many runs and confuses ordering
DROP_SNAPSHOTS = {"7d"}


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
    """
    Save ONLY CSV (per your requirement).
    """
    tables_dir = out_dir / "tables"
    df.to_csv(tables_dir / f"{stem}.csv", index=False)


def save_plot_png(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """
    Save ONLY PNG (no PDFs).
    """
    plots_dir = out_dir / "plots"
    fig.savefig(plots_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Snapshot ordering helpers
# -----------------------------
def snapshot_sort_key(snapshot: str) -> Tuple[int, str]:
    """
    Sort snapshots using your fixed order. Unknown snapshots go to the end (alphabetically).
    """
    if snapshot in SNAPSHOT_ORDER_SET:
        return (SNAPSHOT_ORDER.index(snapshot), snapshot)
    return (10_000, snapshot)


def sort_snapshots(values: List[str]) -> List[str]:
    return sorted(values, key=snapshot_sort_key)


def parse_snapshot_to_hours(snapshot: str) -> Optional[float]:
    """
    Convert snapshot label like "6h", "12h", "1d", "2w" into hours.
    Returns None if it cannot parse.
    """
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
    """
    True if horizon <= 24 hours, based on parse_snapshot_to_hours.
    We explicitly want <= 1d, so this includes: 1d, 12h, 6h (and any future <=24h labels).
    """
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


def plot_bar(counts: pd.Series, title: str, xlabel: str, ylabel: str,
             out_dir: Path, stem: str, rotate: int = 45, order: Optional[List[str]] = None,
             top_n: Optional[int] = None) -> None:
    """
    Basic bar plot. Optionally reorder index and/or take top_n.
    """
    s = counts.copy()

    if order is not None:
        idx = [x for x in order if x in s.index]
        rest = [x for x in s.index if x not in idx]
        s = pd.concat([s.loc[idx], s.loc[sorted(rest)]]) if len(rest) else s.loc[idx]

    if top_n is not None and s.shape[0] > top_n:
        s = s.sort_values(ascending=False).head(top_n)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(s.index.astype(str), s.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    save_plot_png(fig, out_dir, stem)


# -----------------------------
# Data transforms
# -----------------------------
def build_prices_long(prices_df: pd.DataFrame, complement_tolerance_default: float) -> pd.DataFrame:
    """
    Long-format from historical_prices.jsonl:

      market_id, slug, snapshot, yes_price, no_price,
      sum_price, abs_sum_minus_1, violates_tolerance
    """
    records: List[Dict[str, Any]] = []

    for _, row in prices_df.iterrows():
        prices_yes = row.get("prices_yes") or {}
        prices_no = row.get("prices_no") or {}

        labels = set(prices_yes.keys()) | set(prices_no.keys())
        for snap in labels:
            if snap in DROP_SNAPSHOTS:
                continue
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
# Main analysis
# -----------------------------
def run_descriptive_statistics(
    markets_path: Path,
    prices_path: Path,
    validation_path: Path,
    out_dir: Path,
    complement_tolerance_default: float = 0.05,
    top_n_tags: int = 30,
    top_n_tickers: int = 25,
) -> None:
    ensure_clean_dir(out_dir)

    # Load
    markets = pd.DataFrame(read_jsonl(markets_path))
    prices = pd.DataFrame(read_jsonl(prices_path))
    validation = pd.DataFrame(read_jsonl(validation_path))

    # Normalize IDs
    if "id" in markets.columns:
        markets["market_id"] = markets["id"].astype(str)
    elif "market_id" in markets.columns:
        markets["market_id"] = markets["market_id"].astype(str)

    if "market_id" in prices.columns:
        prices["market_id"] = prices["market_id"].astype(str)
    if "market_id" in validation.columns:
        validation["market_id"] = validation["market_id"].astype(str)

    # -----------------------------
    # 0) Dataset sizes + missingness
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

    miss = pd.concat([
        missingness_table(markets, "markets", ["market_id", "slug", "ticker", "resolvedOutcome", "volumeNum", "tags"]),
        missingness_table(prices, "prices", ["market_id", "slug", "observed_span_hours", "prices_yes", "prices_no"]),
        missingness_table(validation, "validation", ["market_id", "slug", "polymarket_estimate", "eikon_eps_mean_estimate",
                                                     "eikon_actual_eps", "surprise", "label"]),
    ], ignore_index=True)

    save_table_csv(miss, out_dir, "00_dataset_sizes_and_missingness")

    # -----------------------------
    # Build long prices
    # -----------------------------
    prices_long = build_prices_long(prices, complement_tolerance_default)

    # -----------------------------
    # 1) Observations per snapshot
    # -----------------------------
    obs_counts = (
        prices_long.assign(
            has_yes=prices_long["yes_price"].notna(),
            has_no=prices_long["no_price"].notna(),
            has_both=prices_long["yes_price"].notna() & prices_long["no_price"].notna(),
        )
        .groupby("snapshot", as_index=False)
        .agg(
            n_yes=("has_yes", "sum"),
            n_no=("has_no", "sum"),
            n_both=("has_both", "sum"),
        )
    )

    # Sort snapshots in required order
    obs_counts = obs_counts.sort_values(by="snapshot", key=lambda s: s.map(lambda x: snapshot_sort_key(str(x))[0]))
    save_table_csv(obs_counts, out_dir, "01_observations_per_snapshot")

    # Plot: n_both by snapshot (ordered)
    n_both_series = obs_counts.set_index("snapshot")["n_both"]
    plot_bar(
        counts=n_both_series,
        title="Number of markets with BOTH YES and NO prices by snapshot",
        xlabel="Snapshot label (ordered)",
        ylabel="Count (both YES & NO available)",
        out_dir=out_dir,
        stem="01_observations_per_snapshot_n_both",
        rotate=45,
        order=SNAPSHOT_ORDER,
    )

    # -----------------------------
    # 2) Distribution of observed_span_hours
    # -----------------------------
    if "observed_span_hours" in prices.columns:
        prices["observed_span_hours"] = safe_numeric(prices["observed_span_hours"])
        save_table_csv(describe_numeric(prices["observed_span_hours"], "observed_span_hours"),
                       out_dir, "02_observed_span_hours_describe")
        plot_hist(
            prices["observed_span_hours"],
            title="Distribution of observed market active span (hours)",
            xlabel="observed_span_hours",
            out_dir=out_dir,
            stem="02_observed_span_hours_hist",
            bins=40,
        )

    # -----------------------------
    # 3) Distribution of outcomes
    #    - Remove markets.png (per request), but keep markets table
    #    - Keep validation plot + table
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
    # 4) Complement tolerance violations (|YES + NO - 1| > tolerance)
    #    Keep ONE informative plot: violation rate (%) by snapshot (ordered)
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

    # Plot: violation rate (%) by snapshot (ordered)
    viol_rate_series = viol_by_snap.set_index("snapshot")["violation_rate_pct"]
    plot_bar(
        counts=viol_rate_series,
        title="Complement violation rate by snapshot (|YES+NO-1| > tolerance)",
        xlabel="Snapshot label (ordered)",
        ylabel="Violation rate (%)",
        out_dir=out_dir,
        stem="04_complement_violations_rate_by_snapshot",
        rotate=45,
        order=SNAPSHOT_ORDER,
    )

    # Overall violations table
    overall_n_both = int((prices_long["yes_price"].notna() & prices_long["no_price"].notna()).sum())
    overall_n_viol = int(prices_long["violates_tolerance"].sum())
    overall_viol_table = pd.DataFrame([{
        "overall_n_rows_prices_long": int(prices_long.shape[0]),
        "overall_n_both_prices": overall_n_both,
        "overall_n_violations": overall_n_viol,
        "overall_violation_rate_pct": (overall_n_viol / overall_n_both * 100.0) if overall_n_both else np.nan,
        "tolerance_default_used_when_missing": complement_tolerance_default,
        "dropped_snapshots": ",".join(sorted(DROP_SNAPSHOTS)) if DROP_SNAPSHOTS else "",
    }])
    save_table_csv(overall_viol_table, out_dir, "04_complement_violations_overall")

    # -----------------------------
    # 5) Tags excluding "earnings"
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
            rotate=60,
            top_n=top_n_tags,
        )

    # -----------------------------
    # 6) Volume distribution (fix log histogram)
    # -----------------------------
    if "volumeNum" in markets.columns:
        markets["volumeNum"] = safe_numeric(markets["volumeNum"])

        save_table_csv(describe_numeric(markets["volumeNum"], "volumeNum"), out_dir, "06_volume_describe")

        # Linear histogram
        plot_hist(
            markets["volumeNum"],
            title="Distribution of market volume (volumeNum)",
            xlabel="volumeNum",
            out_dir=out_dir,
            stem="06_volume_hist_linear",
            bins=40,
        )

        # Correct log-scale histogram: log-spaced bins + log x-axis
        vol = markets["volumeNum"].dropna()
        vol = vol[vol > 0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if vol.empty:
            ax.text(0.5, 0.5, "No positive volume values available", ha="center", va="center")
        else:
            vmin, vmax = float(vol.min()), float(vol.max())
            # log-spaced bins are essential when x-axis is log
            bins = np.logspace(np.log10(vmin), np.log10(vmax), 40)
            ax.hist(vol.values, bins=bins)
            ax.set_xscale("log")
        ax.set_title("Distribution of market volume (volumeNum) — log axis + log-spaced bins")
        ax.set_xlabel("volumeNum (log axis)")
        ax.set_ylabel("Count")
        save_plot_png(fig, out_dir, "06_volume_hist_logbins")

        # Alternative: histogram of log10(volumeNum) on linear axis (often easier to read)
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
    # 7) Eikon mean estimate - Polymarket estimate
    # 8) Eikon actual - Polymarket estimate
    # 9) surprise distribution
    # -----------------------------
    val = validation.copy()
    for c in ["polymarket_estimate", "eikon_eps_mean_estimate", "eikon_actual_eps", "surprise"]:
        if c in val.columns:
            val[c] = safe_numeric(val[c])

    if {"eikon_eps_mean_estimate", "polymarket_estimate"}.issubset(val.columns):
        val["diff_eikon_mean_minus_poly_est"] = val["eikon_eps_mean_estimate"] - val["polymarket_estimate"]
        save_table_csv(describe_numeric(val["diff_eikon_mean_minus_poly_est"], "diff_eikon_mean_minus_poly_est"),
                       out_dir, "07_diff_eikon_mean_minus_polymarket_estimate_describe")
        plot_hist(
            val["diff_eikon_mean_minus_poly_est"],
            title="Eikon mean EPS estimate minus Polymarket estimate",
            xlabel="(eikon_eps_mean_estimate - polymarket_estimate)",
            out_dir=out_dir,
            stem="07_diff_eikon_mean_minus_poly_est_hist",
            bins=40,
        )

    if {"eikon_actual_eps", "polymarket_estimate"}.issubset(val.columns):
        val["diff_eikon_actual_minus_poly_est"] = val["eikon_actual_eps"] - val["polymarket_estimate"]
        save_table_csv(describe_numeric(val["diff_eikon_actual_minus_poly_est"], "diff_eikon_actual_minus_poly_est"),
                       out_dir, "08_diff_eikon_actual_minus_polymarket_estimate_describe")
        plot_hist(
            val["diff_eikon_actual_minus_poly_est"],
            title="Eikon actual EPS minus Polymarket estimate",
            xlabel="(eikon_actual_eps - polymarket_estimate)",
            out_dir=out_dir,
            stem="08_diff_eikon_actual_minus_poly_est_hist",
            bins=40,
        )

    if "surprise" in val.columns:
        save_table_csv(describe_numeric(val["surprise"], "surprise"), out_dir, "09_surprise_describe")
        plot_hist(
            val["surprise"],
            title="Distribution of earnings surprise",
            xlabel="surprise",
            out_dir=out_dir,
            stem="09_surprise_hist",
            bins=40,
        )

    # -----------------------------
    # Extra (still useful): top tickers by market count
    # -----------------------------
    if "ticker" in markets.columns:
        tick_counts = markets["ticker"].astype(str).replace({"nan": np.nan}).dropna().value_counts()
        tick_table = tick_counts.reset_index()
        tick_table.columns = ["ticker", "n_markets"]
        save_table_csv(tick_table, out_dir, "10_ticker_counts_all")

        plot_bar(
            counts=tick_counts,
            title="Top tickers by number of markets",
            xlabel="Ticker",
            ylabel="Number of markets",
            out_dir=out_dir,
            stem="10_top_tickers",
            rotate=60,
            top_n=top_n_tickers,
        )

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
        panel = pd.concat(numeric_panels, ignore_index=True)
        save_table_csv(panel, out_dir, "11_numeric_descriptive_panel")

    # -----------------------------
    # 12) Span vs volume scatter (log-x) + slope line
    #     Fit: y = a + b * log10(x)
    # -----------------------------
    if ("volumeNum" in markets.columns) and ("observed_span_hours" in prices.columns):
        # Merge by market_id + slug when available, else market_id only.
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

            # OLS fit: y = a + b*log10(x)
            # np.polyfit returns [b, a] for degree 1 when x=lx
            b, a = np.polyfit(lx, y, 1)

            # Line across x-range
            x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            y_line = a + b * np.log10(x_line)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x, y)
            ax.plot(x_line, y_line)  # slope line (matplotlib chooses default line color)
            ax.set_xscale("log")
            ax.set_title(f"Market activity span vs volume (log volume) with slope line\nFit: span = {a:.3f} + {b:.3f}*log10(volume)")
            ax.set_xlabel("volumeNum (log axis)")
            ax.set_ylabel("observed_span_hours")
            save_plot_png(fig, out_dir, "12_span_vs_volume_scatter_logx_with_slope")

            # Save slope coefficients to CSV for reporting
            coef_tbl = pd.DataFrame([{"a_intercept": a, "b_slope_log10_volume": b, "n_obs": int(mp.shape[0])}])
            save_table_csv(coef_tbl, out_dir, "12_span_vs_volume_slope_coefficients")

    # -----------------------------
    # 13) NEW: Distance to certainty for snapshots <= 1d
    #     distance = 1 - max(YES, NO)
    # -----------------------------
    # Use only snapshots <= 1d, and require both prices present.
    lt = prices_long.copy()
    lt = lt[lt["snapshot"].map(lambda s: snapshot_leq_1d(str(s)))]
    lt = lt[lt["yes_price"].notna() & lt["no_price"].notna()]

    if not lt.empty:
        lt["max_outcome_price"] = np.maximum(lt["yes_price"], lt["no_price"])
        lt["distance_to_1_from_max_outcome"] = 1.0 - lt["max_outcome_price"]

        # Table by snapshot
        rows = []
        for snap in sort_snapshots(list(lt["snapshot"].unique())):
            sub = lt.loc[lt["snapshot"] == snap, "distance_to_1_from_max_outcome"]
            d = describe_numeric(sub, f"distance_to_1_from_max_outcome__{snap}")
            # flatten for easy table
            r = d.iloc[0].to_dict()
            r["snapshot"] = snap
            rows.append(r)

        dist_tbl = pd.DataFrame(rows)
        # reorder columns a bit
        cols = ["snapshot", "n", "mean", "std", "min", "p25", "median", "p75", "p90", "p95", "p99", "max"]
        dist_tbl = dist_tbl[[c for c in cols if c in dist_tbl.columns] + [c for c in dist_tbl.columns if c not in cols]]
        save_table_csv(dist_tbl, out_dir, "13_distance_to_certainty_by_snapshot_leq_1d")

        # Plot: overlay histograms for each snapshot <= 1d
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for snap in sort_snapshots(list(lt["snapshot"].unique())):
            sub = lt.loc[lt["snapshot"] == snap, "distance_to_1_from_max_outcome"].dropna()
            if sub.empty:
                continue
            ax.hist(sub.values, bins=40, histtype="step", label=str(snap))

        ax.set_title("Distribution of distance to certainty: 1 - max(YES, NO)\nSnapshots with horizon <= 1d")
        ax.set_xlabel("1 - max(YES, NO)")
        ax.set_ylabel("Count")
        ax.legend()
        save_plot_png(fig, out_dir, "13_distance_to_certainty_hist_leq_1d")

    # -----------------------------
    # Manifest
    # -----------------------------
    manifest_lines = [
        "Descriptive statistics outputs written to:",
        str(out_dir),
        "",
        "No PDFs produced. Tables are CSV only.",
        "",
        "Snapshot order enforced:",
        ", ".join(SNAPSHOT_ORDER),
        f"Dropped snapshots: {', '.join(sorted(DROP_SNAPSHOTS)) if DROP_SNAPSHOTS else '(none)'}",
        "",
        "Key outputs:",
        "- 01_observations_per_snapshot (CSV + PNG plot, ordered)",
        "- 02_observed_span_hours (CSV + PNG hist)",
        "- 03_outcomes_distribution_validation (CSV + PNG plot) and markets table-only",
        "- 04_complement_violations_by_snapshot (CSV) + ONE plot: rate by snapshot (ordered)",
        "- 05_tags_distribution_excluding_earnings (CSV + PNG plot)",
        "- 06_volume distributions (CSV + PNG plots; corrected log histogram)",
        "- 07/08 differences vs Polymarket estimate (CSV + PNG hists)",
        "- 09_surprise (CSV + PNG hist)",
        "- 12_span_vs_volume scatter with slope line (PNG + coefficients CSV)",
        "- 13_distance_to_certainty for snapshots <= 1d (CSV + PNG plot)",
    ]
    (out_dir / "logs" / "MANIFEST.txt").write_text("\n".join(manifest_lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create descriptive statistics tables and plots for Polymarket Corporate Earnings dataset."
    )
    p.add_argument("--markets", type=str, default=DEFAULT_MARKETS_PATH, help="Path to markets.jsonl")
    p.add_argument("--prices", type=str, default=DEFAULT_PRICES_PATH, help="Path to historical_prices.jsonl")
    p.add_argument("--validation", type=str, default=DEFAULT_VALIDATION_PATH, help="Path to correct.jsonl")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Output directory (will be overwritten)")
    p.add_argument("--tolerance", type=float, default=0.05, help="Default complement tolerance if missing in price rows")
    p.add_argument("--top-tags", type=int, default=30, help="Top N tags (excluding earnings) to plot")
    p.add_argument("--top-tickers", type=int, default=25, help="Top N tickers to plot")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    run_descriptive_statistics(
        markets_path=Path(args.markets),
        prices_path=Path(args.prices),
        validation_path=Path(args.validation),
        out_dir=Path(args.out),
        complement_tolerance_default=float(args.tolerance),
        top_n_tags=int(args.top_tags),
        top_n_tickers=int(args.top_tickers),
    )

    print(f"[OK] Wrote descriptive statistics outputs to: {args.out}")


if __name__ == "__main__":
    main()
