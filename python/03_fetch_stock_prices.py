#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_fetch_stock_prices.py (FAST, BATCHED, THREAD-SAFE)

Key updates (to match the new per-market corporate info)
--------------------------------------------------------
- Input is now:
    data/corporate_info/corporate_info_by_market.jsonl
  (one JSON object per Polymarket market)

- Each Polymarket market is treated as an independent observation.
  (No company-level aggregation; no "markets[]" flattening needed anymore.)

- For each market we fetch *daily close* stock prices for the market's RIC
  from:
      [umaEndDate - 250 days,  umaEndDate + 10 days]
  (calendar-day window; trading days only appear in the output.)

- We also fetch S&P 500 (RIC: .SPX) daily close prices from:
      2025-01-01  through  today's date
  regardless of the event windows.

- Only daily close prices are used (TR.PriceClose, Frq='D').

Outputs (relative to project root)
---------------------------------
data/stock_prices/
  - stock_prices_daily.csv
  - stock_prices_daily.jsonl
  - stock_prices_daily.json            (nested per-market JSON, optional but kept)
  - stock_prices_summary.txt

Performance
-----------
- Uses ek.get_data in batches (chunked instruments) — fast and thread-safe.
- Buckets markets by close month/quarter to keep date ranges reasonable per request.
- tqdm progress bars.

Requirements
------------
pip install eikon pandas tqdm

Eikon App Key:
- env var EIKON_APP_KEY (recommended), or --app-key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import eikon as ek  # type: ignore
except Exception:
    ek = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


# ---------------------------------------------------------------------
# Suppress noisy warnings from eikon/pandas
# ---------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*errors='ignore'.*deprecated.*to_numeric.*",
    category=FutureWarning,
    module=r"eikon\.data_grid",
)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"eikon\.data_grid")


# ---------------------------------------------------------------------
# Retry settings (same idea as 03_fetch_corp_info.py)
# ---------------------------------------------------------------------
EIKON_RETRIES = 5
EIKON_RETRY_BASE_SLEEP = 0.7


def _looks_like_eikon_network_error(exc: Exception) -> bool:
    s = str(exc)
    return ("Error code 500" in s and "Network Error" in s) or ('"message":"Network Error"' in s)


def eikon_retry_get_data(
    instruments: List[str],
    fields: List[Any],
    parameters: Dict[str, Any],
    *,
    retries: int = EIKON_RETRIES,
) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    """
    Robust wrapper around ek.get_data that retries transient proxy/network errors.
    """
    assert ek is not None
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            df, err = ek.get_data(instruments, fields, parameters=parameters)
            if isinstance(df, pd.DataFrame):
                return df, err
            return None, err
        except Exception as exc:
            last_exc = exc
            time.sleep(EIKON_RETRY_BASE_SLEEP * (2 ** attempt))

    return None, last_exc


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MarketEvent:
    """
    One row per Polymarket market (independent observation).
    """
    ric: str
    ticker: str
    company_name: str
    market_id: str
    slug: str

    uma_end_date: str   # original timestamp string
    close_date: str     # YYYY-MM-DD (derived from uma_end_date)


# ---------------------------------------------------------------------
# Path helpers (deterministic relative paths)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Polymarket-Earnings-Study/


def default_input_path() -> Path:
    return PROJECT_ROOT / "data" / "corporate_info" / "corporate_info_by_market.jsonl"


def default_output_dir() -> Path:
    return PROJECT_ROOT / "data" / "stock_prices"


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {ex}") from ex
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: Path, s: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(s)


# ---------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------
def parse_iso_date(s: Any) -> Optional[date]:
    """
    Parse YYYY-MM-DD from an ISO-like string (timestamps OK).
    """
    if s is None:
        return None
    try:
        return date.fromisoformat(str(s).strip()[0:10])
    except Exception:
        return None


def safe_date_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ---------------------------------------------------------------------
# Parsing corporate_info_by_market.jsonl
# ---------------------------------------------------------------------
def extract_events_by_market(corporate_rows: List[Dict[str, Any]]) -> Tuple[List[MarketEvent], List[str]]:
    """
    Input rows are already per-market (one JSON per market).
    We simply validate + construct MarketEvent objects.
    """
    events: List[MarketEvent] = []
    warnings_out: List[str] = []

    for idx, row in enumerate(corporate_rows, start=1):
        ric = str(row.get("ric") or "").strip()
        ticker = str(row.get("ticker") or "").strip()
        company_name = str(row.get("company_name") or "").strip()
        market_id = str(row.get("market_id") or "").strip()
        slug = str(row.get("slug") or "").strip()
        uma_end = str(row.get("uma_end_date") or row.get("umaEndDate") or "").strip()

        if not ric:
            warnings_out.append(f"Line {idx}: missing ric (skipping).")
            continue
        if not market_id:
            warnings_out.append(f"Line {idx}: missing market_id for ric={ric} (skipping).")
            continue
        if not uma_end:
            warnings_out.append(f"Line {idx}: missing uma_end_date for ric={ric}, market_id={market_id} (skipping).")
            continue

        close_dt = parse_iso_date(uma_end)
        if close_dt is None:
            warnings_out.append(f"Line {idx}: bad uma_end_date='{uma_end}' for market_id={market_id} (skipping).")
            continue

        # ticker/company_name can be empty; keep them as empty strings (consistent schema)
        events.append(
            MarketEvent(
                ric=ric,
                ticker=ticker,
                company_name=company_name,
                market_id=market_id,
                slug=slug,
                uma_end_date=uma_end,
                close_date=close_dt.isoformat(),
            )
        )

    # Hard dedupe on (market_id) if duplicates exist in file
    uniq: Dict[str, MarketEvent] = {}
    for e in events:
        uniq[e.market_id] = e

    return list(uniq.values()), warnings_out


# ---------------------------------------------------------------------
# Eikon init
# ---------------------------------------------------------------------
def set_eikon_app_key(app_key: Optional[str]) -> None:
    if ek is None:
        raise RuntimeError("Python package 'eikon' is not available. Install it first (pip install eikon).")
    key = app_key or os.getenv("EIKON_APP_KEY") or os.getenv("APP_KEY")
    if not key:
        raise RuntimeError("Missing Eikon App Key. Set env EIKON_APP_KEY or pass --app-key.")
    ek.set_app_key(key)

    # Try to quiet SDK logging if supported
    try:
        set_level = getattr(ek, "set_log_level", None)
        if callable(set_level):
            set_level(0)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------
def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _bucket_key(close_d: date, mode: str) -> str:
    """
    Bucket mode reduces date span per request.
    - month:   YYYY-MM
    - quarter: YYYY-Qn
    - all:     single bucket
    """
    if mode == "all":
        return "ALL"
    if mode == "quarter":
        q = (close_d.month - 1) // 3 + 1
        return f"{close_d.year}-Q{q}"
    return f"{close_d.year}-{close_d.month:02d}"


# ---------------------------------------------------------------------
# Column detection (robust vs display headers)
# ---------------------------------------------------------------------
def find_col_by_substrings(columns: List[str], substrings: List[str]) -> Optional[str]:
    low_cols = [c.lower() for c in columns]
    for sub in substrings:
        s = sub.lower()
        for i, c in enumerate(low_cols):
            if s in c:
                return columns[i]
    return None


def get_first_present_column(columns: List[str], preferred_exact: List[str], fallback_substrings: List[str]) -> Optional[str]:
    colset = set(columns)
    for name in preferred_exact:
        if name in colset:
            return name
    if fallback_substrings:
        return find_col_by_substrings(columns, fallback_substrings)
    return None


# ---------------------------------------------------------------------
# FAST price fetch via ek.get_data (batched)
# ---------------------------------------------------------------------
def fetch_close_batch(
    rics: List[str],
    start_d: date,
    end_d: date,
    *,
    frq: str = "D",
    throttle_s: float = 0.0,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch daily CLOSE for many rics in one call.

    Returns standardized columns:
      ric, date, close
    """
    assert ek is not None

    fields: List[Any] = [
        "TR.PriceClose",
        "TR.PriceClose.date",
    ]
    params = {"SDate": start_d.isoformat(), "EDate": end_d.isoformat(), "Frq": frq}

    df, err = eikon_retry_get_data(rics, fields, params)
    if throttle_s > 0:
        time.sleep(throttle_s)

    if df is None or df.empty:
        return pd.DataFrame(columns=["ric", "date", "close"]), str(err)

    cols = list(df.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]
    date_col = get_first_present_column(cols, preferred_exact=["Date"], fallback_substrings=["date"])
    close_col = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["price close", "close"])

    if date_col is None or close_col is None:
        return pd.DataFrame(columns=["ric", "date", "close"]), f"Could not map columns. Columns={cols}"

    out = df[[inst_col, date_col, close_col]].copy()
    out.columns = ["ric", "date", "close"]

    out["ric"] = out["ric"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.drop_duplicates(subset=["ric", "date"]).sort_values(["ric", "date"])
    return out, None


# ---------------------------------------------------------------------
# Main runner (importable)
# ---------------------------------------------------------------------
def run_fetch_stock_prices(
    input_path: Path,
    out_dir: Path,
    app_key: Optional[str] = None,
    pre_days: int = 250,
    post_days: int = 10,
    chunk_size: int = 50,
    bucket_mode: str = "month",  # month | quarter | all
    throttle_s: float = 0.0,     # sleep after each ek.get_data batch
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Fast runner using batched ek.get_data calls.

    Window definition (calendar days):
      per market:
        [close_date - pre_days, close_date + post_days]
      (trading days only appear in output, because that's what Eikon returns for daily closes)

    Benchmark:
      S&P500 (.SPX) from 2025-01-01 through date.today()
    """
    if tqdm is None and show_progress:
        show_progress = False

    BENCHMARK_RIC = ".SPX"
    SPX_START = date(2025, 1, 1)
    SPX_END = date.today()  # "today" is whatever day this script is run

    set_eikon_app_key(app_key)
    ensure_dir(out_dir)

    corporate_rows = read_jsonl(input_path)
    events, parse_warnings = extract_events_by_market(corporate_rows)

    # Bucket markets by close date
    buckets: Dict[str, List[MarketEvent]] = {}
    for e in events:
        close_d = safe_date_ymd(e.close_date)
        k = _bucket_key(close_d, bucket_mode)
        buckets.setdefault(k, []).append(e)

    # Storage for fetched closes by RIC (merged across buckets)
    close_by_ric: Dict[str, pd.DataFrame] = {}
    failures_fetch: List[str] = []
    failures_window: List[str] = []

    # -----------------------------------------------------------------
    # Fetch benchmark SPX once (2025-01-01 -> today), independent of events
    # -----------------------------------------------------------------
    df_spx, err_spx = fetch_close_batch([BENCHMARK_RIC], SPX_START, SPX_END, throttle_s=throttle_s)
    if err_spx:
        failures_fetch.append(f"Benchmark {BENCHMARK_RIC}: {err_spx}")
    else:
        close_by_ric[BENCHMARK_RIC] = df_spx[["date", "close"]].sort_values("date")

    spx_ts = close_by_ric.get(BENCHMARK_RIC)
    spx_map: Dict[date, float] = {}
    if spx_ts is not None and not spx_ts.empty:
        spx_map = dict(zip(spx_ts["date"].tolist(), spx_ts["close"].tolist()))

    # -----------------------------------------------------------------
    # Fetch closes for market RICs, bucketed for efficiency
    # -----------------------------------------------------------------
    bucket_items = sorted(buckets.items(), key=lambda kv: kv[0])
    bucket_iter = bucket_items
    if show_progress:
        bucket_iter = tqdm(bucket_items, desc=f"Buckets ({bucket_mode})", unit="bucket")  # type: ignore

    for bkey, bevents in bucket_iter:  # type: ignore
        close_dates = [safe_date_ymd(e.close_date) for e in bevents]
        start_cal = min(close_dates) - timedelta(days=int(pre_days))
        end_cal = max(close_dates) + timedelta(days=int(post_days))

        rics = sorted({e.ric for e in bevents if e.ric})
        if not rics:
            continue

        ric_chunks = list(chunked(rics, max(1, int(chunk_size))))
        chunk_iter = ric_chunks
        if show_progress:
            chunk_iter = tqdm(ric_chunks, desc=f"close {bkey}", unit="chunk", leave=False)  # type: ignore

        for ric_chunk in chunk_iter:  # type: ignore
            df_batch, err = fetch_close_batch(ric_chunk, start_cal, end_cal, throttle_s=throttle_s)
            if err:
                failures_fetch.append(f"Bucket {bkey} chunk size={len(ric_chunk)}: {err}")
                continue

            for ric, g in df_batch.groupby("ric"):
                g2 = g[["date", "close"]].sort_values("date").copy()
                if ric in close_by_ric:
                    merged = pd.concat([close_by_ric[ric], g2], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
                    close_by_ric[ric] = merged
                else:
                    close_by_ric[ric] = g2

    # -----------------------------------------------------------------
    # Build outputs per market (calendar-window slicing)
    # -----------------------------------------------------------------
    all_rows: List[Dict[str, Any]] = []
    nested_events: List[Dict[str, Any]] = []

    ev_iter = events
    if show_progress:
        ev_iter = tqdm(events, desc="Slicing markets", unit="mkt")  # type: ignore

    for e in ev_iter:  # type: ignore
        ts = close_by_ric.get(e.ric)
        if ts is None or ts.empty:
            failures_window.append(f"RIC {e.ric} market_id={e.market_id}: NO_TS_DATA")
            continue

        close_d = safe_date_ymd(e.close_date)
        start_d = close_d - timedelta(days=int(pre_days))
        end_d = close_d + timedelta(days=int(post_days))

        # Filter to window (trading days only)
        ts_win = ts[(ts["date"] >= start_d) & (ts["date"] <= end_d)].copy()
        if ts_win.empty:
            failures_window.append(
                f"RIC {e.ric} market_id={e.market_id}: empty window after filtering [{start_d}..{end_d}]"
            )
            continue

        # Records: one per trading day in the calendar window
        records: List[Dict[str, Any]] = []
        for _, row in ts_win.iterrows():
            d: date = row["date"]
            offset_day = (d - close_d).days  # calendar-day offset relative to umaEndDate date
            records.append(
                {
                    "date": d.isoformat(),
                    "offset_day": int(offset_day),
                    "close": float(row["close"]) if pd.notna(row["close"]) else None,
                    "spx_close": float(spx_map[d]) if d in spx_map and pd.notna(spx_map[d]) else None,
                }
            )

        # Nested per-market object (kept for convenience)
        event_obj: Dict[str, Any] = {
            "market_id": e.market_id,
            "slug": e.slug,
            "ric": e.ric,
            "ticker": e.ticker,
            "company_name": e.company_name,
            "uma_end_date": e.uma_end_date,
            "close_date": e.close_date,
            "window": {
                "pre_days": int(pre_days),
                "post_days": int(post_days),
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "n_trading_days_returned": int(len(records)),
            },
            "prices": records,
        }
        nested_events.append(event_obj)

        # Long rows (CSV/JSONL)
        for r in records:
            all_rows.append(
                {
                    "ric": e.ric,
                    "ticker": e.ticker,
                    "company_name": e.company_name,
                    "market_id": e.market_id,
                    "slug": e.slug,
                    "uma_end_date": e.uma_end_date,
                    "close_date": e.close_date,
                    "window_start_date": start_d.isoformat(),
                    "window_end_date": end_d.isoformat(),
                    "date": r.get("date"),
                    "offset_day": r.get("offset_day"),
                    "close": r.get("close"),
                    "spx_close": r.get("spx_close"),
                }
            )

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    csv_path = out_dir / "stock_prices_daily.csv"
    jsonl_path = out_dir / "stock_prices_daily.jsonl"
    json_path = out_dir / "stock_prices_daily.json"
    summary_path = out_dir / "stock_prices_summary.txt"

    df_out = pd.DataFrame(all_rows)
    if not df_out.empty:
        df_out.sort_values(["ric", "close_date", "market_id", "offset_day", "date"], inplace=True)
        df_out.to_csv(csv_path, index=False, encoding="utf-8")

    write_jsonl(jsonl_path, all_rows)
    write_json(
        json_path,
        {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "benchmark": {"ric": BENCHMARK_RIC, "start": SPX_START.isoformat(), "end": SPX_END.isoformat()},
            "markets": nested_events,
        },
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    lines: List[str] = []
    lines.append("Polymarket Corporate Earnings — Stock Prices Fetch Summary (PER-MARKET, FAST MODE)")
    lines.append(f"Generated at (UTC): {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"Input:  {input_path}")
    lines.append(f"Output: {out_dir}")
    lines.append("")
    lines.append(f"Input rows:       {len(corporate_rows)}")
    lines.append(f"Markets parsed:   {len(events)}")
    lines.append(f"Unique RICs seen: {len({e.ric for e in events})}")
    lines.append(f"RICs in cache:    {len(close_by_ric)} (includes benchmark if fetched)")
    lines.append(f"Output rows:      {len(all_rows)}")
    lines.append("")
    lines.append(f"Per-market calendar window: pre_days={pre_days}, post_days={post_days}")
    lines.append(f"Bucket mode: {bucket_mode}")
    lines.append(f"Chunk size:  {chunk_size}")
    lines.append(f"Throttle after batch: {throttle_s}s")
    lines.append("")
    lines.append("Benchmark (S&P500):")
    lines.append(f"  RIC:   {BENCHMARK_RIC}")
    lines.append(f"  Start: {SPX_START.isoformat()}")
    lines.append(f"  End:   {SPX_END.isoformat()} (today at runtime)")
    lines.append("")

    if parse_warnings:
        lines.append("PARSE WARNINGS")
        lines.extend([f"- {w}" for w in parse_warnings[:200]])
        if len(parse_warnings) > 200:
            lines.append(f"... {len(parse_warnings) - 200} more omitted")
        lines.append("")

    if failures_fetch:
        lines.append("BATCH FETCH FAILURES")
        lines.extend([f"- {x}" for x in failures_fetch[:200]])
        if len(failures_fetch) > 200:
            lines.append(f"... {len(failures_fetch) - 200} more omitted")
        lines.append("")

    if failures_window:
        lines.append("WINDOW / MARKET ISSUES")
        lines.extend([f"- {x}" for x in failures_window[:200]])
        if len(failures_window) > 200:
            lines.append(f"... {len(failures_window) - 200} more omitted")
        lines.append("")

    write_text(summary_path, "\n".join(lines) + "\n")

    return {
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "markets_total": len(events),
        "rows_total": len(all_rows),
        "rics_in_cache": len(close_by_ric),
        "bucket_mode": bucket_mode,
        "chunk_size": chunk_size,
        "pre_days": int(pre_days),
        "post_days": int(post_days),
        "spx_start": SPX_START.isoformat(),
        "spx_end": SPX_END.isoformat(),
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch daily close stock prices around uma_end_date from Eikon (PER-MARKET, FAST, batched)."
    )
    p.add_argument("--input", type=str, default=str(default_input_path()),
                   help="Path to corporate_info_by_market.jsonl")
    p.add_argument("--outdir", type=str, default=str(default_output_dir()),
                   help="Output directory")
    p.add_argument("--app-key", type=str, default=None,
                   help="Eikon App Key (or set env EIKON_APP_KEY)")

    p.add_argument("--pre-days", type=int, default=250,
                   help="Calendar days before close_date (umaEndDate date)")
    p.add_argument("--post-days", type=int, default=10,
                   help="Calendar days after close_date (umaEndDate date)")

    p.add_argument("--chunk-size", type=int, default=50,
                   help="RICs per ek.get_data call (50 is a good start)")
    p.add_argument("--bucket-mode", type=str, default="month",
                   choices=["month", "quarter", "all"],
                   help="Bucket markets to reduce date span per request")
    p.add_argument("--throttle", type=float, default=0.0,
                   help="Sleep after each batch call (seconds)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bars")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.outdir)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    result = run_fetch_stock_prices(
        input_path=input_path,
        out_dir=out_dir,
        app_key=args.app_key,
        pre_days=int(args.pre_days),
        post_days=int(args.post_days),
        chunk_size=int(args.chunk_size),
        bucket_mode=str(args.bucket_mode),
        throttle_s=float(args.throttle),
        show_progress=not bool(args.no_progress),
    )

    print("DONE")
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
