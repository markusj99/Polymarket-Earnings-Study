#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_fetch_stock_prices.py (FAST + THREAD-SAFE)

Key change vs the threaded version:
- DO NOT parallelize Eikon calls. The eikon SDK is not thread-safe.
- Use ek.get_data() in batches (like your 03_fetch_corp_info.py), which is much faster.

What this script does
---------------------
1) Read corporate_info.jsonl
2) Flatten to per-event rows (ric, market_id, slug, anchor_date, ...)
3) Group events into date buckets (default: month buckets) to avoid huge date ranges per request
4) For each bucket:
   - Fetch daily OHLCV for all RICs in that bucket using batched ek.get_data calls
5) For each event:
   - Find event trading date (anchor date if trading day else next trading day)
   - Slice trading-day window: [-pre_td, +post_td]
   - Emit long rows for CSV/JSONL and nested JSON per event
6) Write outputs + summary txt.

Outputs
-------
data/stock_prices/
  - stock_prices_daily.csv
  - stock_prices_daily.jsonl
  - stock_prices_daily.json
  - stock_prices_summary.txt

Speed knobs
-----------
- --chunk-size (default 50): instruments per get_data call
- --bucket-mode (default month): reduce date span per call
- --no-earnings-time: skip BMO/AMC lookup (recommended for speed)

Earnings time (BMO/AMC)
-----------------------
Historical “exact release time” is often not reliably available via simple TR fields.
This script keeps a BEST-EFFORT per-event lookup, but it is slow and may return TNS.
For speed runs, use --no-earnings-time (recommended), then optionally run a separate
enrichment pass later.

Requirements
------------
pip install eikon pandas tqdm

Eikon App Key:
- env var EIKON_APP_KEY (recommended), or --app-key
"""

from __future__ import annotations

import argparse
import json
import math
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


# -----------------------------------------------------------------------------
# Suppress noisy warnings from eikon/pandas (requested)
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*errors='ignore'.*deprecated.*to_numeric.*",
    category=FutureWarning,
    module=r"eikon\.data_grid",
)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"eikon\.data_grid")


# -----------------------------------------------------------------------------
# Retry settings (borrow the idea from 03_fetch_corp_info.py)
# -----------------------------------------------------------------------------
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
            sleep_s = EIKON_RETRY_BASE_SLEEP * (2 ** attempt)
            time.sleep(sleep_s)

    return None, last_exc


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EarningsEvent:
    ric: str
    ticker: str
    company_name: str
    market_id: str
    slug: str
    anchor_date: str  # YYYY-MM-DD


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def guess_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "data").exists():
            return p
    return here.parent


def default_input_path() -> Path:
    root = guess_project_root()
    return root / "data" / "corporate_info" / "corporate_info.jsonl"


def default_output_dir() -> Path:
    root = guess_project_root()
    return root / "data" / "stock_prices"


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Parsing corporate_info.jsonl
# -----------------------------------------------------------------------------
def extract_events(corporate_rows: List[Dict[str, Any]]) -> Tuple[List[EarningsEvent], List[str]]:
    events: List[EarningsEvent] = []
    warnings_out: List[str] = []

    for idx, row in enumerate(corporate_rows, start=1):
        ric = (row.get("ric") or "").strip()
        ticker = (row.get("ticker") or "").strip()
        company_name = (row.get("company_name") or "").strip()

        if not ric:
            warnings_out.append(f"Line {idx}: missing ric for ticker={ticker} (skipping all its events).")
            continue

        markets = row.get("markets") or []
        if not isinstance(markets, list) or not markets:
            warnings_out.append(f"Line {idx}: no markets list for ric={ric} (nothing to fetch).")
            continue

        for m in markets:
            market_id = str(m.get("market_id") or "").strip()
            slug = str(m.get("slug") or "").strip()
            anchor_date = str(m.get("anchor_date") or "").strip()

            if not market_id or not anchor_date:
                warnings_out.append(
                    f"Line {idx}: missing market_id/anchor_date for ric={ric}, ticker={ticker} (skipping one market)."
                )
                continue

            events.append(
                EarningsEvent(
                    ric=ric,
                    ticker=ticker,
                    company_name=company_name,
                    market_id=market_id,
                    slug=slug,
                    anchor_date=anchor_date,
                )
            )

    # Deduplicate
    uniq: Dict[Tuple[str, str, str], EarningsEvent] = {}
    for e in events:
        uniq[(e.ric, e.market_id, e.anchor_date)] = e

    return list(uniq.values()), warnings_out


# -----------------------------------------------------------------------------
# Eikon init
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Date / window helpers
# -----------------------------------------------------------------------------
def safe_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def calendar_span_for_trading_days(pre_td: int, post_td: int, holiday_buffer_days: int = 10) -> Tuple[int, int]:
    """
    Convert trading-day target window into approximate calendar-day request window.
    With +-30 td you usually only need ~ +/- 55 calendar days; buffer helps holidays.
    """
    lookback = int(math.ceil(pre_td * 7 / 5)) + holiday_buffer_days
    lookahead = int(math.ceil(post_td * 7 / 5)) + holiday_buffer_days
    return lookback, lookahead


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _bucket_key(anchor: date, mode: str) -> str:
    """
    Bucket mode reduces the range per request.
    - month: YYYY-MM
    - quarter: YYYY-Qn
    - all: single bucket
    """
    if mode == "all":
        return "ALL"
    if mode == "quarter":
        q = (anchor.month - 1) // 3 + 1
        return f"{anchor.year}-Q{q}"
    # default month
    return f"{anchor.year}-{anchor.month:02d}"


# -----------------------------------------------------------------------------
# Column detection (robust vs display headers)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# FAST price fetch via ek.get_data (batched)
# -----------------------------------------------------------------------------
def fetch_ohlcv_batch(
    rics: List[str],
    start_d: date,
    end_d: date,
    *,
    frq: str = "D",
    throttle_s: float = 0.0,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch daily OHLCV for many rics in one call.

    Returns dataframe with standardized columns:
      ric, date, open, high, low, close, volume

    On failure returns empty df and error string.
    """
    assert ek is not None

    fields: List[Any] = [
        "TR.PriceOpen",
        "TR.PriceHigh",
        "TR.PriceLow",
        "TR.PriceClose",
        "TR.Volume",
        "TR.PriceClose.date",
    ]
    params = {"SDate": start_d.isoformat(), "EDate": end_d.isoformat(), "Frq": frq}

    df, err = eikon_retry_get_data(rics, fields, params)
    if throttle_s > 0:
        time.sleep(throttle_s)

    if df is None or df.empty:
        return pd.DataFrame(columns=["ric", "date", "open", "high", "low", "close", "volume"]), str(err)

    cols = list(df.columns)

    inst_col = "Instrument" if "Instrument" in cols else cols[0]
    date_col = get_first_present_column(cols, preferred_exact=["Date"], fallback_substrings=["date"])
    open_col = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["price open", "open"])
    high_col = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["price high", "high"])
    low_col = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["price low", "low"])
    close_col = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["price close", "close"])
    vol_col = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["volume"])

    missing = [x for x in [date_col, open_col, high_col, low_col, close_col, vol_col] if x is None]
    if missing:
        return pd.DataFrame(columns=["ric", "date", "open", "high", "low", "close", "volume"]), (
            f"Could not map required columns from Eikon output. Columns={cols}"
        )

    out = df[[inst_col, date_col, open_col, high_col, low_col, close_col, vol_col]].copy()
    out.columns = ["ric", "date", "open", "high", "low", "close", "volume"]

    out["ric"] = out["ric"].astype(str)

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.drop_duplicates(subset=["ric", "date"]).sort_values(["ric", "date"])
    return out, None


# -----------------------------------------------------------------------------
# Event slicing
# -----------------------------------------------------------------------------
def find_event_trading_date(ts_dates: List[date], anchor_d: date) -> Optional[date]:
    """
    If anchor date in dates -> use it
    else -> next date after anchor
    """
    if not ts_dates:
        return None
    if anchor_d in ts_dates:
        return anchor_d
    # next available
    for d in ts_dates:
        if d > anchor_d:
            return d
    return None


def slice_trading_window_dates(ts_dates: List[date], event_d: date, pre_td: int, post_td: int) -> Tuple[List[date], Dict[str, Any]]:
    """
    Slice list of trading dates around event_d by index.
    Returns the window dates and truncation metadata.
    """
    try:
        pos = ts_dates.index(event_d)
    except ValueError:
        return [], {"event_trading_date": None, "truncated_left": True, "truncated_right": True, "available_rows": 0}

    start_pos = pos - pre_td
    end_pos = pos + post_td

    truncated_left = start_pos < 0
    truncated_right = end_pos >= len(ts_dates)

    start_pos = max(0, start_pos)
    end_pos = min(len(ts_dates) - 1, end_pos)

    window = ts_dates[start_pos : end_pos + 1]
    meta = {
        "event_trading_date": event_d.isoformat(),
        "requested_pre_td": pre_td,
        "requested_post_td": post_td,
        "truncated_left": bool(truncated_left),
        "truncated_right": bool(truncated_right),
        "available_rows": int(len(window)),
        "expected_rows": int(pre_td + post_td + 1),
    }
    return window, meta


# -----------------------------------------------------------------------------
# Earnings time best-effort (kept, but slow)
# -----------------------------------------------------------------------------
EARN_TIME_FIELDS = [
    "TR.EarningsAnnouncementDateTime",
    "TR.EarningsAnnouncementTime",
    "TR.EarningsAnnouncementTimeCode",
    "TR.EarningsReleaseTime",
]


def parse_bmo_amc(value: Any) -> Tuple[str, Optional[str]]:
    if value is None:
        return "TNS", None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return "TNS", None

    u = s.upper()
    if u in {"BMO", "BEFORE MARKET OPEN", "PRE-MARKET", "PRE MARKET", "PREOPEN", "PRE-OPEN"}:
        return "BMO", s
    if u in {"AMC", "AFTER MARKET CLOSE", "POST-MARKET", "POST MARKET", "AFTER HOURS", "AFTER-HOURS"}:
        return "AMC", s
    if "BEFORE" in u and "OPEN" in u:
        return "BMO", s
    if "AFTER" in u and ("CLOSE" in u or "MARKET" in u or "HOURS" in u):
        return "AMC", s
    if any(ch.isdigit() for ch in u) and ":" in u:
        return "TNS", s
    return "TNS", s


def fetch_earnings_time_best_effort(ric: str, anchor_date_str: str) -> Dict[str, Any]:
    """
    Slow, best-effort. Many accounts/fields won't provide reliable historical times.
    """
    assert ek is not None
    anchor_d = safe_date(anchor_date_str)

    out = {
        "bmo_amc_tag": "TNS",
        "earnings_time_raw": None,
        "earnings_time_field_used": None,
        "earnings_time_note": None,
    }

    # Try dated query first
    for fld in EARN_TIME_FIELDS:
        try:
            df, err = ek.get_data([ric], [fld], parameters={"SDate": anchor_d.isoformat(), "EDate": anchor_d.isoformat()})
            if df is None or len(df) == 0:
                continue
            v = df.iloc[0].get(fld)
            tag, raw = parse_bmo_amc(v)
            if raw is not None:
                out.update(
                    {
                        "bmo_amc_tag": tag,
                        "earnings_time_raw": raw,
                        "earnings_time_field_used": fld,
                        "earnings_time_note": "Fetched with SDate/EDate params.",
                    }
                )
                return out
        except Exception:
            continue

    # Fallback: undated (often upcoming)
    for fld in EARN_TIME_FIELDS:
        try:
            df, err = ek.get_data([ric], [fld])
            if df is None or len(df) == 0:
                continue
            v = df.iloc[0].get(fld)
            tag, raw = parse_bmo_amc(v)
            if raw is not None:
                out.update(
                    {
                        "bmo_amc_tag": tag,
                        "earnings_time_raw": raw,
                        "earnings_time_field_used": fld,
                        "earnings_time_note": "Fetched without date params (may be upcoming, not historical).",
                    }
                )
                return out
        except Exception:
            continue

    out["earnings_time_note"] = "No usable earnings time fields returned from Eikon API for this event."
    return out


# -----------------------------------------------------------------------------
# Main runner (importable)
# -----------------------------------------------------------------------------
def run_fetch_stock_prices(
    input_path: Path,
    out_dir: Path,
    app_key: Optional[str] = None,
    pre_trading_days: int = 30,
    post_trading_days: int = 30,
    holiday_buffer_days: int = 10,
    chunk_size: int = 50,
    bucket_mode: str = "month",  # month | quarter | all
    throttle_s: float = 0.0,      # sleep after each ek.get_data batch
    include_earnings_time: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Thread-safe fast runner using batched ek.get_data calls.
    """
    if tqdm is None and show_progress:
        show_progress = False

    set_eikon_app_key(app_key)
    ensure_dir(out_dir)

    corporate_rows = read_jsonl(input_path)
    events, parse_warnings = extract_events(corporate_rows)

    # Group events into buckets to avoid huge date spans per request
    buckets: Dict[str, List[EarningsEvent]] = {}
    for e in events:
        k = _bucket_key(safe_date(e.anchor_date), bucket_mode)
        buckets.setdefault(k, []).append(e)

    lookback_days, lookahead_days = calendar_span_for_trading_days(
        pre_td=pre_trading_days,
        post_td=post_trading_days,
        holiday_buffer_days=holiday_buffer_days,
    )

    # Storage for fetched OHLCV by RIC
    ohlcv_by_ric: Dict[str, pd.DataFrame] = {}

    failures_fetch: List[str] = []
    failures_window: List[str] = []
    failures_time: List[str] = []

    # ---- Fetch OHLCV per bucket in batches ----
    bucket_items = sorted(buckets.items(), key=lambda kv: kv[0])
    bucket_iter = bucket_items
    if show_progress:
        bucket_iter = tqdm(bucket_items, desc=f"Buckets ({bucket_mode})", unit="bucket")  # type: ignore

    for bkey, bevents in bucket_iter:  # type: ignore
        anchors = [safe_date(e.anchor_date) for e in bevents]
        start_cal = min(anchors) - timedelta(days=lookback_days)
        end_cal = max(anchors) + timedelta(days=lookahead_days)

        rics = sorted({e.ric for e in bevents if e.ric})
        if not rics:
            continue

        # Fetch in chunks
        chunk_iter = list(chunked(rics, max(1, int(chunk_size))))
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc=f"OHLCV {bkey}", unit="chunk", leave=False)  # type: ignore

        for ric_chunk in chunk_iter:  # type: ignore
            df_batch, err = fetch_ohlcv_batch(ric_chunk, start_cal, end_cal, throttle_s=throttle_s)
            if err:
                failures_fetch.append(f"Bucket {bkey} chunk size={len(ric_chunk)}: {err}")
                continue

            # Merge batch into per-ric dict
            for ric, g in df_batch.groupby("ric"):
                g2 = g.copy()
                # keep only needed columns
                g2 = g2[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
                if ric in ohlcv_by_ric:
                    merged = pd.concat([ohlcv_by_ric[ric], g2], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
                    ohlcv_by_ric[ric] = merged
                else:
                    ohlcv_by_ric[ric] = g2

    # ---- Build outputs per event ----
    all_rows: List[Dict[str, Any]] = []
    nested_events: List[Dict[str, Any]] = []

    ev_iter = events
    if show_progress:
        ev_iter = tqdm(events, desc="Slicing events", unit="evt")  # type: ignore

    for e in ev_iter:  # type: ignore
        ts = ohlcv_by_ric.get(e.ric)
        if ts is None or ts.empty:
            failures_window.append(f"RIC {e.ric} event market_id={e.market_id} anchor_date={e.anchor_date}: NO_TS_DATA")
            continue

        ts_dates = list(ts["date"].tolist())
        anchor_d = safe_date(e.anchor_date)
        event_d = find_event_trading_date(ts_dates, anchor_d)
        if event_d is None:
            failures_window.append(
                f"RIC {e.ric} event market_id={e.market_id} anchor_date={e.anchor_date}: could not find event trading date"
            )
            continue

        window_dates, wmeta = slice_trading_window_dates(ts_dates, event_d, pre_trading_days, post_trading_days)
        if not window_dates:
            failures_window.append(
                f"RIC {e.ric} event market_id={e.market_id} anchor_date={e.anchor_date}: empty window after slicing"
            )
            continue

        # Earnings time (optional, slow)
        tmeta = {
            "bmo_amc_tag": "TNS",
            "earnings_time_raw": None,
            "earnings_time_field_used": None,
            "earnings_time_note": "Skipped by config.",
        }
        if include_earnings_time:
            tmeta = fetch_earnings_time_best_effort(e.ric, e.anchor_date)
            if tmeta.get("bmo_amc_tag") == "TNS":
                failures_time.append(
                    f"RIC {e.ric} event market_id={e.market_id} anchor_date={e.anchor_date}: "
                    f"BMO/AMC unknown ({tmeta.get('earnings_time_note')})"
                )

        # Build window rows with offsets
        # create a lookup for fast access
        ts_map = {d: row for d, row in ts.set_index("date").iterrows()}

        event_pos = ts_dates.index(event_d)
        records: List[Dict[str, Any]] = []
        for d in window_dates:
            pos = ts_dates.index(d)
            offset_td = pos - event_pos
            row = ts_map.get(d)
            records.append(
                {
                    "date": d.isoformat(),
                    "offset_td": int(offset_td),
                    "OPEN": float(row["open"]) if row is not None and pd.notna(row["open"]) else None,
                    "HIGH": float(row["high"]) if row is not None and pd.notna(row["high"]) else None,
                    "LOW": float(row["low"]) if row is not None and pd.notna(row["low"]) else None,
                    "CLOSE": float(row["close"]) if row is not None and pd.notna(row["close"]) else None,
                    "VOLUME": float(row["volume"]) if row is not None and pd.notna(row["volume"]) else None,
                }
            )

        event_obj: Dict[str, Any] = {
            "market_id": e.market_id,
            "slug": e.slug,
            "ric": e.ric,
            "ticker": e.ticker,
            "company_name": e.company_name,
            "anchor_date": e.anchor_date,
            "event_trading_date": wmeta["event_trading_date"],
            "bmo_amc_tag": tmeta.get("bmo_amc_tag"),
            "earnings_time_raw": tmeta.get("earnings_time_raw"),
            "earnings_time_field_used": tmeta.get("earnings_time_field_used"),
            "earnings_time_note": tmeta.get("earnings_time_note"),
            "window_meta": wmeta,
            "prices": records,
        }
        nested_events.append(event_obj)

        for r in records:
            all_rows.append(
                {
                    "ric": e.ric,
                    "ticker": e.ticker,
                    "company_name": e.company_name,
                    "market_id": e.market_id,
                    "slug": e.slug,
                    "anchor_date": e.anchor_date,
                    "event_trading_date": wmeta["event_trading_date"],
                    "bmo_amc_tag": tmeta.get("bmo_amc_tag"),
                    "earnings_time_raw": tmeta.get("earnings_time_raw"),
                    "earnings_time_field_used": tmeta.get("earnings_time_field_used"),
                    "date": r.get("date"),
                    "offset_td": r.get("offset_td"),
                    "open": r.get("OPEN"),
                    "high": r.get("HIGH"),
                    "low": r.get("LOW"),
                    "close": r.get("CLOSE"),
                    "volume": r.get("VOLUME"),
                    "truncated_left": wmeta.get("truncated_left"),
                    "truncated_right": wmeta.get("truncated_right"),
                }
            )

        if wmeta.get("truncated_left") or wmeta.get("truncated_right"):
            failures_window.append(
                f"RIC {e.ric} event market_id={e.market_id} anchor_date={e.anchor_date}: "
                f"TRUNCATED (left={wmeta.get('truncated_left')}, right={wmeta.get('truncated_right')}, "
                f"available_rows={wmeta.get('available_rows')}/{wmeta.get('expected_rows')})"
            )

    # ---- Write outputs ----
    csv_path = out_dir / "stock_prices_daily.csv"
    jsonl_path = out_dir / "stock_prices_daily.jsonl"
    json_path = out_dir / "stock_prices_daily.json"
    summary_path = out_dir / "stock_prices_summary.txt"

    df_out = pd.DataFrame(all_rows)
    if not df_out.empty:
        df_out.sort_values(["ric", "anchor_date", "market_id", "offset_td"], inplace=True)
        df_out.to_csv(csv_path, index=False, encoding="utf-8")

    write_jsonl(jsonl_path, all_rows)
    write_json(json_path, {"generated_at_utc": datetime.utcnow().isoformat() + "Z", "events": nested_events})

    # ---- Summary ----
    lines: List[str] = []
    lines.append("Polymarket Corporate Earnings — Stock Prices Fetch Summary (FAST MODE)")
    lines.append(f"Generated at (UTC): {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"Input:  {input_path}")
    lines.append(f"Output: {out_dir}")
    lines.append("")
    lines.append(f"Corporate rows: {len(corporate_rows)}")
    lines.append(f"Events parsed:  {len(events)}")
    lines.append(f"RICs fetched:   {len(ohlcv_by_ric)}")
    lines.append(f"Output rows:    {len(all_rows)}")
    lines.append("")
    lines.append(f"Trading-day window: pre={pre_trading_days}, post={post_trading_days}")
    lines.append(f"Bucket mode: {bucket_mode}")
    lines.append(f"Chunk size: {chunk_size}")
    lines.append(f"Throttle after batch: {throttle_s}s")
    lines.append(f"Earnings time included: {include_earnings_time}")
    lines.append("")

    if parse_warnings:
        lines.append("PARSE WARNINGS")
        lines.extend([f"- {w}" for w in parse_warnings])
        lines.append("")

    if failures_fetch:
        lines.append("BATCH FETCH FAILURES")
        lines.extend([f"- {x}" for x in failures_fetch[:200]])
        if len(failures_fetch) > 200:
            lines.append(f"... {len(failures_fetch) - 200} more omitted")
        lines.append("")

    if failures_window:
        lines.append("WINDOW ISSUES / EVENT FAILURES")
        lines.extend([f"- {x}" for x in failures_window[:200]])
        if len(failures_window) > 200:
            lines.append(f"... {len(failures_window) - 200} more omitted")
        lines.append("")

    if include_earnings_time:
        lines.append("BMO/AMC CLASSIFICATION ISSUES (best effort)")
        lines.append(f"- Events with missing/unknown time tag: {len(failures_time)}")
        lines.extend([f"  * {x}" for x in failures_time[:200]])
        if len(failures_time) > 200:
            lines.append(f"  ... {len(failures_time) - 200} more omitted")
        lines.append("")

    write_text(summary_path, "\n".join(lines) + "\n")

    return {
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "events_total": len(events),
        "rows_total": len(all_rows),
        "rics_total": len(ohlcv_by_ric),
        "bucket_mode": bucket_mode,
        "chunk_size": chunk_size,
        "include_earnings_time": include_earnings_time,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch daily stock prices around earnings anchor_date from Eikon (FAST, batched).")
    p.add_argument("--input", type=str, default=str(default_input_path()), help="Path to corporate_info.jsonl")
    p.add_argument("--outdir", type=str, default=str(default_output_dir()), help="Output directory")
    p.add_argument("--app-key", type=str, default=None, help="Eikon App Key (or set env EIKON_APP_KEY)")

    p.add_argument("--pre-td", type=int, default=30, help="Trading days before anchor")
    p.add_argument("--post-td", type=int, default=30, help="Trading days after anchor")
    p.add_argument("--holiday-buffer", type=int, default=10, help="Extra calendar-day buffer for holidays/weekends")

    p.add_argument("--chunk-size", type=int, default=50, help="RICs per ek.get_data call (50 is a good start)")
    p.add_argument("--bucket-mode", type=str, default="month", choices=["month", "quarter", "all"], help="Bucket events to reduce date span")
    p.add_argument("--throttle", type=float, default=0.0, help="Sleep after each batch call (seconds)")

    p.add_argument("--earnings-time", action="store_true", help="Enable slow best-effort BMO/AMC lookup")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
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
        pre_trading_days=int(args.pre_td),
        post_trading_days=int(args.post_td),
        holiday_buffer_days=int(args.holiday_buffer),
        chunk_size=int(args.chunk_size),
        bucket_mode=str(args.bucket_mode),
        throttle_s=float(args.throttle),
        include_earnings_time=bool(args.earnings_time),
        show_progress=not bool(args.no_progress),
    )

    print("DONE")
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
