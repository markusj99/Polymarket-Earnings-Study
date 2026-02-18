#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
heckman_selection.py

Heckman selection model data pull (NYSE + Nasdaq) using Refinitiv Eikon/Workspace.

Fixes in this version
---------------------
- Windows asyncio stability: forces WindowsSelectorEventLoopPolicy.
- Eikon SDK thread safety: serializes ek.get_data() calls (no concurrent requests).
- SCREEN bounded and stops when paging is ignored/repeats.
- Handles duplicate 'Date' columns (Eikon often returns multiple columns named 'Date'):
    - market cap uses date #0
    - analysts uses date #1 (fallback to #0)
  Snapshots computed separately and merged by ric.
- 400 Bad Request resilience: splits failed batches to isolate bad instruments.

Mid-date logic (as requested)
-----------------------------
- Read Corporate_Earnings/statistics/descriptive_statistics/tables/02b_market_end_dates_counts.csv
- Exclude the first end date (outlier)
- observed_start = min(remaining)
- observed_end   = max(remaining)
- mid_date       = observed_start + floor((observed_end - observed_start)/2)

Outputs (relative)
------------------
Corporate_Earnings/statistics/heckman_selection_model/
  - screener_universe_rics.csv
  - screener_universe_rics.jsonl
  - heckman_universe_companies.csv
  - heckman_universe_events.csv
  - heckman_universe_events.jsonl
  - heckman_missing_summary.json
  - heckman_report.txt

Run
---
python heckman_selection.py --app-key <KEY>
or (reads env EIKON_APP_KEY):
python heckman_selection.py --app-key

If proxy errors occur mid-run:
  python heckman_selection.py --app-key --min-interval-s 0.6
"""

from __future__ import annotations

# --- MUST be early on Windows BEFORE Eikon touches asyncio internals ---
import sys
if sys.platform.startswith("win"):
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import argparse
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import eikon as ek  # type: ignore
except Exception:
    ek = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


# =========================
# Paths (relative to script)
# =========================

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


DEFAULT_ENDDATES_CSV = (
    project_root()
    / "statistics"
    / "descriptive_statistics"
    / "tables"
    / "02b_market_end_dates_counts.csv"
)

OUT_DIR = project_root() / "data" / "heckman_selection_model"
OUT_EVENTS_JSONL = OUT_DIR / "heckman_universe_events.jsonl"
OUT_EVENTS_CSV = OUT_DIR / "heckman_universe_events.csv"
OUT_COMPANIES_CSV = OUT_DIR / "heckman_universe_companies.csv"
OUT_MISSING_JSON = OUT_DIR / "heckman_missing_summary.json"
OUT_REPORT_TXT = OUT_DIR / "heckman_report.txt"
OUT_SCREEN_RICS_CSV = OUT_DIR / "screener_universe_rics.csv"
OUT_SCREEN_RICS_JSONL = OUT_DIR / "screener_universe_rics.jsonl"


# =========================
# Defaults
# =========================

DEFAULT_LOOKBACK_DAYS = 183
DEFAULT_BUFFER_DAYS = 5
DEFAULT_ASOF_BUFFER_DAYS = 30

BATCH_STATIC = 120
BATCH_ASOF = 120
BATCH_PV = 25
BATCH_EVENTS = 250

DEFAULT_SCREEN_PAGE_SIZE = 1000
DEFAULT_SCREEN_MAX_PAGES = 25
DEFAULT_SCREEN_MAX_INSTRUMENTS = 25000

EIKON_RETRIES = 6
EIKON_RETRY_BASE_SLEEP = 1.0


# =========================
# Logging / warnings
# =========================

LOG = logging.getLogger("heckman_selection")


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg)
            else:
                sys.stderr.write(msg + "\n")
        except Exception:
            pass


def setup_logging_quiet() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.ERROR)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)

    for name in [
        "asyncio",
        "urllib3",
        "requests",
        "eikon",
        "refinitiv",
        "refinitiv.data",
        "refinitiv.data.eikon",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def setup_warnings_suppression() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"eikon\.data_grid")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"eikon(\..*)?")


def require_eikon() -> None:
    if ek is None:
        raise RuntimeError("eikon package not available. Install via: pip install eikon")


def require_tqdm() -> None:
    if tqdm is None:
        raise RuntimeError("tqdm package not available. Install via: pip install tqdm")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def find_col(columns: List[str], *, exact: List[str] = None, contains_any: List[str] = None) -> Optional[str]:
    exact = exact or []
    contains_any = contains_any or []
    colset = set(columns)
    for e in exact:
        if e in colset:
            return e
    low = [str(c).lower() for c in columns]
    for sub in contains_any:
        s = sub.lower()
        for i, c in enumerate(low):
            if s in c:
                return columns[i]
    return None


# =========================
# Observed window -> mid_date
# =========================

@dataclass
class ObservedWindow:
    enddates_path: Path
    excluded_outlier_date: Optional[date]
    observed_start: date
    observed_end: date
    mid_date: date
    n_dates_total: int
    n_dates_used: int


def compute_observed_window_from_enddates(enddates_csv: Path) -> ObservedWindow:
    if not enddates_csv.exists():
        raise FileNotFoundError(f"End dates file not found: {enddates_csv}")

    df = pd.read_csv(enddates_csv)
    if "end_date_utc" not in df.columns:
        raise ValueError(f"CSV missing required column 'end_date_utc': {enddates_csv}")

    df["end_date_utc"] = pd.to_datetime(df["end_date_utc"], errors="coerce").dt.date
    df = df.dropna(subset=["end_date_utc"]).copy()
    df = df.sort_values("end_date_utc")

    if len(df) < 2:
        raise ValueError(f"Need at least 2 valid dates in {enddates_csv}.")

    excluded = df.iloc[0]["end_date_utc"]
    df2 = df.iloc[1:].copy()
    if df2.empty:
        raise ValueError("After excluding first end date, no dates remain.")

    start = df2["end_date_utc"].min()
    end = df2["end_date_utc"].max()
    if start is None or end is None:
        raise ValueError("Could not compute observed start/end from end dates file.")

    mid = start + timedelta(days=(end - start).days // 2)
    return ObservedWindow(
        enddates_path=enddates_csv,
        excluded_outlier_date=excluded,
        observed_start=start,
        observed_end=end,
        mid_date=mid,
        n_dates_total=int(len(df)),
        n_dates_used=int(len(df2)),
    )


# =========================
# Eikon client (SERIALIZED)
# =========================

def _looks_like_proxy_down(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("proxy not running" in s) or ("cannot be reached" in s) or ("no proxy address" in s)


def _looks_like_400(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("400" in s and "bad request" in s) or ("backend error" in s and "400" in s)


class RateLimiter:
    def __init__(self, min_interval_s: float) -> None:
        import threading
        self._min = float(min_interval_s)
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now < self._next_ok:
                time.sleep(self._next_ok - now)
            self._next_ok = time.monotonic() + self._min


@dataclass
class EikonCallResult:
    df: Optional[pd.DataFrame]
    err: Optional[Any]
    exc: Optional[str]


class EikonClient:
    """
    Thread-safety note:
    The Eikon SDK internally uses an async session. On Windows, calling ek.get_data()
    concurrently can trigger asyncio task mismatch errors.

    We enforce single-threaded ek.get_data usage via a lock.
    """
    def __init__(self, app_key: str, *, min_interval_s: float = 0.35) -> None:
        require_eikon()
        ek.set_app_key(app_key)
        try:
            set_level = getattr(ek, "set_log_level", None)
            if callable(set_level):
                set_level(0)
        except Exception:
            pass

        import threading
        self._call_lock = threading.Lock()
        self.limiter = RateLimiter(min_interval_s=float(min_interval_s))

    def get_data(self, instruments: Any, fields: List[Any], parameters: Dict[str, Any]) -> EikonCallResult:
        last_exc: Optional[Exception] = None
        for attempt in range(EIKON_RETRIES):
            try:
                self.limiter.wait()
                with self._call_lock:
                    df, err = ek.get_data(instruments, fields, parameters=parameters)
                if isinstance(df, pd.DataFrame):
                    return EikonCallResult(df=df, err=err, exc=None)
                return EikonCallResult(df=None, err=err, exc=None)
            except Exception as exc:
                last_exc = exc
                if _looks_like_proxy_down(exc):
                    time.sleep(5.0)
                time.sleep(EIKON_RETRY_BASE_SLEEP * (2 ** attempt))
        return EikonCallResult(df=None, err=None, exc=str(last_exc) if last_exc else "Unknown error")


def _tr_field(name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        tf = getattr(ek, "TR_Field", None)
        if callable(tf):
            return tf(name, params) if params else tf(name)
    except Exception:
        pass
    if params:
        inside = ",".join([f"{k}={v}" for k, v in params.items()])
        return f"{name}({inside})"
    return name


# =========================
# Robust batched call with splitting on 400
# =========================

def get_data_batched_split(
    client: EikonClient,
    instruments: List[str],
    fields: List[Any],
    parameters: Dict[str, Any],
    *,
    max_split_depth: int = 4,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    If a batch fails with 400 Bad Request, split it to isolate problematic instruments.
    Returns concatenated DF + problems list.
    """
    problems: List[str] = []

    def _call(batch: List[str], depth: int) -> pd.DataFrame:
        if not batch:
            return pd.DataFrame()

        res = client.get_data(batch, fields, parameters)
        if res.df is not None and not res.df.empty:
            return res.df

        if res.exc:
            problems.append(f"batch_failed size={len(batch)} exc={res.exc}")

        # Split only when the failure smells like 400 (often caused by one bad instrument),
        # and only when we have room to split further.
        if res.exc and "400" in res.exc and len(batch) > 1 and depth > 0:
            mid = len(batch) // 2
            left = _call(batch[:mid], depth - 1)
            right = _call(batch[mid:], depth - 1)
            if left.empty and right.empty:
                return pd.DataFrame()
            if left.empty:
                return right
            if right.empty:
                return left
            return pd.concat([left, right], ignore_index=True)

        return pd.DataFrame()

    df = _call(instruments, max_split_depth)
    return df, problems


# =========================
# SCREEN (bounded + detects paging ignored)
# =========================

@dataclass
class ScreenStats:
    pages_processed: int
    instruments_collected: int
    stop_reason: str


def screen_nyse_nasdaq_rics(
    client: EikonClient,
    *,
    page_size: int,
    max_pages: int,
    max_instruments: int,
) -> Tuple[List[str], List[str], ScreenStats]:
    problems: List[str] = []

    screen = (
        'SCREEN(U(IN(Equity(active,public,primary,countryprimaryquote))/*UNV:Public*/),'
        ' IN(TR.ExchangeMarketIdCode,"XNYS","XNAS"))'
    )
    fields = ["TR.CommonName", "TR.ExchangeMarketIdCode", "TR.PrimaryExchangeName", "TR.TickerSymbol"]

    seen: set[str] = set()
    pages = 0
    start = 0
    stop_reason = "unknown"

    pbar = tqdm(desc="Screening XNYS/XNAS equities (pages)", unit="page") if tqdm else None
    try:
        for _ in range(max_pages):
            pages += 1
            params = {"StartNum": start, "EndNum": start + page_size}
            res = client.get_data(screen, fields, params)
            dfp = res.df

            if dfp is None or dfp.empty:
                stop_reason = "empty_page"
                break

            inst_col = "Instrument" if "Instrument" in dfp.columns else dfp.columns[0]
            page_rics = [str(x).strip() for x in dfp[inst_col].tolist() if str(x).strip()]

            new_count = 0
            for r in page_rics:
                if r not in seen:
                    seen.add(r)
                    new_count += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"rics={len(seen)} new={new_count} rows={len(dfp)}")

            # Detect ignored paging: page_size=1000 but page has >1000 rows.
            # In your run: rows=2132 on page 1 => we stop after page 1.
            if pages == 1 and len(dfp) > page_size:
                stop_reason = "paging_ignored_returned_full_set_on_page1"
                problems.append(
                    f"SCREEN appears to ignore StartNum/EndNum (page1 rows={len(dfp)} > page_size={page_size}). "
                    "Stopping after page 1 to avoid repeats."
                )
                break

            if new_count == 0:
                stop_reason = "no_new_instruments_repeat_page"
                problems.append("SCREEN pagination appears to repeat pages (no new instruments). Stopped.")
                break

            if len(dfp) < page_size:
                stop_reason = "short_last_page"
                break

            if len(seen) >= max_instruments:
                stop_reason = f"hit_max_instruments_{max_instruments}"
                problems.append(f"Hit max_instruments cap ({max_instruments}). Consider tightening SCREEN filters.")
                break

            start += page_size
        else:
            stop_reason = f"hit_max_pages_{max_pages}"
            problems.append(f"Hit max_pages cap ({max_pages}). Consider tightening SCREEN filters.")
    finally:
        if pbar is not None:
            pbar.close()

    rics = sorted(seen)
    if not rics:
        problems.append("SCREEN returned no instruments (check entitlements / proxy / login / screener syntax).")

    stats = ScreenStats(pages_processed=pages, instruments_collected=len(rics), stop_reason=stop_reason)
    return rics, problems, stats


# =========================
# Fetchers (batched + split)
# =========================

def fetch_static_metadata(client: EikonClient, rics: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    problems: List[str] = []
    fields: List[Any] = [
        "TR.RIC",
        "TR.TickerSymbol",
        "TR.CommonName",
        "TR.GICSSector",
        "TR.GICSIndustry",
        "TR.TRBCIndustry",
        "TR.PrimaryExchangeName",
        "TR.ExchangeMarketIdCode",
        "TR.HeadquartersCountry",
        "TR.HQCountryCode",
        "TR.CoRPrimaryCountry",
        "TR.ExchangeCountry",
    ]

    parts: List[pd.DataFrame] = []
    pbar = tqdm(total=(len(rics) + BATCH_STATIC - 1) // BATCH_STATIC, desc="Static metadata (batches)", unit="batch") if tqdm else None
    try:
        for batch in chunked(rics, BATCH_STATIC):
            dfb, probs = get_data_batched_split(client, batch, fields, {}, max_split_depth=4)
            problems.extend(probs)
            if not dfb.empty:
                parts.append(dfb)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), problems


def fetch_asof_marketcap_and_analysts(
    client: EikonClient,
    rics: List[str],
    asof_date: date,
    *,
    buffer_days: int,
) -> Tuple[pd.DataFrame, List[str]]:
    problems: List[str] = []
    sdate = asof_date - timedelta(days=int(buffer_days))
    edate = asof_date

    fields: List[Any] = [
        _tr_field("TR.CompanyMarketCap", {"Curn": "USD"}),
        "TR.CompanyMarketCap.date",
        "TR.NumberOfAnalysts",
        "TR.NumberOfAnalysts.date",
    ]
    params = {"SDate": sdate.isoformat(), "EDate": edate.isoformat(), "Frq": "D"}

    parts: List[pd.DataFrame] = []
    pbar = tqdm(total=(len(rics) + BATCH_ASOF - 1) // BATCH_ASOF, desc="As-of mcap+analysts (batches)", unit="batch") if tqdm else None
    try:
        for batch in chunked(rics, BATCH_ASOF):
            dfb, probs = get_data_batched_split(client, batch, fields, dict(params), max_split_depth=4)
            problems.extend(probs)
            if not dfb.empty:
                parts.append(dfb)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), problems


def fetch_daily_pv(client: EikonClient, rics: List[str], start_date: date, end_date: date) -> Tuple[pd.DataFrame, List[str]]:
    problems: List[str] = []
    fields = ["TR.PriceClose", "TR.Volume", "TR.PriceClose.date"]
    params = {"SDate": start_date.isoformat(), "EDate": end_date.isoformat(), "Frq": "D"}

    parts: List[pd.DataFrame] = []
    pbar = tqdm(total=(len(rics) + BATCH_PV - 1) // BATCH_PV, desc="Daily Price/Volume (batches)", unit="batch") if tqdm else None
    try:
        for batch in chunked(rics, BATCH_PV):
            dfb, probs = get_data_batched_split(client, batch, fields, dict(params), max_split_depth=4)
            problems.extend(probs)
            if not dfb.empty:
                parts.append(dfb)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), problems


def fetch_events_results(client: EikonClient, rics: List[str], start_date: date, end_date: date) -> Tuple[pd.DataFrame, List[str]]:
    problems: List[str] = []
    fields = ["TR.EventStartDate", "TR.EventStartTime", "TR.EventType", "TR.EventTitle"]
    params = {"SDate": start_date.isoformat(), "EDate": end_date.isoformat(), "EventType": "RES", "RH": "IN", "CH": "Fd"}

    parts: List[pd.DataFrame] = []
    pbar = tqdm(total=(len(rics) + BATCH_EVENTS - 1) // BATCH_EVENTS, desc="Events (RES) (batches)", unit="batch") if tqdm else None
    try:
        for batch in chunked(rics, BATCH_EVENTS):
            dfb, probs = get_data_batched_split(client, batch, fields, dict(params), max_split_depth=4)
            problems.extend(probs)
            if not dfb.empty:
                parts.append(dfb)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), problems


# =========================
# Normalization helpers
# =========================

def _as_series_or_first(df: pd.DataFrame, colname: str) -> pd.Series:
    """
    If df has duplicate columns with the same name, df[colname] returns a DataFrame (2D).
    This returns the first column as a Series in that case.
    """
    x = df.loc[:, colname]
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def _extract_all_date_series(df: pd.DataFrame) -> List[pd.Series]:
    """
    Returns ALL date-like columns as Series, in column order.
    Handles duplicated 'Date' columns (common in Eikon results).
    """
    cols = list(df.columns)
    # prioritize exact 'Date' (case-insensitive)
    date_names = [c for c in cols if str(c).strip().lower() == "date"]
    if not date_names:
        # fallback to any column containing "date"
        date_names = [c for c in cols if "date" in str(c).lower()]

    out: List[pd.Series] = []
    for name in date_names:
        x = df.loc[:, name]
        if isinstance(x, pd.DataFrame):
            for j in range(x.shape[1]):
                out.append(x.iloc[:, j])
        else:
            out.append(x)
    return out


def _normalize_static(static_df: pd.DataFrame) -> pd.DataFrame:
    if static_df is None or static_df.empty:
        return pd.DataFrame(columns=[
            "ric", "ticker", "company_name",
            "gics_sector", "gics_industry", "trbc_industry",
            "primary_exchange", "exchange_mic",
            "hq_country", "hq_country_code", "country_of_risk", "exchange_country",
        ])

    cols = list(static_df.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]

    col_ticker = find_col(cols, contains_any=["ticker symbol", "tickersymbol"])
    col_name = find_col(cols, contains_any=["common name"])
    col_gics_sector = find_col(cols, contains_any=["gics sector"])
    col_gics_ind = find_col(cols, contains_any=["gics industry"])
    col_trbc = find_col(cols, contains_any=["trbc industry"])
    col_exch = find_col(cols, contains_any=["primary exchange name", "exchange name"])
    col_mic = find_col(cols, contains_any=["exchange market id code"])
    col_hq_country = find_col(cols, contains_any=["headquarters country", "country of headquarters"])
    col_hq_code = find_col(cols, contains_any=["hqcountrycode", "iso code of headquarters"])
    col_risk = find_col(cols, contains_any=["primary country of risk", "cor primary country"])
    col_exch_country = find_col(cols, contains_any=["exchange country", "country of exchange"])

    out = pd.DataFrame({
        "ric": static_df[inst_col].astype(str),
        "ticker": static_df[col_ticker] if col_ticker else None,
        "company_name": static_df[col_name] if col_name else None,
        "gics_sector": static_df[col_gics_sector] if col_gics_sector else None,
        "gics_industry": static_df[col_gics_ind] if col_gics_ind else None,
        "trbc_industry": static_df[col_trbc] if col_trbc else None,
        "primary_exchange": static_df[col_exch] if col_exch else None,
        "exchange_mic": static_df[col_mic] if col_mic else None,
        "hq_country": static_df[col_hq_country] if col_hq_country else None,
        "hq_country_code": static_df[col_hq_code] if col_hq_code else None,
        "country_of_risk": static_df[col_risk] if col_risk else None,
        "exchange_country": static_df[col_exch_country] if col_exch_country else None,
    }).drop_duplicates(subset=["ric"])

    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).replace({"None": ""}).str.strip()
            out[c] = out[c].where(out[c].str.len() > 0, None)
    return out


def normalize_asof_marketcap(asof_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Return long-form: ric, date, market_cap_usd
    Uses date series #0 (first Date-like column) by convention.
    """
    if asof_raw is None or asof_raw.empty:
        return pd.DataFrame(columns=["ric", "date", "market_cap_usd"])

    cols = list(asof_raw.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]

    col_mcap = find_col(cols, contains_any=["company market cap", "market cap"])
    date_series = _extract_all_date_series(asof_raw)
    ds0 = date_series[0] if date_series else pd.Series([None] * len(asof_raw))

    out = pd.DataFrame({
        "ric": asof_raw[inst_col].astype(str),
        "date": ds0.astype(str).str.slice(0, 10),
        "market_cap_usd": asof_raw[col_mcap] if col_mcap else np.nan,
    })

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["market_cap_usd"] = pd.to_numeric(out["market_cap_usd"], errors="coerce")
    out = out.dropna(subset=["ric", "date"]).sort_values(["ric", "date"]).drop_duplicates(subset=["ric", "date"])
    return out


def normalize_asof_analysts(asof_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Return long-form: ric, date, analysts
    Uses date series #1 if available, else falls back to #0.
    """
    if asof_raw is None or asof_raw.empty:
        return pd.DataFrame(columns=["ric", "date", "analysts"])

    cols = list(asof_raw.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]

    col_an = find_col(cols, contains_any=["number of analysts", "analysts"])
    date_series = _extract_all_date_series(asof_raw)
    if not date_series:
        ds = pd.Series([None] * len(asof_raw))
    else:
        ds = date_series[1] if len(date_series) >= 2 else date_series[0]

    out = pd.DataFrame({
        "ric": asof_raw[inst_col].astype(str),
        "date": ds.astype(str).str.slice(0, 10),
        "analysts": asof_raw[col_an] if col_an else np.nan,
    })

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["analysts"] = pd.to_numeric(out["analysts"], errors="coerce")
    out = out.dropna(subset=["ric", "date"]).sort_values(["ric", "date"]).drop_duplicates(subset=["ric", "date"])
    return out


def snapshot_last_value_asof(df: pd.DataFrame, asof_date: date, value_col: str, out_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ric", out_col])
    sub = df[df["date"] <= asof_date].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ric", out_col])
    sub = sub.sort_values(["ric", "date"])
    last = sub.groupby("ric", sort=False).tail(1)
    out = last[["ric", value_col]].copy().rename(columns={value_col: out_col})
    return out


def _normalize_pv(pv_df: pd.DataFrame) -> pd.DataFrame:
    if pv_df is None or pv_df.empty:
        return pd.DataFrame(columns=["ric", "date", "price", "volume"])

    cols = list(pv_df.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]
    col_date = find_col(cols, exact=["Date"], contains_any=["date"])
    col_price = find_col(cols, contains_any=["price close", "priceclose", "close"])
    col_vol = find_col(cols, contains_any=["volume"])

    # If duplicate 'Date', take first
    if col_date is None:
        date_series = _extract_all_date_series(pv_df)
        ds0 = date_series[0] if date_series else pd.Series([None] * len(pv_df))
        date_str = ds0.astype(str)
    else:
        x = pv_df.loc[:, col_date]
        if isinstance(x, pd.DataFrame):
            date_str = x.iloc[:, 0].astype(str)
        else:
            date_str = x.astype(str)

    out = pd.DataFrame({
        "ric": pv_df[inst_col].astype(str),
        "date": date_str.str.slice(0, 10),
        "price": pv_df[col_price] if col_price else np.nan,
        "volume": pv_df[col_vol] if col_vol else np.nan,
    })

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out = out.dropna(subset=["ric", "date"]).drop_duplicates(subset=["ric", "date"]).sort_values(["ric", "date"])
    return out


def _normalize_events(ev_df: pd.DataFrame) -> pd.DataFrame:
    if ev_df is None or ev_df.empty:
        return pd.DataFrame(columns=["ric", "event_date", "event_time", "event_title", "event_type"])

    cols = list(ev_df.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]
    col_d = find_col(cols, contains_any=["event start date", "eventstartdate"])
    col_t = find_col(cols, contains_any=["event start time", "eventstarttime"])
    col_type = find_col(cols, contains_any=["event type"])
    col_title = find_col(cols, contains_any=["event title", "event name", "event headline"])

    out = pd.DataFrame({
        "ric": ev_df[inst_col].astype(str),
        "event_date": ev_df[col_d] if col_d else None,
        "event_time": ev_df[col_t] if col_t else None,
        "event_type": ev_df[col_type] if col_type else None,
        "event_title": ev_df[col_title] if col_title else None,
    })

    out["event_date"] = out["event_date"].astype(str).str.slice(0, 10)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.date
    out["event_time"] = out["event_time"].astype(str).replace({"None": ""}).str.strip()
    out["event_type"] = out["event_type"].astype(str).replace({"None": ""}).str.strip()
    out["event_title"] = out["event_title"].astype(str).replace({"None": ""}).str.strip()

    out = out.dropna(subset=["ric", "event_date"]).sort_values(["ric", "event_date"])
    out = out.drop_duplicates(subset=["ric", "event_date", "event_time", "event_title"])
    return out


def compute_firm_features_asof(pv: pd.DataFrame, asof_date: date, lookback_days: int) -> pd.DataFrame:
    if pv.empty:
        return pd.DataFrame(columns=[
            "ric",
            "turnover_lookback_window_start_asof_mid",
            "turnover_lookback_window_end_asof_mid",
            "turnover_lookback_sum_volume_asof_mid",
            "turnover_lookback_avg_daily_volume_asof_mid",
            "volatility_lookback_asof_mid",
        ])

    w0 = asof_date - timedelta(days=int(lookback_days))
    w1 = asof_date
    w = pv[(pv["date"] >= w0) & (pv["date"] <= w1)].copy()
    if w.empty:
        return pd.DataFrame(columns=[
            "ric",
            "turnover_lookback_window_start_asof_mid",
            "turnover_lookback_window_end_asof_mid",
            "turnover_lookback_sum_volume_asof_mid",
            "turnover_lookback_avg_daily_volume_asof_mid",
            "volatility_lookback_asof_mid",
        ])

    w = w.sort_values(["ric", "date"])

    agg = (
        w.groupby("ric", sort=False)
         .agg(
             turnover_lookback_sum_volume_asof_mid=("volume", "sum"),
             turnover_lookback_avg_daily_volume_asof_mid=("volume", "mean"),
         )
         .reset_index()
    )

    def vol_fun(g: pd.DataFrame) -> float:
        px = pd.to_numeric(g["price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        px = px[px > 0]
        if len(px) < 3:
            return np.nan
        lr = np.log(px).diff().dropna()
        if len(lr) < 2:
            return np.nan
        return float(lr.std(ddof=1))

    vol = w.groupby("ric", sort=False).apply(vol_fun).rename("volatility_lookback_asof_mid").reset_index()

    out = agg.merge(vol, on="ric", how="left")
    out["turnover_lookback_window_start_asof_mid"] = w0.isoformat()
    out["turnover_lookback_window_end_asof_mid"] = w1.isoformat()
    return out


# =========================
# IO / reporting
# =========================

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_df_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = r.to_dict()
            for k, v in list(obj.items()):
                if isinstance(v, (np.integer,)):
                    obj[k] = int(v)
                if isinstance(v, (np.floating,)):
                    obj[k] = float(v)
                if isinstance(v, (date, datetime)):
                    obj[k] = v.isoformat()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def missing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"generated_at_utc": utc_now_iso(), "rows": int(len(df)), "columns": {}}
    n = len(df)
    for c in df.columns:
        miss = int(df[c].isna().sum()) if hasattr(df[c], "isna") else 0
        out["columns"][c] = {"missing": miss, "total": int(n), "missing_pct": (miss / n * 100.0) if n else None}
    return out


def write_report(path: Path, *, sections: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("=============== HECKMAN UNIVERSE REPORT ===============")
    lines.append(f"Generated at (UTC): {utc_now_iso()}")
    lines.append("=======================================================")
    for title, body in sections:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        lines.append(body.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Args / main
# =========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eikon data pull for Heckman selection model (bounded + Windows-safe).")

    p.add_argument("--app-key", nargs="?", const="__ENV__", default=None,
                   help="Eikon app key. If provided with no value, reads env EIKON_APP_KEY.")
    p.add_argument("--enddates-csv", type=str, default=str(DEFAULT_ENDDATES_CSV))

    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    p.add_argument("--asof-buffer-days", type=int, default=DEFAULT_ASOF_BUFFER_DAYS)

    p.add_argument("--min-interval-s", type=float, default=0.35)

    p.add_argument("--screen-page-size", type=int, default=DEFAULT_SCREEN_PAGE_SIZE)
    p.add_argument("--screen-max-pages", type=int, default=DEFAULT_SCREEN_MAX_PAGES)
    p.add_argument("--screen-max-instruments", type=int, default=DEFAULT_SCREEN_MAX_INSTRUMENTS)

    p.add_argument("--max-rics", type=int, default=None, help="TEST MODE: limit number of RICs after screening.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    require_tqdm()
    setup_logging_quiet()
    setup_warnings_suppression()

    args = parse_args(argv)

    # app key
    if args.app_key is None:
        print("ERROR: Missing --app-key (or use --app-key with no value to read env EIKON_APP_KEY).")
        return 2
    if args.app_key == "__ENV__":
        app_key = os.getenv("EIKON_APP_KEY") or os.getenv("APP_KEY") or ""
        if not app_key:
            print("ERROR: EIKON_APP_KEY (or APP_KEY) not found in environment.")
            return 2
    else:
        app_key = args.app_key

    # observed window + mid date
    try:
        window = compute_observed_window_from_enddates(Path(args.enddates_csv))
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_report(OUT_REPORT_TXT, sections=[("Fatal", f"Failed to compute mid_date from enddates CSV: {exc}")])
        return 2

    observed_start = window.observed_start
    observed_end = window.observed_end
    mid_date = window.mid_date

    lookback_days = int(args.lookback_days)
    buffer_days = int(args.buffer_days)
    asof_buffer_days = int(args.asof_buffer_days)

    client = EikonClient(app_key, min_interval_s=float(args.min_interval_s))

    # SCREEN (bounded)
    rics, screen_problems, screen_stats = screen_nyse_nasdaq_rics(
        client,
        page_size=int(args.screen_page_size),
        max_pages=int(args.screen_max_pages),
        max_instruments=int(args.screen_max_instruments),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ric": rics}).to_csv(OUT_SCREEN_RICS_CSV, index=False, encoding="utf-8")
    write_jsonl(OUT_SCREEN_RICS_JSONL, [{"ric": r} for r in rics])

    if not rics:
        write_report(OUT_REPORT_TXT, sections=[
            ("Observed window / midpoint", "\n".join([
                f"Enddates CSV:              {window.enddates_path}",
                f"Excluded outlier (first):  {window.excluded_outlier_date}",
                f"Observed start / end:      {observed_start} .. {observed_end}",
                f"Mid date (as-of):          {mid_date}",
            ])),
            ("Fatal", "SCREEN returned zero instruments."),
            ("Screener problems", "\n".join(screen_problems) if screen_problems else "(none)"),
        ])
        return 2

    if args.max_rics is not None:
        rics = rics[: int(args.max_rics)]

    # Do NOT fetch in parallel; batching provides safe speed.
    pv_start = mid_date - timedelta(days=lookback_days + buffer_days)
    pv_end = mid_date

    static_raw, static_problems = fetch_static_metadata(client, rics)
    asof_raw, asof_problems = fetch_asof_marketcap_and_analysts(client, rics, mid_date, buffer_days=asof_buffer_days)
    pv_raw, pv_problems = fetch_daily_pv(client, rics, pv_start, pv_end)
    ev_raw, ev_problems = fetch_events_results(client, rics, observed_start, observed_end)

    # normalize
    static_norm = _normalize_static(static_raw)
    pv = _normalize_pv(pv_raw)
    events = _normalize_events(ev_raw)

    # as-of normalization (FIXED: handles duplicate Date columns)
    mcap_long = normalize_asof_marketcap(asof_raw)
    an_long = normalize_asof_analysts(asof_raw)

    mcap_snap = snapshot_last_value_asof(mcap_long, mid_date, "market_cap_usd", "market_cap_usd_asof_mid")
    an_snap = snapshot_last_value_asof(an_long, mid_date, "analysts", "analysts_covering_asof_mid")

    asof_snap = mcap_snap.merge(an_snap, on="ric", how="outer")

    firm_feat = compute_firm_features_asof(pv, asof_date=mid_date, lookback_days=lookback_days)

    # companies
    companies = static_norm.merge(asof_snap, on="ric", how="left").merge(firm_feat, on="ric", how="left")
    companies["asof_mid_date_utc"] = mid_date.isoformat()
    companies["observed_window_start_utc"] = observed_start.isoformat()
    companies["observed_window_end_utc"] = observed_end.isoformat()
    companies["retrieved_at_utc"] = utc_now_iso()

    # events join
    if not events.empty:
        events = events.merge(companies, on="ric", how="left", suffixes=("", "_company"))

    companies.sort_values(["exchange_mic", "ric"]).to_csv(OUT_COMPANIES_CSV, index=False, encoding="utf-8")
    events.to_csv(OUT_EVENTS_CSV, index=False, encoding="utf-8")
    write_df_jsonl(OUT_EVENTS_JSONL, events)

    ms = missing_summary(events) if not events.empty else {"generated_at_utc": utc_now_iso(), "rows": 0, "columns": {}}
    OUT_MISSING_JSON.write_text(json.dumps(ms, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_lines: List[str] = []
    if not events.empty:
        for c, s in sorted(ms["columns"].items(), key=lambda kv: (-kv[1]["missing"], kv[0]))[:60]:
            pct = s["missing_pct"]
            missing_lines.append(
                f"{c:<38} missing={s['missing']:>8} / {s['total']:<8} ({pct:6.2f}%)"
                if pct is not None else f"{c} missing={s['missing']}"
            )

    sections: List[Tuple[str, str]] = []
    sections.append((
        "Observed window / midpoint from end dates file",
        "\n".join([
            f"Enddates CSV:              {window.enddates_path}",
            f"Excluded outlier (first):  {window.excluded_outlier_date}",
            f"Dates total / used:        {window.n_dates_total} / {window.n_dates_used}",
            f"Observed start / end:      {observed_start} .. {observed_end}",
            f"Mid date (as-of):          {mid_date}",
        ])
    ))
    sections.append((
        "SCREEN stats (bounded)",
        "\n".join([
            f"Page size:                 {int(args.screen_page_size)}",
            f"Max pages cap:             {int(args.screen_max_pages)}",
            f"Max instruments cap:       {int(args.screen_max_instruments)}",
            f"Pages processed:           {screen_stats.pages_processed}",
            f"Instruments collected:     {screen_stats.instruments_collected}",
            f"Stop reason:               {screen_stats.stop_reason}",
            f"Universe used downstream:  {len(rics)} (after --max-rics if set)",
        ])
    ))
    sections.append((
        "Top problems / warnings",
        "\n".join([
            *(screen_problems or ["(none from screener)"]),
            *(static_problems or ["(none from static fetch)"]),
            *(asof_problems or ["(none from asof fetch)"]),
            *(pv_problems or ["(none from pv fetch)"]),
            *(ev_problems or ["(none from events fetch)"]),
        ])
    ))
    sections.append((
        "Missing values (top 60 columns)",
        "\n".join(missing_lines) if missing_lines else "(no events or no missingness computed)"
    ))
    write_report(OUT_REPORT_TXT, sections=sections)

    msg = (
        "\n==================== DONE ====================\n"
        f"Output dir:       {OUT_DIR}\n"
        f"Screener RICs:    {OUT_SCREEN_RICS_CSV}\n"
        f"Companies CSV:    {OUT_COMPANIES_CSV}\n"
        f"Events CSV:       {OUT_EVENTS_CSV}\n"
        f"Events JSONL:     {OUT_EVENTS_JSONL}\n"
        f"Missing JSON:     {OUT_MISSING_JSON}\n"
        f"Report TXT:       {OUT_REPORT_TXT}\n"
        f"As-of mid date:   {mid_date}\n"
        f"Observed window:  {observed_start} .. {observed_end}\n"
        "=============================================\n"
    )
    tqdm.write(msg) if tqdm is not None else print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
