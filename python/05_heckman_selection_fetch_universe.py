#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
05_heckman_selection_fetch_universe.py

Heckman selection model data pull (NYSE + Nasdaq) using Refinitiv Eikon/Workspace.

UPDATED (2026-02-18)
--------------------
This version no longer depends on Python-generated descriptive statistics tables.

Instead, it:
- Reads observed market window directly from:
    Polymarket-Earnings-Study/data/markets/markets.jsonl
- Uses the market fields:
    startDate, umaEndDate
- Disregards the FIRST market (outlier) and uses the SECOND earliest startDate
  as the start of the observed window.
- Uses the maximum umaEndDate as the end of the observed window.
- Fetches all earnings events (Eikon RES events) between observed_start and observed_end.
- Computes corporate metrics "as-of" 2 days before each earnings event.
  (asof_date = event_date - 2 days)

Outputs (relative to project root)
----------------------------------
data/heckman_selection_model/
  - screener_universe_rics.csv
  - screener_universe_rics.jsonl
  - heckman_universe_companies.csv
  - heckman_universe_events.csv
  - heckman_universe_events.jsonl
  - heckman_missing_summary.json
  - heckman_report.txt

Run
---
python python/05_heckman_selection_fetch_universe.py --app-key <KEY>
or (reads env EIKON_APP_KEY):
python python/05_heckman_selection_fetch_universe.py --app-key
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


DEFAULT_MARKETS_JSONL = project_root() / "data" / "markets" / "markets.jsonl"

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
# Markets.jsonl -> observed window
# =========================

@dataclass
class ObservedWindow:
    markets_path: Path
    excluded_outlier_start: Optional[date]
    observed_start: date
    observed_end: date
    n_markets_total: int
    n_markets_used: int


def _parse_iso_dt(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    try:
        return pd.to_datetime(str(s), errors="coerce", utc=True).to_pydatetime()
    except Exception:
        return None


def compute_observed_window_from_markets(markets_jsonl: Path) -> ObservedWindow:
    """
    Uses markets.jsonl:
      - observed_start = second earliest startDate (exclude the first market as outlier)
      - observed_end   = max umaEndDate
    """
    if not markets_jsonl.exists():
        raise FileNotFoundError(f"Markets file not found: {markets_jsonl}")

    starts: List[date] = []
    ends: List[date] = []

    n = 0
    with markets_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            obj = json.loads(line)

            sdt = _parse_iso_dt(obj.get("startDate"))
            edt = _parse_iso_dt(obj.get("umaEndDate"))

            if sdt is not None:
                starts.append(sdt.date())
            if edt is not None:
                ends.append(edt.date())

    if n == 0:
        raise ValueError(f"No rows found in {markets_jsonl}.")
    if len(starts) < 2:
        raise ValueError("Need at least 2 valid startDate values to exclude the first (outlier).")
    if len(ends) < 1:
        raise ValueError("Need at least 1 valid umaEndDate to compute observed_end.")

    starts_sorted = sorted(starts)
    excluded = starts_sorted[0]
    observed_start = starts_sorted[1]  # second earliest (exclude first as outlier)
    observed_end = max(ends)

    if observed_end < observed_start:
        raise ValueError(f"Observed end ({observed_end}) is earlier than observed start ({observed_start}).")

    return ObservedWindow(
        markets_path=markets_jsonl,
        excluded_outlier_start=excluded,
        observed_start=observed_start,
        observed_end=observed_end,
        n_markets_total=int(n),
        n_markets_used=int(n - 1),
    )


# =========================
# Eikon client (SERIALIZED)
# =========================

def _looks_like_proxy_down(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("proxy not running" in s) or ("cannot be reached" in s) or ("no proxy address" in s)


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
    problems: List[str] = []

    def _call(batch: List[str], depth: int) -> pd.DataFrame:
        if not batch:
            return pd.DataFrame()

        res = client.get_data(batch, fields, parameters)
        if res.df is not None and not res.df.empty:
            return res.df

        if res.exc:
            problems.append(f"batch_failed size={len(batch)} exc={res.exc}")

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


def fetch_marketcap_and_analysts_series(
    client: EikonClient,
    rics: List[str],
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pull daily series of market cap and analyst counts for the whole window.
    Later we take "last value as-of event_asof_date" per event.
    """
    problems: List[str] = []
    fields: List[Any] = [
        _tr_field("TR.CompanyMarketCap", {"Curn": "USD"}),
        "TR.CompanyMarketCap.date",
        "TR.NumberOfAnalysts",
        "TR.NumberOfAnalysts.date",
    ]
    params = {"SDate": start_date.isoformat(), "EDate": end_date.isoformat(), "Frq": "D"}

    parts: List[pd.DataFrame] = []
    pbar = tqdm(total=(len(rics) + BATCH_ASOF - 1) // BATCH_ASOF, desc="Mcap+analysts series (batches)", unit="batch") if tqdm else None
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

def _extract_all_date_series(df: pd.DataFrame) -> List[pd.Series]:
    cols = list(df.columns)
    date_names = [c for c in cols if str(c).strip().lower() == "date"]
    if not date_names:
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


def normalize_mcap_analysts_long(asof_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two long-form dataframes:
      - mcap_long: ric, date, market_cap_usd
      - an_long:   ric, date, analysts
    Handles duplicate Date columns via _extract_all_date_series.
    """
    if asof_raw is None or asof_raw.empty:
        return (
            pd.DataFrame(columns=["ric", "date", "market_cap_usd"]),
            pd.DataFrame(columns=["ric", "date", "analysts"]),
        )

    cols = list(asof_raw.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]

    col_mcap = find_col(cols, contains_any=["company market cap", "market cap"])
    col_an = find_col(cols, contains_any=["number of analysts", "analysts"])

    date_series = _extract_all_date_series(asof_raw)
    ds0 = date_series[0] if date_series else pd.Series([None] * len(asof_raw))
    ds1 = date_series[1] if len(date_series) >= 2 else ds0

    mcap = pd.DataFrame({
        "ric": asof_raw[inst_col].astype(str),
        "date": ds0.astype(str).str.slice(0, 10),
        "market_cap_usd": asof_raw[col_mcap] if col_mcap else np.nan,
    })
    mcap["date"] = pd.to_datetime(mcap["date"], errors="coerce").dt.date
    mcap["market_cap_usd"] = pd.to_numeric(mcap["market_cap_usd"], errors="coerce")
    mcap = mcap.dropna(subset=["ric", "date"]).sort_values(["ric", "date"]).drop_duplicates(subset=["ric", "date"])

    an = pd.DataFrame({
        "ric": asof_raw[inst_col].astype(str),
        "date": ds1.astype(str).str.slice(0, 10),
        "analysts": asof_raw[col_an] if col_an else np.nan,
    })
    an["date"] = pd.to_datetime(an["date"], errors="coerce").dt.date
    an["analysts"] = pd.to_numeric(an["analysts"], errors="coerce")
    an = an.dropna(subset=["ric", "date"]).sort_values(["ric", "date"]).drop_duplicates(subset=["ric", "date"])

    return mcap, an


def _normalize_pv(pv_df: pd.DataFrame) -> pd.DataFrame:
    if pv_df is None or pv_df.empty:
        return pd.DataFrame(columns=["ric", "date", "price", "volume"])

    cols = list(pv_df.columns)
    inst_col = "Instrument" if "Instrument" in cols else cols[0]
    col_date = find_col(cols, exact=["Date"], contains_any=["date"])
    col_price = find_col(cols, contains_any=["price close", "priceclose", "close"])
    col_vol = find_col(cols, contains_any=["volume"])

    if col_date is None:
        date_series = _extract_all_date_series(pv_df)
        ds0 = date_series[0] if date_series else pd.Series([None] * len(pv_df))
        date_str = ds0.astype(str)
    else:
        x = pv_df.loc[:, col_date]
        date_str = x.iloc[:, 0].astype(str) if isinstance(x, pd.DataFrame) else x.astype(str)

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


# =========================
# Event-level as-of feature construction
# =========================

def _last_value_asof_per_event(
    series_long: pd.DataFrame,
    events: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
    asof_col: str = "asof_date",
) -> pd.DataFrame:
    """
    For each event (ric, asof_date), attach last available value from series_long (ric, date, value_col)
    where series_long.date <= asof_date.

    Returns events with an additional column out_col.
    """
    if events.empty:
        events[out_col] = np.nan
        return events

    if series_long.empty:
        events[out_col] = np.nan
        return events

    series_long = series_long[["ric", "date", value_col]].copy()
    series_long = series_long.dropna(subset=["ric", "date"]).sort_values(["ric", "date"])

    # merge_asof requires sorted keys; do per-ric for correctness
    out_parts: List[pd.DataFrame] = []
    for ric, ev in events.groupby("ric", sort=False):
        ev2 = ev.sort_values(asof_col).copy()
        s = series_long[series_long["ric"] == ric].copy()
        if s.empty:
            ev2[out_col] = np.nan
            out_parts.append(ev2)
            continue

        # convert dates to datetime64 for merge_asof
        ev2["_asof_dt"] = pd.to_datetime(ev2[asof_col].astype(str), errors="coerce")
        s["_dt"] = pd.to_datetime(s["date"].astype(str), errors="coerce")
        s = s.dropna(subset=["_dt"]).sort_values("_dt")
        ev2 = ev2.dropna(subset=["_asof_dt"]).sort_values("_asof_dt")

        merged = pd.merge_asof(
            ev2,
            s[["_dt", value_col]].rename(columns={"_dt": "_asof_merge_key"}),
            left_on="_asof_dt",
            right_on="_asof_merge_key",
            direction="backward",
        )
        merged[out_col] = merged[value_col]
        merged = merged.drop(columns=[value_col, "_asof_dt", "_asof_merge_key"], errors="ignore")
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else events.copy()
    return out


def _event_level_turnover_volatility(
    pv: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lookback_days: int,
    asof_col: str = "asof_date",
) -> pd.DataFrame:
    """
    For each (ric, event), compute:
      - turnover_lookback_sum_volume_asof_evt
      - turnover_lookback_avg_daily_volume_asof_evt
      - volatility_lookback_asof_evt
    using window [asof_date - lookback_days, asof_date] inclusive.
    """
    if events.empty:
        return events

    pv = pv.dropna(subset=["ric", "date"]).copy()
    pv = pv.sort_values(["ric", "date"])

    out_parts: List[pd.DataFrame] = []

    for ric, ev in events.groupby("ric", sort=False):
        ev2 = ev.copy()
        pv2 = pv[pv["ric"] == ric]
        if pv2.empty:
            ev2["turnover_lookback_sum_volume_asof_evt"] = np.nan
            ev2["turnover_lookback_avg_daily_volume_asof_evt"] = np.nan
            ev2["volatility_lookback_asof_evt"] = np.nan
            out_parts.append(ev2)
            continue

        pv2 = pv2.sort_values("date")
        pv2_price = pd.to_numeric(pv2["price"], errors="coerce")
        pv2_vol = pd.to_numeric(pv2["volume"], errors="coerce")

        # Precompute log returns series for volatility
        px = pv2_price.copy()
        px = px.where(px > 0, np.nan)
        log_px = np.log(px)
        lr = log_px.diff()

        # Build a quick lookup by date position
        dates = pv2["date"].tolist()

        def _slice_mask(d0: date, d1: date) -> np.ndarray:
            # boolean mask over pv2 for date range
            # (vectorized compare on pandas series is fine here)
            return (pv2["date"] >= d0) & (pv2["date"] <= d1)

        sums: List[float] = []
        means: List[float] = []
        vols: List[float] = []

        for _, row in ev2.iterrows():
            asof = row.get(asof_col)
            if pd.isna(asof):
                sums.append(np.nan); means.append(np.nan); vols.append(np.nan)
                continue
            asof_d = asof if isinstance(asof, date) else pd.to_datetime(str(asof), errors="coerce").date()
            w0 = asof_d - timedelta(days=int(lookback_days))
            w1 = asof_d

            m = _slice_mask(w0, w1)
            if not bool(m.any()):
                sums.append(np.nan); means.append(np.nan); vols.append(np.nan)
                continue

            vwin = pv2_vol[m]
            sums.append(float(np.nansum(vwin.values)))
            means.append(float(np.nanmean(vwin.values)))

            lrwin = lr[m].replace([np.inf, -np.inf], np.nan).dropna()
            vols.append(float(lrwin.std(ddof=1)) if len(lrwin) >= 2 else np.nan)

        ev2["turnover_lookback_sum_volume_asof_evt"] = sums
        ev2["turnover_lookback_avg_daily_volume_asof_evt"] = means
        ev2["volatility_lookback_asof_evt"] = vols

        ev2["turnover_lookback_window_start_asof_evt"] = (
            pd.to_datetime(ev2[asof_col].astype(str), errors="coerce") - pd.to_timedelta(int(lookback_days), unit="D")
        ).dt.date.astype(str)
        ev2["turnover_lookback_window_end_asof_evt"] = pd.to_datetime(ev2[asof_col].astype(str), errors="coerce").dt.date.astype(str)

        out_parts.append(ev2)

    return pd.concat(out_parts, ignore_index=True) if out_parts else events


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

    p.add_argument("--markets-jsonl", type=str, default=str(DEFAULT_MARKETS_JSONL),
                   help="Path to markets.jsonl used to compute observed window.")

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

    # observed window from markets.jsonl
    try:
        window = compute_observed_window_from_markets(Path(args.markets_jsonl))
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_report(OUT_REPORT_TXT, sections=[("Fatal", f"Failed to compute observed window from markets.jsonl: {exc}")])
        print(f"FATAL: {exc}", file=sys.stderr)
        print(f"See report: {OUT_REPORT_TXT}", file=sys.stderr)
        return 2

    observed_start = window.observed_start
    observed_end = window.observed_end

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
            ("Observed window from markets.jsonl", "\n".join([
                f"Markets JSONL:             {window.markets_path}",
                f"Excluded outlier start:    {window.excluded_outlier_start}",
                f"Observed start / end:      {observed_start} .. {observed_end}",
                f"Markets total / used:      {window.n_markets_total} / {window.n_markets_used}",
            ])),
            ("Fatal", "SCREEN returned zero instruments."),
            ("Screener problems", "\n".join(screen_problems) if screen_problems else "(none)"),
        ])
        return 2

    if args.max_rics is not None:
        rics = rics[: int(args.max_rics)]

    # Fetch events in observed window
    ev_raw, ev_problems = fetch_events_results(client, rics, observed_start, observed_end)
    events = _normalize_events(ev_raw)

    if events.empty:
        write_report(OUT_REPORT_TXT, sections=[
            ("Observed window from markets.jsonl", "\n".join([
                f"Markets JSONL:             {window.markets_path}",
                f"Excluded outlier start:    {window.excluded_outlier_start}",
                f"Observed start / end:      {observed_start} .. {observed_end}",
                f"Markets total / used:      {window.n_markets_total} / {window.n_markets_used}",
            ])),
            ("Fatal", "No RES events returned by Eikon in observed window."),
            ("Top problems / warnings", "\n".join(ev_problems) if ev_problems else "(none)"),
        ])
        return 2

    # As-of date = 2 days before earnings report
    events["asof_date"] = events["event_date"].apply(lambda d: (d - timedelta(days=2)) if isinstance(d, date) else pd.NaT)
    events["retrieved_at_utc"] = utc_now_iso()
    events["observed_window_start_utc"] = observed_start.isoformat()
    events["observed_window_end_utc"] = observed_end.isoformat()

    # Wide enough PV range to compute lookbacks for ALL events
    min_asof = events["asof_date"].min()
    max_asof = events["asof_date"].max()
    if isinstance(min_asof, date) and isinstance(max_asof, date):
        pv_start = min_asof - timedelta(days=lookback_days + buffer_days)
        pv_end = max_asof
    else:
        pv_start = observed_start - timedelta(days=lookback_days + buffer_days)
        pv_end = observed_end

    # Fetch static metadata
    static_raw, static_problems = fetch_static_metadata(client, rics)
    static_norm = _normalize_static(static_raw)

    # Fetch PV and normalize
    pv_raw, pv_problems = fetch_daily_pv(client, rics, pv_start, pv_end)
    pv = _normalize_pv(pv_raw)

    # Fetch marketcap+analysts series over window (with buffer)
    series_start = pv_start - timedelta(days=asof_buffer_days)
    series_end = pv_end
    asof_raw, asof_problems = fetch_marketcap_and_analysts_series(client, rics, series_start, series_end)
    mcap_long, an_long = normalize_mcap_analysts_long(asof_raw)

    # Attach market cap / analysts as-of each event's asof_date
    events = _last_value_asof_per_event(mcap_long, events, value_col="market_cap_usd", out_col="market_cap_usd_asof_evt")
    events = _last_value_asof_per_event(an_long, events, value_col="analysts", out_col="analysts_covering_asof_evt")

    # Attach PV-based features as-of each event
    events = _event_level_turnover_volatility(pv, events, lookback_days=lookback_days)

    # Companies table: keep firm-level static info (no single as-of anymore)
    companies = static_norm.copy()
    companies["retrieved_at_utc"] = utc_now_iso()

    # Join static firm info onto events
    events = events.merge(companies, on="ric", how="left", suffixes=("", "_company"))

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    companies.sort_values(["exchange_mic", "ric"]).to_csv(OUT_COMPANIES_CSV, index=False, encoding="utf-8")
    events.to_csv(OUT_EVENTS_CSV, index=False, encoding="utf-8")
    write_df_jsonl(OUT_EVENTS_JSONL, events)

    ms = missing_summary(events) if not events.empty else {"generated_at_utc": utc_now_iso(), "rows": 0, "columns": {}}
    OUT_MISSING_JSON.write_text(json.dumps(ms, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_lines: List[str] = []
    for c, s in sorted(ms["columns"].items(), key=lambda kv: (-kv[1]["missing"], kv[0]))[:60]:
        pct = s["missing_pct"]
        missing_lines.append(
            f"{c:<45} missing={s['missing']:>8} / {s['total']:<8} ({pct:6.2f}%)"
            if pct is not None else f"{c} missing={s['missing']}"
        )

    sections: List[Tuple[str, str]] = []
    sections.append((
        "Observed window from markets.jsonl",
        "\n".join([
            f"Markets JSONL:             {window.markets_path}",
            f"Excluded outlier start:    {window.excluded_outlier_start}",
            f"Markets total / used:      {window.n_markets_total} / {window.n_markets_used}",
            f"Observed start / end:      {observed_start} .. {observed_end}",
            f"As-of rule:               event_date - 2 days",
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
            *(asof_problems or ["(none from mcap/analysts series fetch)"]),
            *(pv_problems or ["(none from pv fetch)"]),
            *(ev_problems or ["(none from events fetch)"]),
        ])
    ))
    sections.append((
        "Missing values (top 60 columns)",
        "\n".join(missing_lines) if missing_lines else "(no missingness computed)"
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
        f"Observed window:  {observed_start} .. {observed_end}\n"
        "As-of rule:       event_date - 2 days\n"
        "=============================================\n"
    )
    tqdm.write(msg) if tqdm is not None else print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
