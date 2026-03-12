#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
03_fetch_corp_info.py

Fetch *per-market* corporate characteristics from Refinitiv Eikon / Workspace
for the Polymarket Corporate Earnings Study.

Key change (per-market as-of logic)
-----------------------------------
- We treat **each Polymarket market as a separate observation** (no company-level aggregation).
- We use **umaEndDate as the market close time**.
- For every market, we fetch / compute corporate characteristics **as of (umaEndDate - 2 days)**.
  Example: umaEndDate = 2025-12-10 -> as-of date = 2025-12-08.

Inputs (relative to project root)
---------------------------------
data/markets/markets.jsonl

Outputs (relative to project root)
----------------------------------
data/corporate_info/corporate_info_by_market.jsonl
data/corporate_info/corporate_info_by_market.csv
data/corporate_info/corporate_info_by_market_summary.txt

Run:
  python 03_fetch_corp_info.py --app-key <KEY>

Test mode:
  python 03_fetch_corp_info.py --app-key <KEY> --max-markets 10

Importable:
  from 03_fetch_corp_info import main
  main(["--app-key","...","--max-markets","10"])

Notes on performance
--------------------
- Uses batching for Eikon calls (chunk size configurable).
- Uses tqdm progress bars.
- Pulls time-series for the full required window once (global min/max), then does per-market
  snapshotting & window calculations locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass, asdict
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
# Project-relative defaults
# =========================

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


DEFAULT_MARKETS_JSONL = project_root() / "data" / "markets" / "markets.jsonl"

DEFAULT_OUT_JSONL = project_root() / "data" / "corporate_info" / "corporate_info_by_market.jsonl"
DEFAULT_OUT_CSV = project_root() / "data" / "corporate_info" / "corporate_info_by_market.csv"
DEFAULT_SUMMARY_TXT = project_root() / "data" / "corporate_info" / "corporate_info_by_market_summary.txt"

DEFAULT_LOOKBACK_DAYS = 183  # ~6 months
DEFAULT_ASOF_LAG_DAYS = 2    # as-of = umaEndDate - 2 days


# =========================
# Eikon retry / proxy config
# =========================

EIKON_RETRIES = 5
EIKON_RETRY_BASE_SLEEP = 0.7
DEFAULT_EIKON_PORT_CANDIDATES = [9000, 9060]
EIKON_STATUS_PATHS = ["/api/status", "/api/handshake"]


# =========================
# Quiet logging
# =========================

LOG = logging.getLogger("corp_info")


class NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "HTTP Request:" in msg:
            return False
        if ("Error code 500" in msg and "Network Error" in msg) or ('"message":"Network Error"' in msg):
            return False
        return True


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


def _suppress_noisy_third_party_loggers() -> None:
    for name in [
        "urllib3",
        "requests",
        "websockets",
        "httpx",
        "httpcore",
        "eikon",
        "refinitiv",
        "refinitiv.data",
        "refinitiv.data.eikon",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def setup_logging_quiet() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.ERROR)

    handler = TqdmLoggingHandler()
    handler.addFilter(NoiseFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)

    _suppress_noisy_third_party_loggers()


def setup_warnings_suppression() -> None:
    """Suppress noisy eikon/pandas warnings."""
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"eikon\.data_grid")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"eikon(\..*)?")


# =========================
# Exceptions
# =========================

class FatalEikonNetworkError(RuntimeError):
    pass


def _looks_like_eikon_network_error(exc: Exception) -> bool:
    s = str(exc)
    return ("Error code 500" in s and "Network Error" in s) or ('"message":"Network Error"' in s)


# =========================
# Basic helpers
# =========================

def require_eikon() -> None:
    if ek is None:
        raise RuntimeError("eikon package not available. Install via: pip install eikon")


def require_tqdm() -> None:
    if tqdm is None:
        raise RuntimeError("tqdm package not available. Install via: pip install tqdm")


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def parse_iso_date(s: Any) -> Optional[date]:
    """Parse YYYY-MM-DD from an ISO-like string (timestamps OK)."""
    if s is None:
        return None
    try:
        return date.fromisoformat(str(s).strip()[0:10])
    except Exception:
        return None


def _is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]

def find_col_by_substrings(columns: List[Any], substrings: List[str]) -> Optional[Any]:
    """
    Find a column whose name contains any of the provided substrings.
    Safely ignores None / blank / non-string column names.
    """
    cleaned: List[Tuple[Any, str]] = []
    for c in columns:
        if c is None:
            continue
        c_text = str(c).strip()
        if not c_text:
            continue
        cleaned.append((c, c_text.lower()))

    for sub in substrings:
        s = str(sub).lower()
        for original_col, lower_col in cleaned:
            if s in lower_col:
                return original_col
    return None


def get_first_present_column(
    columns: List[Any],
    preferred_exact: List[str],
    fallback_substrings: List[str],
) -> Optional[Any]:
    """
    Prefer exact header matches first, then fallback to substring-based matching.
    Safely ignores None / blank / non-string column names.
    """
    cleaned: List[Tuple[Any, str]] = []
    for c in columns:
        if c is None:
            continue
        c_text = str(c).strip()
        if not c_text:
            continue
        cleaned.append((c, c_text))

    exact_map = {text: original for original, text in cleaned}

    for name in preferred_exact:
        if name in exact_map:
            return exact_map[name]

    if fallback_substrings:
        return find_col_by_substrings([original for original, _ in cleaned], fallback_substrings)

    return None

def fetch_earnings_release_datetime(
    ric: str,
    around_date: date,
    *,
    fail_fast: bool,
) -> Optional[str]:
    """
    Fetch the earnings release date+time nearest to around_date.
    Returns:
        YYYY-MM-DDTHH:MM:SS  if time is available
        YYYY-MM-DD           if only date is available
        None                 if nothing is found
    """
    start = around_date - timedelta(days=14)
    end = around_date + timedelta(days=1)

    fields = [
        "TR.EventStartDate",
        "TR.EventStartTime",
        "TR.EventType",
        "TR.EventTitle",
    ]
    params = {
        "SDate": start.isoformat(),
        "EDate": end.isoformat(),
        "EventType": "RES",
    }

    df, err = eikon_retry_get_data(
        [ric],
        fields,
        params,
        retries=EIKON_RETRIES,
        fail_fast=fail_fast,
    )
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    cols = [c for c in df.columns if c is not None and str(c).strip() != ""]
    if not cols:
        return None
    dcol = get_first_present_column(
        cols,
        preferred_exact=[],
        fallback_substrings=["event start date", "start date"],
    )
    tcol = get_first_present_column(
        cols,
        preferred_exact=[],
        fallback_substrings=["event start time", "start time"],
    )

    if dcol is None:
        return None

    tmp = df.copy()
    tmp["event_date"] = pd.to_datetime(tmp[dcol], errors="coerce").dt.date
    tmp = tmp.dropna(subset=["event_date"])
    if tmp.empty:
        return None

    tmp["distance_days"] = tmp["event_date"].apply(lambda d: abs((d - around_date).days))
    row = tmp.sort_values(["distance_days"]).iloc[0]

    event_date = row["event_date"].isoformat()

    if tcol is not None and not _is_missing_value(row.get(tcol)):
        event_time = str(row[tcol]).strip()
        if event_time:
            return f"{event_date}T{event_time}"

    return event_date


# =========================
# Proxy helpers
# =========================

def _http_get_text(url: str, timeout_s: float = 1.5) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-eikon-proxy-check"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read(4096)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _read_port_inuse_file() -> Optional[int]:
    appdata = os.getenv("APPDATA")
    if not appdata:
        return None

    candidates = [
        Path(appdata) / "Thomson Reuters" / "Eikon API Proxy" / ".portInUse",
        Path(appdata) / "Refinitiv" / "Eikon API Proxy" / ".portInUse",
        Path(appdata) / "Thomson Reuters" / "Refinitiv Workspace" / "Eikon API Proxy" / ".portInUse",
    ]
    for p in candidates:
        try:
            if p.exists():
                raw = p.read_text(encoding="utf-8", errors="ignore").strip()
                port = int(raw)
                if 1 <= port <= 65535:
                    return port
        except Exception:
            continue
    return None


def detect_eikon_proxy_port(extra_ports: Optional[List[int]] = None) -> Optional[int]:
    ports: List[int] = []
    file_port = _read_port_inuse_file()
    if file_port:
        ports.append(file_port)
    ports.extend(DEFAULT_EIKON_PORT_CANDIDATES)
    if extra_ports:
        ports.extend([p for p in extra_ports if isinstance(p, int) and 1 <= p <= 65535])

    seen = set()
    uniq: List[int] = []
    for p in ports:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    for port in uniq:
        for path in EIKON_STATUS_PATHS:
            url = f"http://127.0.0.1:{port}{path}"
            if _http_get_text(url) is not None:
                return port
    return None


def init_eikon(app_key: str, eikon_port: Optional[int], require_proxy: bool) -> None:
    require_eikon()
    ek.set_app_key(app_key)

    # Try to silence SDK logging
    try:
        set_level = getattr(ek, "set_log_level", None)
        if callable(set_level):
            set_level(0)
    except Exception:
        pass

    if eikon_port is not None:
        setter = getattr(ek, "set_port_number", None)
        if callable(setter):
            try:
                setter(int(eikon_port))
            except Exception:
                pass

    if require_proxy and eikon_port is None:
        detected = detect_eikon_proxy_port()
        if detected is None:
            raise RuntimeError(
                "Could not detect a running local Eikon/Workspace Data API Proxy on localhost.\n"
                "Start Workspace/Eikon, log in, ensure the Data API proxy is running, then retry.\n"
                "If your proxy uses a non-standard port, pass --eikon-port <PORT>."
            )
        setter = getattr(ek, "set_port_number", None)
        if callable(setter):
            try:
                setter(int(detected))
            except Exception:
                pass


# =========================
# Eikon calls with retry
# =========================

def eikon_retry_get_data(
    instruments: List[str],
    fields: List[Any],
    parameters: Dict[str, Any],
    *,
    retries: int,
    fail_fast: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    last_exc: Optional[Exception] = None
    network_error_seen = False

    for attempt in range(retries):
        try:
            df, err = ek.get_data(instruments, fields, parameters=parameters)
            if isinstance(df, pd.DataFrame):
                return df, err
            return None, err
        except Exception as exc:
            last_exc = exc
            if _looks_like_eikon_network_error(exc):
                network_error_seen = True
            time.sleep(EIKON_RETRY_BASE_SLEEP * (2 ** attempt))

    if network_error_seen and fail_fast:
        raise FatalEikonNetworkError(
            "Eikon/Workspace proxy repeatedly returned 500 'Network Error'. "
            "Most often: logged out/offline, VPN/proxy/firewall blocks, or backend outage."
        )

    return None, last_exc


# =========================
# Data models
# =========================

@dataclass
class MarketObs:
    market_id: str
    slug: Optional[str]
    ticker: Optional[str]
    ric: str

    uma_end_date_raw: str       # original umaEndDate string
    close_date: str             # YYYY-MM-DD
    asof_date: str              # YYYY-MM-DD (close_date - 2 days)


@dataclass
class CorporateInfoByMarket:
    # Market identifiers
    market_id: str
    slug: Optional[str]
    ticker: Optional[str]
    ric: str

    uma_end_date: str
    close_date: str
    asof_date: str

    earnings_release_datetime: Optional[str]

    # Corporate characteristics (as-of asof_date where applicable)
    company_name: Optional[str]

    market_cap_usd_asof: Optional[float]
    analysts_covering_asof: Optional[float]

    gics_sector: Optional[str]
    gics_industry: Optional[str]
    trbc_industry: Optional[str]

    hq_country: Optional[str]
    hq_country_code: Optional[str]
    country_of_risk: Optional[str]
    exchange_country: Optional[str]
    country_source: Optional[str]

    primary_exchange: Optional[str]

    # Market/stock-derived features (window ends at asof_date)
    turnover_6m_window_start: Optional[str]
    turnover_6m_window_end: Optional[str]
    turnover_6m_sum_volume: Optional[float]
    turnover_6m_avg_daily_volume: Optional[float]
    volatility_6m: Optional[float]

    retrieved_at_utc: str
    notes: List[str]


# =========================
# IO
# =========================

def iter_jsonl(path: Path, max_lines: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: Path, records: List[CorporateInfoByMarket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[CorporateInfoByMarket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])

    # Store notes list as compact JSON string for CSV safety
    if "notes" in df.columns:
        df["notes"] = df["notes"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)

    df.to_csv(path, index=False, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =========================
# Market loading (from markets.jsonl)
# =========================

def load_markets_marketsjsonl(
    markets_jsonl: Path,
    *,
    max_markets: Optional[int],
    asof_lag_days: int,
) -> Tuple[List[MarketObs], Dict[str, int]]:
    """
    Load Polymarket markets from data/markets/markets.jsonl.

    We require:
      - a market id (val_market_id or id)
      - a RIC (val_ric or ric)
      - umaEndDate (used as market close time)
    """
    skipped: Dict[str, int] = {
        "missing_market_id": 0,
        "missing_ric": 0,
        "missing_umaEndDate": 0,
        "bad_umaEndDate": 0,
    }

    out: List[MarketObs] = []
    for obj in iter_jsonl(markets_jsonl, max_markets):
        market_id = _safe_str(obj.get("val_market_id")) or _safe_str(obj.get("market_id")) or _safe_str(obj.get("id"))
        if not market_id:
            skipped["missing_market_id"] += 1
            continue

        ric = _safe_str(obj.get("val_ric")) or _safe_str(obj.get("ric"))
        if not ric:
            skipped["missing_ric"] += 1
            continue

        uma_end = _safe_str(obj.get("umaEndDate"))
        if not uma_end:
            skipped["missing_umaEndDate"] += 1
            continue

        close_dt = parse_iso_date(uma_end)
        if not close_dt:
            skipped["bad_umaEndDate"] += 1
            continue

        asof_dt = close_dt - timedelta(days=int(asof_lag_days))

        out.append(
            MarketObs(
                market_id=market_id,
                slug=_safe_str(obj.get("val_slug")) or _safe_str(obj.get("slug")),
                ticker=_safe_str(obj.get("val_ticker")) or _safe_str(obj.get("ticker")),
                ric=ric,
                uma_end_date_raw=uma_end,
                close_date=close_dt.isoformat(),
                asof_date=asof_dt.isoformat(),
            )
        )

    return out, skipped


# =========================
# Eikon field helpers
# =========================

def _tr_field(name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Use ek.TR_Field when available (recommended for parameterized TR fields).
    Falls back to string form if TR_Field isn't available.
    """
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


def fetch_static_metadata(rics: List[str], *, fail_fast: bool) -> pd.DataFrame:
    """
    STATIC request (no SDate/EDate) => one row per instrument.

    We keep the HQ/country fields here because they are stable and historically
    the "display header" mapping is easiest to handle with a single static snapshot.
    """
    fields: List[Any] = [
        # name + classification
        "TR.CommonName",
        "TR.GICSSector",
        "TR.GICSIndustry",
        "TR.TRBCIndustry",
        # primary exchange
        "TR.PrimaryExchangeName",
        "TR.ExchangeName",
        # country fields (often returned as display headers)
        "TR.HeadquartersCountry",
        "TR.HQCountryCode",
        "TR.CoRPrimaryCountry",
        "TR.ExchangeCountry",
    ]
    df, err = eikon_retry_get_data(rics, fields, parameters={}, retries=EIKON_RETRIES, fail_fast=fail_fast)
    if df is None:
        raise RuntimeError(f"Failed to fetch static metadata. Last error: {err}")
    return df


def fetch_timeseries_batch(
    rics: List[str],
    fields: List[Any],
    start: date,
    end: date,
    *,
    fail_fast: bool,
) -> pd.DataFrame:
    """
    Generic daily time series fetch for a batch of instruments.
    Returns the raw Eikon DataFrame (instrument + date + value columns).
    """
    params = {"SDate": start.isoformat(), "EDate": end.isoformat(), "Frq": "D"}
    df, _err = eikon_retry_get_data(rics, fields, params, retries=EIKON_RETRIES, fail_fast=fail_fast)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _instrument_col(df: pd.DataFrame) -> str:
    return "Instrument" if "Instrument" in df.columns else df.columns[0]


def _date_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return get_first_present_column(cols, preferred_exact=["Date"], fallback_substrings=[".date", " date", "date"])


def compute_window_features(pv: pd.DataFrame, window_start: date, window_end: date) -> Dict[str, Optional[float]]:
    """
    pv must have columns: date, price, volume
    """
    if pv.empty:
        return {"sum_volume": None, "avg_daily_volume": None, "volatility": None}

    w = pv[(pv["date"] >= window_start) & (pv["date"] <= window_end)].copy()
    if w.empty:
        return {"sum_volume": None, "avg_daily_volume": None, "volatility": None}

    vol_s = pd.to_numeric(w["volume"], errors="coerce")
    sum_volume = float(vol_s.sum(skipna=True)) if vol_s.notna().any() else None
    avg_daily_volume = float(vol_s.mean(skipna=True)) if vol_s.notna().any() else None

    px = pd.to_numeric(w["price"], errors="coerce")
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]

    volatility = None
    if len(px) >= 3:
        logret = np.log(px).diff().dropna()
        if len(logret) >= 2:
            volatility = float(logret.std(ddof=1))

    return {"sum_volume": sum_volume, "avg_daily_volume": avg_daily_volume, "volatility": volatility}


def snapshot_asof(df: pd.DataFrame, asof: date, value_col: str) -> Optional[float]:
    """
    df must have columns: date, <value_col>
    Returns last observation on or before asof.
    """
    if df.empty:
        return None
    sub = df[df["date"] <= asof]
    if sub.empty:
        return None
    try:
        v = sub.iloc[-1][value_col]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


# =========================
# Missing summary (TXT only)
# =========================

def build_missing_summary_txt(
    records: List[CorporateInfoByMarket],
    *,
    skipped_counts: Dict[str, int],
    markets_input_path: Path,
) -> str:
    """
    Human-readable missingness summary for the final per-market dataset.
    """
    n = len(records)
    header = [
        "==================== CORPORATE INFO (PER-MARKET) SUMMARY ====================",
        f"Generated at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Input markets.jsonl: {markets_input_path}",
        f"Output rows (markets): {n}",
        "",
        "Skipped input rows:",
    ]
    for k, v in skipped_counts.items():
        header.append(f"  - {k}: {v}")
    header.append("=============================================================================")
    header.append("")

    if n == 0:
        return "\n".join(header + ["No output records generated (n=0)."])

    # Compute missing counts per field
    field_names = list(asdict(records[0]).keys())
    stats: List[Tuple[str, int, float]] = []
    for f in field_names:
        miss = 0
        for r in records:
            v = getattr(r, f)
            if f == "notes":
                # notes is allowed to be empty list; not "missing"
                continue
            if _is_missing_value(v):
                miss += 1
        pct = (miss / n * 100.0) if n else 0.0
        stats.append((f, miss, pct))

    stats.sort(key=lambda x: (-x[1], x[0]))

    lines = header + ["Missingness by variable (sorted by missing count):", "-" * 72]
    for i, (f, miss, pct) in enumerate(stats, start=1):
        lines.append(f"{i:>3}. {f:<35} missing={miss:>6} / {n:<6} ({pct:6.2f}%)")
    lines.append("")
    return "\n".join(lines)


# =========================
# Args / Main
# =========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch per-market corporate info (as-of umaEndDate-2d) from Eikon.")
    p.add_argument("--markets-jsonl", type=str, default=str(DEFAULT_MARKETS_JSONL))
    p.add_argument("--out-jsonl", type=str, default=str(DEFAULT_OUT_JSONL))
    p.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    p.add_argument("--summary-txt", type=str, default=str(DEFAULT_SUMMARY_TXT))

    p.add_argument("--max-markets", type=int, default=None, help="TEST MODE: only first X lines of markets.jsonl")

    p.add_argument(
        "--app-key",
        nargs="?",
        const="__ENV__",
        default=None,
        help="Eikon app key. If provided with no value, reads env EIKON_APP_KEY.",
    )
    p.add_argument("--eikon-port", type=int, default=None)
    p.add_argument("--skip-proxy-check", action="store_true")
    p.add_argument("--no-fail-fast", action="store_true")

    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--asof-lag-days", type=int, default=DEFAULT_ASOF_LAG_DAYS)
    p.add_argument("--batch-size", type=int, default=50, help="Eikon instrument batch size (typical 25-100).")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    require_tqdm()
    setup_logging_quiet()
    setup_warnings_suppression()

    args = parse_args(argv)
    fail_fast = (not args.no_fail_fast)

    markets_jsonl = Path(args.markets_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    summary_txt = Path(args.summary_txt)

    if not markets_jsonl.exists():
        LOG.error("Input file not found: %s", markets_jsonl)
        return 2

    # Resolve app key
    if args.app_key is None:
        LOG.error("Missing --app-key. Provide it or use '--app-key' (no value) to read env EIKON_APP_KEY.")
        return 2
    if args.app_key == "__ENV__":
        app_key = os.getenv("EIKON_APP_KEY") or os.getenv("APP_KEY") or ""
        if not app_key:
            LOG.error("EIKON_APP_KEY (or APP_KEY) not found in environment.")
            return 2
    else:
        app_key = args.app_key

    # Init Eikon
    try:
        init_eikon(app_key, eikon_port=args.eikon_port, require_proxy=(not args.skip_proxy_check))
    except Exception as exc:
        LOG.error("Eikon initialization failed: %s", exc)
        return 2

    # Load markets (1 obs per market)
    markets, skipped_counts = load_markets_marketsjsonl(
        markets_jsonl,
        max_markets=args.max_markets,
        asof_lag_days=int(args.asof_lag_days),
    )
    if not markets:
        LOG.error("No usable markets loaded from %s", markets_jsonl)
        write_text(summary_txt, build_missing_summary_txt([], skipped_counts=skipped_counts, markets_input_path=markets_jsonl))
        return 2

    # Unique RICs + global date window (for time series)
    rics = sorted({m.ric for m in markets})
    asof_dates = [parse_iso_date(m.asof_date) for m in markets]
    asof_dates = [d for d in asof_dates if d is not None]
    if not asof_dates:
        LOG.error("No parsable asof_date values found.")
        return 2

    global_end = max(asof_dates)
    global_start = min(asof_dates) - timedelta(days=int(args.lookback_days) + 5)

    batch_size = int(args.batch_size)
    lookback_days = int(args.lookback_days)

    # -------------------------
    # 1) Static metadata (batched)
    # -------------------------
    static_parts: List[pd.DataFrame] = []
    with tqdm(total=len(rics), desc="Eikon static metadata", unit="ric") as pbar:
        for batch in chunked(rics, batch_size):
            static_parts.append(fetch_static_metadata(batch, fail_fast=fail_fast))
            pbar.update(len(batch))
    static_df = pd.concat(static_parts, ignore_index=True) if static_parts else pd.DataFrame()
    if static_df.empty:
        LOG.error("Static metadata fetch returned empty DataFrame.")
        return 2

    inst_col = _instrument_col(static_df)
    static_df["_RIC_"] = static_df[inst_col].astype(str)

    cols = list(static_df.columns)

    COL_COMPANY_NAME = get_first_present_column(
        cols,
        preferred_exact=["Company Common Name", "Company Name", "Common Name"],
        fallback_substrings=["company common name", "common name"],
    )
    COL_GICS_SECTOR = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["gics sector"])
    COL_GICS_INDUSTRY = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["gics industry"])
    COL_TRBC_INDUSTRY = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["trbc industry"])
    COL_PRIMARY_EXCH = get_first_present_column(
        cols,
        preferred_exact=["Primary Exchange Name"],
        fallback_substrings=["primary exchange", "exchange name"],
    )

    # Country columns: prefer exact display names (robust to Eikon headers)
    COL_HQ_COUNTRY = get_first_present_column(
        cols,
        preferred_exact=["Country of Headquarters"],
        fallback_substrings=["country of headquarters", "headquarters country"],
    )
    COL_HQ_CODE = get_first_present_column(
        cols,
        preferred_exact=["Country ISO Code of Headquarters"],
        fallback_substrings=["iso code of headquarters", "hqcountrycode", "country code of headquarters"],
    )
    COL_RISK = get_first_present_column(
        cols,
        preferred_exact=["Primary Country of Risk"],
        fallback_substrings=["primary country of risk", "country of primary risk", "cor primary"],
    )
    COL_EXCH_COUNTRY = get_first_present_column(
        cols,
        preferred_exact=["Country of Exchange"],
        fallback_substrings=["country of exchange", "exchange country"],
    )

    def static_row(ric: str) -> pd.Series:
        sub = static_df[static_df["_RIC_"] == ric]
        return sub.iloc[0] if not sub.empty else pd.Series(dtype=object)

    def sget(srow: pd.Series, col: Optional[str]) -> Any:
        if col is None or srow is None or getattr(srow, "empty", False):
            return None
        return srow.get(col)

    # -------------------------
    # 2) Time series (batched, global window)
    # -------------------------
    # Price + Volume (for turnover & volatility)
    pv_raw_parts: List[pd.DataFrame] = []
    pv_fields: List[Any] = ["TR.PriceClose", "TR.Volume", "TR.PriceClose.date"]
    with tqdm(total=len(rics), desc="Eikon price+volume (daily)", unit="ric") as pbar:
        for batch in chunked(rics, batch_size):
            pv_raw_parts.append(
                fetch_timeseries_batch(batch, pv_fields, global_start, global_end, fail_fast=fail_fast)
            )
            pbar.update(len(batch))
    pv_raw = pd.concat(pv_raw_parts, ignore_index=True) if pv_raw_parts else pd.DataFrame()

    # Analysts (daily)
    an_raw_parts: List[pd.DataFrame] = []
    an_fields: List[Any] = ["TR.NumberOfAnalysts", "TR.NumberOfAnalysts.date"]
    with tqdm(total=len(rics), desc="Eikon analysts (daily)", unit="ric") as pbar:
        for batch in chunked(rics, batch_size):
            an_raw_parts.append(
                fetch_timeseries_batch(batch, an_fields, global_start, global_end, fail_fast=fail_fast)
            )
            pbar.update(len(batch))
    an_raw = pd.concat(an_raw_parts, ignore_index=True) if an_raw_parts else pd.DataFrame()

    # Market cap (daily, USD) — for as-of market cap per market
    mc_raw_parts: List[pd.DataFrame] = []
    mc_fields: List[Any] = [
        _tr_field("TR.CompanyMarketCap", {"Curn": "USD"}),
        "TR.CompanyMarketCap.date",
    ]
    with tqdm(total=len(rics), desc="Eikon market cap (daily, USD)", unit="ric") as pbar:
        for batch in chunked(rics, batch_size):
            mc_raw_parts.append(
                fetch_timeseries_batch(batch, mc_fields, global_start, global_end, fail_fast=fail_fast)
            )
            pbar.update(len(batch))
    mc_raw = pd.concat(mc_raw_parts, ignore_index=True) if mc_raw_parts else pd.DataFrame()

    # -------------------------
    # 3) Normalize time series to dict[ric] -> DataFrame(date, value)
    # -------------------------
    pv_by_ric: Dict[str, pd.DataFrame] = {}
    if not pv_raw.empty:
        inst = _instrument_col(pv_raw)
        dcol = _date_col(pv_raw)
        cols_pv = list(pv_raw.columns)
        pcol = get_first_present_column(cols_pv, preferred_exact=[], fallback_substrings=["price close", "priceclose", "close"])
        vcol = get_first_present_column(cols_pv, preferred_exact=[], fallback_substrings=["volume"])

        if dcol and pcol and vcol:
            tmp = pv_raw[[inst, dcol, pcol, vcol]].copy()
            tmp["date"] = tmp[dcol].apply(parse_iso_date)
            tmp["price"] = pd.to_numeric(tmp[pcol], errors="coerce")
            tmp["volume"] = pd.to_numeric(tmp[vcol], errors="coerce")
            tmp = tmp.dropna(subset=["date"])
            for ric, g in tmp.groupby(inst, sort=False):
                df = g[["date", "price", "volume"]].drop_duplicates(subset=["date"]).sort_values("date")
                pv_by_ric[str(ric)] = df.reset_index(drop=True)

    an_by_ric: Dict[str, pd.DataFrame] = {}
    if not an_raw.empty:
        inst = _instrument_col(an_raw)
        dcol = _date_col(an_raw)
        cols_an = list(an_raw.columns)
        acol = get_first_present_column(cols_an, preferred_exact=[], fallback_substrings=["number of analysts", "analysts"])
        if dcol and acol:
            tmp = an_raw[[inst, dcol, acol]].copy()
            tmp["date"] = tmp[dcol].apply(parse_iso_date)
            tmp["analysts"] = pd.to_numeric(tmp[acol], errors="coerce")
            tmp = tmp.dropna(subset=["date"])
            for ric, g in tmp.groupby(inst, sort=False):
                df = g[["date", "analysts"]].drop_duplicates(subset=["date"]).sort_values("date")
                an_by_ric[str(ric)] = df.reset_index(drop=True)

    mc_by_ric: Dict[str, pd.DataFrame] = {}
    if not mc_raw.empty:
        inst = _instrument_col(mc_raw)
        dcol = _date_col(mc_raw)
        cols_mc = list(mc_raw.columns)
        mcol = get_first_present_column(cols_mc, preferred_exact=[], fallback_substrings=["company market cap", "market cap"])
        if dcol and mcol:
            tmp = mc_raw[[inst, dcol, mcol]].copy()
            tmp["date"] = tmp[dcol].apply(parse_iso_date)
            tmp["market_cap_usd"] = pd.to_numeric(tmp[mcol], errors="coerce")
            tmp = tmp.dropna(subset=["date"])
            for ric, g in tmp.groupby(inst, sort=False):
                df = g[["date", "market_cap_usd"]].drop_duplicates(subset=["date"]).sort_values("date")
                mc_by_ric[str(ric)] = df.reset_index(drop=True)

    # -------------------------
    # 4) Build per-market output rows
    # -------------------------
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: List[CorporateInfoByMarket] = []

    with tqdm(total=len(markets), desc="Assembling per-market rows", unit="market") as pbar:
        for m in markets:
            pbar.update(1)
            notes: List[str] = []

            srow = static_row(m.ric)

            company_name = _safe_str(sget(srow, COL_COMPANY_NAME))
            gics_sector = _safe_str(sget(srow, COL_GICS_SECTOR))
            gics_industry = _safe_str(sget(srow, COL_GICS_INDUSTRY))
            trbc_industry = _safe_str(sget(srow, COL_TRBC_INDUSTRY))
            primary_exchange = _safe_str(sget(srow, COL_PRIMARY_EXCH))

            # Country mapping (display-header aware)
            hq_country_name = _safe_str(sget(srow, COL_HQ_COUNTRY))
            hq_country_code = _safe_str(sget(srow, COL_HQ_CODE))
            country_of_risk = _safe_str(sget(srow, COL_RISK))
            exchange_country = _safe_str(sget(srow, COL_EXCH_COUNTRY))

            if hq_country_name:
                hq_country = hq_country_name
                country_source = "Country of Headquarters"
            elif hq_country_code:
                hq_country = hq_country_code
                country_source = "Country ISO Code of Headquarters"
            else:
                hq_country = None
                country_source = None
                notes.append("hq_country_missing")

            asof_dt = parse_iso_date(m.asof_date)
            if not asof_dt:
                # should be rare (already validated when loading)
                notes.append("bad_asof_date")
                asof_dt = date.today()

            close_dt_for_event = parse_iso_date(m.close_date)
            earnings_release_datetime = None
            if close_dt_for_event is not None:
                earnings_release_datetime = fetch_earnings_release_datetime(
                    m.ric,
                    close_dt_for_event,
                    fail_fast=fail_fast,
                )
            if earnings_release_datetime is None:
                notes.append("earnings_release_time_missing")

            # As-of snapshots (market cap + analysts)
            market_cap_asof = snapshot_asof(mc_by_ric.get(m.ric, pd.DataFrame()), asof_dt, "market_cap_usd")
            if market_cap_asof is None:
                notes.append("market_cap_missing_asof")

            analysts_asof = snapshot_asof(an_by_ric.get(m.ric, pd.DataFrame()), asof_dt, "analysts")
            if analysts_asof is None:
                notes.append("analysts_missing_asof")

            # Window features from price+volume (ends at asof_dt)
            w0 = asof_dt - timedelta(days=lookback_days)
            w1 = asof_dt
            pv = pv_by_ric.get(m.ric, pd.DataFrame())
            feats = compute_window_features(pv, w0, w1)
            if feats["sum_volume"] is None:
                notes.append("turnover_missing_window")
            if feats["volatility"] is None:
                notes.append("volatility_missing_window")

            results.append(
                CorporateInfoByMarket(
                    market_id=m.market_id,
                    slug=m.slug,
                    ticker=m.ticker,
                    ric=m.ric,
                    uma_end_date=m.uma_end_date_raw,
                    close_date=m.close_date,
                    asof_date=m.asof_date,
                    earnings_release_datetime=earnings_release_datetime,
                    company_name=company_name,
                    market_cap_usd_asof=market_cap_asof,
                    analysts_covering_asof=analysts_asof,
                    gics_sector=gics_sector,
                    gics_industry=gics_industry,
                    trbc_industry=trbc_industry,
                    hq_country=hq_country,
                    hq_country_code=hq_country_code,
                    country_of_risk=country_of_risk,
                    exchange_country=exchange_country,
                    country_source=country_source,
                    primary_exchange=primary_exchange,
                    turnover_6m_window_start=w0.isoformat(),
                    turnover_6m_window_end=w1.isoformat(),
                    turnover_6m_sum_volume=feats["sum_volume"],
                    turnover_6m_avg_daily_volume=feats["avg_daily_volume"],
                    volatility_6m=feats["volatility"],
                    retrieved_at_utc=now_utc,
                    notes=notes,
                )
            )

    # -------------------------
    # 5) Write outputs (ONLY per-market files)
    # -------------------------
    write_jsonl(out_jsonl, results)
    write_csv(out_csv, results)

    summary_text = build_missing_summary_txt(
        results,
        skipped_counts=skipped_counts,
        markets_input_path=markets_jsonl,
    )
    write_text(summary_txt, summary_text)

    msg = (
        "\n==================== DONE ====================\n"
        f"Input:        {markets_jsonl}\n"
        f"Output JSONL: {out_jsonl}\n"
        f"Output CSV:   {out_csv}\n"
        f"Summary TXT:  {summary_txt}\n"
        f"Markets out:  {len(results)}\n"
        f"RICs used:    {len(rics)}\n"
        f"As-of lag:    {args.asof_lag_days} day(s) (asof = umaEndDate - lag)\n"
        f"Lookback:     {lookback_days} day(s)\n"
        "=============================================\n"
    )
    tqdm.write(msg) if tqdm is not None else print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
