#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
03_fetch_corp_info.py

Fetch corporate characteristics from Refinitiv Eikon / Workspace for thesis analysis.

IMPORTANT FIX (HQ country)
--------------------------
Eikon returns *display column names* (e.g. "Country of Headquarters") rather than TR.* codes.
Your test confirms these exact headers exist. This script now maps fields using those headers
(with additional robust substring fallbacks).

INPUT (relative paths)
----------------------
data/validation/correct.jsonl
data/validation/incorrectly_resolved.jsonl

OUTPUTS (relative paths)
------------------------
data/corporate_info/corporate_info.jsonl
data/corporate_info/missing_summary.json
data/corporate_info/missing_summary.txt

Run:
  python 03_fetch_corp_info.py --app-key <KEY>

Test mode:
  python 03_fetch_corp_info.py --app-key <KEY> --max-markets 10

Importable:
  from 03_fetch_corp_info import main
  main(["--app-key","...","--max-markets","10"])
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
from dataclasses import dataclass, asdict, fields as dataclass_fields
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
    return Path(__file__).resolve().parent


DEFAULT_CORRECT_JSONL = project_root() / "data" / "validation" / "correct.jsonl"
DEFAULT_INCORRECT_JSONL = project_root() / "data" / "validation" / "incorrectly_resolved.jsonl"
DEFAULT_OUT_JSONL = project_root() / "data" / "corporate_info" / "corporate_info.jsonl"
DEFAULT_SUMMARY_JSON = project_root() / "data" / "corporate_info" / "missing_summary.json"
DEFAULT_SUMMARY_TXT = project_root() / "data" / "corporate_info" / "missing_summary.txt"

DEFAULT_LOOKBACK_DAYS = 183  # ~6 months


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
    """
    Suppress noisy eikon/pandas FutureWarnings (like the one you saw in test.py).
    """
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


def median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return float(ys[mid]) if n % 2 == 1 else float(0.5 * (ys[mid - 1] + ys[mid]))


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def find_col_by_substrings(columns: List[str], substrings: List[str]) -> Optional[str]:
    """
    Find a column whose lower-cased name contains ANY of the provided substrings.
    """
    low_cols = [c.lower() for c in columns]
    for sub in substrings:
        s = sub.lower()
        for i, c in enumerate(low_cols):
            if s in c:
                return columns[i]
    return None


def get_first_present_column(columns: List[str], preferred_exact: List[str], fallback_substrings: List[str]) -> Optional[str]:
    """
    Prefer exact header matches first (since your output shows exact display names),
    then fallback to substring-based matching.
    """
    colset = set(columns)
    for name in preferred_exact:
        if name in colset:
            return name
    if fallback_substrings:
        return find_col_by_substrings(columns, fallback_substrings)
    return None


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
class MarketRef:
    market_id: Optional[str]
    slug: Optional[str]
    ticker: Optional[str]
    ric: str
    anchor_date: Optional[str]
    bucket: str  # "correct" or "incorrect"
    status: Optional[str]
    polymarket_resolved_outcome: Optional[str]
    expected_resolution: Optional[str]
    label: Optional[str]


@dataclass
class CorporateInfoRecord:
    ric: str
    ticker: Optional[str]
    company_name: Optional[str]

    market_cap_usd: Optional[float]
    gics_sector: Optional[str]
    gics_industry: Optional[str]
    trbc_industry: Optional[str]

    # Country fields (now correctly mapped to returned display headers)
    hq_country: Optional[str]
    hq_country_code: Optional[str]
    country_of_risk: Optional[str]
    exchange_country: Optional[str]
    country_source: Optional[str]

    primary_exchange: Optional[str]

    analysts_covering_latest: Optional[float]
    analysts_covering_sample_mean: Optional[float]
    analysts_covering_sample_median: Optional[float]

    turnover_6m_sum_volume_mean: Optional[float]
    turnover_6m_sum_volume_median: Optional[float]
    volatility_6m_mean: Optional[float]
    volatility_6m_median: Optional[float]

    sample_markets_n: int
    market_ids: List[str]
    slugs: List[str]
    markets: List[Dict[str, Any]]

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

def load_grouped_by_ric(
    correct_jsonl: Path,
    incorrect_jsonl: Path,
    max_markets: Optional[int],
) -> Dict[str, List[MarketRef]]:
    grouped: Dict[str, List[MarketRef]] = {}
    seen: set[tuple[str, str]] = set()  # (ric, market_id) dedupe
    total = 0

    def ingest(path: Path, bucket: str) -> None:
        nonlocal total
        if not path.exists():
            return

        for obj in iter_jsonl(path, None):
            if max_markets is not None and total >= max_markets:
                return

            ric = _safe_str(obj.get("ric"))
            if not ric:
                continue

            market_id = _safe_str(obj.get("market_id"))
            if market_id:
                key = (ric, market_id)
                if key in seen:
                    continue
                seen.add(key)

            m = MarketRef(
                market_id=market_id,
                slug=_safe_str(obj.get("slug")),
                ticker=_safe_str(obj.get("ticker")),
                ric=ric,
                anchor_date=_safe_str(obj.get("anchor_date")),
                bucket=bucket,
                status=_safe_str(obj.get("status")),
                polymarket_resolved_outcome=_safe_str(obj.get("polymarket_resolved_outcome")),
                expected_resolution=_safe_str(obj.get("expected_resolution")),
                label=_safe_str(obj.get("label")),
            )
            grouped.setdefault(ric, []).append(m)
            total += 1

    ingest(correct_jsonl, "correct")
    ingest(incorrect_jsonl, "incorrect")
    return grouped


def write_jsonl(path: Path, records: List[CorporateInfoRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =========================
# Missing summary
# =========================

def build_missing_summary(records: List[CorporateInfoRecord]) -> Dict[str, Any]:
    n_companies = len(records)
    per_market_rows: List[Dict[str, Any]] = []
    for r in records:
        per_market_rows.extend(r.markets or [])
    n_markets = len(per_market_rows)

    exclude_top = {"market_ids", "slugs", "markets", "notes", "retrieved_at_utc"}
    top_field_names = [f.name for f in dataclass_fields(CorporateInfoRecord) if f.name not in exclude_top]

    top_stats: Dict[str, Dict[str, Any]] = {}
    for name in top_field_names:
        total = n_companies
        missing = 0
        for r in records:
            if _is_missing_value(getattr(r, name)):
                missing += 1
        top_stats[name] = {
            "missing": missing,
            "total": total,
            "missing_pct": (missing / total * 100.0) if total else None,
        }

    key_union: List[str] = sorted({k for row in per_market_rows for k in row.keys()})
    per_market_stats: Dict[str, Dict[str, Any]] = {}
    for k in key_union:
        total = n_markets
        missing = 0
        for row in per_market_rows:
            if k not in row or _is_missing_value(row.get(k)):
                missing += 1
        per_market_stats[k] = {
            "missing": missing,
            "total": total,
            "missing_pct": (missing / total * 100.0) if total else None,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "companies": n_companies,
        "per_market_rows": n_markets,
        "top_level": top_stats,
        "per_market": per_market_stats,
    }


def format_missing_summary_txt(summary: Dict[str, Any]) -> str:
    def block(title: str, stats: Dict[str, Dict[str, Any]], top_n: int = 120) -> str:
        items = sorted(stats.items(), key=lambda kv: (-kv[1]["missing"], kv[0]))
        lines = [title, "-" * len(title)]
        for i, (k, v) in enumerate(items[:top_n], start=1):
            miss = v["missing"]
            tot = v["total"]
            pct = v["missing_pct"]
            pct_s = f"{pct:6.2f}%" if pct is not None else "  n/a "
            lines.append(f"{i:>3}. {k:<42} missing={miss:>6} / {tot:<6} ({pct_s})")
        if len(items) > top_n:
            lines.append(f"... ({len(items) - top_n} more omitted)")
        return "\n".join(lines)

    header = (
        "==================== MISSING SUMMARY ====================\n"
        f"Generated at (UTC): {summary.get('generated_at_utc')}\n"
        f"Companies:          {summary.get('companies')}\n"
        f"Per-market rows:    {summary.get('per_market_rows')}\n"
        "=========================================================\n"
    )
    top_block = block("Top-level variables (company)", summary.get("top_level", {}))
    pm_block = block("Per-market variables (markets[])", summary.get("per_market", {}))
    return header + "\n\n" + top_block + "\n\n" + pm_block + "\n"


# =========================
# Eikon data fetchers
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
    This is required for stable retrieval of HQ country fields.
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

        # market cap
        _tr_field("TR.CompanyMarketCap", {"Curn": "USD"}),

        # analysts
        "TR.NumberOfAnalysts",

        # country fields (your test proves these return as display headers)
        "TR.HeadquartersCountry",
        "TR.HQCountryCode",
        "TR.CoRPrimaryCountry",
        "TR.ExchangeCountry",
    ]
    df, err = eikon_retry_get_data(rics, fields, parameters={}, retries=EIKON_RETRIES, fail_fast=fail_fast)
    if df is None:
        raise RuntimeError(f"Failed to fetch static metadata. Last error: {err}")
    return df


def fetch_daily_price_volume(ric: str, start: date, end: date, *, fail_fast: bool) -> pd.DataFrame:
    fields = ["TR.PriceClose", "TR.Volume", "TR.PriceClose.date"]
    params = {"SDate": start.isoformat(), "EDate": end.isoformat(), "Frq": "D"}
    df, _err = eikon_retry_get_data([ric], fields, params, retries=EIKON_RETRIES, fail_fast=fail_fast)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "price", "volume"])

    cols = list(df.columns)
    col_date = get_first_present_column(cols, preferred_exact=["Date"], fallback_substrings=["date"])
    col_price = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["price close", "priceclose", "close"])
    col_vol = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["volume"])

    rows: List[Tuple[date, float, float]] = []
    for _, r in df.iterrows():
        d = parse_iso_date(r[col_date]) if col_date else None
        if not d:
            continue
        px = pd.to_numeric(pd.Series([r[col_price]]), errors="coerce").iloc[0] if col_price else np.nan
        vol = pd.to_numeric(pd.Series([r[col_vol]]), errors="coerce").iloc[0] if col_vol else np.nan
        rows.append((d, float(px) if not pd.isna(px) else np.nan, float(vol) if not pd.isna(vol) else np.nan))

    out = pd.DataFrame(rows, columns=["date", "price", "volume"]).drop_duplicates(subset=["date"])
    out.sort_values("date", inplace=True)
    return out


def fetch_daily_analyst_coverage(ric: str, start: date, end: date, *, fail_fast: bool) -> pd.DataFrame:
    fields = ["TR.NumberOfAnalysts", "TR.NumberOfAnalysts.date"]
    params = {"SDate": start.isoformat(), "EDate": end.isoformat(), "Frq": "D"}
    df, _err = eikon_retry_get_data([ric], fields, params, retries=EIKON_RETRIES, fail_fast=fail_fast)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "analysts"])

    cols = list(df.columns)
    col_date = get_first_present_column(cols, preferred_exact=["Date"], fallback_substrings=["date"])
    col_val = get_first_present_column(cols, preferred_exact=[], fallback_substrings=["number of analysts", "analysts"])

    rows: List[Tuple[date, float]] = []
    for _, r in df.iterrows():
        d = parse_iso_date(r[col_date]) if col_date else None
        if not d:
            continue
        v = pd.to_numeric(pd.Series([r[col_val]]), errors="coerce").iloc[0] if col_val else np.nan
        if pd.isna(v):
            continue
        rows.append((d, float(v)))

    out = pd.DataFrame(rows, columns=["date", "analysts"]).drop_duplicates(subset=["date"])
    out.sort_values("date", inplace=True)
    return out


def snapshot_asof(df: pd.DataFrame, asof: date, value_col: str) -> Optional[float]:
    if df.empty:
        return None
    sub = df[df["date"] <= asof]
    if sub.empty:
        return None
    try:
        return float(sub.iloc[-1][value_col])
    except Exception:
        return None


def compute_window_features(pv: pd.DataFrame, window_start: date, window_end: date) -> Dict[str, Optional[float]]:
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


# =========================
# Args / Main
# =========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch corporate info from Eikon for Polymarket earnings thesis.")
    p.add_argument("--correct-jsonl", type=str, default=str(DEFAULT_CORRECT_JSONL))
    p.add_argument("--out-jsonl", type=str, default=str(DEFAULT_OUT_JSONL))
    p.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    p.add_argument("--summary-txt", type=str, default=str(DEFAULT_SUMMARY_TXT))
    p.add_argument("--max-markets", type=int, default=None, help="TEST MODE: only first X lines of correct.jsonl")
    p.add_argument("--incorrect-jsonl", type=str, default=str(DEFAULT_INCORRECT_JSONL))

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
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    require_tqdm()
    setup_logging_quiet()
    setup_warnings_suppression()

    args = parse_args(argv)
    fail_fast = (not args.no_fail_fast)

    correct_jsonl = Path(args.correct_jsonl)
    incorrect_jsonl = Path(getattr(args, "incorrect_jsonl", "")) if hasattr(args, "incorrect_jsonl") else None

    out_jsonl = Path(args.out_jsonl)
    summary_json = Path(args.summary_json)
    summary_txt = Path(args.summary_txt)

    # Require at least one input file to exist
    have_correct = correct_jsonl.exists()
    have_incorrect = bool(incorrect_jsonl and incorrect_jsonl.exists())
    if not have_correct and not have_incorrect:
        LOG.error(
            "No inputs found. Expected at least one of:\n  %s\n  %s",
            str(correct_jsonl),
            str(incorrect_jsonl) if incorrect_jsonl else "(missing --incorrect-jsonl arg)",
        )
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

    # Load markets from BOTH correct + incorrect
    grouped = load_grouped_by_ric(
        correct_jsonl=correct_jsonl,
        incorrect_jsonl=incorrect_jsonl if incorrect_jsonl is not None else Path(""),
        max_markets=args.max_markets,
    )

    rics = sorted(grouped.keys())
    if not rics:
        LOG.error("No RICs found across inputs.")
        return 2

    # ---- STATIC METADATA (HQ COUNTRY FIX) ----
    static_parts: List[pd.DataFrame] = []
    for batch in chunked(rics, 50):
        static_parts.append(fetch_static_metadata(batch, fail_fast=fail_fast))
    static_df = pd.concat(static_parts, ignore_index=True) if static_parts else pd.DataFrame()
    if static_df.empty:
        LOG.error("Static metadata fetch returned empty DataFrame.")
        return 2

    # Instrument column
    inst_col = "Instrument" if "Instrument" in static_df.columns else static_df.columns[0]
    static_df["_RIC_"] = static_df[inst_col].astype(str)

    # Map static columns using YOUR confirmed headers first.
    cols = list(static_df.columns)

    COL_COMPANY_NAME = get_first_present_column(
        cols,
        preferred_exact=["Company Common Name", "Company Name", "Common Name"],
        fallback_substrings=["company common name", "common name"],
    )
    COL_MARKET_CAP = get_first_present_column(
        cols,
        preferred_exact=[],
        fallback_substrings=["company market cap", "market cap"],
    )
    COL_GICS_SECTOR = get_first_present_column(
        cols, preferred_exact=[], fallback_substrings=["gics sector"]
    )
    COL_GICS_INDUSTRY = get_first_present_column(
        cols, preferred_exact=[], fallback_substrings=["gics industry"]
    )
    COL_TRBC_INDUSTRY = get_first_present_column(
        cols, preferred_exact=[], fallback_substrings=["trbc industry"]
    )
    COL_PRIMARY_EXCH = get_first_present_column(
        cols,
        preferred_exact=["Primary Exchange Name"],
        fallback_substrings=["primary exchange", "exchange name"],
    )
    COL_ANALYSTS = get_first_present_column(
        cols, preferred_exact=[], fallback_substrings=["number of analysts"]
    )

    # Country columns: use exact names from your test output first.
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

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: List[CorporateInfoRecord] = []

    with tqdm(total=len(rics), desc="Fetching corporate info", unit="company") as pbar:
        for ric in rics:
            pbar.update(1)
            markets = grouped[ric]
            notes: List[str] = []

            srow = static_row(ric)

            company_name = _safe_str(sget(srow, COL_COMPANY_NAME))

            market_cap_usd = None
            try:
                v = sget(srow, COL_MARKET_CAP)
                market_cap_usd = float(v) if v is not None and str(v).strip() and str(v).lower() != "nan" else None
            except Exception:
                market_cap_usd = None

            gics_sector = _safe_str(sget(srow, COL_GICS_SECTOR))
            gics_industry = _safe_str(sget(srow, COL_GICS_INDUSTRY))
            trbc_industry = _safe_str(sget(srow, COL_TRBC_INDUSTRY))

            primary_exchange = _safe_str(sget(srow, COL_PRIMARY_EXCH))

            analysts_covering_latest = None
            try:
                v = sget(srow, COL_ANALYSTS)
                analysts_covering_latest = float(v) if v is not None and str(v).strip() and str(v).lower() != "nan" else None
            except Exception:
                analysts_covering_latest = None

            # ---- Country: now mapped to actual returned headers ----
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

            # Determine lookback window from anchors
            anchor_dates = [parse_iso_date(m.anchor_date) for m in markets]
            anchor_dates = [d for d in anchor_dates if d is not None]
            if not anchor_dates:
                notes.append("no_anchor_dates_in_group")
                anchor_min = date.today()
                anchor_max = date.today()
            else:
                anchor_min = min(anchor_dates)
                anchor_max = max(anchor_dates)

            ts_start = anchor_min - timedelta(days=args.lookback_days + 5)
            ts_end = anchor_max

            pv = fetch_daily_price_volume(ric, ts_start, ts_end, fail_fast=fail_fast)
            an = fetch_daily_analyst_coverage(ric, ts_start, ts_end, fail_fast=fail_fast)

            per_market: List[Dict[str, Any]] = []
            sum_vols: List[float] = []
            vols: List[float] = []
            analysts_asof_vals: List[float] = []

            for m in markets:
                ad = parse_iso_date(m.anchor_date)

                # include match bucket/info if present on MarketRef
                row: Dict[str, Any] = {
                    "market_id": m.market_id,
                    "slug": m.slug,
                    "ticker": m.ticker,
                    "anchor_date": m.anchor_date,
                    "bucket": getattr(m, "bucket", None),
                    "status": getattr(m, "status", None),
                    "polymarket_resolved_outcome": getattr(m, "polymarket_resolved_outcome", None),
                    "expected_resolution": getattr(m, "expected_resolution", None),
                    "label": getattr(m, "label", None),
                }

                if ad:
                    w0 = ad - timedelta(days=args.lookback_days)
                    w1 = ad
                    feats = compute_window_features(pv, w0, w1)
                    row.update({
                        "turnover_6m_window_start": w0.isoformat(),
                        "turnover_6m_window_end": w1.isoformat(),
                        "turnover_6m_sum_volume": feats["sum_volume"],
                        "turnover_6m_avg_daily_volume": feats["avg_daily_volume"],
                        "volatility_6m": feats["volatility"],
                    })
                    if feats["sum_volume"] is not None:
                        sum_vols.append(float(feats["sum_volume"]))
                    if feats["volatility"] is not None:
                        vols.append(float(feats["volatility"]))

                    a_asof = snapshot_asof(an, ad, "analysts")
                    row["analysts_covering_asof_anchor"] = a_asof
                    if a_asof is not None:
                        analysts_asof_vals.append(float(a_asof))
                else:
                    row["turnover_6m_sum_volume"] = None
                    row["turnover_6m_avg_daily_volume"] = None
                    row["volatility_6m"] = None
                    row["analysts_covering_asof_anchor"] = None

                per_market.append(row)

            analysts_mean = float(np.mean(analysts_asof_vals)) if analysts_asof_vals else None
            analysts_med = median(analysts_asof_vals) if analysts_asof_vals else None

            sum_vol_mean = float(np.mean(sum_vols)) if sum_vols else None
            sum_vol_med = median(sum_vols) if sum_vols else None
            vol_mean = float(np.mean(vols)) if vols else None
            vol_med = median(vols) if vols else None

            tickers = [m.ticker for m in markets if m.ticker]
            ticker = tickers[0] if tickers else None

            market_ids = [m.market_id for m in markets if m.market_id]
            slugs = [m.slug for m in markets if m.slug]

            results.append(
                CorporateInfoRecord(
                    ric=ric,
                    ticker=ticker,
                    company_name=company_name,
                    market_cap_usd=market_cap_usd,
                    gics_sector=gics_sector,
                    gics_industry=gics_industry,
                    trbc_industry=trbc_industry,
                    hq_country=hq_country,
                    hq_country_code=hq_country_code,
                    country_of_risk=country_of_risk,
                    exchange_country=exchange_country,
                    country_source=country_source,
                    primary_exchange=primary_exchange,
                    analysts_covering_latest=analysts_covering_latest,
                    analysts_covering_sample_mean=analysts_mean,
                    analysts_covering_sample_median=analysts_med,
                    turnover_6m_sum_volume_mean=sum_vol_mean,
                    turnover_6m_sum_volume_median=sum_vol_med,
                    volatility_6m_mean=vol_mean,
                    volatility_6m_median=vol_med,
                    sample_markets_n=len(markets),
                    market_ids=market_ids,
                    slugs=slugs,
                    markets=per_market,
                    retrieved_at_utc=now_utc,
                    notes=notes,
                )
            )

    # Write outputs
    write_jsonl(out_jsonl, results)

    summary = build_missing_summary(results)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text(summary_txt, format_missing_summary_txt(summary))

    msg = (
        "\n==================== DONE ====================\n"
        f"Inputs:\n"
        f"  - correct:   {correct_jsonl} ({'OK' if have_correct else 'MISSING'})\n"
        f"  - incorrect: {incorrect_jsonl} ({'OK' if have_incorrect else 'MISSING'})\n"
        f"Output:       {out_jsonl}\n"
        f"Summary JSON: {summary_json}\n"
        f"Summary TXT:  {summary_txt}\n"
        f"RICs:         {len(rics)}\n"
        "=============================================\n"
    )
    tqdm.write(msg) if tqdm is not None else print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
