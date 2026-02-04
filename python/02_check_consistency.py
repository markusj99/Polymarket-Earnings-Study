#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
01_optional_check_consistency.py

Validate Polymarket "beat earnings / beat EPS estimate" market resolutions against Refinitiv Eikon
(Earnings Surprise / Estimates data) AND WRITE RESULTS BACK INTO THE ORIGINAL MARKETS FILES.

WHAT THIS SCRIPT DOES (UPDATED)
-------------------------------
This script reads your existing:
  - Corporate_Earnings/data/markets/markets.jsonl
  - Corporate_Earnings/data/markets/markets.csv

It then:
  1) Identifies “earnings/EPS” candidate markets.
  2) Resolves tickers -> RICs (robust batching + bisect salvage).
  3) Fetches EPS actual/estimates history from Eikon (robust batching + bisect salvage).
  4) Matches the closest relevant earnings event around an “anchor date” inferred from slug/question/json fields.
  5) Computes whether the market SHOULD have resolved YES/NO (based on actual vs estimate).
  6) **Updates the original markets.jsonl and markets.csv in place** by adding a set of prefixed fields
     (val_*) that contain the same information that previously lived in separate output files.

IN-PLACE UPDATE OUTPUT
----------------------
- markets.jsonl is overwritten (atomic replace) with the same market objects plus new fields:
    val_status, val_skip_reason, val_ric, val_anchor_date, val_polymarket_estimate, val_label,
    val_expected_resolution, val_matched_announce_date, val_eikon_actual_eps, etc. (see ResultRecord).

- markets.csv is overwritten (atomic replace) to mirror markets.jsonl, preserving existing column order
  and appending any new val_* columns at the end.

ADDITIONAL OUTPUTS (still written)
----------------------------------
1) Unmatched markets for manual investigation:
   - Corporate_Earnings/data/validation/unmatched.jsonl
   - Corporate_Earnings/data/validation/unmatched.csv

2) A human-readable summary .txt that includes how many markets were resolved incorrectly:
   - Corporate_Earnings/data/validation/consistency_summary.txt

NOTES
-----
- Requires Eikon Desktop / Refinitiv Workspace running + logged in, with Data API proxy available.
- This script does NOT create new “correct/incorrect” datasets anymore; those results are now stored in-place
  inside markets.jsonl / markets.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import eikon as ek  # type: ignore
except Exception:
    ek = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


# =========================
# Defaults / Config
# =========================

# Project root is one level above /python
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

DEFAULT_MARKETS_JSONL_PATH = PROJECT_ROOT / "data" / "markets" / "markets.jsonl"
DEFAULT_MARKETS_CSV_PATH   = PROJECT_ROOT / "data" / "markets" / "markets.csv"

# Prefer NYSE (.N) before NASDAQ (.O) in suffix guessing.
RIC_SUFFIX_GUESSES = [".N", ".O", ".A", ".L", ".K"]

DEFAULT_EVENT_PRE_DAYS = 10
DEFAULT_EVENT_POST_DAYS = 30
DEFAULT_MAX_EVENT_DISTANCE_DAYS = 60

EPS_TIE_TOL = 1e-6

EIKON_RETRIES = 5
EIKON_RETRY_BASE_SLEEP = 0.7

DEFAULT_EIKON_PORT_CANDIDATES = [9000, 9060]
EIKON_STATUS_PATHS = ["/api/status", "/api/handshake"]

# Batching knobs (tunable via CLI)
DEFAULT_SYMBOLOGY_CHUNK_SIZE = 250     # symbols per ek.get_symbology call
DEFAULT_VALIDATE_CHUNK_SIZE = 200      # instruments per ek.get_data validation call
DEFAULT_EPS_CHUNK_SIZE = 25            # RICs per EPS history call

# New: prefix for fields written back into markets.* files
VAL_PREFIX = "val_"


# =========================
# Logging (quiet)
# =========================

LOG = logging.getLogger("eikon_eps_validator")


class NoiseFilter(logging.Filter):
    """Drop known noisy messages so the tqdm bar stays clean."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "HTTP Request:" in msg:
            return False
        if ("Error code 500" in msg and "Network Error" in msg) or ('"message":"Network Error"' in msg):
            return False
        return True


class TqdmLoggingHandler(logging.Handler):
    """Log via tqdm.write() so logs don't corrupt the progress bar."""

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


def setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.ERROR)

    handler = TqdmLoggingHandler()
    handler.addFilter(NoiseFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)

    _suppress_noisy_third_party_loggers()


def setup_warnings_suppression() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"eikon\.data_grid",
    )


def is_missing_value(x: Any) -> bool:
    if x is None:
        return True
    # pandas NA / NaN
    try:
        import pandas as _pd  # type: ignore
        if _pd.isna(x):
            return True
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return True
    return s.lower() in {"nan", "<na>", "none", "null"}


# =========================
# Exceptions / Overrides
# =========================

class FatalEikonNetworkError(RuntimeError):
    """Raised when Eikon repeatedly returns 500 'Network Error' and fail-fast is enabled."""


# Eikon RICs can be case-sensitive. Berkshire Hathaway Class A is BRKa in Eikon.
TICKER_TO_RIC_OVERRIDES: Dict[str, str] = {
    "BRK.A": "BRKa",
    # Optional (if you ever see it): "BRK.B": "BRKb",
}


# =========================
# Data models
# =========================

@dataclass
class EarningsEvent:
    announce_date: date
    fperiod: Optional[str]
    period_end_date: Optional[date]
    actual_eps: Optional[float]
    mean_estimate: Optional[float]
    high_estimate: Optional[float]
    low_estimate: Optional[float]
    stddev_estimate: Optional[float]


@dataclass
class ResultRecord:
    line_no: int
    market_id: Optional[str]
    slug: Optional[str]
    question: Optional[str]
    ticker: Optional[str]
    ric: Optional[str]
    polymarket_resolved_outcome: Optional[str]

    anchor_date: Optional[str]
    polymarket_estimate: Optional[float]
    polymarket_estimate_source: Optional[str]

    yes_semantics: Optional[str]
    inline_counts_as: Optional[str]

    matched_announce_date: Optional[str]
    matched_fperiod: Optional[str]
    matched_period_end_date: Optional[str]

    eikon_actual_eps: Optional[float]
    eikon_eps_mean_estimate: Optional[float]
    eikon_eps_high_estimate: Optional[float]
    eikon_eps_low_estimate: Optional[float]
    eikon_eps_stddev_estimate: Optional[float]

    estimate_used: Optional[float]
    estimate_used_source: Optional[str]

    surprise: Optional[float]
    label: Optional[str]
    expected_resolution: Optional[str]
    match_method: Optional[str]

    status: str  # MATCHED_CORRECT | MATCHED_INCORRECT | UNMATCHED
    skip_reason: Optional[str]


@dataclass
class PendingMarket:
    base: ResultRecord
    anchor_dt: date


# =========================
# Parsing helpers
# =========================

def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _to_date_iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def parse_any_datetime_to_date(value: Any) -> Optional[date]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date()
    except Exception:
        pass

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None

    return None


def parse_anchor_date(
    slug: Optional[str],
    question: Optional[str],
    raw_market: Dict[str, Any],
) -> Tuple[Optional[date], Optional[str]]:
    if slug:
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", slug)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "slug_ymd"
            except Exception:
                pass

        m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", slug)
        if m2:
            try:
                return date(int(m2.group(3)), int(m2.group(1)), int(m2.group(2))), "slug_mdy"
            except Exception:
                pass

    if question:
        m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", question)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "question_ymd"
            except Exception:
                pass

    candidate_keys = [
        "endDate",
        "end_date",
        "closeTime",
        "close_time",
        "resolutionTime",
        "resolution_time",
        "resolvedTime",
        "resolved_time",
        "expiresAt",
        "expires_at",
        "expiration",
        "expirationTime",
        "expiration_time",
    ]
    for k in candidate_keys:
        if k in raw_market:
            d = parse_any_datetime_to_date(raw_market.get(k))
            if d:
                return d, f"json_field:{k}"

    for k, v in raw_market.items():
        if isinstance(v, str) and ("T" in v or v.endswith("Z")):
            d = parse_any_datetime_to_date(v)
            if d:
                return d, f"json_scan:{k}"

    return None, None


def extract_estimate_from_question(question: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not question:
        return None, None

    pattern = (
        r"\(\s*(?P<sign>-)?\s*\$?\s*(?P<num>\d+(?:\.\d+)?)\s*EPS\s*\)"
        r"|(?P<sign2>-)?\s*\$?\s*(?P<num2>\d+(?:\.\d+)?)\s*EPS\b"
    )
    m = re.search(pattern, question, flags=re.IGNORECASE)
    if not m:
        return None, None

    sign = m.group("sign") or m.group("sign2")
    num = m.group("num") or m.group("num2")
    try:
        val = float(num)
        if sign:
            val = -val
        return val, "question_number"
    except Exception:
        return None, None


def extract_estimate_from_slug(slug: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not slug:
        return None, None

    s = slug.lower()

    pt_tokens = re.findall(r"(?:neg)?\d+pt\d+", s)
    if pt_tokens:
        tok = pt_tokens[-1]
        neg = tok.startswith("neg")
        tok2 = tok[3:] if neg else tok
        m = re.match(r"(\d+)pt(\d+)", tok2)
        if m:
            whole = int(m.group(1))
            dec = m.group(2)
            val = whole + int(dec) / (10 ** len(dec))
            if neg:
                val = -val
            return float(val), "slug_pt_number"

    parts = s.split("-")
    if "eps" in parts and parts:
        tail = parts[-1]
        if re.fullmatch(r"\d{1,3}", tail):
            has_date = bool(
                re.search(r"\d{4}-\d{1,2}-\d{1,2}", s) or re.search(r"\d{1,2}-\d{1,2}-\d{4}", s)
            )
            if has_date:
                try:
                    v = float(int(tail))
                    if 0 <= v <= 100:
                        return v, "slug_trailing_int"
                except Exception:
                    pass

    return None, None


def infer_yes_semantics(question: Optional[str], slug: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join([question or "", slug or ""]).lower()
    tie_counts_as = "NO"
    if "miss" in text or "below" in text or "under" in text:
        return "YES_MEANS_MISS", tie_counts_as
    return "YES_MEANS_BEAT", tie_counts_as


def is_candidate_earnings_market(question: Optional[str], slug: Optional[str]) -> bool:
    t = " ".join([question or "", slug or ""]).lower()
    if not t.strip():
        return False
    if "revenue" in t or "sales" in t:
        return "eps" in t
    return ("earnings" in t) or ("eps" in t) or ("forecast" in t) or ("estimate" in t)


def normalize_outcome(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"YES", "NO"}:
        return s
    if s == "TRUE":
        return "YES"
    if s == "FALSE":
        return "NO"
    return None


def extract_resolved_outcome(raw_market: Dict[str, Any]) -> Optional[str]:
    for k in [
        "resolvedOutcome",
        "resolved_outcome",
        "resolution",
        "outcome",
        "resolved",
        "finalOutcome",
        "final_outcome",
    ]:
        if k in raw_market:
            out = normalize_outcome(raw_market.get(k))
            if out:
                return out
    if isinstance(raw_market.get("result"), dict):
        out = normalize_outcome(raw_market["result"].get("outcome"))
        if out:
            return out
    return None


def extract_market_id(raw_market: Dict[str, Any]) -> Optional[str]:
    for k in ["id", "market_id", "conditionId", "condition_id", "slugId"]:
        v = _safe_str(raw_market.get(k))
        if v:
            return v
    return None


def extract_ticker(raw_market: Dict[str, Any], question: Optional[str]) -> Optional[str]:
    t = _safe_str(raw_market.get("ticker"))
    if t:
        return t.upper()
    if question:
        m = re.search(r"\(([A-Z0-9.\-]{1,12})\)", question)
        if m:
            return m.group(1).upper()
    return None


# =========================
# Eikon proxy helpers
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
    uniq_ports: List[int] = []
    for p in ports:
        if p not in seen:
            uniq_ports.append(p)
            seen.add(p)

    for port in uniq_ports:
        for path in EIKON_STATUS_PATHS:
            url = f"http://127.0.0.1:{port}{path}"
            if _http_get_text(url) is not None:
                return port
    return None


# =========================
# Eikon SDK helpers
# =========================

def require_eikon() -> None:
    if ek is None:
        raise RuntimeError("eikon package not available. Install via: pip install eikon")


def require_tqdm() -> None:
    if tqdm is None:
        raise RuntimeError("tqdm package not available. Install via: pip install tqdm")


def require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas not available. Install via: pip install pandas")


def init_eikon(app_key: str, eikon_port: Optional[int], require_proxy: bool) -> None:
    require_eikon()
    ek.set_app_key(app_key)

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


def _looks_like_eikon_network_error(exc: Exception) -> bool:
    s = str(exc)
    return ("Error code 500" in s and "Network Error" in s) or ('"message":"Network Error"' in s)


def eikon_retry_get_data(
    instruments: List[str],
    fields: List[str],
    parameters: Dict[str, Any],
    *,
    retries: int,
    fail_fast: bool,
) -> Tuple[Optional[Any], Optional[Any]]:
    last_exc: Optional[Exception] = None
    network_error_seen = False

    for attempt in range(retries):
        try:
            df, err = ek.get_data(instruments, fields, parameters=parameters)  # type: ignore
            return df, err
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


def _merge_err(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k not in out:
                out[k] = v
            else:
                out[k] = [out[k], v]
        return out
    return [a, b]


def safe_get_data(
    instruments: List[str],
    fields: List[str],
    parameters: Dict[str, Any],
    *,
    retries: int,
    fail_fast: bool,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Robust ek.get_data:
    - Try the batch.
    - If the batch fails (df is None) and len(instruments)>1, bisect and salvage.
    """
    if not instruments:
        return None, None

    df, err = eikon_retry_get_data(instruments, fields, parameters, retries=retries, fail_fast=fail_fast)
    if df is not None:
        return df, err

    if len(instruments) <= 1:
        return None, err

    mid = len(instruments) // 2
    df1, err1 = safe_get_data(instruments[:mid], fields, parameters, retries=retries, fail_fast=fail_fast)
    df2, err2 = safe_get_data(instruments[mid:], fields, parameters, retries=retries, fail_fast=fail_fast)

    if df1 is None and df2 is None:
        return None, _merge_err(err1, err2)

    if df1 is None:
        return df2, _merge_err(err1, err2)
    if df2 is None:
        return df1, _merge_err(err1, err2)

    require_pandas()
    try:
        return pd.concat([df1, df2], ignore_index=True, sort=False), _merge_err(err1, err2)  # type: ignore
    except Exception:
        try:
            return df1._append(df2, ignore_index=True), _merge_err(err1, err2)
        except Exception:
            return df1, _merge_err(err1, err2)


def find_col(df, needles: List[str]) -> Optional[str]:
    if df is None:
        return None
    cols = list(df.columns)
    lower_cols = [str(c).lower() for c in cols]
    for n in needles:
        n2 = n.lower()
        for i, c in enumerate(lower_cols):
            if n2 in c:
                return cols[i]
    return None


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    n = max(1, int(n))
    for i in range(0, len(xs), n):
        yield xs[i: i + n]


# =========================
# Robust symbology (batched + bisect on failure)
# =========================

def _safe_get_symbology_df(symbols: List[str], *, best_match: bool) -> Optional[Any]:
    """
    Robust ek.get_symbology:
    - Try the batch.
    - If it raises, bisect and salvage.
    """
    if not symbols:
        return None

    try:
        return ek.get_symbology(  # type: ignore
            symbols,
            from_symbol_type="ticker",
            to_symbol_type="RIC",
            best_match=best_match,
        )
    except Exception:
        if len(symbols) <= 1:
            return None
        mid = len(symbols) // 2
        df1 = _safe_get_symbology_df(symbols[:mid], best_match=best_match)
        df2 = _safe_get_symbology_df(symbols[mid:], best_match=best_match)
        if df1 is None:
            return df2
        if df2 is None:
            return df1
        require_pandas()
        try:
            return pd.concat([df1, df2], ignore_index=True, sort=False)  # type: ignore
        except Exception:
            try:
                return df1._append(df2, ignore_index=True)
            except Exception:
                return df1


def batch_get_symbology_best_match(
    symbols: List[str],
    *,
    chunk_size: int,
) -> Dict[str, Optional[str]]:
    """
    Batched ek.get_symbology(..., best_match=True).

    Returns: input_symbol -> ric (or None)

    Critical fixes:
    - Treat pandas NA / NaN / "<NA>" as missing (do NOT stringify into "<NA>").
    - Case-insensitive output parsing.
    - If input column is missing, fall back to row-order mapping ONLY when lengths match.
    - Ensure every input symbol is present in the output dict.
    """
    out: Dict[str, Optional[str]] = {}
    if not symbols:
        return out

    def _is_missing_value(x: Any) -> bool:
        return is_missing_value(x)

    # De-dupe while preserving order (normalize to UPPER because caller treats symbols case-insensitively)
    seen: set[str] = set()
    uniq: List[str] = []
    for s in symbols:
        s2 = (s or "").strip().upper()
        if not s2:
            continue
        if s2 not in seen:
            uniq.append(s2)
            seen.add(s2)

    for chunk in chunked(uniq, chunk_size):
        df = _safe_get_symbology_df(chunk, best_match=True)

        if df is None or getattr(df, "empty", True):
            for s in chunk:
                out.setdefault(s, None)
            continue

        cols = list(df.columns)
        cols_lc = {str(c).strip().lower(): c for c in cols}

        # Find RIC column (case-insensitive)
        ric_col = None
        for key in ["ric", "to ric", "to_ric", "to symbol", "tosymbol", "to"]:
            if key in cols_lc:
                ric_col = cols_lc[key]
                break
        if ric_col is None:
            for c in cols:
                if str(c).strip().lower() == "ric":
                    ric_col = c
                    break
        if ric_col is None:
            for s in chunk:
                out.setdefault(s, None)
            continue

        # Find input column (case-insensitive)
        in_col = None
        for key in ["ticker", "symbol", "from", "input", "fromsymbol", "from symbol", "instrument", "original", "source"]:
            if key in cols_lc:
                in_col = cols_lc[key]
                break

        # Case 1: explicit input column exists
        if in_col is not None:
            try:
                for _, row in df.iterrows():
                    k_raw = row.get(in_col, None)
                    if _is_missing_value(k_raw):
                        continue
                    k = str(k_raw).strip().upper()

                    v_raw = row.get(ric_col, None)
                    if _is_missing_value(v_raw):
                        out.setdefault(k, None)
                        continue

                    v = str(v_raw).strip()
                    out[k] = v if v else None
            except Exception:
                for s in chunk:
                    out.setdefault(s, None)

            for s in chunk:
                out.setdefault(s, None)
            continue

        # Case 2: No input column. Use row-order mapping ONLY if lengths match.
        try:
            if len(df) == len(chunk):
                for i, s in enumerate(chunk):
                    v_raw = df.iloc[i][ric_col]
                    if _is_missing_value(v_raw):
                        out[s] = None
                    else:
                        v = str(v_raw).strip()
                        out[s] = v if v else None
            else:
                for s in chunk:
                    out.setdefault(s, None)
        except Exception:
            for s in chunk:
                out.setdefault(s, None)

    return out


# =========================
# Ticker -> RIC resolution (batched)
# =========================

_TICKER_TO_RIC_CACHE: Dict[str, Optional[str]] = {}
_INSTRUMENT_VALID_CACHE: Dict[str, bool] = {}
_EVENTS_CACHE: Dict[str, List[EarningsEvent]] = {}


def _mark_valid_cache(requested: List[str], valid: set[str]) -> None:
    req_set = set(requested)
    for inst in req_set:
        _INSTRUMENT_VALID_CACHE[inst] = (inst in valid)


def batch_validate_instruments(
    instruments: List[str],
    *,
    fail_fast: bool,
    chunk_size: int,
) -> set[str]:
    """
    Validate instruments/RICs in batches.

    Fixes vs old batching:
    - Uses safe_get_data() so one failing instrument doesn't kill the whole chunk.
    - Uses a stricter heuristic to avoid "false valid" rows:
        valid if at least one of TR.CommonName or TR.ExchangeName is non-empty.
    """
    valid: set[str] = set()
    if not instruments:
        return valid

    seen: set[str] = set()
    uniq: List[str] = []
    for x in instruments:
        x2 = (x or "").strip()
        if not x2:
            continue
        if x2 not in seen:
            uniq.append(x2)
            seen.add(x2)

    validate_fields = ["TR.CommonName", "TR.ExchangeName"]

    for ch in chunked(uniq, chunk_size):
        df, _err = safe_get_data(ch, validate_fields, {}, retries=EIKON_RETRIES, fail_fast=fail_fast)
        if df is None or getattr(df, "empty", True):
            _mark_valid_cache(ch, set())
            continue

        cols = list(df.columns)
        inst_col = "Instrument" if "Instrument" in cols else (cols[0] if cols else None)
        if inst_col is None:
            _mark_valid_cache(ch, set())
            continue

        common_col = "TR.CommonName" if "TR.CommonName" in cols else find_col(df, ["common name"])
        exch_col = "TR.ExchangeName" if "TR.ExchangeName" in cols else find_col(df, ["exchange name"])

        found: set[str] = set()
        try:
            for _, row in df.iterrows():
                inst = row.get(inst_col)
                if inst is None:
                    continue
                s_inst = str(inst).strip()
                if not s_inst or s_inst.lower() == "nan":
                    continue

                common = str(row.get(common_col, "")).strip() if common_col else ""
                exch = str(row.get(exch_col, "")).strip() if exch_col else ""

                if (common and common.lower() != "nan") or (exch and exch.lower() != "nan"):
                    found.add(s_inst)
        except Exception:
            found = set()

        _mark_valid_cache(ch, found)
        valid.update(found)

    return valid


def _symbology_inputs_for_ticker(t: str) -> List[str]:
    t = (t or "").strip().upper()
    if not t:
        return []

    cands: List[str] = [t]

    if "." in t:
        cands.append(t.replace(".", "-"))
        cands.append(t.replace(".", ""))

    if "-" in t:
        cands.append(t.replace("-", "."))
        cands.append(t.replace("-", ""))

    m = re.fullmatch(r"([A-Z0-9]+)\.([A-Z])", t)
    if m:
        root = m.group(1)
        cls = m.group(2)
        cands.append(f"{root}{cls}")          # BRKA
        cands.append(f"{root}-{cls}")         # BRK-A
        cands.append(f"{root}{cls.lower()}")  # BRKa

    seen: set[str] = set()
    out: List[str] = []
    for x in cands:
        x2 = x.strip().upper()
        if x2 and x2 not in seen:
            out.append(x2)
            seen.add(x2)
    return out


def _ric_guess_candidates_for_ticker(t: str) -> List[str]:
    t_raw = (t or "").strip()
    t_up = t_raw.upper()
    if not t_up:
        return []

    roots: List[str] = [t_up]

    if "." in t_up:
        roots.extend([t_up.replace(".", ""), t_up.replace(".", "-")])

    if "-" in t_up:
        roots.extend([t_up.replace("-", ""), t_up.replace("-", ".")])

    m = re.fullmatch(r"([A-Z0-9]+)\.([A-Z])", t_up)
    if m:
        root = m.group(1)
        cls = m.group(2)
        roots.insert(1, f"{root}{cls.lower()}")  # e.g. BRKa

    seen: set[str] = set()
    uniq_roots: List[str] = []
    for r in roots:
        r2 = r.strip()
        if r2 and r2 not in seen:
            uniq_roots.append(r2)
            seen.add(r2)

    out: List[str] = []
    for root in uniq_roots:
        for suf in RIC_SUFFIX_GUESSES:
            out.append(f"{root}{suf}")

    return out


def resolve_tickers_to_rics_batched(
    tickers: List[str],
    *,
    fail_fast: bool,
    symbology_chunk_size: int,
    validate_chunk_size: int,
    show_progress: bool,
) -> Dict[str, Optional[str]]:
    """
    Resolve many tickers to RICs using batching.

    Fixes:
    - Never treat pandas NA / NaN / "<NA>" as a valid resolved RIC.
    - Normalize keys consistently to UPPER.
    - Validate only non-missing returned RIC strings.
    - Ensure suffix-guess fallback runs when symbology returns missing.
    """
    def _is_missing_value(x: Any) -> bool:
        return is_missing_value(x)

    seen: set[str] = set()
    uniq: List[str] = []
    for t in tickers:
        t2 = (t or "").strip().upper()
        if not t2:
            continue
        if t2 not in seen:
            uniq.append(t2)
            seen.add(t2)

    out: Dict[str, Optional[str]] = {t: None for t in uniq}

    remaining: List[str] = []
    for t in uniq:
        if t in _TICKER_TO_RIC_CACHE:
            v = _TICKER_TO_RIC_CACHE[t]
            out[t] = None if _is_missing_value(v) else str(v).strip()
        else:
            remaining.append(t)

    if not remaining:
        return out

    direct_rics: List[str] = [t for t in remaining if ("." in t or t.endswith("=R"))]
    if direct_rics:
        valid_direct = batch_validate_instruments(
            direct_rics, fail_fast=fail_fast, chunk_size=validate_chunk_size
        )
        for t in direct_rics:
            if t in valid_direct:
                out[t] = t
                _TICKER_TO_RIC_CACHE[t] = t

    unresolved: List[str] = [t for t in remaining if out.get(t) is None]

    sym_inputs: List[str] = []
    sym_inputs_by_ticker: Dict[str, List[str]] = {}
    for t in unresolved:
        cands = _symbology_inputs_for_ticker(t)
        cands_u = [c.strip().upper() for c in cands if c and c.strip()]
        sym_inputs_by_ticker[t] = cands_u
        sym_inputs.extend(cands_u)

    sym_map: Dict[str, Optional[str]] = {}
    if sym_inputs:
        if show_progress and tqdm is not None:
            sym_chunks = list(chunked(sym_inputs, symbology_chunk_size))
            for ch in tqdm(sym_chunks, desc="Eikon symbology (best_match)", unit="chunk"):
                part = batch_get_symbology_best_match(ch, chunk_size=symbology_chunk_size)
                sym_map.update(part)
        else:
            sym_map = batch_get_symbology_best_match(sym_inputs, chunk_size=symbology_chunk_size)

    returned_rics: List[str] = []
    for v in sym_map.values():
        if _is_missing_value(v):
            continue
        r = str(v).strip()
        if r:
            returned_rics.append(r)

    valid_returned: set[str] = set()
    if returned_rics:
        valid_returned = batch_validate_instruments(
            returned_rics, fail_fast=fail_fast, chunk_size=validate_chunk_size
        )

    for t in unresolved:
        for sym in sym_inputs_by_ticker.get(t, []):
            ric_raw = sym_map.get(sym)
            if _is_missing_value(ric_raw):
                continue
            ric = str(ric_raw).strip()
            if ric and (ric in valid_returned):
                out[t] = ric
                _TICKER_TO_RIC_CACHE[t] = ric
                break

    unresolved2: List[str] = [t for t in unresolved if out.get(t) is None]

    if unresolved2:
        guess_map: Dict[str, List[str]] = {t: _ric_guess_candidates_for_ticker(t) for t in unresolved2}
        all_guesses: List[str] = []
        for guesses in guess_map.values():
            all_guesses.extend([g for g in guesses if g and g.strip()])

        valid_guesses = batch_validate_instruments(
            all_guesses, fail_fast=fail_fast, chunk_size=validate_chunk_size
        )

        for t in unresolved2:
            for g in guess_map.get(t, []):
                if g in valid_guesses:
                    out[t] = g
                    _TICKER_TO_RIC_CACHE[t] = g
                    break

    unresolved3: List[str] = [t for t in unresolved2 if out.get(t) is None]

    if unresolved3:
        valid_fallback = batch_validate_instruments(
            unresolved3, fail_fast=fail_fast, chunk_size=validate_chunk_size
        )
        for t in unresolved3:
            if t in valid_fallback:
                out[t] = t
                _TICKER_TO_RIC_CACHE[t] = t
            else:
                out[t] = None
                _TICKER_TO_RIC_CACHE[t] = None

    for t, ric_override in TICKER_TO_RIC_OVERRIDES.items():
        t2 = (t or "").strip().upper()
        if not t2:
            continue
        if t2 in out:
            out[t2] = ric_override
            _TICKER_TO_RIC_CACHE[t2] = ric_override

    return out


# =========================
# Earnings data retrieval (BATCHED + robust split)
# =========================

def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return None
        return float(x)
    except Exception:
        return None


def fetch_eps_events_batched(
    rics: List[str],
    *,
    fail_fast: bool,
    lookback_years: int,
    eps_chunk_size: int,
    show_progress: bool,
) -> Dict[str, List[EarningsEvent]]:
    to_fetch: List[str] = []
    for r in rics:
        r2 = (r or "").strip()
        if not r2:
            continue
        if r2 in _EVENTS_CACHE:
            continue
        to_fetch.append(r2)

    for r in to_fetch:
        _EVENTS_CACHE.setdefault(r, [])

    if not to_fetch:
        return {r: _EVENTS_CACHE.get(r, []) for r in rics}

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=lookback_years * 365)
    end = today + timedelta(days=365)

    fields = [
        "TR.EPSActValue",
        "TR.EPSActValue.date",
        "TR.EPSActValue.fperiod",
        "TR.EPSActValue.PeriodEndDate",
        "TR.EPSMean",
        "TR.EPSHigh",
        "TR.EPSLow",
        "TR.EPSStdDev",
    ]
    params = {"SDate": start.isoformat(), "EDate": end.isoformat(), "Period": "FQ0", "Frq": "FQ"}

    chunks = list(chunked(to_fetch, eps_chunk_size))
    chunk_iter: Iterable[List[str]] = chunks
    if show_progress and tqdm is not None:
        chunk_iter = tqdm(chunks, desc="Eikon EPS history", unit="chunk")

    for ric_chunk in chunk_iter:
        df, _err = safe_get_data(
            ric_chunk,
            fields,
            params,
            retries=EIKON_RETRIES,
            fail_fast=fail_fast,
        )
        if df is None or getattr(df, "empty", True):
            continue

        cols = list(df.columns)
        inst_col = "Instrument" if "Instrument" in cols else (cols[0] if cols else None)
        if inst_col is None:
            continue

        col_actual = find_col(df, ["earnings per share - actual", "eps - actual", "eps actual", "tr.epsactvalue"])
        col_date = find_col(df, ["tr.epsactvalue.date", " date"])
        col_fperiod = find_col(df, ["financial period absolute", "fperiod", "tr.epsactvalue.fperiod"])
        col_ped = find_col(df, ["period end date", "tr.epsactvalue.periodenddate"])

        col_mean = find_col(df, ["earnings per share - mean", "eps - mean", "eps mean", "tr.epsmean"])
        col_high = find_col(df, ["earnings per share - high", "eps - high", "eps high", "tr.epshigh"])
        col_low = find_col(df, ["earnings per share - low", "eps - low", "eps low", "tr.epslow"])
        col_std = find_col(df, ["standard deviation", "std dev", "stdev", "tr.epsstddev"])

        tmp: Dict[str, List[EarningsEvent]] = {}
        for _, row in df.iterrows():
            try:
                inst = row.get(inst_col)
                if inst is None:
                    continue
                ric = str(inst).strip()
                if not ric or ric.lower() == "nan":
                    continue

                d_raw = row.get(col_date) if col_date else None
                d_dt = parse_any_datetime_to_date(d_raw)
                if not d_dt:
                    continue

                actual = _parse_float(row.get(col_actual)) if col_actual else None
                fperiod = _safe_str(row.get(col_fperiod)) if col_fperiod else None
                ped = parse_any_datetime_to_date(row.get(col_ped)) if col_ped else None

                mean = _parse_float(row.get(col_mean)) if col_mean else None
                high = _parse_float(row.get(col_high)) if col_high else None
                low = _parse_float(row.get(col_low)) if col_low else None
                std = _parse_float(row.get(col_std)) if col_std else None

                tmp.setdefault(ric, []).append(EarningsEvent(d_dt, fperiod, ped, actual, mean, high, low, std))
            except Exception:
                continue

        for ric, evs in tmp.items():
            evs.sort(key=lambda e: e.announce_date)
            _EVENTS_CACHE[ric] = evs

    return {r: _EVENTS_CACHE.get(r, []) for r in rics}


def match_event_by_anchor_date(
    events: List[EarningsEvent],
    anchor: date,
    pre_days: int,
    post_days: int,
    max_distance_days: int,
) -> Tuple[Optional[EarningsEvent], Optional[str]]:
    if not events:
        return None, None

    lo = anchor - timedelta(days=pre_days)
    hi = anchor + timedelta(days=post_days)

    candidates = [e for e in events if lo <= e.announce_date <= hi]
    if not candidates:
        closest = min(events, key=lambda e: abs((e.announce_date - anchor).days))
        if abs((closest.announce_date - anchor).days) <= max_distance_days:
            return closest, "closest_overall"
        return None, None

    best = min(candidates, key=lambda e: abs((e.announce_date - anchor).days))
    if abs((best.announce_date - anchor).days) > max_distance_days:
        return None, None
    return best, "announce_date"


# =========================
# Validation logic
# =========================

def decide_label(actual: float, estimate: float, tie_tol: float = EPS_TIE_TOL) -> str:
    if actual > estimate + tie_tol:
        return "BEAT"
    if actual < estimate - tie_tol:
        return "MISS"
    return "TIE"


def expected_resolution_from_label(label: str, yes_semantics: str, tie_counts_as: str) -> str:
    tie_yes = (tie_counts_as.upper() == "YES")

    if yes_semantics == "YES_MEANS_BEAT":
        if label == "BEAT":
            return "YES"
        if label == "TIE":
            return "YES" if tie_yes else "NO"
        return "NO"

    if yes_semantics == "YES_MEANS_MISS":
        if label == "MISS":
            return "YES"
        if label == "TIE":
            return "YES" if tie_yes else "NO"
        return "NO"

    return "NO"


# =========================
# I/O helpers
# =========================

def write_jsonl_records(path: Path, records: List[ResultRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_csv_records(path: Path, records: List[ResultRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ResultRecord.__annotations__.keys()))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """
    Atomic-ish replacement:
    - On Windows, os.replace is atomic for same-volume moves.
    """
    os.replace(str(tmp_path), str(final_path))


def write_markets_jsonl_inplace(path: Path, markets: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in markets:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _atomic_replace(tmp, path)


def _read_existing_csv_columns(csv_path: Path) -> Optional[List[str]]:
    """
    Read column order from existing markets.csv, if available.
    If pandas is installed, use it; otherwise use csv module.
    """
    if not csv_path.exists():
        return None

    try:
        if pd is not None:
            df0 = pd.read_csv(csv_path, nrows=1)  # type: ignore
            return list(df0.columns)
    except Exception:
        pass

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                return [h.strip() for h in header]
    except Exception:
        return None

    return None


def write_markets_csv_inplace(csv_path: Path, markets: List[Dict[str, Any]]) -> None:
    """
    Overwrite markets.csv to mirror markets.jsonl.

    - Preserves existing column order if markets.csv exists.
    - Appends any new columns at the end (sorted for stability).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_cols = _read_existing_csv_columns(csv_path)

    # Collect all keys
    all_keys: set[str] = set()
    for m in markets:
        all_keys.update(m.keys())

    if existing_cols:
        base_cols = [c for c in existing_cols if c in all_keys]
        new_cols = sorted([c for c in all_keys if c not in set(existing_cols)])
        cols = base_cols + new_cols
    else:
        cols = sorted(all_keys)

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")

    if pd is not None:
        try:
            df = pd.DataFrame(markets)  # type: ignore
            # Ensure all columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            df.to_csv(tmp, index=False, encoding="utf-8")  # type: ignore
            _atomic_replace(tmp, csv_path)
            return
        except Exception:
            # fallback below
            pass

    # csv module fallback
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in markets:
            w.writerow({k: row.get(k, None) for k in cols})
    _atomic_replace(tmp, csv_path)


# =========================
# Runner (importable)
# =========================

def run_optional_check_consistency_inplace(
    *,
    markets_jsonl_path: Path,
    markets_csv_path: Path,
    validation_dir: Path,
    app_key: str,
    eikon_port: Optional[int],
    require_proxy: bool,
    fail_fast: bool,
    max_markets: Optional[int],
    event_pre_days: int,
    event_post_days: int,
    max_event_distance_days: int,
    symbology_chunk_size: int,
    validate_chunk_size: int,
    eps_chunk_size: int,
    show_progress: bool,
) -> Dict[str, Any]:
    """
    Main entrypoint.

    Reads markets_jsonl_path, validates candidate earnings markets,
    then writes results back into:
      - markets_jsonl_path (overwrite)
      - markets_csv_path (overwrite)

    Still writes:
      - validation_dir/unmatched.jsonl + .csv
      - validation_dir/consistency_summary.txt
    """
    out_unmatched = validation_dir / "unmatched.jsonl"
    out_unmatched_csv = validation_dir / "unmatched.csv"
    out_summary_txt = validation_dir / "consistency_summary.txt"

    unmatched_records: List[ResultRecord] = []

    lines_scanned = 0
    markets_parsed = 0
    considered = 0
    ignored_non_candidate = 0
    matched = 0
    correct_n = 0
    incorrect_n = 0
    unmatched_n = 0

    pending: List[PendingMarket] = []
    tickers_needed: set[str] = set()

    if not markets_jsonl_path.exists():
        raise FileNotFoundError(f"Markets JSONL file not found: {markets_jsonl_path}")

    init_eikon(app_key, eikon_port=eikon_port, require_proxy=require_proxy)

    # Read ALL markets into memory (so we can overwrite in place)
    all_markets: List[Dict[str, Any]] = []
    line_no_to_idx: Dict[int, int] = {}

    # PASS A: parse + pre-filter (and store all markets)
    parse_bar_ctx = None
    if show_progress and tqdm is not None:
        # If max_markets is set, use that; otherwise we still need a pass to count lines,
        # but to avoid a second pass we just use "unknown total" (tqdm without total).
        parse_bar_ctx = tqdm(total=(max_markets if max_markets else None), desc="Parsing markets", unit="line")

    try:
        with markets_jsonl_path.open("r", encoding="utf-8", errors="strict") as f:
            for line_no, line in enumerate(f, start=1):
                if max_markets is not None and line_no > max_markets:
                    break

                lines_scanned += 1
                if parse_bar_ctx is not None:
                    parse_bar_ctx.update(1)

                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except Exception as exc:
                    # We abort to avoid rewriting a file with dropped/altered invalid lines.
                    raise RuntimeError(
                        f"Invalid JSON on line {line_no} in {markets_jsonl_path}. "
                        "Fix the file before running in-place updates."
                    ) from exc

                markets_parsed += 1
                idx = len(all_markets)
                all_markets.append(raw)
                line_no_to_idx[line_no] = idx

                slug = _safe_str(raw.get("slug"))
                question = _safe_str(raw.get("question")) or _safe_str(raw.get("title"))
                market_id = extract_market_id(raw)
                resolved_outcome = extract_resolved_outcome(raw)
                ticker = extract_ticker(raw, question)

                if not is_candidate_earnings_market(question, slug):
                    ignored_non_candidate += 1
                    continue

                considered += 1

                anchor_dt, _anchor_src = parse_anchor_date(slug, question, raw)

                q_est, q_src = extract_estimate_from_question(question)
                s_est, s_src = extract_estimate_from_slug(slug)
                polymarket_est = q_est if q_est is not None else s_est
                polymarket_est_src = q_src if q_est is not None else s_src

                yes_semantics, tie_counts_as = infer_yes_semantics(question, slug)

                base = ResultRecord(
                    line_no=line_no,
                    market_id=market_id,
                    slug=slug,
                    question=question,
                    ticker=ticker,
                    ric=None,
                    polymarket_resolved_outcome=resolved_outcome,
                    anchor_date=_to_date_iso(anchor_dt),
                    polymarket_estimate=polymarket_est,
                    polymarket_estimate_source=polymarket_est_src,
                    yes_semantics=yes_semantics,
                    inline_counts_as=tie_counts_as,
                    matched_announce_date=None,
                    matched_fperiod=None,
                    matched_period_end_date=None,
                    eikon_actual_eps=None,
                    eikon_eps_mean_estimate=None,
                    eikon_eps_high_estimate=None,
                    eikon_eps_low_estimate=None,
                    eikon_eps_stddev_estimate=None,
                    estimate_used=None,
                    estimate_used_source=None,
                    surprise=None,
                    label=None,
                    expected_resolution=None,
                    match_method=None,
                    status="UNMATCHED",
                    skip_reason=None,
                )

                if not resolved_outcome:
                    base.skip_reason = "unresolved_or_missing_outcome"
                    unmatched_records.append(base)
                    unmatched_n += 1
                    continue
                if not ticker:
                    base.skip_reason = "no_ticker"
                    unmatched_records.append(base)
                    unmatched_n += 1
                    continue
                if not anchor_dt:
                    base.skip_reason = "no_anchor_date"
                    unmatched_records.append(base)
                    unmatched_n += 1
                    continue

                pending.append(PendingMarket(base=base, anchor_dt=anchor_dt))
                tickers_needed.add(ticker.upper())
    finally:
        if parse_bar_ctx is not None:
            parse_bar_ctx.close()

    # PASS B: batch resolve tickers -> RICs
    ticker_to_ric = resolve_tickers_to_rics_batched(
        sorted(tickers_needed),
        fail_fast=fail_fast,
        symbology_chunk_size=symbology_chunk_size,
        validate_chunk_size=validate_chunk_size,
        show_progress=show_progress,
    )

    pending2: List[PendingMarket] = []
    for pm in pending:
        t = (pm.base.ticker or "").upper()
        ric = ticker_to_ric.get(t)
        if not ric:
            pm.base.skip_reason = "no_ric"
            unmatched_records.append(pm.base)
            unmatched_n += 1
            continue
        pm.base.ric = ric
        pending2.append(pm)

    # PASS C: batch fetch EPS events for RICs (robust: bisects failing batches)
    rics_needed = sorted({pm.base.ric for pm in pending2 if pm.base.ric})
    fetch_eps_events_batched(
        rics_needed,
        fail_fast=fail_fast,
        lookback_years=12,
        eps_chunk_size=eps_chunk_size,
        show_progress=show_progress,
    )

    # PASS D: finalize record classification
    iter_pm: Iterable[PendingMarket] = tqdm(pending2, desc="Classifying", unit="mkt") if (show_progress and tqdm is not None) else pending2

    # We'll keep a dict of line_no -> ResultRecord to write back into all_markets
    results_by_line_no: Dict[int, ResultRecord] = {}
    for r in unmatched_records:
        results_by_line_no[r.line_no] = r

    for pm in iter_pm:
        base = pm.base
        anchor_dt = pm.anchor_dt
        ric = base.ric or ""

        events = _EVENTS_CACHE.get(ric, [])
        if not events:
            base.skip_reason = "no_events_returned"
            unmatched_records.append(base)
            unmatched_n += 1
            results_by_line_no[base.line_no] = base
            continue

        ev, method = match_event_by_anchor_date(
            events,
            anchor_dt,
            pre_days=event_pre_days,
            post_days=event_post_days,
            max_distance_days=max_event_distance_days,
        )
        if not ev:
            base.skip_reason = "no_event_match"
            unmatched_records.append(base)
            unmatched_n += 1
            results_by_line_no[base.line_no] = base
            continue

        base.matched_announce_date = _to_date_iso(ev.announce_date)
        base.matched_fperiod = ev.fperiod
        base.matched_period_end_date = _to_date_iso(ev.period_end_date)
        base.eikon_actual_eps = ev.actual_eps
        base.eikon_eps_mean_estimate = ev.mean_estimate
        base.eikon_eps_high_estimate = ev.high_estimate
        base.eikon_eps_low_estimate = ev.low_estimate
        base.eikon_eps_stddev_estimate = ev.stddev_estimate
        base.match_method = method

        if ev.actual_eps is None:
            base.skip_reason = "no_actual_eps"
            unmatched_records.append(base)
            unmatched_n += 1
            results_by_line_no[base.line_no] = base
            continue

        polymarket_est = base.polymarket_estimate
        polymarket_est_src = base.polymarket_estimate_source

        estimate_used: Optional[float] = None
        estimate_used_source: Optional[str] = None

        if polymarket_est is not None:
            estimate_used = float(polymarket_est)
            estimate_used_source = polymarket_est_src
        elif ev.mean_estimate is not None:
            estimate_used = float(ev.mean_estimate)
            estimate_used_source = "eikon_mean"

        if estimate_used is None:
            base.skip_reason = "no_estimate"
            unmatched_records.append(base)
            unmatched_n += 1
            results_by_line_no[base.line_no] = base
            continue

        matched += 1

        base.estimate_used = estimate_used
        base.estimate_used_source = estimate_used_source

        lab = decide_label(float(ev.actual_eps), estimate_used)
        base.label = lab
        base.surprise = float(ev.actual_eps) - estimate_used
        base.expected_resolution = expected_resolution_from_label(
            lab,
            (base.yes_semantics or "YES_MEANS_BEAT"),
            (base.inline_counts_as or "NO"),
        )

        if base.expected_resolution == base.polymarket_resolved_outcome:
            base.status = "MATCHED_CORRECT"
            correct_n += 1
        else:
            base.status = "MATCHED_INCORRECT"
            incorrect_n += 1

        results_by_line_no[base.line_no] = base

    # Write unmatched output files (still produced)
    write_jsonl_records(out_unmatched, unmatched_records)
    write_csv_records(out_unmatched_csv, unmatched_records)

    # Apply results back into the in-memory markets
    updated_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    updated_count = 0

    def _apply_result_to_market(market_obj: Dict[str, Any], rec: ResultRecord) -> None:
        """
        Write the validation results back to the market dict using a stable prefix (val_*).
        """
        d = asdict(rec)
        for k, v in d.items():
            market_obj[f"{VAL_PREFIX}{k}"] = v
        market_obj[f"{VAL_PREFIX}updated_utc"] = updated_utc
        market_obj[f"{VAL_PREFIX}script"] = Path(__file__).name

    for ln, rec in results_by_line_no.items():
        idx = line_no_to_idx.get(ln)
        if idx is None:
            continue
        _apply_result_to_market(all_markets[idx], rec)
        updated_count += 1

    # Overwrite markets.jsonl and markets.csv in place
    write_markets_jsonl_inplace(markets_jsonl_path, all_markets)
    write_markets_csv_inplace(markets_csv_path, all_markets)

    # Write summary txt (requested)
    summary_txt = (
        "==================== CONSISTENCY SUMMARY ====================\n"
        f"Timestamp (UTC):                    {updated_utc}\n"
        f"Markets JSONL updated:              {markets_jsonl_path}\n"
        f"Markets CSV updated:                {markets_csv_path}\n"
        "\n"
        f"Lines scanned:                      {lines_scanned}\n"
        f"Markets parsed (valid JSON):        {markets_parsed}\n"
        f"Ignored (non-earnings candidate):   {ignored_non_candidate}\n"
        f"Considered (earnings-like):         {considered}\n"
        f"Matched (event+actual+estimate):    {matched}\n"
        f"  - Correctly resolved:             {correct_n}\n"
        f"  - Incorrectly resolved:           {incorrect_n}\n"
        f"Unmatched (skip_reason set):        {unmatched_n}\n"
        "\n"
        f"Markets updated with val_* fields:  {updated_count}\n"
        "\n"
        "Manual investigation outputs:\n"
        f"  - {out_unmatched}\n"
        f"  - {out_unmatched_csv}\n"
        "\n"
        "Batching:\n"
        f"  - symbology_chunk_size:           {int(symbology_chunk_size)}\n"
        f"  - validate_chunk_size:            {int(validate_chunk_size)}\n"
        f"  - eps_chunk_size:                 {int(eps_chunk_size)}\n"
        "=============================================================\n"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)
    out_summary_txt.write_text(summary_txt, encoding="utf-8")

    return {
        "lines_scanned": lines_scanned,
        "markets_parsed": markets_parsed,
        "ignored_non_candidate": ignored_non_candidate,
        "considered": considered,
        "matched": matched,
        "correct": correct_n,
        "incorrect": incorrect_n,
        "unmatched": unmatched_n,
        "updated_count": updated_count,
        "markets_jsonl_path": str(markets_jsonl_path),
        "markets_csv_path": str(markets_csv_path),
        "out_unmatched": str(out_unmatched),
        "out_unmatched_csv": str(out_unmatched_csv),
        "out_summary_txt": str(out_summary_txt),
        "batching": {
            "symbology_chunk_size": int(symbology_chunk_size),
            "validate_chunk_size": int(validate_chunk_size),
            "eps_chunk_size": int(eps_chunk_size),
        },
    }


# Backwards-compatible alias (old name)
def run_optional_check_consistency(**kwargs: Any) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper.

    If you previously imported run_optional_check_consistency, it now performs the in-place update.
    """
    return run_optional_check_consistency_inplace(**kwargs)


# =========================
# CLI
# =========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Polymarket earnings markets vs Refinitiv Eikon EPS actual/estimates (in-place update of markets files)."
    )
    p.add_argument(
        "--markets-jsonl",
        type=str,
        default=str(DEFAULT_MARKETS_JSONL_PATH),
        help=f"Path to markets.jsonl (default: {DEFAULT_MARKETS_JSONL_PATH})",
    )
    p.add_argument(
        "--markets-csv",
        type=str,
        default=str(DEFAULT_MARKETS_CSV_PATH),
        help=f"Path to markets.csv to overwrite/update (default: {DEFAULT_MARKETS_CSV_PATH})",
    )
    p.add_argument("--max-markets", type=int, default=None, help="Max number of markets (lines) to scan (debug).")
    p.add_argument(
        "--app-key",
        nargs="?",
        const="__ENV__",
        default=None,
        help="Eikon app key. If provided with no value, reads EIKON_APP_KEY from env.",
    )
    p.add_argument("--eikon-port", type=int, default=None, help="Force Eikon/Workspace API proxy port (e.g., 9000/9060).")
    p.add_argument("--skip-proxy-check", action="store_true", help="Skip early proxy reachability check.")
    p.add_argument("--no-fail-fast", action="store_true", help="Do NOT abort early on repeated Eikon 500 'Network Error'.")
    p.add_argument("--validation-dir", type=str, default=str(DEFAULT_VALIDATION_DIR), help="Output dir for unmatched + summary.txt")

    p.add_argument("--event-pre-days", type=int, default=DEFAULT_EVENT_PRE_DAYS)
    p.add_argument("--event-post-days", type=int, default=DEFAULT_EVENT_POST_DAYS)
    p.add_argument("--max-event-distance-days", type=int, default=DEFAULT_MAX_EVENT_DISTANCE_DAYS)

    p.add_argument("--symbology-chunk-size", type=int, default=DEFAULT_SYMBOLOGY_CHUNK_SIZE,
                   help="Symbols per ek.get_symbology batch (default: %(default)s)")
    p.add_argument("--validate-chunk-size", type=int, default=DEFAULT_VALIDATE_CHUNK_SIZE,
                   help="Instruments per ek.get_data validation batch (default: %(default)s)")
    p.add_argument("--eps-chunk-size", type=int, default=DEFAULT_EPS_CHUNK_SIZE,
                   help="RICs per EPS history ek.get_data batch (default: %(default)s)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    require_tqdm()
    setup_logging()
    setup_warnings_suppression()

    args = parse_args(argv)
    fail_fast = (not args.no_fail_fast)

    markets_jsonl_path = Path(args.markets_jsonl)
    markets_csv_path = Path(args.markets_csv)
    validation_dir = Path(args.validation_dir)

    if args.app_key is None:
        LOG.error("Missing --app-key. Provide it or use '--app-key' (no value) to read from env EIKON_APP_KEY.")
        return 2
    if args.app_key == "__ENV__":
        app_key = os.getenv("EIKON_APP_KEY") or os.getenv("APP_KEY") or ""
        if not app_key:
            LOG.error("EIKON_APP_KEY (or APP_KEY) not found in environment.")
            return 2
    else:
        app_key = args.app_key

    try:
        summary = run_optional_check_consistency_inplace(
            markets_jsonl_path=markets_jsonl_path,
            markets_csv_path=markets_csv_path,
            validation_dir=validation_dir,
            app_key=app_key,
            eikon_port=args.eikon_port,
            require_proxy=(not args.skip_proxy_check),
            fail_fast=fail_fast,
            max_markets=args.max_markets,
            event_pre_days=args.event_pre_days,
            event_post_days=args.event_post_days,
            max_event_distance_days=args.max_event_distance_days,
            symbology_chunk_size=args.symbology_chunk_size,
            validate_chunk_size=args.validate_chunk_size,
            eps_chunk_size=args.eps_chunk_size,
            show_progress=(not args.no_progress),
        )
    except FatalEikonNetworkError as exc:
        LOG.error(
            "%s\n\n"
            "Fix likely involves:\n"
            "  - Ensure Workspace/Eikon is running AND logged in\n"
            "  - Verify the API proxy is running (port 9000/9060)\n"
            "  - If on VPN/corporate proxy, allowlist/firewall may be required\n",
            exc,
        )
        return 2
    except Exception as exc:
        LOG.error("Fatal error: %s", exc)
        return 2

    printable = (
        "\n"
        "==================== SUMMARY ====================\n"
        f"Markets updated (JSONL):            {summary['markets_jsonl_path']}\n"
        f"Markets updated (CSV):              {summary['markets_csv_path']}\n"
        "\n"
        f"Lines scanned:                      {summary['lines_scanned']}\n"
        f"Markets parsed (valid JSON):        {summary['markets_parsed']}\n"
        f"Ignored (non-earnings candidate):   {summary['ignored_non_candidate']}\n"
        f"Considered (earnings-like):         {summary['considered']}\n"
        f"Matched (event+actual+estimate):    {summary['matched']}\n"
        f"  - Correctly resolved:             {summary['correct']}\n"
        f"  - Incorrectly resolved:           {summary['incorrect']}\n"
        f"Unmatched:                          {summary['unmatched']}\n"
        f"Markets updated with val_* fields:  {summary['updated_count']}\n"
        "\n"
        "Manual investigation:\n"
        f"  - Unmatched JSONL:                {summary['out_unmatched']}\n"
        f"  - Unmatched CSV:                  {summary['out_unmatched_csv']}\n"
        f"Summary TXT:                        {summary['out_summary_txt']}\n"
        "=================================================\n"
    )
    if tqdm is not None and not args.no_progress:
        tqdm.write(printable)
    else:
        print(printable)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
