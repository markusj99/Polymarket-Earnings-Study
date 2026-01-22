#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
01_optional_check_consistency.py

Validate Polymarket "beat earnings / beat EPS estimate" market resolutions against Refinitiv Eikon
(Earnings Surprise / Estimates data).

OUTPUTS (always written)
------------------------
1) correct.jsonl
   - Matched to an Eikon event + estimate and Polymarket resolution agrees with expected outcome.

2) incorrectly_resolved.jsonl
   - Matched to an Eikon event + estimate but Polymarket resolution disagrees with expected outcome.

3) unmatched.jsonl
   - Could not match to Eikon (missing ticker/anchor/ric/event/actual/estimate/etc). skip_reason explains why.

CONSOLE OUTPUT
--------------
- Shows a tqdm progress bar during processing.
- Prints ONE summary block at the end (after all markets are processed).
- Prints ERROR messages only when something catastrophic prevents the script from continuing.
- Suppresses noisy third-party HTTP logs and the Eikon internal FutureWarning spam.

FAIL-FAST ON EIKON "500 Network Error" (optional)
-------------------------------------------------
If Eikon repeatedly returns error code 500 with message "Network Error", that typically means
Workspace/Eikon cannot reach upstream services (logged out/offline, VPN/proxy/firewall, or outage).
By default, this script aborts early in that scenario and writes partial outputs.

Disable with: --no-fail-fast

NOTES
-----
- Requires Eikon Desktop / Refinitiv Workspace running + logged in, with Data API proxy available.
- Entitlements vary: TR.EPSMean may be unavailable; the script falls back to Polymarket estimate
  from question/slug when possible.
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
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_VALIDATION_DIR = Path(__file__).resolve().parent / "data" / "validation"
DEFAULT_MARKETS_PATH = Path(__file__).resolve().parent / "data" / "markets" / "markets.jsonl"

RIC_SUFFIX_GUESSES = [".O", ".N", ".A", ".L"]

DEFAULT_EVENT_PRE_DAYS = 10
DEFAULT_EVENT_POST_DAYS = 30
DEFAULT_MAX_EVENT_DISTANCE_DAYS = 60

EPS_TIE_TOL = 1e-6

EIKON_RETRIES = 5
EIKON_RETRY_BASE_SLEEP = 0.7

DEFAULT_EIKON_PORT_CANDIDATES = [9000, 9060]
EIKON_STATUS_PATHS = ["/api/status", "/api/handshake"]


# =========================
# Logging (quiet)
# =========================

LOG = logging.getLogger("eikon_eps_validator")


class NoiseFilter(logging.Filter):
    """
    Drop known noisy messages so the tqdm bar stays clean.
    NOTE: Python warnings are not handled here (use warnings.filterwarnings).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        # Common spam from HTTP stacks
        if "HTTP Request:" in msg:
            return False

        # Repeated Eikon "Network Error" spam; we emit a single fatal error ourselves if needed.
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
    """
    Raise verbosity thresholds of known chatty stacks.
    This prevents things like "HTTP Request: POST http://127.0.0.1:9000/..." from printing.
    """
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
    """
    Catastrophic-only logging:
    - root level ERROR
    - handler has NoiseFilter to drop spam
    """
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
    Suppress the specific eikon/pandas FutureWarning that breaks tqdm output, e.g.:
      eikon/data_grid.py: FutureWarning: errors='ignore' is deprecated ...
    """
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"eikon\.data_grid",
    )


# =========================
# Exceptions
# =========================


class FatalEikonNetworkError(RuntimeError):
    """Raised when Eikon repeatedly returns 500 'Network Error' and fail-fast is enabled."""


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

    estimate_used: Optional[float]
    estimate_used_source: Optional[str]

    surprise: Optional[float]
    label: Optional[str]
    expected_resolution: Optional[str]
    match_method: Optional[str]

    status: str  # MATCHED_CORRECT | MATCHED_INCORRECT | UNMATCHED
    skip_reason: Optional[str]


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
    """
    Attempt to read the Eikon API Proxy port from common Windows locations.
    """
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
    """
    Detect a working localhost port by probing /api/status and /api/handshake.
    """
    ports: List[int] = []
    file_port = _read_port_inuse_file()
    if file_port:
        ports.append(file_port)
    ports.extend(DEFAULT_EIKON_PORT_CANDIDATES)
    if extra_ports:
        ports.extend([p for p in extra_ports if isinstance(p, int) and 1 <= p <= 65535])

    # Deduplicate while preserving order
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


def init_eikon(app_key: str, eikon_port: Optional[int], require_proxy: bool) -> None:
    """
    Initialize Eikon SDK.

    We intentionally do not print INFO logs here. Any catastrophic issue raises.
    """
    require_eikon()
    ek.set_app_key(app_key)

    # Attempt to disable internal SDK logging if available (helps keep output clean).
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
    """
    Wrapper around ek.get_data with retries/backoff.

    - Silent on intermediate failures to keep output clean.
    - If all retries fail with 500 "Network Error" and fail_fast=True, raises FatalEikonNetworkError.
    """
    last_exc: Optional[Exception] = None
    network_error_seen = False

    for attempt in range(retries):
        try:
            df, err = ek.get_data(instruments, fields, parameters=parameters)
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


_TICKER_TO_RIC_CACHE: Dict[str, Optional[str]] = {}
_INSTRUMENT_VALID_CACHE: Dict[str, bool] = {}


def _is_valid_instrument(inst: str, *, fail_fast: bool) -> bool:
    try:
        df, _err = eikon_retry_get_data(
            [inst],
            ["TR.CommonName"],
            {},
            retries=EIKON_RETRIES,
            fail_fast=fail_fast,
        )
        if df is None or getattr(df, "empty", True):
            return False
        name_col = next((c for c in df.columns if str(c).lower() != "instrument"), None)
        if not name_col:
            return False
        val = df.iloc[0][name_col]
        return bool(str(val).strip()) and str(val).strip().lower() != "nan"
    except FatalEikonNetworkError:
        raise
    except Exception:
        return False


def _cached_is_valid(inst: str, *, fail_fast: bool) -> bool:
    if inst in _INSTRUMENT_VALID_CACHE:
        return _INSTRUMENT_VALID_CACHE[inst]
    ok = _is_valid_instrument(inst, fail_fast=fail_fast)
    _INSTRUMENT_VALID_CACHE[inst] = ok
    return ok


def ticker_to_ric_best_effort(ticker: str, *, fail_fast: bool) -> Optional[str]:
    """
    Robust mapping ticker -> Refinitiv RIC/instrument.
    """
    
    t = (ticker or "").strip().upper()
    if not t:
        return None

    # Cache hit
    if t in _TICKER_TO_RIC_CACHE:
        return _TICKER_TO_RIC_CACHE[t]

    # 1) If it already looks like a RIC, validate and return.
    #    (Note: this can be too permissive for "BRK.A" because it's a ticker format,
    #     but validation via _cached_is_valid prevents false positives.)
    if "." in t or t.endswith("=R"):
        if _cached_is_valid(t, fail_fast=fail_fast):
            _TICKER_TO_RIC_CACHE[t] = t
            return t

    # Build a list of candidate "symbology input" strings (things we feed to ek.get_symbology).
    # Keep order stable; we dedupe later.
    candidates: List[str] = [t]

    # If there is a dot in the ticker, try alternative separators that symbology sometimes matches.
    # Example: BRK.A -> BRK-A, BRKA
    if "." in t:
        candidates.append(t.replace(".", "-"))
        candidates.append(t.replace(".", ""))

    # If there is a hyphen, also try dot and concatenation.
    # Example: BRK-A -> BRK.A, BRKA
    if "-" in t:
        candidates.append(t.replace("-", "."))
        candidates.append(t.replace("-", ""))

    # 2) Special handling for share-class tickers in the form "XXXX.A" or "XXXX.B"
    #    Refinitiv often uses a lowercase class letter in the RIC root: "XXXXa.N" / "XXXXb.N"
    #    (Berkshire is the canonical example: BRK.A -> BRKa.N).
    m = re.fullmatch(r"([A-Z0-9]+)\.([A-Z])", t)
    if m:
        root = m.group(1)
        cls = m.group(2)

        # Lowercase class letter root (XXXXa, XXXXb, ...)
        root_lc = f"{root}{cls.lower()}"
        # Also include a concatenated uppercase root (XXXXA) and dashed version (XXXX-A)
        # because different symbology backends may match differently.
        root_uc = f"{root}{cls}"
        root_dash = f"{root}-{cls}"

        # Add as symbology candidates
        candidates.extend([root_lc, root_uc, root_dash])

        # Additionally, we can try direct validated RIC guesses from these roots.
        # This is often faster than symbology and fixes BRK.A quickly.
        for suf in RIC_SUFFIX_GUESSES:
            guess = f"{root_lc}{suf}"
            if _cached_is_valid(guess, fail_fast=fail_fast):
                _TICKER_TO_RIC_CACHE[t] = guess
                return guess

    # De-duplicate candidates preserving order
    seen: set[str] = set()
    uniq_candidates: List[str] = []
    for c in candidates:
        c2 = c.strip().upper()
        if not c2:
            continue
        if c2 not in seen:
            uniq_candidates.append(c2)
            seen.add(c2)

    # 3) Symbology best match
    for sym in uniq_candidates:
        try:
            df = ek.get_symbology(
                [sym],
                from_symbol_type="ticker",
                to_symbol_type="RIC",
                best_match=True,
            )
            if df is not None and not df.empty and "RIC" in df.columns:
                ric = str(df.iloc[0]["RIC"]).strip()
                if ric and _cached_is_valid(ric, fail_fast=fail_fast):
                    _TICKER_TO_RIC_CACHE[t] = ric
                    return ric
        except FatalEikonNetworkError:
            # Bubble up if fail_fast triggered deeper in validation.
            raise
        except Exception:
            # Ignore and continue to next candidate.
            pass

    # 4) Symbology all matches + preference ordering
    for sym in uniq_candidates:
        try:
            df = ek.get_symbology(
                [sym],
                from_symbol_type="ticker",
                to_symbol_type="RIC",
                best_match=False,
            )
            if df is None or df.empty or "RIC" not in df.columns:
                continue

            rics = [
                str(x).strip()
                for x in df["RIC"].tolist()
                if isinstance(x, str) and x.strip()
            ]

            # Prefer common suffixes first (NASDAQ/NYSE/AMEX/LSE).
            for suf in RIC_SUFFIX_GUESSES:
                for r in rics:
                    if r.endswith(suf) and _cached_is_valid(r, fail_fast=fail_fast):
                        _TICKER_TO_RIC_CACHE[t] = r
                        return r

            # Otherwise, first valid result
            for r in rics:
                if _cached_is_valid(r, fail_fast=fail_fast):
                    _TICKER_TO_RIC_CACHE[t] = r
                    return r

        except FatalEikonNetworkError:
            raise
        except Exception:
            pass

    # 5) Suffix guesses (helps when symbology fails, e.g., "MLKN" -> "MLKN.O")
    #    We try for the original ticker *and* common normalization variants.
    guess_roots: List[str] = [t]
    if "." in t:
        guess_roots.extend([t.replace(".", ""), t.replace(".", "-")])
    if "-" in t:
        guess_roots.extend([t.replace("-", ""), t.replace("-", ".")])

    # De-dupe guess roots preserving order
    seen2: set[str] = set()
    uniq_roots: List[str] = []
    for r in guess_roots:
        r2 = r.strip()
        if not r2:
            continue
        if r2 not in seen2:
            uniq_roots.append(r2)
            seen2.add(r2)

    for root in uniq_roots:
        for suf in RIC_SUFFIX_GUESSES:
            guess = f"{root}{suf}"
            if _cached_is_valid(guess, fail_fast=fail_fast):
                _TICKER_TO_RIC_CACHE[t] = guess
                return guess

    # 6) Final fallback: try the ticker itself as an instrument (some work without suffix)
    if _cached_is_valid(t, fail_fast=fail_fast):
        _TICKER_TO_RIC_CACHE[t] = t
        return t

    _TICKER_TO_RIC_CACHE[t] = None
    return None


# =========================
# Earnings data retrieval
# =========================

_EVENTS_CACHE: Dict[str, List[EarningsEvent]] = {}


def fetch_eps_events(ric: str, *, fail_fast: bool, lookback_years: int = 12) -> List[EarningsEvent]:
    """
    Fetch quarterly EPS actual events (and mean estimates if available) for an instrument.
    Cached per RIC for speed.
    """
    if ric in _EVENTS_CACHE:
        return _EVENTS_CACHE[ric]

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=lookback_years * 365)
    end = today + timedelta(days=365)

    fields = [
        "TR.EPSActValue",
        "TR.EPSActValue.date",
        "TR.EPSActValue.fperiod",
        "TR.EPSActValue.PeriodEndDate",
        "TR.EPSMean",
    ]
    params = {"SDate": start.isoformat(), "EDate": end.isoformat(), "Period": "FQ0", "Frq": "FQ"}

    df, _err = eikon_retry_get_data([ric], fields, params, retries=EIKON_RETRIES, fail_fast=fail_fast)

    events: List[EarningsEvent] = []
    if df is None or getattr(df, "empty", True):
        _EVENTS_CACHE[ric] = []
        return []

    col_actual = find_col(df, ["earnings per share - actual", "eps - actual", "eps actual", "tr.epsactvalue"])
    col_date = find_col(df, ["tr.epsactvalue.date", " date"])
    col_fperiod = find_col(df, ["financial period absolute", "fperiod", "tr.epsactvalue.fperiod"])
    col_ped = find_col(df, ["period end date", "tr.epsactvalue.periodenddate"])
    col_mean = find_col(df, ["earnings per share - mean", "eps - mean", "eps mean", "tr.epsmean"])

    for _, row in df.iterrows():
        try:
            d_raw = row[col_date] if col_date else None
            d_dt = parse_any_datetime_to_date(d_raw)
            if not d_dt:
                continue

            a = None
            if col_actual:
                try:
                    a = float(row[col_actual])
                except Exception:
                    a = None

            fp = _safe_str(row[col_fperiod]) if col_fperiod else None

            ped = None
            if col_ped:
                ped = parse_any_datetime_to_date(row[col_ped])

            mean = None
            if col_mean:
                try:
                    mean = float(row[col_mean])
                except Exception:
                    mean = None

            events.append(EarningsEvent(d_dt, fp, ped, a, mean))
        except Exception:
            continue

    events.sort(key=lambda e: e.announce_date)
    _EVENTS_CACHE[ric] = events
    return events


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


def write_jsonl(path: Path, records: List[ResultRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def write_csv(path: Path, records: List[ResultRecord]) -> None:
    """
    Write records to CSV (same fields as ResultRecord).
    Lists/None are handled as simple strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ResultRecord.__annotations__.keys()))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))



def count_lines_for_progress(path: Path, max_markets: Optional[int]) -> int:
    """
    tqdm total:
      - if --max-markets is set, use that
      - otherwise count physical file lines (fast enough for typical JSONL files)
    """
    if max_markets is not None:
        return max_markets
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


# =========================
# Main
# =========================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Polymarket earnings markets vs Refinitiv Eikon EPS actual/estimates."
    )
    p.add_argument(
        "--markets",
        type=str,
        default=str(DEFAULT_MARKETS_PATH),
        help=f"Path to markets.jsonl (default: {DEFAULT_MARKETS_PATH})",
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
    p.add_argument("--validation-dir", type=str, default=str(DEFAULT_VALIDATION_DIR), help="Output dir for jsonl results")
    p.add_argument("--event-pre-days", type=int, default=DEFAULT_EVENT_PRE_DAYS)
    p.add_argument("--event-post-days", type=int, default=DEFAULT_EVENT_POST_DAYS)
    p.add_argument("--max-event-distance-days", type=int, default=DEFAULT_MAX_EVENT_DISTANCE_DAYS)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    require_tqdm()
    setup_logging()
    setup_warnings_suppression()

    args = parse_args(argv)
    fail_fast = (not args.no_fail_fast)

    markets_path = Path(args.markets)
    validation_dir = Path(args.validation_dir)

    out_correct = validation_dir / "correct.jsonl"
    out_incorrect = validation_dir / "incorrectly_resolved.jsonl"
    out_unmatched = validation_dir / "unmatched.jsonl"
    out_correct_csv = validation_dir / "correct.csv"
    out_incorrect_csv = validation_dir / "incorrectly_resolved.csv"
    out_unmatched_csv = validation_dir / "unmatched.csv"


    # Buckets (always written)
    correct_records: List[ResultRecord] = []
    incorrect_records: List[ResultRecord] = []
    unmatched_records: List[ResultRecord] = []

    # Summary counters
    lines_scanned = 0          # physical lines scanned (up to max_markets)
    markets_parsed = 0         # JSON objects successfully parsed
    considered = 0             # earnings-like markets
    ignored_non_candidate = 0  # parsed JSON objects ignored by candidate filter
    matched = 0                # matched to event+actual+estimate
    correct_n = 0
    incorrect_n = 0
    unmatched_n = 0            # considered but failed matching somewhere

    if not markets_path.exists():
        LOG.error("Markets file not found: %s", markets_path)
        return 2

    # Resolve app key
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

    # Init Eikon (catastrophic if fails)
    try:
        init_eikon(app_key, eikon_port=args.eikon_port, require_proxy=(not args.skip_proxy_check))
    except Exception as exc:
        LOG.error("Eikon initialization failed: %s", exc)
        return 2

    total_for_bar = count_lines_for_progress(markets_path, args.max_markets)

    try:
        with tqdm(total=total_for_bar, desc="Validating markets", unit="line") as pbar:
            with markets_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    if args.max_markets is not None and line_no > args.max_markets:
                        break

                    lines_scanned += 1
                    pbar.update(1)

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        raw = json.loads(line)
                    except Exception:
                        # Keep output clean: silently skip invalid JSON lines
                        continue

                    markets_parsed += 1

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
                        estimate_used=None,
                        estimate_used_source=None,
                        surprise=None,
                        label=None,
                        expected_resolution=None,
                        match_method=None,
                        status="UNMATCHED",
                        skip_reason=None,
                    )

                    # Unmatched conditions (considered markets only)
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

                    ric = ticker_to_ric_best_effort(ticker, fail_fast=fail_fast)
                    if not ric:
                        base.skip_reason = "no_ric"
                        unmatched_records.append(base)
                        unmatched_n += 1
                        continue
                    base.ric = ric

                    events = fetch_eps_events(ric, fail_fast=fail_fast)
                    if not events:
                        base.skip_reason = "no_events_returned"
                        unmatched_records.append(base)
                        unmatched_n += 1
                        continue

                    ev, method = match_event_by_anchor_date(
                        events,
                        anchor_dt,
                        pre_days=args.event_pre_days,
                        post_days=args.event_post_days,
                        max_distance_days=args.max_event_distance_days,
                    )
                    if not ev:
                        base.skip_reason = "no_event_match"
                        unmatched_records.append(base)
                        unmatched_n += 1
                        continue

                    base.matched_announce_date = _to_date_iso(ev.announce_date)
                    base.matched_fperiod = ev.fperiod
                    base.matched_period_end_date = _to_date_iso(ev.period_end_date)
                    base.eikon_actual_eps = ev.actual_eps
                    base.eikon_eps_mean_estimate = ev.mean_estimate
                    base.match_method = method

                    if ev.actual_eps is None:
                        base.skip_reason = "no_actual_eps"
                        unmatched_records.append(base)
                        unmatched_n += 1
                        continue

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
                        continue

                    # Matched at this point
                    matched += 1

                    base.estimate_used = estimate_used
                    base.estimate_used_source = estimate_used_source

                    lab = decide_label(float(ev.actual_eps), estimate_used)
                    base.label = lab
                    base.surprise = float(ev.actual_eps) - estimate_used
                    base.expected_resolution = expected_resolution_from_label(
                        lab,
                        yes_semantics or "YES_MEANS_BEAT",
                        tie_counts_as or "NO",
                    )

                    if base.expected_resolution == resolved_outcome:
                        base.status = "MATCHED_CORRECT"
                        correct_records.append(base)
                        correct_n += 1
                    else:
                        base.status = "MATCHED_INCORRECT"
                        incorrect_records.append(base)
                        incorrect_n += 1

    except FatalEikonNetworkError as exc:
        # Catastrophic: write partial outputs and print one error
        write_jsonl(out_correct, correct_records)
        write_jsonl(out_incorrect, incorrect_records)
        write_jsonl(out_unmatched, unmatched_records)

        write_csv(out_correct_csv, correct_records)
        write_csv(out_incorrect_csv, incorrect_records)
        write_csv(out_unmatched_csv, unmatched_records)


        LOG.error(
            "%s\n\n"
            "Partial outputs were written. Fix likely involves:\n"
            "  - Ensure Workspace/Eikon is running AND logged in\n"
            "  - Verify the API proxy is running (port 9000/9060)\n"
            "  - If on VPN/corporate proxy, allowlist/firewall may be required\n",
            exc,
        )
        return 2

    # Always write all three files
    write_jsonl(out_correct, correct_records)
    write_jsonl(out_incorrect, incorrect_records)
    write_jsonl(out_unmatched, unmatched_records)

    write_csv(out_correct_csv, correct_records)
    write_csv(out_incorrect_csv, incorrect_records)
    write_csv(out_unmatched_csv, unmatched_records)


    # End-of-run summary (prints once)
    summary = (
        "\n"
        "==================== SUMMARY ====================\n"
        f"Lines scanned:                       {lines_scanned}\n"
        f"Markets parsed (valid JSON):         {markets_parsed}\n"
        f"Ignored (non-earnings candidate):    {ignored_non_candidate}\n"
        f"Considered (earnings-like):          {considered}\n"
        f"Matched (event+actual+estimate):     {matched}\n"
        f"  - Correct:                         {correct_n}\n"
        f"  - Incorrectly resolved:            {incorrect_n}\n"
        f"Unmatched (skip_reason set):         {unmatched_n}\n"
        "\n"
        "Outputs:\n"
        f"  - {out_correct}\n"
        f"  - {out_incorrect}\n"
        f"  - {out_unmatched}\n"
        "=================================================\n"
    )
    tqdm.write(summary) if tqdm is not None else print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
