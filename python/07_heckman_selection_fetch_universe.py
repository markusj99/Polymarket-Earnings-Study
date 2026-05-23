#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
05_heckman_selection_fetch_universe.py

Heckman selection model universe pull for U.S. equities using Refinitiv
Eikon / Workspace.

UPDATED (2026-03-30)
--------------------
Main goals of this rewrite:
1. Increase company / event coverage for matching against Polymarket markets.
2. Keep the final universe restricted to U.S.-listed equities.
3. Preserve transparent, well-documented, batch-safe behaviour.
4. Capture Nasdaq-listed equities more completely by screening Nasdaq segment
   MICs as well as the operating MIC.
5. Avoid the repeated RuntimeWarning / FutureWarning messages observed in the
   prior run.

Relative to the earlier version, this script now:
- Screens a broader U.S. primary-listing universe using separate MIC passes.
- Explicitly includes Nasdaq segment MICs (XNGS, XNMS, XNCM) in addition to
  XNAS. In practice, many Nasdaq-listed common equities are assigned a segment
  MIC rather than the operating MIC, so screening only XNAS can severely
  undercount Nasdaq names.
- Supplements the screened universe with RICs already observed in the
  Polymarket dataset.
- Fetches static metadata for the combined seed set first, then filters the
  downstream fetch universe to U.S. equities.
- Writes a richer screener universe file with source flags.
- Uses safer concatenation and volatility logic to avoid pandas / numpy
  warnings from empty concat inputs and log(0).

Observed window logic
---------------------
The observed market window is still read from:
    data/markets/markets.jsonl
using:
- observed_start = second earliest startDate (exclude the first outlier market)
- observed_end   = maximum umaEndDate

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
import hashlib
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
DEFAULT_COMPLETE_DATASET_CSV = project_root() / "data" / "complete_dataset_long.csv"
DEFAULT_COMPLETE_DATASET_JSONL = project_root() / "data" / "complete_dataset_long.jsonl"

OUT_DIR = project_root() / "data" / "heckman_selection_model"
OUT_EVENTS_JSONL = OUT_DIR / "heckman_universe_events.jsonl"
OUT_EVENTS_CSV = OUT_DIR / "heckman_universe_events.csv"
OUT_COMPANIES_CSV = OUT_DIR / "heckman_universe_companies.csv"
OUT_MISSING_JSON = OUT_DIR / "heckman_missing_summary.json"
OUT_REPORT_TXT = OUT_DIR / "heckman_report.txt"
OUT_SCREEN_RICS_CSV = OUT_DIR / "screener_universe_rics.csv"
OUT_SCREEN_RICS_JSONL = OUT_DIR / "screener_universe_rics.jsonl"

RESUME_DIR = OUT_DIR / "_resume"
RESUME_VERSION = "2026-03-31-resume-v1"

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
DEFAULT_SCREEN_MAX_PAGES_PER_MIC = 25
DEFAULT_SCREEN_MAX_INSTRUMENTS_PER_MIC = 25000

# Use primary-listing style venues. Nasdaq is intentionally represented by both
# the operating MIC (XNAS) and the main segment MICs because the segment MICs
# often carry the actual listing in Refinitiv metadata.
DEFAULT_US_EQUITY_MICS = [
    "XNYS",  # NYSE
    "XASE",  # NYSE American / AMEX
    "XNAS",  # Nasdaq operating MIC / all markets
    "XNGS",  # Nasdaq Global Select Market
    "XNMS",  # Nasdaq Global Market
    "XNCM",  # Nasdaq Capital Market
]

# Optional alias expansion. If the user includes XNAS only, expand it to the
# segment MICs too so Nasdaq coverage is not accidentally incomplete.
MIC_EXPANSION_MAP: Dict[str, List[str]] = {
    "XNAS": ["XNAS", "XNGS", "XNMS", "XNCM"],
}

# Fallback suffixes sometimes seen on U.S.-listed equities in RIC format.
# This is used only as a conservative backup when static metadata is missing.
DEFAULT_US_RIC_SUFFIXES = [
    ".N",   # NYSE
    ".O",   # Nasdaq
    ".A",   # NYSE American
    ".P",   # NYSE Arca / regional
    ".K",   # Cboe / other U.S. venue variants seen in Refinitiv
]

EIKON_RETRIES = 6
EIKON_RETRY_BASE_SLEEP = 1.0

# =========================
# Helpers: JSON, CSV, DataFrame I/O
# =========================

def _json_default(x: Any) -> Any:
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
    )

def atomic_write_df_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def atomic_write_df_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = {k: _json_default(v) for k, v in r.to_dict().items()}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def stage_signature(
    *,
    stage_name: str,
    instruments: Sequence[str],
    fields: Sequence[Any],
    parameters: Dict[str, Any],
    batch_size: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "resume_version": RESUME_VERSION,
        "stage_name": stage_name,
        "instruments": [str(x) for x in instruments],
        "fields": [str(x) for x in fields],
        "parameters": parameters,
        "batch_size": int(batch_size),
        "extra": extra or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

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


def stringify_list(xs: Sequence[str]) -> str:
    return ", ".join(str(x) for x in xs)


def default_complete_dataset_path() -> Path:
    if DEFAULT_COMPLETE_DATASET_CSV.exists():
        return DEFAULT_COMPLETE_DATASET_CSV
    return DEFAULT_COMPLETE_DATASET_JSONL


def normalize_ric_string(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s or s in {"NA", "NAN", "NULL", "NONE"}:
        return None
    return s


def normalize_ticker_string(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s or s in {"NA", "NAN", "NULL", "NONE"}:
        return None
    return s


def looks_like_us_ric(ric: Optional[str], us_suffixes: Sequence[str]) -> bool:
    if ric is None:
        return False
    ric_u = ric.upper()
    return any(ric_u.endswith(sfx.upper()) for sfx in us_suffixes)


def concat_non_empty(parts: Sequence[pd.DataFrame], empty_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    usable = [x for x in parts if isinstance(x, pd.DataFrame) and not x.empty]
    if usable:
        return pd.concat(usable, ignore_index=True)
    if empty_columns is None:
        return pd.DataFrame()
    return pd.DataFrame(columns=list(empty_columns))


def expand_exchange_mics(exchange_mics: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for mic in exchange_mics:
        mic_u = str(mic).strip().upper()
        if not mic_u:
            continue
        expanded = MIC_EXPANSION_MAP.get(mic_u, [mic_u])
        for item in expanded:
            item_u = str(item).strip().upper()
            if item_u and item_u not in seen:
                seen.add(item_u)
                out.append(item_u)
    return out


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


@dataclass
class SeedUniverseSummary:
    n_screen_rics_raw: int
    n_polymarket_seed_rics_raw: int
    n_combined_seed_rics_raw: int
    n_static_rows_raw: int
    n_static_rows_normalized: int
    n_us_equity_rics_after_static_filter: int
    n_us_equity_rics_after_fallback_filter: int


@dataclass
class ScreenStats:
    label: str
    pages_processed: int
    instruments_collected: int
    stop_reason: str


def _parse_iso_dt(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    try:
        return pd.to_datetime(str(s), errors="coerce", utc=True).to_pydatetime()
    except Exception:
        return None


def compute_observed_window_from_markets(markets_jsonl: Path) -> ObservedWindow:
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
    observed_start = starts_sorted[1]
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
) -> Tuple[pd.DataFrame, List[str], bool]:
    problems: List[str] = []

    def _call(batch: List[str], depth: int) -> Tuple[pd.DataFrame, bool]:
        if not batch:
            return pd.DataFrame(), False

        res = client.get_data(batch, fields, parameters)

        # Non-fatal field/data issues can still be logged, but they do NOT mean
        # the batch must be re-fetched.
        if res.err is not None:
            problems.append(f"batch_err size={len(batch)} err={res.err}")

        # No exception => request completed, even if the returned frame is empty
        # or values are missing.
        if res.exc is None:
            if isinstance(res.df, pd.DataFrame):
                return res.df, False
            return pd.DataFrame(), False

        problems.append(f"batch_failed size={len(batch)} exc={res.exc}")

        if "400" in res.exc and len(batch) > 1 and depth > 0:
            mid = len(batch) // 2
            left_df, left_failed = _call(batch[:mid], depth - 1)
            right_df, right_failed = _call(batch[mid:], depth - 1)
            return concat_non_empty([left_df, right_df]), (left_failed or right_failed)

        return pd.DataFrame(), True

    df, had_failure = _call(instruments, max_split_depth)
    return df, problems, had_failure

def fetch_batched_resumable(
    client: EikonClient,
    *,
    stage_name: str,
    instruments: List[str],
    fields: List[Any],
    parameters: Dict[str, Any],
    batch_size: int,
    empty_columns: Optional[Sequence[str]] = None,
    max_split_depth: int = 4,
    progress_desc: Optional[str] = None,
    extra_signature_data: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    instruments = [str(x) for x in instruments]
    sig = stage_signature(
        stage_name=stage_name,
        instruments=instruments,
        fields=fields,
        parameters=parameters,
        batch_size=batch_size,
        extra=extra_signature_data,
    )

    stage_dir = RESUME_DIR / stage_name / sig
    manifest_path = stage_dir / "manifest.json"
    full_csv_path = stage_dir / f"{stage_name}.csv"
    full_jsonl_path = stage_dir / f"{stage_name}.jsonl"

    # Fast path: completed stage already consolidated
    manifest = read_json(manifest_path, default={})
    if manifest.get("stage_complete") and full_csv_path.exists():
        df = pd.read_csv(full_csv_path)
        return df, manifest.get("problems", [])

    stage_dir.mkdir(parents=True, exist_ok=True)

    batches = list(chunked(instruments, batch_size))
    problems: List[str] = []
    parts: List[pd.DataFrame] = []

    pbar = tqdm(total=len(batches), desc=(progress_desc or stage_name), unit="batch") if tqdm else None
    try:
        for batch_idx, batch in enumerate(batches):
            batch_meta_path = stage_dir / f"batch_{batch_idx:05d}.json"
            batch_csv_path = stage_dir / f"batch_{batch_idx:05d}.csv"

            batch_meta = read_json(batch_meta_path, default=None)

            # Resume path: previously completed batch
            if (
                isinstance(batch_meta, dict)
                and batch_meta.get("success") is True
                and batch_meta.get("requested_instruments") == batch
            ):
                if batch_csv_path.exists():
                    try:
                        df_old = pd.read_csv(batch_csv_path)
                        if not df_old.empty:
                            parts.append(df_old)
                    except Exception:
                        # Corrupt cached CSV => re-fetch this batch
                        batch_meta = None
                if batch_meta is not None:
                    if pbar is not None:
                        pbar.update(1)
                    continue

            dfb, probs, had_failure = get_data_batched_split(
                client,
                batch,
                fields,
                dict(parameters),
                max_split_depth=max_split_depth,
            )
            problems.extend(probs)

            meta_out = {
                "stage_name": stage_name,
                "signature": sig,
                "batch_index": batch_idx,
                "requested_instruments": batch,
                "requested_count": len(batch),
                "success": (not had_failure),
                "row_count": int(len(dfb)),
                "completed_at_utc": utc_now_iso() if not had_failure else None,
                "problems_tail": probs[-20:],
            }

            if had_failure:
                atomic_write_json(batch_meta_path, meta_out)
            else:
                # Success, even if dfb is empty
                if not dfb.empty:
                    atomic_write_df_csv(batch_csv_path, dfb)
                    parts.append(dfb)
                else:
                    if batch_csv_path.exists():
                        batch_csv_path.unlink()
                atomic_write_json(batch_meta_path, meta_out)

            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    completed_batches = 0
    for batch_idx in range(len(batches)):
        meta = read_json(stage_dir / f"batch_{batch_idx:05d}.json", default={})
        if isinstance(meta, dict) and meta.get("success") is True:
            completed_batches += 1

    full_df = concat_non_empty(parts, empty_columns=empty_columns)

    manifest = {
        "stage_name": stage_name,
        "signature": sig,
        "resume_version": RESUME_VERSION,
        "requested_batches": len(batches),
        "completed_batches": completed_batches,
        "stage_complete": completed_batches == len(batches),
        "batch_size": int(batch_size),
        "generated_at_utc": utc_now_iso(),
        "problems": problems[-200:],
    }

    if manifest["stage_complete"]:
        atomic_write_df_csv(full_csv_path, full_df)
        atomic_write_df_jsonl(full_jsonl_path, full_df)

    atomic_write_json(manifest_path, manifest)
    return full_df, problems

# =========================
# SCREEN helpers
# =========================

def build_us_equity_screen_for_mic(mic: str) -> str:
    """
    Build a screen expression for active public primary equities whose primary
    quote is on a given exchange MIC.

    Important practical note:
    Nasdaq coverage often requires segment MICs such as XNGS, XNMS, and XNCM,
    not just the operating MIC XNAS.
    """
    mic_clean = str(mic).strip().upper()
    return (
        'SCREEN(U(IN(Equity(active,public,primary,countryprimaryquote))/*UNV:Public*/),'
        f' IN(TR.ExchangeMarketIdCode,"{mic_clean}"))'
    )


def screen_rics_from_expression(
    client: EikonClient,
    *,
    label: str,
    screen_expr: str,
    page_size: int,
    max_pages: int,
    max_instruments: int,
) -> Tuple[List[str], List[str], ScreenStats]:
    problems: List[str] = []
    fields = ["TR.CommonName", "TR.ExchangeMarketIdCode", "TR.PrimaryExchangeName", "TR.TickerSymbol"]

    seen: set[str] = set()
    pages = 0
    start = 0
    stop_reason = "unknown"

    pbar = tqdm(desc=f"Screening {label} (pages)", unit="page") if tqdm else None
    try:
        for _ in range(max_pages):
            pages += 1
            params = {"StartNum": start, "EndNum": start + page_size}
            res = client.get_data(screen_expr, fields, params)
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
                    f"[{label}] SCREEN appears to ignore StartNum/EndNum "
                    f"(page1 rows={len(dfp)} > page_size={page_size}). Stopping after page 1 to avoid repeats."
                )
                break

            if new_count == 0:
                stop_reason = "no_new_instruments_repeat_page"
                problems.append(f"[{label}] SCREEN pagination appears to repeat pages (no new instruments). Stopped.")
                break

            if len(dfp) < page_size:
                stop_reason = "short_last_page"
                break

            if len(seen) >= max_instruments:
                stop_reason = f"hit_max_instruments_{max_instruments}"
                problems.append(f"[{label}] Hit max_instruments cap ({max_instruments}).")
                break

            start += page_size
        else:
            stop_reason = f"hit_max_pages_{max_pages}"
            problems.append(f"[{label}] Hit max_pages cap ({max_pages}).")
    finally:
        if pbar is not None:
            pbar.close()

    stats = ScreenStats(
        label=label,
        pages_processed=pages,
        instruments_collected=len(seen),
        stop_reason=stop_reason,
    )
    return sorted(seen), problems, stats


def screen_us_equity_rics(
    client: EikonClient,
    *,
    exchange_mics: Sequence[str],
    page_size: int,
    max_pages_per_mic: int,
    max_instruments_per_mic: int,
) -> Tuple[Dict[str, List[str]], List[str], List[ScreenStats]]:
    all_problems: List[str] = []
    all_stats: List[ScreenStats] = []
    by_mic: Dict[str, List[str]] = {}

    for mic in exchange_mics:
        label = f"US equities on {mic}"
        screen_expr = build_us_equity_screen_for_mic(mic)
        rics_mic, probs_mic, stats_mic = screen_rics_from_expression(
            client,
            label=label,
            screen_expr=screen_expr,
            page_size=page_size,
            max_pages=max_pages_per_mic,
            max_instruments=max_instruments_per_mic,
        )
        by_mic[mic] = rics_mic
        all_problems.extend(probs_mic)
        all_stats.append(stats_mic)

    return by_mic, all_problems, all_stats


# =========================
# Polymarket seed extraction
# =========================

def read_complete_dataset_seeds(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    problems: List[str] = []

    if not path.exists():
        problems.append(f"Complete dataset file not found: {path}")
        return pd.DataFrame(columns=["ric", "ticker"]), problems

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            problems.append(f"Unsupported complete dataset file extension: {path.suffix}")
            return pd.DataFrame(columns=["ric", "ticker"]), problems
    except Exception as exc:
        problems.append(f"Failed to read complete dataset seeds from {path}: {exc}")
        return pd.DataFrame(columns=["ric", "ticker"]), problems

    out = pd.DataFrame({
        "ric": df["ric"] if "ric" in df.columns else None,
        "ticker": df["ticker"] if "ticker" in df.columns else None,
    })
    out["ric"] = out["ric"].map(normalize_ric_string)
    out["ticker"] = out["ticker"].map(normalize_ticker_string)
    out = out.dropna(subset=["ric"]).drop_duplicates(subset=["ric"]).reset_index(drop=True)

    return out, problems


# =========================
# Fetchers (batched + split)
# =========================

def fetch_static_metadata(client: EikonClient, rics: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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
    return fetch_batched_resumable(
        client,
        stage_name="static_metadata",
        instruments=rics,
        fields=fields,
        parameters={},
        batch_size=BATCH_STATIC,
        progress_desc="Static metadata (batches)",
    )

def fetch_marketcap_and_analysts_series(
    client: EikonClient,
    rics: List[str],
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, List[str]]:
    fields: List[Any] = [
        _tr_field("TR.CompanyMarketCap", {"Curn": "USD"}),
        "TR.CompanyMarketCap.date",
        "TR.NumberOfAnalysts",
        "TR.NumberOfAnalysts.date",
    ]
    params = {"SDate": start_date.isoformat(), "EDate": end_date.isoformat(), "Frq": "D"}

    return fetch_batched_resumable(
        client,
        stage_name="marketcap_analysts_series",
        instruments=rics,
        fields=fields,
        parameters=params,
        batch_size=BATCH_ASOF,
        progress_desc="Mcap+analysts series (batches)",
        extra_signature_data={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
    )

def fetch_daily_pv(client: EikonClient, rics: List[str], start_date: date, end_date: date) -> Tuple[pd.DataFrame, List[str]]:
    fields = ["TR.PriceClose", "TR.Volume", "TR.PriceClose.date"]
    params = {"SDate": start_date.isoformat(), "EDate": end_date.isoformat(), "Frq": "D"}

    return fetch_batched_resumable(
        client,
        stage_name="daily_price_volume",
        instruments=rics,
        fields=fields,
        parameters=params,
        batch_size=BATCH_PV,
        progress_desc="Daily Price/Volume (batches)",
        extra_signature_data={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
    )

def fetch_events_results(client: EikonClient, rics: List[str], start_date: date, end_date: date) -> Tuple[pd.DataFrame, List[str]]:
    fields = ["TR.EventStartDate", "TR.EventStartTime", "TR.EventType", "TR.EventTitle"]
    params = {"SDate": start_date.isoformat(), "EDate": end_date.isoformat(), "EventType": "RES", "RH": "IN", "CH": "Fd"}

    return fetch_batched_resumable(
        client,
        stage_name="events_res",
        instruments=rics,
        fields=fields,
        parameters=params,
        batch_size=BATCH_EVENTS,
        progress_desc="Events (RES) (batches)",
        extra_signature_data={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
    )

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


def select_us_equity_rics(
    static_norm: pd.DataFrame,
    *,
    combined_seed_rics: Sequence[str],
    exchange_mics: Sequence[str],
    us_suffixes: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:
    static_use = static_norm.copy()
    static_use["ric"] = static_use["ric"].map(normalize_ric_string)
    static_use["exchange_mic"] = static_use["exchange_mic"].map(normalize_ric_string)

    mic_set = {str(x).strip().upper() for x in exchange_mics}

    static_use["is_us_equity_from_static"] = (
        static_use["exchange_country"].fillna("").str.upper().eq("UNITED STATES OF AMERICA")
        | static_use["exchange_mic"].fillna("").isin(mic_set)
    )

    rics_from_static = set(static_use.loc[static_use["is_us_equity_from_static"], "ric"].dropna().astype(str))

    all_seed_rics = {str(r).strip().upper() for r in combined_seed_rics if str(r).strip()}
    static_ric_set = set(static_use["ric"].dropna().astype(str))
    missing_from_static = sorted(all_seed_rics - static_ric_set)
    fallback_rics = {r for r in missing_from_static if looks_like_us_ric(r, us_suffixes)}

    keep_rics = sorted(rics_from_static | fallback_rics)

    audit = pd.DataFrame({"ric": sorted(all_seed_rics)})
    audit["seen_in_static"] = audit["ric"].isin(static_ric_set).astype(int)
    audit["kept_from_static_rule"] = audit["ric"].isin(rics_from_static).astype(int)
    audit["kept_from_suffix_fallback"] = audit["ric"].isin(fallback_rics).astype(int)
    audit["kept_final_us_equity_universe"] = audit["ric"].isin(set(keep_rics)).astype(int)

    return keep_rics, audit


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
    if events.empty:
        events[out_col] = np.nan
        return events

    if series_long.empty:
        events[out_col] = np.nan
        return events

    series_long = series_long[["ric", "date", value_col]].copy()
    series_long = series_long.dropna(subset=["ric", "date"]).sort_values(["ric", "date"])

    out_parts: List[pd.DataFrame] = []
    for ric, ev in events.groupby("ric", sort=False):
        ev2 = ev.sort_values(asof_col).copy()
        s = series_long[series_long["ric"] == ric].copy()
        if s.empty:
            ev2[out_col] = np.nan
            out_parts.append(ev2)
            continue

        ev2["_asof_dt"] = pd.to_datetime(ev2[asof_col].astype(str), errors="coerce")
        s["_dt"] = pd.to_datetime(s["date"].astype(str), errors="coerce")
        s = s.dropna(subset=["_dt"]).sort_values("_dt")

        ev_missing = ev2[ev2["_asof_dt"].isna()].copy()
        if not ev_missing.empty:
            ev_missing[out_col] = np.nan
            ev_missing = ev_missing.drop(columns=["_asof_dt"], errors="ignore")
            out_parts.append(ev_missing)

        ev_non_missing = ev2[ev2["_asof_dt"].notna()].sort_values("_asof_dt").copy()
        if ev_non_missing.empty:
            continue

        merged = pd.merge_asof(
            ev_non_missing,
            s[["_dt", value_col]].rename(columns={"_dt": "_asof_merge_key"}),
            left_on="_asof_dt",
            right_on="_asof_merge_key",
            direction="backward",
        )
        merged[out_col] = merged[value_col]
        merged = merged.drop(columns=[value_col, "_asof_dt", "_asof_merge_key"], errors="ignore")
        out_parts.append(merged)

    out = concat_non_empty(out_parts, empty_columns=list(events.columns) + [out_col])
    return out


def _event_level_turnover_volatility(
    pv: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lookback_days: int,
    asof_col: str = "asof_date",
) -> pd.DataFrame:
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
        pv2_vol = pd.to_numeric(pv2["volume"], errors="coerce")

        px = pd.to_numeric(pv2["price"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        px[~np.isfinite(px) | (px <= 0.0)] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            log_px = np.log(px)
        lr = pd.Series(log_px, index=pv2.index, dtype="float64").diff()

        def _slice_mask(d0: date, d1: date) -> pd.Series:
            return (pv2["date"] >= d0) & (pv2["date"] <= d1)

        sums: List[float] = []
        means: List[float] = []
        vols: List[float] = []

        for _, row in ev2.iterrows():
            asof = row.get(asof_col)
            if pd.isna(asof):
                sums.append(np.nan)
                means.append(np.nan)
                vols.append(np.nan)
                continue
            asof_d = asof if isinstance(asof, date) else pd.to_datetime(str(asof), errors="coerce").date()
            w0 = asof_d - timedelta(days=int(lookback_days))
            w1 = asof_d

            m = _slice_mask(w0, w1)
            if not bool(m.any()):
                sums.append(np.nan)
                means.append(np.nan)
                vols.append(np.nan)
                continue

            vvals = pd.to_numeric(pv2_vol.loc[m], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            valid_v = vvals[np.isfinite(vvals)]
            if valid_v.size > 0:
                sums.append(float(valid_v.sum()))
                means.append(float(valid_v.mean()))
            else:
                sums.append(np.nan)
                means.append(np.nan)

            lrwin = lr.loc[m].replace([np.inf, -np.inf], np.nan).dropna()
            vols.append(float(lrwin.std(ddof=1)) if len(lrwin) >= 2 else np.nan)

        ev2["turnover_lookback_sum_volume_asof_evt"] = sums
        ev2["turnover_lookback_avg_daily_volume_asof_evt"] = means
        ev2["volatility_lookback_asof_evt"] = vols

        ev2["turnover_lookback_window_start_asof_evt"] = (
            pd.to_datetime(ev2[asof_col].astype(str), errors="coerce") - pd.to_timedelta(int(lookback_days), unit="D")
        ).dt.date.astype(str)
        ev2["turnover_lookback_window_end_asof_evt"] = pd.to_datetime(ev2[asof_col].astype(str), errors="coerce").dt.date.astype(str)

        out_parts.append(ev2)

    return concat_non_empty(out_parts, empty_columns=list(events.columns))


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
    p = argparse.ArgumentParser(description="Eikon data pull for Heckman selection model focused on U.S. equities.")

    p.add_argument("--app-key", nargs="?", const="__ENV__", default=None,
                   help="Eikon app key. If provided with no value, reads env EIKON_APP_KEY.")

    p.add_argument("--markets-jsonl", type=str, default=str(DEFAULT_MARKETS_JSONL),
                   help="Path to markets.jsonl used to compute observed window.")

    p.add_argument("--complete-dataset", type=str, default=str(default_complete_dataset_path()),
                   help="Path to complete_dataset_long.csv or complete_dataset_long.jsonl used to seed RICs already observed in Polymarket.")

    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    p.add_argument("--asof-buffer-days", type=int, default=DEFAULT_ASOF_BUFFER_DAYS)

    p.add_argument("--min-interval-s", type=float, default=0.35)

    p.add_argument("--exchange-mics", type=str, default=",".join(DEFAULT_US_EQUITY_MICS),
                   help="Comma-separated list of exchange MICs to screen separately. XNAS is automatically expanded to include XNGS, XNMS, and XNCM.")
    p.add_argument("--screen-page-size", type=int, default=DEFAULT_SCREEN_PAGE_SIZE)
    p.add_argument("--screen-max-pages-per-mic", type=int, default=DEFAULT_SCREEN_MAX_PAGES_PER_MIC)
    p.add_argument("--screen-max-instruments-per-mic", type=int, default=DEFAULT_SCREEN_MAX_INSTRUMENTS_PER_MIC)

    p.add_argument("--no-polymarket-seeds", action="store_true",
                   help="Do not supplement the screener universe with RICs already present in complete_dataset_long.")
    p.add_argument("--max-rics", type=int, default=None, help="TEST MODE: limit number of downstream RICs after U.S.-equity filtering.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    require_tqdm()
    setup_logging_quiet()
    setup_warnings_suppression()

    args = parse_args(argv)

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

    requested_exchange_mics = [x.strip().upper() for x in str(args.exchange_mics).split(",") if x.strip()]
    exchange_mics = expand_exchange_mics(requested_exchange_mics)
    if not exchange_mics:
        print("ERROR: --exchange-mics produced an empty list.")
        return 2

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

    by_mic, screen_problems, screen_stats = screen_us_equity_rics(
        client,
        exchange_mics=exchange_mics,
        page_size=int(args.screen_page_size),
        max_pages_per_mic=int(args.screen_max_pages_per_mic),
        max_instruments_per_mic=int(args.screen_max_instruments_per_mic),
    )

    screen_ric_records: List[Dict[str, Any]] = []
    screen_ric_set: set[str] = set()
    for mic, rics_mic in by_mic.items():
        for ric in rics_mic:
            ric_norm = normalize_ric_string(ric)
            if ric_norm is None:
                continue
            screen_ric_set.add(ric_norm)
            screen_ric_records.append({
                "ric": ric_norm,
                "source_exchange_mic": mic,
                "from_exchange_screen": 1,
            })

    seed_problems: List[str] = []
    polymarket_seed_df = pd.DataFrame(columns=["ric", "ticker"])
    if not args.no_polymarket_seeds:
        polymarket_seed_df, seed_problems = read_complete_dataset_seeds(Path(args.complete_dataset))

    polymarket_seed_rics = sorted(set(polymarket_seed_df["ric"].dropna().astype(str)))
    combined_seed_rics = sorted(screen_ric_set | set(polymarket_seed_rics))

    if not combined_seed_rics:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_report(OUT_REPORT_TXT, sections=[
            ("Observed window from markets.jsonl", "\n".join([
                f"Markets JSONL:             {window.markets_path}",
                f"Excluded outlier start:    {window.excluded_outlier_start}",
                f"Observed start / end:      {observed_start} .. {observed_end}",
                f"Markets total / used:      {window.n_markets_total} / {window.n_markets_used}",
            ])),
            ("Fatal", "Combined seed universe is empty after exchange screening and Polymarket seed extraction."),
            ("Problems", "\n".join(screen_problems + seed_problems) if (screen_problems or seed_problems) else "(none)"),
        ])
        return 2

    static_raw, static_problems = fetch_static_metadata(client, combined_seed_rics)
    static_norm = _normalize_static(static_raw)
    us_equity_rics, us_equity_audit = select_us_equity_rics(
        static_norm,
        combined_seed_rics=combined_seed_rics,
        exchange_mics=exchange_mics,
        us_suffixes=DEFAULT_US_RIC_SUFFIXES,
    )

    if args.max_rics is not None:
        us_equity_rics = us_equity_rics[: int(args.max_rics)]

    if not us_equity_rics:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_report(OUT_REPORT_TXT, sections=[
            ("Observed window from markets.jsonl", "\n".join([
                f"Markets JSONL:             {window.markets_path}",
                f"Excluded outlier start:    {window.excluded_outlier_start}",
                f"Observed start / end:      {observed_start} .. {observed_end}",
                f"Markets total / used:      {window.n_markets_total} / {window.n_markets_used}",
            ])),
            ("Fatal", "No U.S. equity RICs remained after applying the U.S.-equity filter."),
            ("Problems", "\n".join(screen_problems + seed_problems + static_problems) if (screen_problems or seed_problems or static_problems) else "(none)"),
        ])
        return 2

    source_flags = us_equity_audit.copy()
    source_flags["from_polymarket_seed"] = source_flags["ric"].isin(set(polymarket_seed_rics)).astype(int)
    source_flags["from_exchange_screen_any"] = source_flags["ric"].isin(screen_ric_set).astype(int)
    by_mic_sets = {mic: set(vals) for mic, vals in by_mic.items()}
    source_flags["source_exchange_mics"] = source_flags["ric"].map(
        lambda ric: ",".join(sorted([mic for mic, vals in by_mic_sets.items() if ric in vals]))
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_flags.sort_values(["kept_final_us_equity_universe", "ric"], ascending=[False, True]).to_csv(
        OUT_SCREEN_RICS_CSV, index=False, encoding="utf-8"
    )
    write_df_jsonl(OUT_SCREEN_RICS_JSONL, source_flags.sort_values("ric").reset_index(drop=True))

    ev_raw, ev_problems = fetch_events_results(client, us_equity_rics, observed_start, observed_end)
    events = _normalize_events(ev_raw)
    events = events[events["ric"].isin(set(us_equity_rics))].copy()

    if events.empty:
        write_report(OUT_REPORT_TXT, sections=[
            ("Observed window from markets.jsonl", "\n".join([
                f"Markets JSONL:             {window.markets_path}",
                f"Excluded outlier start:    {window.excluded_outlier_start}",
                f"Observed start / end:      {observed_start} .. {observed_end}",
                f"Markets total / used:      {window.n_markets_total} / {window.n_markets_used}",
            ])),
            ("Fatal", "No RES events returned by Eikon in observed window for the filtered U.S. equity universe."),
            ("Top problems / warnings", "\n".join(screen_problems + seed_problems + static_problems + ev_problems) if (screen_problems or seed_problems or static_problems or ev_problems) else "(none)"),
        ])
        return 2

    events["asof_date"] = events["event_date"].apply(lambda d: (d - timedelta(days=2)) if isinstance(d, date) else pd.NaT)
    events["retrieved_at_utc"] = utc_now_iso()
    events["observed_window_start_utc"] = observed_start.isoformat()
    events["observed_window_end_utc"] = observed_end.isoformat()

    min_asof = events["asof_date"].min()
    max_asof = events["asof_date"].max()
    if isinstance(min_asof, date) and isinstance(max_asof, date):
        pv_start = min_asof - timedelta(days=lookback_days + buffer_days)
        pv_end = max_asof
    else:
        pv_start = observed_start - timedelta(days=lookback_days + buffer_days)
        pv_end = observed_end

    pv_raw, pv_problems = fetch_daily_pv(client, us_equity_rics, pv_start, pv_end)
    pv = _normalize_pv(pv_raw)
    pv = pv[pv["ric"].isin(set(us_equity_rics))].copy()

    series_start = pv_start - timedelta(days=asof_buffer_days)
    series_end = pv_end
    asof_raw, asof_problems = fetch_marketcap_and_analysts_series(client, us_equity_rics, series_start, series_end)
    mcap_long, an_long = normalize_mcap_analysts_long(asof_raw)
    mcap_long = mcap_long[mcap_long["ric"].isin(set(us_equity_rics))].copy()
    an_long = an_long[an_long["ric"].isin(set(us_equity_rics))].copy()

    companies = static_norm[static_norm["ric"].isin(set(us_equity_rics))].copy()
    companies["retrieved_at_utc"] = utc_now_iso()

    events = _last_value_asof_per_event(mcap_long, events, value_col="market_cap_usd", out_col="market_cap_usd_asof_evt")
    events = _last_value_asof_per_event(an_long, events, value_col="analysts", out_col="analysts_covering_asof_evt")
    events = _event_level_turnover_volatility(pv, events, lookback_days=lookback_days)

    events = events.merge(companies, on="ric", how="left", suffixes=("", "_company"))

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

    screen_lines: List[str] = [
        f"Requested exchange MICs:     {stringify_list(requested_exchange_mics)}",
        f"Expanded exchange MICs:      {stringify_list(exchange_mics)}",
        f"Page size per MIC:           {int(args.screen_page_size)}",
        f"Max pages per MIC:           {int(args.screen_max_pages_per_mic)}",
        f"Max instruments per MIC:     {int(args.screen_max_instruments_per_mic)}",
        f"Raw unique RICs from screen: {len(screen_ric_set)}",
    ]
    for st in screen_stats:
        screen_lines.append(
            f"  - {st.label:<26} pages={st.pages_processed:<3} rics={st.instruments_collected:<6} stop={st.stop_reason}"
        )

    seed_summary = SeedUniverseSummary(
        n_screen_rics_raw=len(screen_ric_set),
        n_polymarket_seed_rics_raw=len(polymarket_seed_rics),
        n_combined_seed_rics_raw=len(combined_seed_rics),
        n_static_rows_raw=int(len(static_raw)),
        n_static_rows_normalized=int(len(static_norm)),
        n_us_equity_rics_after_static_filter=int(us_equity_audit["kept_from_static_rule"].sum()),
        n_us_equity_rics_after_fallback_filter=int(us_equity_audit["kept_final_us_equity_universe"].sum()),
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
    sections.append(("SCREEN stats (separate MIC passes)", "\n".join(screen_lines)))
    sections.append((
        "Seed-universe expansion summary",
        "\n".join([
            f"Polymarket seed file:               {args.complete_dataset if not args.no_polymarket_seeds else '(disabled)'}",
            f"Raw screen RICs:                    {seed_summary.n_screen_rics_raw}",
            f"Raw Polymarket-seed RICs:           {seed_summary.n_polymarket_seed_rics_raw}",
            f"Combined raw seed RICs:             {seed_summary.n_combined_seed_rics_raw}",
            f"Static metadata rows (raw):         {seed_summary.n_static_rows_raw}",
            f"Static metadata rows (normalized):  {seed_summary.n_static_rows_normalized}",
            f"U.S. equities kept from static:     {seed_summary.n_us_equity_rics_after_static_filter}",
            f"Final U.S.-equity RIC universe:     {seed_summary.n_us_equity_rics_after_fallback_filter}",
            f"Downstream U.S.-equity RICs used:   {len(us_equity_rics)}",
        ])
    ))
    sections.append((
        "Output counts",
        "\n".join([
            f"Companies rows written:             {len(companies)}",
            f"Event rows written:                 {len(events)}",
            f"PV rows normalized:                 {len(pv)}",
            f"Market-cap series rows:             {len(mcap_long)}",
            f"Analyst series rows:                {len(an_long)}",
        ])
    ))
    sections.append((
        "Top problems / warnings",
        "\n".join([
            *(screen_problems or ["(none from screener)"]),
            *(seed_problems or ["(none from Polymarket seed extraction)"]),
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
        f"U.S. equity RICs: {len(us_equity_rics)}\n"
        "As-of rule:       event_date - 2 days\n"
        "=============================================\n"
    )
    tqdm.write(msg) if tqdm is not None else print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
