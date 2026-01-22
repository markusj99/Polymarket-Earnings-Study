#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_build_resolved_markets.py

Polymarket Earnings (Resolved Markets) — Dataset Builder
========================================================

What you asked for (implemented)
--------------------------------
1) Standalone-friendly:
   - All default settings are defined at the TOP of the script (edit them and run).

2) Import-safe + callable:
   - Core pipeline is `build_resolved_earnings_dataset(...)`.
   - Another script can call it and override *only* what it wants (e.g., out_dir).

3) Output path configurable:
   - `out_dir` can be passed from another script.
   - When run standalone, it uses the default OUT_DIR defined below.

4) Heartbeat removed:
   - No heartbeat; tqdm covers progress visibility.

5) Optional CSV:
   - `WRITE_CSV` default at top
   - Override with `write_csv=...` in function call, or `--no-csv` in CLI.

Outputs
-------
Always:
- markets.jsonl
- discarded_markets.jsonl
- summary.txt

Optional:
- markets.csv  (when write_csv=True)

Dependencies
------------
pip install requests tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests
from tqdm import tqdm


# =============================================================================
# DEFAULT CONFIG (edit these when running standalone)
# =============================================================================

# -------------------------
# TEST MODE, limits pages/markets fetched
# -------------------------
TEST = False
TEST_MAX_PAGES = 2
TEST_MAX_MARKETS = 10

# -------------------------
# INCREMENTAL MODE, if False will re-fetch everything
# -------------------------
INCREMENTAL_MODE = True

# -------------------------
# OUTPUT
# -------------------------
# Default output dir is RELATIVE to this script, for portability/sharing.
# You can edit this for standalone runs or pass out_dir=... when calling from another script.
OUT_DIR = Path(__file__).resolve().parent / "data" / "markets"

# Output filenames
FNAME_MARKETS_JSONL = "markets.jsonl"
FNAME_MARKETS_CSV = "markets.csv"
FNAME_DISCARDED_JSONL = "discarded_markets.jsonl"
FNAME_SUMMARY_TXT = "summary.txt"

# Whether to write CSV output (JSONL + summary always written)
WRITE_CSV = True

# -------------------------
# API / performance
# -------------------------
GAMMA = "https://gamma-api.polymarket.com"
HTTP_TIMEOUT = 25
RETRIES = 3
RETRY_SLEEP_S = 0.8
PRINT_RETRY_DETAILS = True

# /markets pagination
PAGE_LIMIT = 200

# Threads for parallel detail fetch
MAX_WORKERS = 15

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "polymarket-earnings-resolved-fetcher/2.1",
}

# -------------------------
# Fetch controls, date range
# -------------------------
END_DATE_MIN: Optional[str] = None
END_DATE_MAX: Optional[str] = None

# Include Gamma related_tags expansion
RELATED_TAGS = True

# Console verbosity
VERBOSE = True


# =============================================================================
# Market classifier (rule-based filter for earnings beat estimate markets)
# =============================================================================

TICKER_RE = re.compile(r"\(([A-Z0-9]{1,7}(?:[.\-][A-Z0-9]{1,4})?)\)")
BEAT_RE = re.compile(r"\bbeat\b", re.IGNORECASE)
EARNINGS_OR_EPS_RE = re.compile(r"\b(earnings|eps)\b", re.IGNORECASE)
ESTIMATE_RE = re.compile(r"\b(consensus|estimate|estimated)\b", re.IGNORECASE)
YESNO_SET = {"yes", "no"}

# -------------------------------------------------------------------------
# Legacy / odd-structure markets
# -------------------------------------------------------------------------
# Market 549606 ("will-broadcom-beat-q2-earnings-estimate") is a known legacy market:
# - The question does NOT contain "(AVGO)" so the normal ticker extractor fails.
# - The description contains "(NASDAQ: AVGO)" so we can safely map it.
SPECIAL_TICKER_BY_MARKET_ID: Dict[str, str] = {
    "549606": "AVGO",
}

# Description patterns like "NASDAQ: AVGO", "NYSE: BRK.A", etc.
EXCHANGE_TICKER_RE = re.compile(
    r"\b(?:NASDAQ|NYSE|AMEX|ARCA|BATS|CBOE|OTC|TSX|LSE|ASX|HKEX|FWB|SWX|TSE)\s*:\s*"
    r"([A-Z0-9]{1,7}(?:[.\-][A-Z0-9]{1,4})?)\b"
)

# =============================================================================
# Utilities
# =============================================================================

def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to disk (write tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def log(msg: str, *, verbose: bool = True) -> None:
    """Simple timestamped log."""
    if not verbose:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 strings (including 'Z') into UTC datetime objects."""
    if not s or not isinstance(s, str):
        return None
    try:
        ss = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def format_dt_utc(dt: Optional[datetime]) -> str:
    """Render datetime as readable UTC string for summary output."""
    if not dt:
        return "N/A"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def parse_json_list_maybe(v: Any) -> Optional[List[Any]]:
    """
    Normalize Gamma fields that may be:
      - native list
      - JSON-encoded list string
    """
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                out = json.loads(s)
                return out if isinstance(out, list) else None
            except Exception:
                return None
    return None


def normalize_yes_no(outcomes: List[Any]) -> Optional[List[str]]:
    """Return standardized order ["Yes","No"] iff outcomes are exactly yes/no (case-insensitive)."""
    outs = [str(x).strip() for x in outcomes]
    if len(outs) != 2:
        return None
    lower = [o.lower() for o in outs]
    if set(lower) != YESNO_SET:
        return None
    return ["Yes", "No"]


def extract_ticker_from_question(question: str) -> Optional[str]:
    """Extract ticker from '(TICKER)' in question."""
    if not isinstance(question, str):
        return None
    m = TICKER_RE.search(question.strip())
    if not m:
        return None
    t = (m.group(1) or "").strip()
    return t or None

def extract_ticker_from_description(description: str) -> Optional[str]:
    """
    Fallback ticker extraction from description.

    Legacy / early Polymarket markets sometimes omit '(TICKER)' from the question,
    but include a canonical exchange reference in the description like:
      - "(NASDAQ: AVGO)"
      - "(NYSE: BRK.A)"
    """
    if not isinstance(description, str):
        return None
    m = EXCHANGE_TICKER_RE.search(description)
    if not m:
        return None
    t = (m.group(1) or "").strip()
    return t or None

def extract_ticker_best_effort(m: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort ticker extraction with a small, explicit exception list.

    Priority:
      1) Hard-coded known legacy exceptions (by market id)
      2) Standard "(TICKER)" from question
      3) Exchange-based pattern from description (e.g., "NASDAQ: AVGO")
    """
    mid = str(m.get("id", "")).strip()
    if mid in SPECIAL_TICKER_BY_MARKET_ID:
        return SPECIAL_TICKER_BY_MARKET_ID[mid]

    q = str(m.get("question") or "")
    t = extract_ticker_from_question(q)
    if t:
        return t

    desc = str(m.get("description") or "")
    t2 = extract_ticker_from_description(desc)
    if t2:
        return t2

    return None


def extract_tag_slugs(tags_payload: Any) -> List[str]:
    """Extract tag slugs from Gamma tag objects."""
    if not isinstance(tags_payload, list):
        return []
    out: List[str] = []
    for t in tags_payload:
        if isinstance(t, dict) and isinstance(t.get("slug"), str):
            out.append(t["slug"])
    return out


def record_discard(discarded: List[Dict[str, Any]], m: Dict[str, Any], reason: str, stage: str) -> None:
    """Append a discard record for auditability."""
    discarded.append({
        "stage": stage,
        "discard_reason": reason,
        "id": str(m.get("id")) if m.get("id") is not None else None,
        "slug": m.get("slug"),
        "question": m.get("question"),
        "endDate": m.get("endDate"),
        "closed": m.get("closed"),
        "active": m.get("active"),
        "archived": m.get("archived"),
        "category": m.get("category"),
        "resolutionSource": m.get("resolutionSource"),
        "description": m.get("description"),
        "tags": extract_tag_slugs(m.get("tags")),
    })


# =============================================================================
# HTTP helpers
# =============================================================================

def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = HTTP_TIMEOUT,
    retries: int = RETRIES,
    retry_sleep_s: float = RETRY_SLEEP_S,
    print_retry_details: bool = PRINT_RETRY_DETAILS,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Perform an HTTP request with retries and return (payload, error).
    Retries: 429, 500, 502, 503, 504 + network exceptions.
    """
    last_err: Optional[Dict[str, Any]] = None
    hdrs = headers or HEADERS

    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                headers=hdrs,
                timeout=timeout_s,
            )

            try:
                payload = resp.json()
            except Exception:
                payload = resp.text

            if 200 <= resp.status_code < 300:
                return payload, None

            last_err = {
                "status_code": resp.status_code,
                "url": url,
                "params": params,
                "response": payload if isinstance(payload, (dict, list)) else str(payload)[:800],
            }

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                if print_retry_details:
                    tqdm.write(
                        f"Retrying ({attempt+1}/{retries}) after HTTP {resp.status_code} "
                        f"for {url} params={params}"
                    )
                time.sleep(retry_sleep_s * (attempt + 1))
                continue

            return None, last_err

        except Exception as e:
            last_err = {"status_code": None, "url": url, "params": params, "exception": repr(e)}
            if attempt < retries:
                if print_retry_details:
                    tqdm.write(f"Retrying ({attempt+1}/{retries}) after exception for {url}: {repr(e)}")
                time.sleep(retry_sleep_s * (attempt + 1))
                continue
            return None, last_err

    return None, last_err


def gamma_get(
    path: str,
    *,
    base_url: str = GAMMA,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = HTTP_TIMEOUT,
    retries: int = RETRIES,
    retry_sleep_s: float = RETRY_SLEEP_S,
    print_retry_details: bool = PRINT_RETRY_DETAILS,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Convenience wrapper: Gamma GET request."""
    base = base_url.rstrip("/")
    return _request_json(
        "GET",
        f"{base}{path}",
        params=params,
        headers=headers,
        timeout_s=timeout_s,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
        print_retry_details=print_retry_details,
    )


# =============================================================================
# Existing output loading (incremental mode)
# =============================================================================

def load_existing_resolved_jsonl(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Load existing markets.jsonl into dict keyed by market id.
    Returns: (records_by_id, parse_error_count)
    """
    records: Dict[str, Dict[str, Any]] = {}
    parse_errors = 0

    if not path.exists():
        return records, parse_errors

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if not isinstance(obj, dict):
                        parse_errors += 1
                        continue
                    mid = str(obj.get("id", "")).strip()
                    if not mid:
                        parse_errors += 1
                        continue
                    records[mid] = obj
                except Exception:
                    parse_errors += 1
                    continue
    except Exception:
        return {}, 1

    return records, parse_errors


# =============================================================================
# Tag id lookup
# =============================================================================

def get_tag_id_by_slug(
    slug: str,
    *,
    gamma_base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    retries: int,
    retry_sleep_s: float,
    print_retry_details: bool,
    verbose: bool,
) -> Optional[int]:
    """
    Gamma requires tag_id for /markets queries.

    Preferred:
      GET /tags/slug/{slug}

    Fallback:
      paginate GET /tags and search for slug
    """
    log(f"Looking up tag id for slug='{slug}' via /tags/slug/{slug} ...", verbose=verbose)
    payload, err = gamma_get(
        f"/tags/slug/{slug}",
        base_url=gamma_base_url,
        headers=headers,
        timeout_s=timeout_s,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
        print_retry_details=print_retry_details,
    )
    if err is None and isinstance(payload, dict) and payload.get("id") is not None:
        try:
            tag_id = int(payload["id"])
            log(f"Found tag_id={tag_id} for slug='{slug}'.", verbose=verbose)
            return tag_id
        except Exception:
            return None

    log("Direct tag lookup failed; falling back to scanning /tags pages...", verbose=verbose)
    offset = 0
    with tqdm(desc="Scanning tags", unit="page", dynamic_ncols=True) as pbar:
        while True:
            page, page_err = gamma_get(
                "/tags",
                base_url=gamma_base_url,
                params={"limit": 200, "offset": offset},
                headers=headers,
                timeout_s=timeout_s,
                retries=retries,
                retry_sleep_s=retry_sleep_s,
                print_retry_details=print_retry_details,
            )
            pbar.update(1)

            if page_err or not isinstance(page, list):
                return None
            if not page:
                return None

            for t in page:
                if isinstance(t, dict) and str(t.get("slug", "")).lower() == slug.lower():
                    try:
                        tag_id = int(t["id"])
                        log(f"Found tag_id={tag_id} for slug='{slug}' via scan.", verbose=verbose)
                        return tag_id
                    except Exception:
                        return None

            offset += 200


# =============================================================================
# Earnings beat classifier (sample definition)
# =============================================================================

def looks_like_earnings_beat_market(m: Dict[str, Any]) -> Tuple[bool, str]:
    """Apply strict rules; returns (ok, reason)."""
    q = (m.get("question") or "").strip()
    slug = (m.get("slug") or "").strip()
    desc = (m.get("description") or "").strip()

    if not q or not slug:
        return False, "missing_question_or_slug"

    outs = parse_json_list_maybe(m.get("outcomes"))
    if not isinstance(outs, list) or not normalize_yes_no(outs):
        return False, "not_binary_yes_no"

    if not extract_ticker_best_effort(m):
        return False, "no_ticker_detected"

    if not BEAT_RE.search(q):
        return False, "missing_beat_keyword"

    combined = f"{q}\n{desc}"
    if not EARNINGS_OR_EPS_RE.search(combined):
        return False, "missing_earnings_or_eps_context"

    if not ESTIMATE_RE.search(combined):
        return False, "missing_estimate_hint"

    if not desc:
        return False, "missing_description"

    if m.get("closed") is not True:
        return False, "not_closed"

    return True, "ok"


def decode_resolved_outcome(m: Dict[str, Any]) -> Optional[str]:
    """Best-effort resolved outcome ("Yes"/"No") from explicit fields or outcomePrices fallback."""
    for k in ("winningOutcome", "resolvedOutcome", "resolution", "result"):
        v = m.get(k)
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("yes", "no"):
                return "Yes" if vv == "yes" else "No"

    outs = parse_json_list_maybe(m.get("outcomes"))
    prices = parse_json_list_maybe(m.get("outcomePrices"))
    if isinstance(outs, list) and isinstance(prices, list) and len(outs) == 2 and len(prices) == 2:
        try:
            p0 = float(prices[0])
            p1 = float(prices[1])
            original = [str(x).strip().lower() for x in outs]
            if p0 >= 0.999 and p1 <= 0.001:
                return "Yes" if original[0] == "yes" else "No"
            if p1 >= 0.999 and p0 <= 0.001:
                return "Yes" if original[1] == "yes" else "No"
        except Exception:
            pass

    return None


# =============================================================================
# Fetch markets list (paged)
# =============================================================================

def list_closed_markets_by_tag(
    tag_id: int,
    *,
    gamma_base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    retries: int,
    retry_sleep_s: float,
    print_retry_details: bool,
    page_limit: int,
    related_tags: bool,
    end_date_min: Optional[str],
    end_date_max: Optional[str],
    test: bool,
    test_max_pages: int,
) -> Iterable[Dict[str, Any]]:
    """
    Stream closed markets under tag_id from Gamma /markets.
    Ordering: endDate,id ascending for stability.
    """
    offset = 0
    pages = 0

    while True:
        params: Dict[str, Any] = {
            "limit": page_limit,
            "offset": offset,
            "tag_id": tag_id,
            "closed": True,
            "include_tag": True,
            "related_tags": bool(related_tags),
            "order": "endDate,id",
            "ascending": True,
        }
        if end_date_min:
            params["end_date_min"] = end_date_min
        if end_date_max:
            params["end_date_max"] = end_date_max

        payload, err = gamma_get(
            "/markets",
            base_url=gamma_base_url,
            params=params,
            headers=headers,
            timeout_s=timeout_s,
            retries=retries,
            retry_sleep_s=retry_sleep_s,
            print_retry_details=print_retry_details,
        )
        if err:
            raise RuntimeError(f"/markets error offset={offset}: {err}")

        if not isinstance(payload, list) or not payload:
            break

        for m in payload:
            if isinstance(m, dict):
                yield m

        offset += page_limit
        pages += 1
        if test and pages >= test_max_pages:
            break


def fetch_market_detail(
    market_id: str,
    *,
    gamma_base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    retries: int,
    retry_sleep_s: float,
    print_retry_details: bool,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Fetch full detail for a given market id."""
    payload, err = gamma_get(
        f"/markets/{market_id}",
        base_url=gamma_base_url,
        headers=headers,
        timeout_s=timeout_s,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
        print_retry_details=print_retry_details,
    )
    if err or not isinstance(payload, dict):
        return market_id, None, err or {"note": "non-dict payload"}
    return market_id, payload, None


def fetch_market_tags(
    market_id: str,
    *,
    gamma_base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    retries: int,
    retry_sleep_s: float,
    print_retry_details: bool,
) -> List[Dict[str, Any]]:
    """Fallback endpoint when detail payload tags are missing."""
    payload, _ = gamma_get(
        f"/markets/{market_id}/tags",
        base_url=gamma_base_url,
        headers=headers,
        timeout_s=timeout_s,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
        print_retry_details=print_retry_details,
    )
    return payload if isinstance(payload, list) else []


# =============================================================================
# Public callable function (for running from another script)
# =============================================================================

def build_resolved_earnings_dataset(
    *,
    out_dir: Union[str, Path, None] = None,
    write_csv: Optional[bool] = None,
    incremental_mode: Optional[bool] = None,
    test: Optional[bool] = None,
    test_max_pages: Optional[int] = None,
    test_max_markets: Optional[int] = None,
    gamma_base_url: Optional[str] = None,
    http_timeout: Optional[int] = None,
    retries: Optional[int] = None,
    retry_sleep_s: Optional[float] = None,
    print_retry_details: Optional[bool] = None,
    page_limit: Optional[int] = None,
    max_workers: Optional[int] = None,
    related_tags: Optional[bool] = None,
    end_date_min: Optional[str] = None,
    end_date_max: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build resolved Polymarket earnings "beat estimate" dataset.

    - Uses TOP-OF-SCRIPT defaults if parameters are None.
    - Overridable from another script by passing the parameters you want.

    Returns a dict with:
      - paths
      - counts
      - cancelled
      - summary_text
    """
    # Resolve defaults
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    write_csv = WRITE_CSV if write_csv is None else bool(write_csv)
    incremental_mode = INCREMENTAL_MODE if incremental_mode is None else bool(incremental_mode)
    test = TEST if test is None else bool(test)
    test_max_pages = TEST_MAX_PAGES if test_max_pages is None else int(test_max_pages)
    test_max_markets = TEST_MAX_MARKETS if test_max_markets is None else int(test_max_markets)

    gamma_base_url = GAMMA if gamma_base_url is None else str(gamma_base_url)
    http_timeout = HTTP_TIMEOUT if http_timeout is None else int(http_timeout)
    retries = RETRIES if retries is None else int(retries)
    retry_sleep_s = RETRY_SLEEP_S if retry_sleep_s is None else float(retry_sleep_s)
    print_retry_details = PRINT_RETRY_DETAILS if print_retry_details is None else bool(print_retry_details)

    page_limit = PAGE_LIMIT if page_limit is None else int(page_limit)
    max_workers = MAX_WORKERS if max_workers is None else int(max_workers)

    related_tags = RELATED_TAGS if related_tags is None else bool(related_tags)
    # If caller did not pass end_date_min/max, fall back to script defaults
    end_date_min = END_DATE_MIN if end_date_min is None else end_date_min
    end_date_max = END_DATE_MAX if end_date_max is None else end_date_max

    verbose = VERBOSE if verbose is None else bool(verbose)

    # Compute output paths (based on chosen out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / FNAME_MARKETS_JSONL
    out_csv = out_dir / FNAME_MARKETS_CSV
    out_discarded_jsonl = out_dir / FNAME_DISCARDED_JSONL
    out_summary = out_dir / FNAME_SUMMARY_TXT

    started = time.time()
    cancelled = False
    cancel_stage: Optional[str] = None

    discard = Counter()
    discarded_markets: List[Dict[str, Any]] = []

    scanned = 0
    skipped_existing = 0

    candidates: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}
    detail_errors: List[Dict[str, Any]] = []

    final_new: List[Dict[str, Any]] = []
    new_yes = 0
    new_no = 0

    existing_by_id: Dict[str, Dict[str, Any]] = {}
    existing_parse_errors = 0
    existing_loaded = 0

    if incremental_mode:
        existing_by_id, existing_parse_errors = load_existing_resolved_jsonl(out_jsonl)
        existing_loaded = len(existing_by_id)

        # Best-effort backfill ticker in existing records
        upgraded = 0
        for _, rec in existing_by_id.items():
            if not isinstance(rec, dict) or rec.get("ticker"):
                continue
            q = rec.get("question") or ""
            t = extract_ticker_best_effort(rec)
            if t:
                rec["ticker"] = t
                upgraded += 1

        if existing_loaded > 0:
            log(
                f"INCREMENTAL_MODE=True: loaded {existing_loaded} existing records "
                f"(parse_errors={existing_parse_errors}, ticker_backfilled={upgraded}).",
                verbose=verbose,
            )
        else:
            log(f"INCREMENTAL_MODE=True: no existing records found at {out_jsonl}.", verbose=verbose)

    log(
        f"Starting. TEST={test}. INCREMENTAL_MODE={incremental_mode}. Output dir: {out_dir}",
        verbose=verbose,
    )

    try:
        # Step 1: Resolve earnings tag_id
        earnings_tag_id = get_tag_id_by_slug(
            "earnings",
            gamma_base_url=gamma_base_url,
            headers=HEADERS,
            timeout_s=http_timeout,
            retries=retries,
            retry_sleep_s=retry_sleep_s,
            print_retry_details=print_retry_details,
            verbose=verbose,
        )
        if earnings_tag_id is None:
            raise SystemExit("Could not resolve earnings tag id. (Network / API issue)")

        # Step 2: Scan markets
        cancel_stage = "scan_markets"
        log("Scanning closed markets with Earnings tag...", verbose=verbose)

        scan_pbar = tqdm(desc="Scanning markets", unit="mkt", dynamic_ncols=True)
        try:
            for m in list_closed_markets_by_tag(
                earnings_tag_id,
                gamma_base_url=gamma_base_url,
                headers=HEADERS,
                timeout_s=http_timeout,
                retries=retries,
                retry_sleep_s=retry_sleep_s,
                print_retry_details=print_retry_details,
                page_limit=page_limit,
                related_tags=related_tags,
                end_date_min=end_date_min,
                end_date_max=end_date_max,
                test=test,
                test_max_pages=test_max_pages,
            ):
                scanned += 1
                scan_pbar.update(1)

                mid = str(m.get("id", "")).strip()
                if incremental_mode and mid and mid in existing_by_id:
                    skipped_existing += 1
                    continue

                ok, reason = looks_like_earnings_beat_market(m)
                if not ok:
                    discard[reason] += 1
                    record_discard(discarded_markets, m, reason, stage="scan")
                    continue

                candidates.append(m)

                scan_pbar.set_postfix({
                    "candidates": len(candidates),
                    "skipped": skipped_existing,
                    "discarded": sum(discard.values()),
                })

                if test and len(candidates) >= test_max_markets:
                    tqdm.write(f"TEST mode: reached {test_max_markets} candidates, stopping scan early.")
                    break
        finally:
            scan_pbar.close()

        log(
            f"Scan done. scanned={scanned}, skipped_existing={skipped_existing}, "
            f"candidates={len(candidates)}, discards={sum(discard.values())}",
            verbose=verbose,
        )

        # Step 3: Fetch details
        cancel_stage = "fetch_details"
        if candidates:
            ids: List[str] = []
            for m in candidates:
                mid = str(m.get("id", "")).strip()
                if not mid:
                    discard["missing_id"] += 1
                    continue
                ids.append(mid)

            log(f"Fetching market details for {len(ids)} candidates (threads={max_workers}) ...", verbose=verbose)

            use_threads = (len(ids) >= 10 and max_workers > 1)

            if use_threads:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [
                        ex.submit(
                            fetch_market_detail,
                            mid,
                            gamma_base_url=gamma_base_url,
                            headers=HEADERS,
                            timeout_s=http_timeout,
                            retries=retries,
                            retry_sleep_s=retry_sleep_s,
                            print_retry_details=print_retry_details,
                        )
                        for mid in ids
                    ]
                    with tqdm(total=len(futs), desc="Fetching details", unit="mkt", dynamic_ncols=True) as pbar:
                        try:
                            for fut in as_completed(futs):
                                mid, payload, err = fut.result()
                                pbar.update(1)

                                if err or payload is None:
                                    detail_errors.append({"market_id": mid, "error": err})
                                    continue

                                details[mid] = payload
                                pbar.set_postfix({"ok": len(details), "err": len(detail_errors)})

                        except KeyboardInterrupt:
                            cancelled = True
                            tqdm.write("Ctrl+C received during detail fetch. Will save what we have so far...")
                            ex.shutdown(wait=False, cancel_futures=True)
            else:
                with tqdm(total=len(ids), desc="Fetching details", unit="mkt", dynamic_ncols=True) as pbar:
                    for mid in ids:
                        try:
                            mid, payload, err = fetch_market_detail(
                                mid,
                                gamma_base_url=gamma_base_url,
                                headers=HEADERS,
                                timeout_s=http_timeout,
                                retries=retries,
                                retry_sleep_s=retry_sleep_s,
                                print_retry_details=print_retry_details,
                            )
                        except KeyboardInterrupt:
                            cancelled = True
                            tqdm.write("Ctrl+C received during detail fetch. Will save what we have so far...")
                            break

                        pbar.update(1)
                        if err or payload is None:
                            detail_errors.append({"market_id": mid, "error": err})
                            continue

                        details[mid] = payload
                        pbar.set_postfix({"ok": len(details), "err": len(detail_errors)})

            log(f"Detail fetch done. ok={len(details)} errors={len(detail_errors)}", verbose=verbose)

        # Step 4: Finalize
        cancel_stage = "finalize"
        log("Building final resolved dataset (new markets this run)...", verbose=verbose)

        with tqdm(total=len(candidates), desc="Finalizing", unit="mkt", dynamic_ncols=True) as pbar:
            for m in candidates:
                pbar.update(1)

                mid = str(m.get("id", "")).strip()
                d = details.get(mid)
                if not d:
                    discard["detail_missing"] += 1
                    record_discard(discarded_markets, m, "detail_missing", stage="detail")
                    continue

                ok, reason = looks_like_earnings_beat_market(d)
                if not ok:
                    discard[f"detail_{reason}"] += 1
                    record_discard(discarded_markets, d, f"detail_{reason}", stage="detail")
                    continue

                resolved = decode_resolved_outcome(d)
                if resolved not in ("Yes", "No"):
                    discard["no_confident_resolved_outcome"] += 1
                    record_discard(discarded_markets, d, "no_confident_resolved_outcome", stage="detail")
                    continue

                q = (d.get("question") or "").strip()
                ticker = extract_ticker_best_effort(d)
                if not ticker:
                    discard["no_ticker_detected"] += 1
                    record_discard(discarded_markets, d, "no_ticker_detected", stage="finalize")
                    continue

                if resolved == "Yes":
                    new_yes += 1
                else:
                    new_no += 1

                tag_slugs = extract_tag_slugs(d.get("tags"))
                if not tag_slugs:
                    tags = fetch_market_tags(
                        mid,
                        gamma_base_url=gamma_base_url,
                        headers=HEADERS,
                        timeout_s=http_timeout,
                        retries=retries,
                        retry_sleep_s=retry_sleep_s,
                        print_retry_details=print_retry_details,
                    )
                    tag_slugs = [
                        t.get("slug") for t in tags
                        if isinstance(t, dict) and isinstance(t.get("slug"), str)
                    ]

                if "earnings" not in {str(x).lower() for x in tag_slugs if isinstance(x, str)}:
                    discard["missing_earnings_tag_after_detail"] += 1
                    record_discard(discarded_markets, d, "missing_earnings_tag_after_detail", stage="detail")
                    continue

                outs = parse_json_list_maybe(d.get("outcomes")) or ["Yes", "No"]
                outs_norm = normalize_yes_no(outs) or ["Yes", "No"]

                rec = {
                    "id": mid,
                    "ticker": ticker,
                    "slug": d.get("slug"),
                    "question": d.get("question"),
                    "outcomes": outs_norm,
                    "endDate": d.get("endDate"),
                    "resolutionSource": d.get("resolutionSource"),
                    "resolvedOutcome": resolved,
                    "closed": d.get("closed"),
                    "active": d.get("active"),
                    "archived": d.get("archived"),
                    "restricted": d.get("restricted"),
                    "resolvedBy": d.get("resolvedBy"),
                    "outcomePrices": d.get("outcomePrices"),
                    "category": d.get("category"),
                    "createdAt": d.get("createdAt"),
                    "updatedAt": d.get("updatedAt"),
                    "volumeNum": d.get("volumeNum"),
                    "liquidityNum": d.get("liquidityNum"),
                    "tags": tag_slugs,
                }
                final_new.append(rec)

                pbar.set_postfix({
                    "kept_new": len(final_new),
                    "discarded": len(discarded_markets),
                    "yes": new_yes,
                    "no": new_no,
                })

    except KeyboardInterrupt:
        cancelled = True
        tqdm.write("Ctrl+C received. Will save what we have so far...")

    finally:
        # Merge outputs
        def sort_key(r: Dict[str, Any]) -> Tuple[int, datetime]:
            dt = parse_iso_dt(r.get("endDate"))
            return (0, dt) if dt else (1, datetime.max.replace(tzinfo=timezone.utc))

        if incremental_mode:
            merged_by_id: Dict[str, Dict[str, Any]] = dict(existing_by_id)
            added = 0
            for r in final_new:
                rid = str(r.get("id", "")).strip()
                if rid and rid not in merged_by_id:
                    merged_by_id[rid] = r
                    added += 1
            output_records = list(merged_by_id.values())
            new_added = added
        else:
            output_records = list(final_new)
            new_added = len(final_new)

        # Backfill ticker if missing (best effort)
        upgraded_missing_ticker = 0
        for r in output_records:
            if r.get("ticker"):
                continue
            t = extract_ticker_best_effort(r)
            if t:
                r["ticker"] = t
                upgraded_missing_ticker += 1

        output_records.sort(key=sort_key)

        # Write JSONL outputs
        log(f"Writing JSONL -> {out_jsonl} (records={len(output_records)})", verbose=verbose)
        atomic_write_text(out_jsonl, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output_records))

        log(f"Writing discarded JSONL -> {out_discarded_jsonl} (records={len(discarded_markets)})", verbose=verbose)
        atomic_write_text(out_discarded_jsonl, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in discarded_markets))

        # Optional CSV output
        if write_csv:
            log(f"Writing CSV -> {out_csv}", verbose=verbose)
            fields = [
                "id", "ticker", "slug", "question", "endDate", "resolutionSource",
                "resolvedOutcome", "closed", "active", "archived", "restricted",
                "resolvedBy", "category", "volumeNum", "liquidityNum", "tags"
            ]
            tmp_csv = out_csv.with_suffix(".csv.tmp")
            with tmp_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in output_records:
                    row = dict(r)
                    row["tags"] = ",".join([t for t in (r.get("tags") or []) if isinstance(t, str)])
                    w.writerow({k: row.get(k) for k in fields})
            tmp_csv.replace(out_csv)

        # Summary
        end_dates = [parse_iso_dt(r.get("endDate")) for r in output_records]
        end_dates = [d for d in end_dates if d is not None]
        earliest_dt = min(end_dates) if end_dates else None
        latest_dt = max(end_dates) if end_dates else None

        total_yes = sum(1 for r in output_records if r.get("resolvedOutcome") == "Yes")
        total_no = sum(1 for r in output_records if r.get("resolvedOutcome") == "No")

        volumes: List[float] = []
        for r in output_records:
            v = r.get("volumeNum")
            if isinstance(v, (int, float)) and math.isfinite(v):
                volumes.append(float(v))

        def stats(xs: List[float]) -> Dict[str, Any]:
            if not xs:
                return {"count": 0}
            xs2 = sorted(xs)
            n = len(xs2)
            return {
                "count": n,
                "min": xs2[0],
                "max": xs2[-1],
                "mean": sum(xs2) / n,
                "median": xs2[n // 2] if n % 2 else (xs2[n // 2 - 1] + xs2[n // 2]) / 2.0,
            }

        vstats = stats(volumes)

        finished_dt = datetime.now(timezone.utc)
        mode_str = "incremental" if incremental_mode else "full_refresh"

        lines: List[str] = []
        lines.append("Polymarket Earnings Resolved Markets — Summary")
        lines.append("=" * 52)
        lines.append(f"Generated (UTC): {format_dt_utc(finished_dt)}")
        lines.append(f"Mode: {mode_str}")
        lines.append(f"write_csv: {write_csv}")
        if incremental_mode:
            lines.append(f"Existing loaded: {existing_loaded} (parse_errors={existing_parse_errors})")
            lines.append(f"New markets added this run: {new_added}")
            if upgraded_missing_ticker:
                lines.append(f"Existing records ticker backfilled during run: {upgraded_missing_ticker}")
        lines.append("")
        lines.append("Outputs")
        lines.append(f"- Markets JSONL:     {out_jsonl}")
        lines.append(f"- Markets CSV:       {out_csv if write_csv else '(disabled)'}")
        lines.append(f"- Discarded JSONL:   {out_discarded_jsonl} (this run)")
        lines.append(f"- Summary TXT:       {out_summary}")
        lines.append("")
        lines.append("Resolution time range (based on endDate)")
        lines.append(f"- Earliest resolved: {format_dt_utc(earliest_dt)}")
        lines.append(f"- Latest resolved:   {format_dt_utc(latest_dt)}")
        lines.append("")
        lines.append("Counts (overall output)")
        lines.append(f"- Total resolved markets saved: {len(output_records)}")
        lines.append(f"- Total Yes: {total_yes}")
        lines.append(f"- Total No:  {total_no}")
        lines.append("")
        lines.append("This run (scan/fetch/build)")
        lines.append(f"- Scanned closed earnings-tag markets: {scanned}")
        lines.append(f"- Skipped already-known markets:       {skipped_existing}")
        lines.append(f"- Candidates after filter:             {len(candidates)}")
        lines.append(f"- Details fetched OK:                  {len(details)}")
        lines.append(f"- Detail errors:                       {len(detail_errors)}")
        lines.append(f"- New resolved markets built:          {len(final_new)} (Yes={new_yes}, No={new_no})")
        lines.append(f"- Discarded markets logged:            {len(discarded_markets)}")
        lines.append("")
        lines.append("Discard reasons (this run)")
        if discard:
            for k, v in sorted(discard.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {k}: {v}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("volumeNum stats (overall output)")
        if vstats.get("count", 0) == 0:
            lines.append("- count: 0")
        else:
            lines.append(f"- count:  {vstats['count']}")
            lines.append(f"- min:    {vstats['min']}")
            lines.append(f"- max:    {vstats['max']}")
            lines.append(f"- mean:   {vstats['mean']}")
            lines.append(f"- median: {vstats['median']}")
        lines.append("")
        lines.append("Run metadata")
        lines.append(f"- Cancelled: {cancelled} (stage={cancel_stage})")
        lines.append(f"- Elapsed seconds: {round(time.time() - started, 3)}")
        lines.append(f"- Page limit: {page_limit}")
        lines.append(f"- Max workers: {max_workers}")
        lines.append(f"- END_DATE_MIN: {end_date_min}")
        lines.append(f"- END_DATE_MAX: {end_date_max}")
        lines.append(f"- RELATED_TAGS: {related_tags}")
        lines.append("")

        summary_text = "\n".join(lines)
        log(f"Writing summary -> {out_summary}", verbose=verbose)
        atomic_write_text(out_summary, summary_text)

        if verbose:
            log("DONE (or cancelled). Summary:", verbose=True)
            print(summary_text, flush=True)

        return {
            "paths": {
                "out_dir": str(out_dir),
                "markets_jsonl": str(out_jsonl),
                "markets_csv": str(out_csv) if write_csv else None,
                "discarded_jsonl": str(out_discarded_jsonl),
                "summary_txt": str(out_summary),
            },
            "counts": {
                "output_records": len(output_records),
                "output_yes": total_yes,
                "output_no": total_no,
                "scanned": scanned,
                "skipped_existing": skipped_existing,
                "candidates": len(candidates),
                "details_ok": len(details),
                "detail_errors": len(detail_errors),
                "new_built": len(final_new),
                "discarded_logged": len(discarded_markets),
            },
            "cancelled": cancelled,
            "summary_text": summary_text,
        }


# =============================================================================
# CLI (optional) — overrides the TOP-OF-SCRIPT defaults
# =============================================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build resolved Polymarket earnings dataset (JSONL + optional CSV)."
    )
    p.add_argument("--out-dir", type=str, default=None, help="Override OUT_DIR.")
    p.add_argument("--no-csv", action="store_true", help="Disable CSV output.")
    p.add_argument("--full-refresh", action="store_true", help="Disable incremental mode.")
    p.add_argument("--test", action="store_true", help="Enable test mode.")
    p.add_argument("--end-date-min", type=str, default=None, help="Override END_DATE_MIN.")
    p.add_argument("--end-date-max", type=str, default=None, help="Override END_DATE_MAX.")
    p.add_argument("--max-workers", type=int, default=None, help="Override MAX_WORKERS.")
    p.add_argument("--page-limit", type=int, default=None, help="Override PAGE_LIMIT.")
    p.add_argument("--quiet", action="store_true", help="Reduce console logs.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Only override what the CLI explicitly provides; otherwise use top defaults.
    res = build_resolved_earnings_dataset(
        out_dir=args.out_dir if args.out_dir is not None else None,
        write_csv=False if args.no_csv else None,
        incremental_mode=False if args.full_refresh else None,
        test=True if args.test else None,
        end_date_min=args.end_date_min if args.end_date_min is not None else None,
        end_date_max=args.end_date_max if args.end_date_max is not None else None,
        max_workers=args.max_workers if args.max_workers is not None else None,
        page_limit=args.page_limit if args.page_limit is not None else None,
        verbose=False if args.quiet else None,
    )
    _ = res
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C). Exiting.", flush=True)
        raise SystemExit(1)
