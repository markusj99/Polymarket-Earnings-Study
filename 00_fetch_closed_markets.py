#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket Earnings (Resolved Markets) — Dataset Builder
========================================================

Purpose
-------
This script builds a clean dataset of *resolved earnings “beat estimate”* markets from Polymarket
by using Polymarket's Gamma API.

It is intended for research use (e.g., a Master's thesis) where we need:
- a stable, reproducible list of markets,
- consistent filtering rules (so the sample definition is clear),
- a record of what we discarded and why (auditability).

High-level output
-----------------
The script produces four files:

1) markets.jsonl
   - Final dataset (one market per line, JSON).
2) markets.csv
   - Same dataset as CSV (easier to inspect in Excel / Stata / R).
3) discarded_markets.jsonl
   - Markets that were scanned but discarded, with a reason and stage.
     This is critical for transparency and reproducibility.
4) summary.txt
   - Human-readable summary of the run: counts, date range, discard reasons, etc.

Key definitions
---------------
- "Resolved market" in this script means:
    closed == True
  (Gamma uses "closed" for markets no longer trading / resolved.)

- Sample definition (what we keep):
  A market must satisfy ALL of:
    1) It is tagged under "earnings" (and optionally related tags if RELATED_TAGS=True)
    2) It is closed
    3) It is a binary Yes/No market
    4) The question contains a ticker in parentheses, e.g. "(AAPL)" or "(BRK.A)"
    5) The question contains the word "beat"
    6) The question/description contains "earnings" or "eps"
    7) The question/description contains "estimate"/"estimated"/"consensus"
    8) The description is non-empty (helps ensure the market is a proper earnings market)
    9) The market has an identifiable resolved outcome (Yes/No)

Incremental mode
----------------
If INCREMENTAL_MODE=True, the script will:
- load existing markets.jsonl (if present)
- avoid adding markets already present by id
- append only newly discovered markets this run

This is useful when the dataset grows over time and you want to re-run without reprocessing
everything.

Progress bars
-------------
The script uses tqdm progress bars for:
- scanning markets pages (unknown total, so tqdm without total)
- fetching details (known total)
- finalizing records (known total)

Audit trail
-----------
Every time the script discards a market, it records:
- stage: "scan" / "detail" / "finalize"
- discard_reason: a machine-readable reason string
- plus identifying fields (id, slug, question, endDate, tags, etc.)

This ensures supervisors / reviewers can see exactly why markets were included/excluded.

Dependencies
------------
pip install requests tqdm
"""

from __future__ import annotations

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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# -------------------------
# TEST MODE
# -------------------------
# TEST=True is useful for quickly verifying that the script works end-to-end.
# When TEST=True, the script stops scanning after TEST_MAX_PAGES pages or after
# collecting TEST_MAX_MARKETS candidate markets (whichever triggers first).
TEST = False
TEST_MAX_PAGES = 2
TEST_MAX_MARKETS = 10

# -------------------------
# INCREMENTAL MODE
# -------------------------
# True  => load existing OUT_JSONL (if present) and only add markets not already included
# False => fetch all markets again (full rebuild; ignores existing OUT_JSONL contents)
INCREMENTAL_MODE = True

# -------------------------
# STATUS / LOGGING
# -------------------------
# Heartbeat is a lightweight "still alive" mechanism. It's less important now that
# we use tqdm, but retained for occasional long-running steps (e.g., tag scanning fallback).
STATUS_EVERY_SECONDS = 15
PRINT_RETRY_DETAILS = True

# -------------------------
# API / performance
# -------------------------
GAMMA = "https://gamma-api.polymarket.com"
HTTP_TIMEOUT = 25
RETRIES = 3
RETRY_SLEEP_S = 0.8

# Gamma /markets pagination
PAGE_LIMIT = 200

# Thread count for parallel market detail fetch
MAX_WORKERS = 15

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "polymarket-earnings-resolved-fetcher/1.7",
}

# -------------------------
# Output paths (REQUESTED)
# -------------------------
OUT_DIR = Path(r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\markets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "markets.jsonl"
OUT_CSV = OUT_DIR / "markets.csv"
OUT_SUMMARY = OUT_DIR / "summary.txt"
OUT_DISCARDED_JSONL = OUT_DIR / "discarded_markets.jsonl"

# -------------------------
# Fetch controls
# -------------------------
# Optional: restrict the markets fetched by endDate.
# Gamma expects ISO strings. Example:
# END_DATE_MIN = "2022-01-01T00:00:00Z"
END_DATE_MIN: Optional[str] = None
END_DATE_MAX: Optional[str] = None

# Whether to include markets tagged via "related tags" (Gamma feature).
RELATED_TAGS = True


# =============================================================================
# Market classifier (rule-based filter for earnings beat estimate markets)
# =============================================================================
# Ticker requirement: question must include "(TICKER)".
# Supports tickers like BRK.A or RDS-B etc.
TICKER_RE = re.compile(r"\(([A-Z0-9]{1,7}(?:[.\-][A-Z0-9]{1,4})?)\)")

# Keyword rules:
BEAT_RE = re.compile(r"\bbeat\b", re.IGNORECASE)
EARNINGS_OR_EPS_RE = re.compile(r"\b(earnings|eps)\b", re.IGNORECASE)
ESTIMATE_RE = re.compile(r"\b(consensus|estimate|estimated)\b", re.IGNORECASE)

# Outcomes must be exactly Yes/No (binary market).
YESNO_SET = {"yes", "no"}


# =============================================================================
# Small utilities (file writing, logging, heartbeat)
# =============================================================================
def atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to disk:
      1) write to a temp file next to the destination
      2) rename temp -> destination

    This prevents partially-written files if the script is interrupted.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def log(msg: str) -> None:
    """
    Simple timestamped log line.

    Note: We use print() here rather than tqdm.write() because most logging is
    outside active progress bars, and tqdm bars in this script use set_postfix
    for live state. You *can* switch to tqdm.write() if preferred.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class Heartbeat:
    """
    Emits log lines at most once every N seconds.

    Useful when:
    - scanning unknown-length endpoints
    - fallback loops without tqdm totals
    """

    def __init__(self, every_s: int) -> None:
        self.every_s = every_s
        self._last = time.time()

    def maybe(self, msg: str) -> None:
        now = time.time()
        if now - self._last >= self.every_s:
            self._last = now
            log(msg)


# =============================================================================
# HTTP helpers
# =============================================================================
def _request_json(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Perform an HTTP request with retries and return structured results.

    Returns:
      (payload, error)

    payload:
      - parsed JSON (dict/list) if response is JSON
      - otherwise raw text

    error:
      - None on success (HTTP 2xx)
      - dict on failure with status_code, url, params, and response/exception

    Retries:
      - HTTP: 429, 500, 502, 503, 504
      - network exceptions

    We use backoff: RETRY_SLEEP_S * (attempt+1)
    """
    last_err: Optional[Dict[str, Any]] = None
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                headers=HEADERS,
                timeout=HTTP_TIMEOUT,
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

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                if PRINT_RETRY_DETAILS:
                    tqdm.write(
                        f"Retrying ({attempt+1}/{RETRIES}) after HTTP {resp.status_code} "
                        f"for {url} params={params}"
                    )
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
                continue

            return None, last_err

        except Exception as e:
            last_err = {"status_code": None, "url": url, "params": params, "exception": repr(e)}
            if attempt < RETRIES:
                if PRINT_RETRY_DETAILS:
                    tqdm.write(f"Retrying ({attempt+1}/{RETRIES}) after exception for {url}: {repr(e)}")
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
                continue
            return None, last_err

    return None, last_err


def gamma_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Convenience wrapper: Gamma GET request.
    """
    return _request_json("GET", f"{GAMMA}{path}", params=params)


# =============================================================================
# Parsing helpers
# =============================================================================
def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO-8601 strings (including 'Z') into UTC datetime objects.
    Returns None if parsing fails.
    """
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
    """
    Render datetime as a readable UTC string for summary output.
    """
    if not dt:
        return "N/A"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def parse_json_list_maybe(v: Any) -> Optional[List[Any]]:
    """
    Some Gamma fields may appear as:
      - a native list
      - or a JSON-encoded string representing a list

    This function normalizes either representation to a Python list.
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
    """
    Ensure the market outcomes are exactly {"yes", "no"} (case-insensitive),
    and return them in a standardized display order ["Yes", "No"].

    If the market is not binary YES/NO, returns None.
    """
    outs = [str(x).strip() for x in outcomes]
    if len(outs) != 2:
        return None
    lower = [o.lower() for o in outs]
    if set(lower) != YESNO_SET:
        return None
    return ["Yes", "No"]


def extract_ticker_from_question(question: str) -> Optional[str]:
    """
    Extract a ticker symbol from a market question.
    Expected format: "... (AAPL) ..."

    Returns the ticker string (e.g. "AAPL" or "BRK.A") or None.
    """
    if not isinstance(question, str):
        return None
    m = TICKER_RE.search(question.strip())
    if not m:
        return None
    t = (m.group(1) or "").strip()
    return t or None


def extract_tag_slugs(tags_payload: Any) -> List[str]:
    """
    Extract tag slugs from Gamma tag objects:
      [{"id":..., "slug":"earnings", ...}, ...] -> ["earnings", ...]
    """
    if not isinstance(tags_payload, list):
        return []
    out: List[str] = []
    for t in tags_payload:
        if isinstance(t, dict) and isinstance(t.get("slug"), str):
            out.append(t["slug"])
    return out


def record_discard(discarded: List[Dict[str, Any]], m: Dict[str, Any], reason: str, stage: str) -> None:
    """
    Append a discard record for auditability.

    stage:
      - "scan": discarded while scanning /markets list (lightweight info)
      - "detail": discarded after fetching full detail
      - "finalize": discarded while building the final record

    reason:
      - machine-readable string (e.g., "not_binary_yes_no")
    """
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
# Existing output loading (incremental mode)
# =============================================================================
def load_existing_resolved_jsonl(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Load existing markets.jsonl into a dict keyed by market id.

    Returns:
      (records_by_id, parse_error_count)

    Any invalid JSON lines are ignored (counted as parse errors).
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
        # If the file cannot be read at all, treat as empty but indicate failure
        return {}, 1

    return records, parse_errors


# =============================================================================
# Tag id lookup
# =============================================================================
def get_tag_id_by_slug(slug: str) -> Optional[int]:
    """
    Gamma requires tag_id for /markets queries.

    Preferred path:
      GET /tags/slug/{slug}  (fast lookup)

    Fallback:
      Paginate over GET /tags (slower) and search for matching slug.
    """
    log(f"Looking up tag id for slug='{slug}' via /tags/slug/{slug} ...")
    payload, err = gamma_get(f"/tags/slug/{slug}")
    if err is None and isinstance(payload, dict) and payload.get("id") is not None:
        try:
            tag_id = int(payload["id"])
            log(f"Found tag_id={tag_id} for slug='{slug}'.")
            return tag_id
        except Exception:
            return None

    log("Direct tag lookup failed; falling back to scanning /tags pages...")
    offset = 0
    hb = Heartbeat(STATUS_EVERY_SECONDS)
    while True:
        hb.maybe(f"Scanning /tags offset={offset} ...")
        page, page_err = gamma_get("/tags", params={"limit": 200, "offset": offset})
        if page_err or not isinstance(page, list):
            return None
        if not page:
            return None

        for t in page:
            if isinstance(t, dict) and str(t.get("slug", "")).lower() == slug.lower():
                try:
                    tag_id = int(t["id"])
                    log(f"Found tag_id={tag_id} for slug='{slug}' via scan.")
                    return tag_id
                except Exception:
                    return None

        offset += 200


# =============================================================================
# Earnings beat classifier (sample definition)
# =============================================================================
def looks_like_earnings_beat_market(m: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Apply strict rules to decide whether a market looks like an earnings beat estimate market.

    Returns:
      (ok, reason)

    If ok=False, reason describes the first rule violated.
    """
    q = (m.get("question") or "").strip()
    slug = (m.get("slug") or "").strip()
    desc = (m.get("description") or "").strip()

    if not q or not slug:
        return False, "missing_question_or_slug"

    outs = parse_json_list_maybe(m.get("outcomes"))
    if not isinstance(outs, list) or not normalize_yes_no(outs):
        return False, "not_binary_yes_no"

    # Ticker must be present in question like "(GS)".
    if not extract_ticker_from_question(q):
        return False, "no_ticker_in_question"

    # Must mention "beat"
    if not BEAT_RE.search(q):
        return False, "missing_beat_keyword"

    # Must mention earnings/eps somewhere in question or description
    combined = f"{q}\n{desc}"
    if not EARNINGS_OR_EPS_RE.search(combined):
        return False, "missing_earnings_or_eps_context"

    # Must mention estimate/consensus somewhere in question or description
    if not ESTIMATE_RE.search(combined):
        return False, "missing_estimate_hint"

    # Require non-empty description to avoid low-information / misclassified markets
    if not desc:
        return False, "missing_description"

    # Must be closed (resolved)
    if m.get("closed") is not True:
        return False, "not_closed"

    return True, "ok"


# =============================================================================
# Resolution outcome extraction (best effort)
# =============================================================================
def decode_resolved_outcome(m: Dict[str, Any]) -> Optional[str]:
    """
    Attempt to determine resolved outcome ("Yes" or "No") from market detail.

    Preferred:
      - winningOutcome / resolvedOutcome / resolution / result fields if present

    Fallback:
      - infer from outcomePrices if one side is ~1.0 and the other ~0.0

    Returns:
      "Yes", "No", or None if uncertain.
    """
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
def list_closed_markets_by_tag(tag_id: int) -> Iterable[Dict[str, Any]]:
    """
    Stream closed markets under a given tag_id from Gamma.

    Note:
    - total number of pages is not known in advance, so scanning uses tqdm without a total.
    - ordering: endDate,id ascending for stability.

    Yields:
      dict market objects as returned by Gamma /markets endpoint.
    """
    offset = 0
    pages = 0

    while True:
        params: Dict[str, Any] = {
            "limit": PAGE_LIMIT,
            "offset": offset,
            "tag_id": tag_id,
            "closed": True,
            "include_tag": True,
            "related_tags": bool(RELATED_TAGS),
            "order": "endDate,id",
            "ascending": True,
        }

        if END_DATE_MIN:
            params["end_date_min"] = END_DATE_MIN
        if END_DATE_MAX:
            params["end_date_max"] = END_DATE_MAX

        payload, err = gamma_get("/markets", params=params)
        if err:
            raise RuntimeError(f"/markets error offset={offset}: {err}")

        # No more results
        if not isinstance(payload, list) or not payload:
            break

        for m in payload:
            if isinstance(m, dict):
                yield m

        offset += PAGE_LIMIT
        pages += 1

        if TEST and pages >= TEST_MAX_PAGES:
            break


# =============================================================================
# Market detail fetch (thread worker)
# =============================================================================
def fetch_market_detail(market_id: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Fetch full detail for a given market id.

    Returns:
      (market_id, payload_dict_or_None, error_or_None)
    """
    payload, err = gamma_get(f"/markets/{market_id}")
    if err or not isinstance(payload, dict):
        return market_id, None, err or {"note": "non-dict payload"}
    return market_id, payload, None


def fetch_market_tags(market_id: str) -> List[Dict[str, Any]]:
    """
    Optional helper: fetch market tags. Used as a fallback when tag list
    in the detail response is missing/empty.
    """
    payload, _ = gamma_get(f"/markets/{market_id}/tags")
    return payload if isinstance(payload, list) else []


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """
    Orchestrates the complete pipeline:

    1) (Optional incremental) Load existing markets.jsonl.
    2) Resolve the "earnings" tag_id (required for scanning /markets).
    3) Scan closed earnings-tag markets.
       - Filter quickly using looks_like_earnings_beat_market() on list payload.
       - Keep candidate markets for detail fetch.
       - Record discards with reasons.
    4) Fetch full details for each candidate (often required because list payloads can be incomplete).
    5) Finalize:
       - re-apply filter rules on full detail payload
       - decode resolved outcome
       - verify earnings tag still present
       - build final records
    6) Merge:
       - INCREMENTAL_MODE: existing + new (no overwrites)
       - full refresh: new only
    7) Write outputs: JSONL, CSV, discarded JSONL, summary.txt
    """
    started = time.time()
    cancelled = False
    cancel_stage = None

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

    # Incremental mode: load existing output
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    existing_parse_errors = 0
    existing_loaded = 0

    if INCREMENTAL_MODE:
        existing_by_id, existing_parse_errors = load_existing_resolved_jsonl(OUT_JSONL)
        existing_loaded = len(existing_by_id)

        # Best-effort upgrade: backfill ticker in existing records if missing
        upgraded = 0
        for mid, rec in existing_by_id.items():
            if not isinstance(rec, dict):
                continue
            if rec.get("ticker"):
                continue
            q = rec.get("question") or ""
            t = extract_ticker_from_question(str(q))
            if t:
                rec["ticker"] = t
                upgraded += 1

        if existing_loaded > 0:
            log(
                f"INCREMENTAL_MODE=True: loaded {existing_loaded} existing records "
                f"(parse_errors={existing_parse_errors}, ticker_backfilled={upgraded})."
            )
        else:
            log(f"INCREMENTAL_MODE=True: no existing records found at {OUT_JSONL}.")

    log(f"Starting. TEST={TEST}. INCREMENTAL_MODE={INCREMENTAL_MODE}. Output dir: {OUT_DIR}")

    try:
        # ---------------------------------------------------------------------
        # Step 1: Resolve earnings tag_id
        # ---------------------------------------------------------------------
        earnings_tag_id = get_tag_id_by_slug("earnings")
        if earnings_tag_id is None:
            raise SystemExit("Could not resolve earnings tag id. (Network / API issue)")

        # ---------------------------------------------------------------------
        # Step 2: Scan markets (streaming)
        # ---------------------------------------------------------------------
        cancel_stage = "scan_markets"
        log("Scanning closed markets with Earnings tag...")

        scan_pbar = tqdm(desc="Scanning markets", unit="mkt", dynamic_ncols=True)
        try:
            for m in list_closed_markets_by_tag(earnings_tag_id):
                scanned += 1
                scan_pbar.update(1)

                # Incremental mode: skip market ids we already have
                mid = str(m.get("id", "")).strip()
                if INCREMENTAL_MODE and mid and mid in existing_by_id:
                    skipped_existing += 1
                    continue

                # Apply quick filter rules on scan payload
                ok, reason = looks_like_earnings_beat_market(m)
                if not ok:
                    discard[reason] += 1
                    record_discard(discarded_markets, m, reason, stage="scan")
                    continue

                # Candidate saved for detail fetch
                candidates.append(m)

                scan_pbar.set_postfix({
                    "candidates": len(candidates),
                    "skipped": skipped_existing,
                    "discarded": sum(discard.values()),
                })

                # Optional test early stop
                if TEST and len(candidates) >= TEST_MAX_MARKETS:
                    tqdm.write(f"TEST mode: reached {TEST_MAX_MARKETS} candidates, stopping scan early.")
                    break
        finally:
            scan_pbar.close()

        log(
            f"Scan done. scanned={scanned}, skipped_existing={skipped_existing}, "
            f"candidates={len(candidates)}, discards={sum(discard.values())}"
        )

        # ---------------------------------------------------------------------
        # Step 3: Fetch full details for each candidate
        # ---------------------------------------------------------------------
        cancel_stage = "fetch_details"
        if candidates:
            ids: List[str] = []
            for m in candidates:
                mid = str(m.get("id", "")).strip()
                if not mid:
                    discard["missing_id"] += 1
                    continue
                ids.append(mid)

            log(f"Fetching market details for {len(ids)} candidates (threads={MAX_WORKERS}) ...")

            use_threads = (len(ids) >= 10 and MAX_WORKERS > 1)

            if use_threads:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = [ex.submit(fetch_market_detail, mid) for mid in ids]
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
                # Serial fallback (useful for debugging or small lists)
                with tqdm(total=len(ids), desc="Fetching details", unit="mkt", dynamic_ncols=True) as pbar:
                    for mid in ids:
                        try:
                            mid, payload, err = fetch_market_detail(mid)
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

            log(f"Detail fetch done. ok={len(details)} errors={len(detail_errors)}")

        # ---------------------------------------------------------------------
        # Step 4: Finalize (build output records)
        # ---------------------------------------------------------------------
        cancel_stage = "finalize"
        log("Building final resolved dataset (new markets this run)...")

        with tqdm(total=len(candidates), desc="Finalizing", unit="mkt", dynamic_ncols=True) as pbar:
            for m in candidates:
                pbar.update(1)

                mid = str(m.get("id", "")).strip()
                d = details.get(mid)
                if not d:
                    discard["detail_missing"] += 1
                    record_discard(discarded_markets, m, "detail_missing", stage="detail")
                    continue

                # Re-apply strict filter on full detail payload
                ok, reason = looks_like_earnings_beat_market(d)
                if not ok:
                    discard[f"detail_{reason}"] += 1
                    record_discard(discarded_markets, d, f"detail_{reason}", stage="detail")
                    continue

                # Determine resolved outcome (Yes/No)
                resolved = decode_resolved_outcome(d)
                if resolved not in ("Yes", "No"):
                    discard["no_confident_resolved_outcome"] += 1
                    record_discard(discarded_markets, d, "no_confident_resolved_outcome", stage="detail")
                    continue

                # Ticker extraction (hard requirement)
                q = (d.get("question") or "").strip()
                ticker = extract_ticker_from_question(q)
                if not ticker:
                    discard["no_ticker_in_question"] += 1
                    record_discard(discarded_markets, d, "no_ticker_in_question", stage="finalize")
                    continue

                if resolved == "Yes":
                    new_yes += 1
                else:
                    new_no += 1

                # Ensure earnings tag present after detail
                tag_slugs = extract_tag_slugs(d.get("tags"))
                if not tag_slugs:
                    # Fallback API call if tags are missing
                    tags = fetch_market_tags(mid)
                    tag_slugs = [
                        t.get("slug") for t in tags
                        if isinstance(t, dict) and isinstance(t.get("slug"), str)
                    ]

                if "earnings" not in {str(x).lower() for x in tag_slugs if isinstance(x, str)}:
                    discard["missing_earnings_tag_after_detail"] += 1
                    record_discard(discarded_markets, d, "missing_earnings_tag_after_detail", stage="detail")
                    continue

                # Normalize outcomes for clean downstream use
                outs = parse_json_list_maybe(d.get("outcomes")) or ["Yes", "No"]
                outs_norm = normalize_yes_no(outs) or ["Yes", "No"]

                # Build final record
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
        # ---------------------------------------------------------------------
        # Step 5: Merge outputs (incremental or full refresh)
        # ---------------------------------------------------------------------
        def sort_key(r: Dict[str, Any]) -> Tuple[int, datetime]:
            dt = parse_iso_dt(r.get("endDate"))
            return (0, dt) if dt else (1, datetime.max.replace(tzinfo=timezone.utc))

        if INCREMENTAL_MODE:
            merged_by_id: Dict[str, Dict[str, Any]] = dict(existing_by_id)

            added = 0
            for r in final_new:
                rid = str(r.get("id", "")).strip()
                if not rid:
                    continue
                # Requirement: do not overwrite existing records
                if rid in merged_by_id:
                    continue
                merged_by_id[rid] = r
                added += 1

            output_records = list(merged_by_id.values())
            new_added = added
        else:
            output_records = list(final_new)
            new_added = len(final_new)

        # Best-effort backfill ticker for any record missing it
        upgraded_missing_ticker = 0
        for r in output_records:
            if r.get("ticker"):
                continue
            q = r.get("question") or ""
            t = extract_ticker_from_question(str(q))
            if t:
                r["ticker"] = t
                upgraded_missing_ticker += 1

        # Sort output for stability (by endDate)
        output_records.sort(key=sort_key)

        # ---------------------------------------------------------------------
        # Step 6: Write outputs (JSONL, discarded JSONL, CSV)
        # ---------------------------------------------------------------------
        log(f"Writing JSONL -> {OUT_JSONL} (records={len(output_records)})")
        jsonl_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output_records)
        atomic_write_text(OUT_JSONL, jsonl_text)

        log(f"Writing discarded JSONL -> {OUT_DISCARDED_JSONL} (records={len(discarded_markets)})")
        discarded_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in discarded_markets)
        atomic_write_text(OUT_DISCARDED_JSONL, discarded_text)

        log(f"Writing CSV -> {OUT_CSV}")
        fields = [
            "id", "ticker", "slug", "question", "endDate", "resolutionSource",
            "resolvedOutcome", "closed", "active", "archived", "restricted",
            "resolvedBy", "category", "volumeNum", "liquidityNum", "tags"
        ]
        tmp_csv = OUT_CSV.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in output_records:
                row = dict(r)
                row["tags"] = ",".join([t for t in (r.get("tags") or []) if isinstance(t, str)])
                w.writerow({k: row.get(k) for k in fields})
        tmp_csv.replace(OUT_CSV)

        # ---------------------------------------------------------------------
        # Step 7: Summary statistics (for human inspection)
        # ---------------------------------------------------------------------
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
        mode_str = "incremental" if INCREMENTAL_MODE else "full_refresh"

        lines: List[str] = []
        lines.append("Polymarket Earnings Resolved Markets — Summary")
        lines.append("=" * 52)
        lines.append(f"Generated (UTC): {format_dt_utc(finished_dt)}")
        lines.append(f"Mode: {mode_str}")
        if INCREMENTAL_MODE:
            lines.append(f"Existing loaded: {existing_loaded} (parse_errors={existing_parse_errors})")
            lines.append(f"New markets added this run: {new_added}")
            if upgraded_missing_ticker:
                lines.append(f"Existing records ticker backfilled during write: {upgraded_missing_ticker}")
        lines.append("")
        lines.append("Outputs")
        lines.append(f"- Markets JSONL:     {OUT_JSONL}")
        lines.append(f"- Markets CSV:       {OUT_CSV}")
        lines.append(f"- Discarded JSONL:   {OUT_DISCARDED_JSONL} (this run)")
        lines.append(f"- Summary TXT:       {OUT_SUMMARY}")
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
        lines.append(f"- Page limit: {PAGE_LIMIT}")
        lines.append(f"- Max workers: {MAX_WORKERS}")
        lines.append(f"- END_DATE_MIN: {END_DATE_MIN}")
        lines.append(f"- END_DATE_MAX: {END_DATE_MAX}")
        lines.append(f"- RELATED_TAGS: {RELATED_TAGS}")
        lines.append("")

        summary_text = "\n".join(lines)

        log(f"Writing summary -> {OUT_SUMMARY}")
        atomic_write_text(OUT_SUMMARY, summary_text)

        log("DONE (or cancelled). Summary:")
        print(summary_text, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(1)
