#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =========================
# TEST MODE
# =========================
TEST = True
TEST_MAX_PAGES = 2
TEST_MAX_CANDIDATES = 60

# =========================
# STATUS / LOGGING
# =========================
STATUS_EVERY_SECONDS = 15          # heartbeat
PRINT_RETRY_DETAILS = True         # print when retrying (429/5xx/network)

# =========================
# API / performance
# =========================
GAMMA = "https://gamma-api.polymarket.com"
HTTP_TIMEOUT = 25
RETRIES = 3
RETRY_SLEEP_S = 0.8

PAGE_LIMIT = 200
MAX_WORKERS = 3

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "polymarket-earnings-resolved-fetcher/1.2",
}

# =========================
# Output paths (script lives in Corporate_Earnings/)
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "earnings_resolved_markets.jsonl"
OUT_CSV = OUT_DIR / "earnings_resolved_markets.csv"
OUT_SUMMARY = OUT_DIR / "earnings_resolved_summary.json"
OUT_DISCARDED_JSONL = OUT_DIR / "earnings_discarded_markets.jsonl"

# =========================
# Strict filters (discard if unclear)
# =========================
TICKER_RE = re.compile(r"\(([A-Z]{1,6})\)")
BEAT_EARNINGS_RE = re.compile(r"\bbeat\b.*\bearnings\b", re.IGNORECASE)
QUARTERLY_HINT_RE = re.compile(r"\bquarterly\b", re.IGNORECASE)
SLUG_HINT_RE = re.compile(r"quarterly-earnings", re.IGNORECASE)
ESTIMATE_HINT_RE = re.compile(r"\b(consensus|estimate|estimated)\b", re.IGNORECASE)

# =========================
# Progress guard (avoid infinite paging)
# =========================
MAX_STALE_PAGES = 5   # stop after this many pages with 0 new market ids

# =========================
# Save logs when script stops
# =========================
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

# ---------------------------
# Logging helpers
# ---------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class Heartbeat:
    def __init__(self, every_s: int) -> None:
        self.every_s = every_s
        self._last = time.time()

    def maybe(self, msg: str) -> None:
        now = time.time()
        if now - self._last >= self.every_s:
            self._last = now
            log(msg)


# ---------------------------
# HTTP helpers
# ---------------------------

def _request_json(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
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
                "response": payload if isinstance(payload, (dict, list)) else str(payload)[:500],
            }

            # Retry on 429/5xx
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                if PRINT_RETRY_DETAILS:
                    log(f"Retrying ({attempt+1}/{RETRIES}) after HTTP {resp.status_code} for {url} params={params}")
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
                continue

            return None, last_err

        except Exception as e:
            last_err = {"status_code": None, "url": url, "params": params, "exception": repr(e)}
            if attempt < RETRIES:
                if PRINT_RETRY_DETAILS:
                    log(f"Retrying ({attempt+1}/{RETRIES}) after exception for {url}: {repr(e)}")
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
                continue
            return None, last_err

    return None, last_err


def gamma_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    return _request_json("GET", f"{GAMMA}{path}", params=params)


# ---------------------------
# Parsing helpers
# ---------------------------

def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
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

def record_discard(discarded: List[Dict[str, Any]], m: Dict[str, Any], reason: str, stage: str) -> None:
    discarded.append({
        "stage": stage,                 # "scan" or "detail" or "finalize"
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
        # Optional but helpful for later inspection:
        "description": m.get("description"),
        "tags": extract_tag_slugs(m.get("tags")),
    })

def parse_json_list_maybe(v: Any) -> Optional[List[Any]]:
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
    outs = [str(x).strip() for x in outcomes]
    if len(outs) != 2:
        return None
    lower = [o.lower() for o in outs]
    if set(lower) != {"yes", "no"}:
        return None
    return ["Yes", "No"]


def extract_tag_slugs(tags_payload: Any) -> List[str]:
    if not isinstance(tags_payload, list):
        return []
    out: List[str] = []
    for t in tags_payload:
        if isinstance(t, dict) and isinstance(t.get("slug"), str):
            out.append(t["slug"])
    return out


# ---------------------------
# Tag id lookup
# ---------------------------

def get_tag_id_by_slug(slug: str) -> Optional[int]:
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


# ---------------------------
# Strict market classifier
# ---------------------------

def looks_like_strict_earnings_beat_market(m: Dict[str, Any]) -> Tuple[bool, str]:
    q = (m.get("question") or "").strip()
    slug = (m.get("slug") or "").strip()
    desc = (m.get("description") or "").strip()

    if not q or not slug:
        return False, "missing_question_or_slug"

    outs = parse_json_list_maybe(m.get("outcomes"))
    if not isinstance(outs, list) or not normalize_yes_no(outs):
        return False, "not_binary_yes_no"

    if not TICKER_RE.search(q):
        return False, "no_ticker_in_question"

    if not BEAT_EARNINGS_RE.search(q):
        return False, "question_not_beat_earnings"

    if not (SLUG_HINT_RE.search(slug) or QUARTERLY_HINT_RE.search(q)):
        return False, "no_quarterly_hint"

    if not desc:
        return False, "missing_description"
    if not ESTIMATE_HINT_RE.search(desc):
        return False, "missing_estimate_hint"

    if m.get("closed") is not True:
        return False, "not_closed"

    return True, "ok"


# ---------------------------
# Resolution outcome extraction (best effort)
# ---------------------------

def decode_resolved_outcome(m: Dict[str, Any]) -> Optional[str]:
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


# ---------------------------
# Fetch markets list (paged)
# ---------------------------

def list_closed_markets_by_tag(tag_id: int) -> Iterable[Dict[str, Any]]:
    offset = 0
    pages = 0
    hb = Heartbeat(STATUS_EVERY_SECONDS)

    seen_ids: set[str] = set()
    stale_pages = 0

    while True:
        params = {
            "limit": PAGE_LIMIT,
            "offset": offset,
            "tag_id": tag_id,
            "closed": True,
            "include_tag": True,
            "order": "endDate",
            "ascending": True,
        }

        hb.maybe(f"Fetching /markets page={pages+1} offset={offset} ...")
        payload, err = gamma_get("/markets", params=params)
        if err:
            raise RuntimeError(f"/markets error offset={offset}: {err}")

        if not isinstance(payload, list) or not payload:
            log(f"/markets returned no more data at offset={offset}.")
            break

        # progress guard: count how many *new* ids we got on this page
        page_ids = []
        for m in payload:
            if isinstance(m, dict) and m.get("id") is not None:
                page_ids.append(str(m["id"]))

        new_ids = [mid for mid in page_ids if mid not in seen_ids]
        if len(new_ids) == 0:
            stale_pages += 1
            log(f"WARNING: No new market IDs on page={pages+1} offset={offset} (stale_pages={stale_pages}/{MAX_STALE_PAGES}).")
            if stale_pages >= MAX_STALE_PAGES:
                log("Stopping pagination due to no progress (repeated pages / no new IDs).")
                break
        else:
            stale_pages = 0
            for mid in new_ids:
                seen_ids.add(mid)

        log(f"Fetched /markets page={pages+1} -> {len(payload)} markets (new_ids={len(new_ids)}).")
        for m in payload:
            if isinstance(m, dict):
                yield m

        offset += PAGE_LIMIT
        pages += 1

        if TEST and pages >= TEST_MAX_PAGES:
            log(f"TEST mode: stopping after {pages} pages.")
            break


def fetch_market_detail(market_id: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    payload, err = gamma_get(f"/markets/{market_id}")
    if err or not isinstance(payload, dict):
        return market_id, None, err or {"note": "non-dict payload"}
    return market_id, payload, None


def fetch_market_tags(market_id: str) -> List[Dict[str, Any]]:
    payload, _ = gamma_get(f"/markets/{market_id}/tags")
    return payload if isinstance(payload, list) else []


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    started = time.time()
    hb = Heartbeat(STATUS_EVERY_SECONDS)
    cancelled = False
    cancel_stage = None

    discard = Counter()
    discarded_markets: List[Dict[str, Any]] = []
    scanned = 0
    candidates: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}
    detail_errors: List[Dict[str, Any]] = []
    final: List[Dict[str, Any]] = []
    yes_count = 0
    no_count = 0

    log(f"Starting. TEST={TEST}. Output dir: {OUT_DIR}")

    try:
        earnings_tag_id = get_tag_id_by_slug("earnings")
        if earnings_tag_id is None:
            raise SystemExit("Could not resolve earnings tag id. (Network / API issue)")

        # 1) Scan closed markets under Earnings tag
        cancel_stage = "scan_markets"
        log("Scanning closed markets with Earnings tag...")
        for m in list_closed_markets_by_tag(earnings_tag_id):
            scanned += 1
            if scanned % 50 == 0:
                hb.maybe(f"Scan progress: scanned={scanned}, candidates={len(candidates)}, discards={sum(discard.values())}")

            ok, reason = looks_like_strict_earnings_beat_market(m)
            if not ok:
                discard[reason] += 1
                record_discard(discarded_markets, m, reason, stage="scan")
                continue

            candidates.append(m)

            if TEST and len(candidates) >= TEST_MAX_CANDIDATES:
                log(f"TEST mode: reached {TEST_MAX_CANDIDATES} candidates, stopping scan early.")
                break

        log(f"Scan done. scanned={scanned}, candidates={len(candidates)}, discards={sum(discard.values())}")

        # 2) Detail fetch
        cancel_stage = "fetch_details"
        if candidates:
            ids = []
            for m in candidates:
                mid = str(m.get("id", "")).strip()
                if not mid:
                    discard["missing_id"] += 1
                    continue
                ids.append(mid)

            log(f"Fetching market details for {len(ids)} candidates (threads={MAX_WORKERS}) ...")
            completed = 0

            use_threads = (len(ids) >= 10 and MAX_WORKERS > 1)
            if use_threads:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = [ex.submit(fetch_market_detail, mid) for mid in ids]
                    try:
                        for fut in as_completed(futs):
                            mid, payload, err = fut.result()
                            completed += 1
                            if completed % 10 == 0:
                                hb.maybe(f"Detail progress: {completed}/{len(ids)} done, ok={len(details)}, errors={len(detail_errors)}")

                            if err or payload is None:
                                detail_errors.append({"market_id": mid, "error": err})
                                continue
                            details[mid] = payload
                    except KeyboardInterrupt:
                        cancelled = True
                        log("Ctrl+C received أثناء detail fetch. Will save what we have so far...")
                        # Cancel pending futures if possible (Python 3.9+)
                        ex.shutdown(wait=False, cancel_futures=True)
            else:
                for mid in ids:
                    try:
                        mid, payload, err = fetch_market_detail(mid)
                    except KeyboardInterrupt:
                        cancelled = True
                        log("Ctrl+C received أثناء detail fetch. Will save what we have so far...")
                        break

                    completed += 1
                    hb.maybe(f"Detail progress: {completed}/{len(ids)} done, ok={len(details)}, errors={len(detail_errors)}")
                    if err or payload is None:
                        detail_errors.append({"market_id": mid, "error": err})
                        continue
                    details[mid] = payload

            log(f"Detail fetch done. ok={len(details)} errors={len(detail_errors)}")

        # 3) Build final records
        cancel_stage = "finalize"
        log("Building final resolved dataset (strict)...")

        for i, m in enumerate(candidates, start=1):
            hb.maybe(f"Finalize progress: {i}/{len(candidates)} processed, kept={len(final)}")

            mid = str(m.get("id", "")).strip()
            d = details.get(mid)
            if not d:
                discard["detail_missing"] += 1
                continue

            ok, reason = looks_like_strict_earnings_beat_market(d)
            if not ok:
                discard[f"detail_{reason}"] += 1
                record_discard(discarded_markets, d, f"detail_{reason}", stage="detail")
                continue

            resolved = decode_resolved_outcome(d)
            if resolved not in ("Yes", "No"):
                discard["no_confident_resolved_outcome"] += 1
                record_discard(discarded_markets, d, f"detail_{reason}", stage="detail")
                continue

            if resolved == "Yes":
                yes_count += 1
            else:
                no_count += 1

            tag_slugs = extract_tag_slugs(d.get("tags"))
            if not tag_slugs:
                tags = fetch_market_tags(mid)
                tag_slugs = [t.get("slug") for t in tags if isinstance(t, dict) and isinstance(t.get("slug"), str)]

            if "earnings" not in {str(x).lower() for x in tag_slugs if isinstance(x, str)}:
                discard["missing_earnings_tag_after_detail"] += 1
                record_discard(discarded_markets, d, f"detail_{reason}", stage="detail")
                continue

            outs = parse_json_list_maybe(d.get("outcomes")) or ["Yes", "No"]
            outs_norm = normalize_yes_no(outs) or ["Yes", "No"]

            rec = {
                "id": mid,
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
            final.append(rec)

    except KeyboardInterrupt:
        cancelled = True
        log("Ctrl+C received. Will save what we have so far...")

    finally:
        # Always sort + write what we have so far
        def sort_key(r: Dict[str, Any]) -> Tuple[int, datetime]:
            dt = parse_iso_dt(r.get("endDate"))
            return (0, dt) if dt else (1, datetime.max.replace(tzinfo=timezone.utc))

        final.sort(key=sort_key)

        log(f"Writing JSONL -> {OUT_JSONL} (records={len(final)})")
        jsonl_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final)
        atomic_write_text(OUT_JSONL, jsonl_text)

        log(f"Writing discarded JSONL -> {OUT_DISCARDED_JSONL} (records={len(discarded_markets)})")
        discarded_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in discarded_markets)
        atomic_write_text(OUT_DISCARDED_JSONL, discarded_text)

        log(f"Writing CSV -> {OUT_CSV}")
        fields = [
            "id", "slug", "question", "endDate", "resolutionSource",
            "resolvedOutcome", "closed", "active", "archived", "restricted",
            "resolvedBy", "category", "volumeNum", "liquidityNum", "tags"
        ]
        # atomic CSV write
        tmp_csv = OUT_CSV.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in final:
                row = dict(r)
                row["tags"] = ",".join([t for t in (r.get("tags") or []) if isinstance(t, str)])
                w.writerow({k: row.get(k) for k in fields})
        tmp_csv.replace(OUT_CSV)

        end_dates = [parse_iso_dt(r.get("endDate")) for r in final]
        end_dates = [d for d in end_dates if d is not None]
        earliest = min(end_dates).isoformat().replace("+00:00", "Z") if end_dates else None
        latest = max(end_dates).isoformat().replace("+00:00", "Z") if end_dates else None

        volumes = []
        for r in final:
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

        summary = {
            "run": {
                "test_mode": TEST,
                "cancelled": cancelled,
                "cancel_stage": cancel_stage,
                "started_at_unix": int(started),
                "finished_at_unix": int(time.time()),
                "elapsed_seconds": round(time.time() - started, 3),
                "page_limit": PAGE_LIMIT,
                "max_workers": MAX_WORKERS,
            },
            "counts": {
                "scanned_closed_earnings_tag_markets": scanned,
                "candidates_after_strict_filter": len(candidates),
                "details_fetched": len(details),
                "final_resolved_markets_saved": len(final),
            },
            "endDate_range": {"earliest_endDate": earliest, "latest_endDate": latest},
            "volumeNum_stats": stats(volumes),
            "discarded_jsonl": str(OUT_DISCARDED_JSONL),
            "discard_reasons": dict(discard),
            "detail_errors_count": len(detail_errors),
            "outputs": {"jsonl": str(OUT_JSONL), "csv": str(OUT_CSV), "summary": str(OUT_SUMMARY)},
            "discarded_markets_logged": len(discarded_markets),
        }

        log(f"Writing summary -> {OUT_SUMMARY}")
        atomic_write_text(OUT_SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2))

        log("DONE (or cancelled). Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(1)
