#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_single_market_by_slug.py

Fetch as much information as possible about ONE Polymarket market, identified by slug.

What it pulls (best effort):
- Gamma:
  - GET /markets/slug/{slug}
  - GET /markets/{id}
  - GET /markets/{id}/tags
  - GET /comments?parent_entity_type=market&parent_entity_id={id} (paginated)
  - For each related event:
      - GET /events/{id}
      - GET /events/{id}/tags
  - For each related series:
      - GET /series/{id}
- CLOB (optional):
  - GET /markets/{condition_id}
  - GET /book?token_id=...
  - GET /price?token_id=...&side=BUY|SELL
  - GET /prices-history?market=...&interval=...&fidelity=...

Prints a single JSON object to console.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import requests

# =========================
# Configure the slug here
# =========================
MARKET_SLUG = "payx-quarterly-earnings-nongaap-eps-12-18-2025-1pt23"

# =========================
# Optional knobs
# =========================
INCLUDE_COMMENTS = False
MAX_COMMENTS_TOTAL = 300  # can be big; increase if you really want "everything"

INCLUDE_CLOB = True
INCLUDE_PRICE_HISTORY = False
PRICE_HISTORY_INTERVAL = "1w"   # one of: 1m, 1w, 1d, 6h, 1h, max
PRICE_HISTORY_FIDELITY = 15     # minutes per datapoint (smaller -> more points)

# =========================
# API configuration
# =========================
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

HTTP_TIMEOUT = 20
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "polymarket-single-market-fetcher/1.0",
}

# ---------------------------
# HTTP helpers
# ---------------------------

def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = HTTP_TIMEOUT,
    retries: int = 2,
    retry_sleep_s: float = 0.6,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Returns (json_or_text, error_dict).
    error_dict is None on success.
    """
    h = dict(HEADERS)
    if headers:
        h.update(headers)

    last_err: Optional[Dict[str, Any]] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data,
                headers=h,
                timeout=timeout,
            )
            # Try to parse JSON either way (even errors)
            content_type = (resp.headers.get("Content-Type") or "").lower()
            payload: Any
            if "application/json" in content_type:
                try:
                    payload = resp.json()
                except Exception:
                    payload = resp.text
            else:
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
                "response": payload,
            }

            # Retry on 429/5xx
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(retry_sleep_s * (attempt + 1))
                continue

            return None, last_err

        except Exception as e:
            last_err = {
                "status_code": None,
                "url": url,
                "params": params,
                "exception": repr(e),
            }
            if attempt < retries:
                time.sleep(retry_sleep_s * (attempt + 1))
                continue
            return None, last_err

    return None, last_err


def _gamma_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    return _request_json("GET", f"{GAMMA}{path}", params=params)


def _clob_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    return _request_json("GET", f"{CLOB}{path}", params=params)


# ---------------------------
# Parsing helpers
# ---------------------------

def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _normalize_token_ids(clob_token_ids: Any) -> List[str]:
    """
    Gamma sometimes returns clobTokenIds/clobTokenIDs as:
    - list[str]
    - list[int]
    - JSON-encoded string of a list
    - comma-separated string
    - single string token id
    """
    if clob_token_ids is None:
        return []

    if isinstance(clob_token_ids, list):
        out: List[str] = []
        for x in clob_token_ids:
            if x is None:
                continue
            out.append(str(x))
        return out

    if isinstance(clob_token_ids, (int, float)):
        return [str(int(clob_token_ids))]

    if isinstance(clob_token_ids, str):
        s = clob_token_ids.strip()
        if not s:
            return []

        # Try JSON list
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"[') and s.endswith(']"')):
            try:
                decoded = json.loads(s)
                return _normalize_token_ids(decoded)
            except Exception:
                pass

        # Common delimiters
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            return [p for p in parts if p]

        # Fallback single token id
        return [s]

    # Unknown type
    return [str(clob_token_ids)]


def _collect_event_ids_from_market(market_obj: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for ev in _as_list(market_obj.get("events")):
        if isinstance(ev, dict) and ev.get("id") is not None:
            ids.add(str(ev["id"]))
    return ids


def _collect_series_ids_from_event(event_obj: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for s in _as_list(event_obj.get("series")):
        if isinstance(s, dict) and s.get("id") is not None:
            ids.add(str(s["id"]))
    return ids


# ---------------------------
# Gamma fetchers
# ---------------------------

def fetch_market_by_slug(slug: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"slug": slug}

    payload, err = _gamma_get(f"/markets/slug/{slug}")
    out["market_by_slug"] = payload
    out["market_by_slug_error"] = err

    if not isinstance(payload, dict):
        # Can't go further reliably
        return out

    market_id = payload.get("id")
    out["market_id"] = market_id

    if market_id is None:
        return out

    market_id_str = str(market_id)

    payload2, err2 = _gamma_get(f"/markets/{market_id_str}")
    out["market_by_id"] = payload2
    out["market_by_id_error"] = err2

    tags, err_tags = _gamma_get(f"/markets/{market_id_str}/tags")
    out["market_tags"] = tags
    out["market_tags_error"] = err_tags

    # Comments (optional)
    if INCLUDE_COMMENTS:
        comments: List[Any] = []
        errors: List[Dict[str, Any]] = []
        limit = 100
        offset = 0

        while len(comments) < MAX_COMMENTS_TOTAL:
            page, page_err = _gamma_get(
                "/comments",
                params={
                    "limit": str(limit),
                    "offset": str(offset),
                    "parent_entity_type": "market",
                    "parent_entity_id": str(market_id),
                    "ascending": "false",
                    "get_positions": "true",
                },
            )
            if page_err:
                errors.append(page_err)
                break

            if not isinstance(page, list) or not page:
                break

            comments.extend(page)
            if offering_done := (len(page) < limit):
                break

            offset += limit

        out["market_comments"] = comments
        out["market_comments_errors"] = errors

    # Related events (from the richest market object we have)
    base_market_obj = payload2 if isinstance(payload2, dict) else payload
    event_ids = _collect_event_ids_from_market(base_market_obj)

    out["related_events"] = {}
    out["related_event_tags"] = {}
    series_ids: Set[str] = set()

    for eid in sorted(event_ids):
        ev, ev_err = _gamma_get(f"/events/{eid}")
        out["related_events"][eid] = {"data": ev, "error": ev_err}

        ev_tags, ev_tags_err = _gamma_get(f"/events/{eid}/tags")
        out["related_event_tags"][eid] = {"data": ev_tags, "error": ev_tags_err}

        if isinstance(ev, dict):
            series_ids |= _collect_series_ids_from_event(ev)

    # Series details
    out["related_series"] = {}
    for sid in sorted(series_ids):
        s, s_err = _gamma_get(f"/series/{sid}")
        out["related_series"][sid] = {"data": s, "error": s_err}

    return out


# ---------------------------
# CLOB fetchers (optional)
# ---------------------------

def fetch_clob_for_market(gamma_market_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses Gamma's conditionId and clobTokenIds to pull public CLOB data.
    """
    out: Dict[str, Any] = {}

    condition_id = gamma_market_obj.get("conditionId")
    out["condition_id"] = condition_id

    token_ids = _normalize_token_ids(
        gamma_market_obj.get("clobTokenIds", gamma_market_obj.get("clob_token_ids"))
    )
    out["token_ids"] = token_ids

    # Market details by condition id
    if condition_id:
        m, m_err = _clob_get(f"/markets/{condition_id}")
        out["clob_market"] = m
        out["clob_market_error"] = m_err
    else:
        out["clob_market"] = None
        out["clob_market_error"] = {"note": "No conditionId found in Gamma market payload."}

    # Order books + prices
    out["books"] = {}
    out["prices"] = {}
    out["price_history"] = {}

    for tid in token_ids:
        # Book
        b, b_err = _clob_get("/book", params={"token_id": tid})
        out["books"][tid] = {"data": b, "error": b_err}

        # Best prices (public)
        p_buy, p_buy_err = _clob_get("/price", params={"token_id": tid, "side": "BUY"})
        p_sell, p_sell_err = _clob_get("/price", params={"token_id": tid, "side": "SELL"})
        out["prices"][tid] = {
            "BUY": {"data": p_buy, "error": p_buy_err},
            "SELL": {"data": p_sell, "error": p_sell_err},
        }

        # Price history
        if INCLUDE_PRICE_HISTORY:
            h, h_err = _clob_get(
                "/prices-history",
                params={
                    "market": tid,
                    "interval": PRICE_HISTORY_INTERVAL,
                    "fidelity": str(PRICE_HISTORY_FIDELITY),
                },
            )
            out["price_history"][tid] = {"data": h, "error": h_err}

    return out


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    started = time.time()

    gamma_bundle = fetch_market_by_slug(MARKET_SLUG)

    result: Dict[str, Any] = {
        "requested_slug": MARKET_SLUG,
        "fetched_at_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started, 3),
        "gamma": gamma_bundle,
    }

    # If we have a Gamma market object, fetch CLOB extras (optional)
    if INCLUDE_CLOB:
        gamma_market_obj = gamma_bundle.get("market_by_slug")
        if isinstance(gamma_market_obj, dict):
            result["clob"] = fetch_clob_for_market(gamma_market_obj)
        else:
            result["clob"] = {
                "error": "Gamma market_by_slug was not an object; cannot derive conditionId/clobTokenIds."
            }

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False))


if __name__ == "__main__":
    if MARKET_SLUG == "put-market-slug-here":
        raise SystemExit("Set MARKET_SLUG at the top of the script before running.")
    main()
