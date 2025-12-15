#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_markets.py

Modes
-----
- bettable: closed=false + local bettable checks (default, same as before)
- closed:   closed=true (all expired/closed markets)
- resolved: closed=true + uma_resolution_status=resolved (server-side if supported)
            + local defensive "resolved-ish" check

Usage
-----
  python fetch_markets.py --mode bettable
  python fetch_markets.py --mode closed
  python fetch_markets.py --mode resolved
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# --------------------------- Configuration -----------------------------

GAMMA = "https://gamma-api.polymarket.com"
HTTP_TIMEOUT = 20
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "polymarket-market-fetcher/1.0"
}

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / "config" / ".env"

TEST_MARKET_LIMIT = 5

MARKETS_PAGE_LIMIT = 200
MAX_PAGES = 1000  # closed/resolved history is large; bump this up

# --------------------------- Helpers -----------------------------------

def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return r.text


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        val = v.strip().lower()
        if val in ("true", "1", "yes", "y"):
            return True
        if val in ("false", "0", "no", "n"):
            return False
    return bool(v)


def _load_test_mode_from_env(env_path: Path) -> bool:
    if not env_path.exists():
        return False
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("TEST_MODE"):
                    _, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    return value.lower() == "true"
    except Exception:
        return False
    return False


# --------------------------- Gamma fetchers ----------------------------

def _fetch_markets_page(
    limit: int = MARKETS_PAGE_LIMIT,
    offset: int = 0,
    closed: bool = False,
    uma_resolution_status: Optional[str] = None,
    order: str = "id",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch a single page of markets from Gamma.

    Gamma supports `closed` and also `uma_resolution_status` as a filter. :contentReference[oaicite:1]{index=1}
    """
    url = f"{GAMMA}/markets"
    params: Dict[str, Any] = {
        "limit": str(limit),
        "offset": str(offset),
        "closed": "true" if closed else "false",
        "order": order,
        "ascending": "true" if ascending else "false",
    }
    if uma_resolution_status:
        params["uma_resolution_status"] = uma_resolution_status

    payload = _get(url, params=params)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    return []


def _fetch_all_markets(
    *,
    closed: bool,
    uma_resolution_status: Optional[str] = None,
    max_pages: int = MAX_PAGES,
) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    limit = MARKETS_PAGE_LIMIT
    offset = 0
    pages = 0

    while pages < max_pages:
        page = _fetch_markets_page(
            limit=limit,
            offset=offset,
            closed=closed,
            uma_resolution_status=uma_resolution_status,
        )
        if not page:
            break

        markets.extend(page)
        pages += 1

        if len(page) < limit:
            break

        offset += limit

    return markets


# --------------------------- Filters -----------------------------------

def is_bettable(market: Dict[str, Any]) -> bool:
    if _coerce_bool(market.get("closed"), False):
        return False
    if not _coerce_bool(market.get("enableOrderBook"), False):
        return False
    if "active" in market and not _coerce_bool(market.get("active"), True):
        return False
    if "acceptingOrders" in market and not _coerce_bool(market.get("acceptingOrders"), True):
        return False
    return True


def is_resolved(market: Dict[str, Any]) -> bool:
    """
    Defensive "resolved" check, because different Gamma payloads/versions may expose
    different fields. We treat a market as resolved if it's closed AND has strong
    resolution signals (resolved flag, outcome, resolvedBy, or umaResolutionStatus).
    """
    if not _coerce_bool(market.get("closed"), False):
        return False

    if _coerce_bool(market.get("resolved"), False):
        return True

    if market.get("outcome") not in (None, "", "null"):
        return True

    if market.get("resolvedBy") not in (None, "", "null"):
        return True

    urs = (market.get("umaResolutionStatus") or "").strip().lower()
    if urs in {"resolved", "finalized", "settled", "invalid", "canceled", "cancelled"}:
        return True

    return False


# --------------------------- Data shaping ------------------------------

def _extract_resolution_time(market: Dict[str, Any]) -> Optional[str]:
    end_date = market.get("endDate")
    return end_date if isinstance(end_date, str) else None


def build_output(mode: str) -> Dict[str, Any]:
    test_mode = _load_test_mode_from_env(ENV_PATH)

    if mode == "bettable":
        all_markets = _fetch_all_markets(closed=False)
        picked = [m for m in all_markets if isinstance(m, dict) and is_bettable(m)]

    elif mode == "closed":
        all_markets = _fetch_all_markets(closed=True)
        picked = [m for m in all_markets if isinstance(m, dict) and _coerce_bool(m.get("closed"), False)]

    elif mode == "resolved":
        # Try server-side narrowing first, then defensive local check.
        # `uma_resolution_status` is documented as a filter param. :contentReference[oaicite:2]{index=2}
        all_markets = _fetch_all_markets(closed=True, uma_resolution_status="resolved")
        picked = [m for m in all_markets if isinstance(m, dict) and is_resolved(m)]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if test_mode:
        picked = picked[:TEST_MARKET_LIMIT]

    markets_out: List[Dict[str, Any]] = []
    for m in picked:
        markets_out.append({
            "id": m.get("id"),
            "slug": m.get("slug"),
            "question": m.get("question"),
            "endDate": _extract_resolution_time(m),
            "active": m.get("active"),
            "closed": m.get("closed"),
            "enableOrderBook": m.get("enableOrderBook"),
            "acceptingOrders": m.get("acceptingOrders"),
            "resolved": m.get("resolved"),
            "resolvedBy": m.get("resolvedBy"),
            "umaResolutionStatus": m.get("umaResolutionStatus"),
            "outcome": m.get("outcome"),
            "category": m.get("category"),
            "liquidity": m.get("liquidity"),
            "volume": m.get("volume"),
            "clobTokenIds": m.get("clobTokenIds"),
            "shortOutcomes": m.get("shortOutcomes"),
            "outcomes": m.get("outcomes"),
        })

    return {
        "mode": mode,
        "test_mode": test_mode,
        "markets_returned": len(markets_out),
        "markets": markets_out,
    }


# --------------------------- CLI / Entry -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket markets from Gamma.")
    parser.add_argument(
        "--mode",
        choices=["bettable", "closed", "resolved"],
        default="bettable",
        help="bettable=active tradable, closed=all closed/expired, resolved=resolved subset of closed",
    )
    args = parser.parse_args()

    result = build_output(args.mode)
    print(json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False))


if __name__ == "__main__":
    main()
