#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_poly_prices.py

Polymarket Earnings — Historical Snapshot Prices (YES + NO)

What this script does
---------------------
For each Polymarket orderbook market in an input JSONL file, this script fetches
historical price series for the YES and NO outcome tokens from the Polymarket
CLOB API and produces snapshot prices at fixed offsets before an anchor time.

Key behavior
------------
- Snapshot anchor uses the **OBSERVED market end** = last timestamp found in
  YES/NO history (instead of relying solely on Gamma endDate).
- Snapshot targets are (observed_end_ts - offset_seconds).
- Snapshot price is the last price with ts <= target_ts.
- Complement check flags labels where |YES + NO - 1| > tolerance.

Outputs
-------
- poly_prices.jsonl   (success records)
- failed_poly_markets.jsonl      (failure records)
- summary.txt               (human-readable run summary)

Notes for thesis/review
-----------------------
- Per-market outputs are JSONL (machine-readable).
- Summary remains TXT (human-readable).
- All timestamps in JSON outputs are UTC. Optional local debug fields are off by default.

"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

# -------------------------
# Snapshot spec (fixed)
# -------------------------
SNAPSHOTS: List[Tuple[str, int]] = [
    ("4w", 4 * 7 * 24 * 3600),
    ("3w", 3 * 7 * 24 * 3600),
    ("2w", 2 * 7 * 24 * 3600),
    ("1w", 1 * 7 * 24 * 3600),
    ("6d", 6 * 24 * 3600),
    ("5d", 5 * 24 * 3600),
    ("4d", 4 * 24 * 3600),
    ("3d", 3 * 24 * 3600),
    ("2d", 2 * 24 * 3600),
    ("1d", 1 * 24 * 3600),
    ("12h", 12 * 3600),
    ("6h", 6 * 3600),
]
MAX_OFFSET_SECONDS = max(s for _, s in SNAPSHOTS)


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class Config:
    gamma_base: str
    clob_base: str

    max_workers: int
    http_timeout: float
    retries: int
    retry_sleep_s: float

    price_fidelity_min: int
    min_fidelity_closed_min: int
    buffer_seconds: int

    complement_tolerance: float

    include_local_debug_fields: bool
    local_tz_name: str

    test_mode: bool
    test_max_markets: int

    user_agent: str


# -------------------------
# Timezone helpers
# -------------------------
def _get_zoneinfo(name: str):
    """
    Return ZoneInfo(name) if available (py3.9+), else None.
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return ZoneInfo(name)
    except Exception:
        return None


def parse_iso_dt(s: Any) -> Optional[datetime]:
    """
    Parse an ISO-8601 string into an aware UTC datetime.
    Returns None on failure.
    """
    if not s or not isinstance(s, str):
        return None
    try:
        ss = s.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def fmt_dt_utc(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def ts_to_dt(ts: Optional[int]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


# -------------------------
# File helpers (atomic)
# -------------------------
def atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Atomically write JSONL to `path` by streaming into a temp file then replacing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to `path` (UTF-8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts (skips invalid lines).
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


# -------------------------
# HTTP helpers
# -------------------------
def _preview_payload(payload: Any, limit: int = 1200) -> Any:
    if isinstance(payload, (dict, list)):
        return payload
    s = str(payload)
    return s[:limit]


def _request_json(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]],
    headers: Dict[str, str],
    timeout: float,
    retries: int,
    retry_sleep_s: float,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Make an HTTP request and parse JSON when possible.

    Retries on:
      - 429, 500, 502, 503, 504
      - network exceptions

    Returns: (payload, error_dict)
    """
    last_err: Optional[Dict[str, Any]] = None

    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
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
                "response": _preview_payload(payload),
            }

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                # exponential backoff with jitter
                sleep_s = retry_sleep_s * (2 ** attempt) * (0.8 + 0.4 * random.random())
                time.sleep(sleep_s)
                continue

            return None, last_err

        except Exception as e:
            last_err = {"status_code": None, "url": url, "params": params, "exception": repr(e)}
            if attempt < retries:
                sleep_s = retry_sleep_s * (2 ** attempt) * (0.8 + 0.4 * random.random())
                time.sleep(sleep_s)
                continue
            return None, last_err

    return None, last_err


def gamma_get(cfg: Config, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    headers = {"Accept": "application/json", "User-Agent": cfg.user_agent}
    return _request_json("GET", f"{cfg.gamma_base}{path}", params, headers, cfg.http_timeout, cfg.retries, cfg.retry_sleep_s)


def clob_get(cfg: Config, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    headers = {"Accept": "application/json", "User-Agent": cfg.user_agent}
    return _request_json("GET", f"{cfg.clob_base}{path}", params, headers, cfg.http_timeout, cfg.retries, cfg.retry_sleep_s)


# -------------------------
# Polymarket parsing helpers
# -------------------------
def parse_json_list_maybe(v: Any) -> Optional[List[Any]]:
    """
    Gamma sometimes returns JSON-encoded lists as strings; accept both.
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


def get_yes_no_token_ids(detail: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    From Gamma market detail, identify YES and NO clob token IDs.
    Returns (yes_id, no_id) or (None, None) if missing.
    """
    outcomes = parse_json_list_maybe(detail.get("outcomes")) or []
    token_ids = parse_json_list_maybe(detail.get("clobTokenIds")) or []

    if len(outcomes) < 2 or len(token_ids) < 2:
        return None, None

    outs_lower = [str(x).strip().lower() for x in outcomes]
    yes_id = None
    no_id = None

    if "yes" in outs_lower:
        i = outs_lower.index("yes")
        yes_id = str(token_ids[i]).strip() if token_ids[i] is not None else None

    if "no" in outs_lower:
        i = outs_lower.index("no")
        no_id = str(token_ids[i]).strip() if token_ids[i] is not None else None

    return yes_id, no_id


def _normalize_epoch_seconds(t_raw: int) -> int:
    """
    CLOB sometimes returns ms timestamps; normalize to seconds.
    """
    if t_raw > 10_000_000_000:  # very likely milliseconds
        return int(t_raw // 1000)
    return int(t_raw)


def build_series(history: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    """
    Convert CLOB history points into sorted (ts_list, p_list).
    Each point is expected to have:
      - t (epoch seconds or ms)
      - p (price)
    """
    pairs: List[Tuple[int, float]] = []
    for pt in history:
        try:
            t = _normalize_epoch_seconds(int(pt.get("t")))
            p = float(pt.get("p"))
            pairs.append((t, p))
        except Exception:
            continue

    if not pairs:
        return [], []
    pairs.sort(key=lambda x: x[0])
    return [t for t, _ in pairs], [p for _, p in pairs]


def pick_price_from_series(ts_list: List[int], p_list: List[float], target_ts: int) -> Tuple[Optional[float], Optional[int]]:
    """
    Return the last price with ts <= target_ts (right-continuous step series).
    """
    if not ts_list:
        return None, None
    idx = bisect_right(ts_list, int(target_ts)) - 1
    if idx < 0:
        return None, None
    return p_list[idx], ts_list[idx]


def any_price_present(prices: Dict[str, Optional[float]]) -> bool:
    return any(v is not None for v in prices.values())


def choose_query_end_ts(input_market: Dict[str, Any], gamma_detail: Dict[str, Any]) -> Optional[int]:
    """
    Choose a conservative end timestamp (UTC epoch seconds) for querying CLOB history.

    We prefer the *latest* timestamp among known close/resolution/end fields to reduce the
    risk of querying too early and missing late trading.

    Returns None if no timestamps are available.
    """
    candidates: List[datetime] = []

    # From input JSONL
    candidates += [
        parse_iso_dt(input_market.get("endDate")),
        parse_iso_dt(input_market.get("closedTime")),
        parse_iso_dt(input_market.get("resolvedTime")),
        parse_iso_dt(input_market.get("resolutionTime")),
    ]

    # From Gamma detail (field names vary across datasets)
    for k in ("endDate", "closedTime", "resolvedTime", "resolutionTime", "resolveTime", "end_time", "closeTime"):
        candidates.append(parse_iso_dt(gamma_detail.get(k)))

    dts = [d for d in candidates if d is not None]
    if not dts:
        return None
    return int(max(dts).timestamp())


def choose_created_ts(input_market: Dict[str, Any], gamma_detail: Dict[str, Any]) -> Optional[int]:
    """
    Choose an approximate created timestamp (UTC epoch seconds), if available.
    """
    candidates: List[datetime] = []
    candidates += [parse_iso_dt(input_market.get("createdAt")), parse_iso_dt(input_market.get("createdDate"))]
    for k in ("createdAt", "createdDate", "created_time"):
        candidates.append(parse_iso_dt(gamma_detail.get(k)))
    dts = [d for d in candidates if d is not None]
    if not dts:
        return None
    return int(min(dts).timestamp())


def clamp_start_ts(end_ts: int, created_ts: Optional[int], buffer_seconds: int) -> int:
    """
    Compute a start_ts that covers all snapshot offsets (max offset + buffer)
    while not going before market creation time (when available).
    """
    start_ts = end_ts - MAX_OFFSET_SECONDS - buffer_seconds
    if created_ts is not None:
        start_ts = max(start_ts, created_ts)
    return max(0, start_ts)


def fetch_prices_history_token(
    cfg: Config,
    token_id: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Fetch price history for a CLOB token.

    Strategy:
      1) Try range query (startTs/endTs) at fidelity_min, then fallback fidelity_closed
      2) Try interval=max at fidelity_min, then fallback fidelity_closed
      3) Try range without fidelity
      4) Try interval=max without fidelity

    Returns:
      - (history_list, None) on success (history may be empty list)
      - (None, error_dict) on hard failure
    """
    attempts: List[Dict[str, Any]] = []

    def try_call(params: Dict[str, Any], tag: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        payload, err = clob_get(cfg, "/prices-history", params=params)
        rec: Dict[str, Any] = {"tag": tag, "params": params, "err": err}
        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            rec["history_len"] = len(payload["history"])
            attempts.append(rec)
            return payload["history"], None
        rec["payload_type"] = type(payload).__name__ if payload is not None else None
        attempts.append(rec)
        return None, err or {"error": "unexpected_payload", "payload_preview": _preview_payload(payload, 500)}

    fids: List[int] = [int(cfg.price_fidelity_min)]
    if cfg.price_fidelity_min < cfg.min_fidelity_closed_min:
        fids.append(int(cfg.min_fidelity_closed_min))

    last_http_err: Optional[Dict[str, Any]] = None

    # Range queries (only if we have start/end)
    if start_ts is not None and end_ts is not None:
        for fid in fids:
            hist, e = try_call(
                {"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts), "fidelity": int(fid)},
                f"range_fid_{fid}",
            )
            if e is not None:
                last_http_err = e
                continue
            if hist is not None and len(hist) > 0:
                return hist, None

    # interval=max
    for fid in fids:
        hist, e = try_call({"market": token_id, "interval": "max", "fidelity": int(fid)}, f"max_fid_{fid}")
        if e is not None:
            last_http_err = e
            continue
        if hist is not None and len(hist) > 0:
            return hist, None

    # Range without fidelity (if possible)
    if start_ts is not None and end_ts is not None:
        hist3, e3 = try_call({"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts)}, "range_no_fid")
        if e3 is None and hist3 is not None and len(hist3) > 0:
            return hist3, None

    # interval=max without fidelity
    hist4, e4 = try_call({"market": token_id, "interval": "max"}, "max_no_fid")
    if e4 is None and hist4 is not None and len(hist4) > 0:
        return hist4, None

    # If any attempt succeeded (err=None) but returned empty history, treat as success with [].
    for a in attempts:
        if a.get("err") is None and isinstance(a.get("history_len"), int):
            return [], None

    return None, {"last_http_error": last_http_err, "attempts": attempts}


# -------------------------
# Worker
# -------------------------
def process_market(cfg: Config, m: Dict[str, Any], run_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single market record.

    Returns:
      (success_record, None) or (None, failure_record)
    """
    mid = str(m.get("id", "")).strip()
    slug = str(m.get("slug", "")).strip()

    if not mid or not slug:
        return None, {"run_id": run_id, "market_id": mid or None, "slug": slug or None, "reason": "missing_id_or_slug"}

    # 1) Gamma market detail (required for token ids)
    detail, derr = gamma_get(cfg, f"/markets/{mid}")
    if derr or not isinstance(detail, dict):
        return None, {"run_id": run_id, "market_id": mid, "slug": slug, "reason": "gamma_market_detail_failed", "error": derr}

    enable_ob = detail.get("enableOrderBook")
    if enable_ob is not True:
        return None, {"run_id": run_id, "market_id": mid, "slug": slug, "reason": "not_orderbook_market", "enableOrderBook": enable_ob}

    yes_token_id, no_token_id = get_yes_no_token_ids(detail)
    if not yes_token_id or not no_token_id:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "missing_yes_or_no_token_id",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
        }

    # 2) Choose query window for CLOB fetch
    query_end_ts = choose_query_end_ts(m, detail)
    created_ts = choose_created_ts(m, detail)

    # If we can't determine any end time, we fall back to interval=max only.
    if query_end_ts is not None:
        start_ts = clamp_start_ts(query_end_ts, created_ts, cfg.buffer_seconds)
        end_ts = query_end_ts
    else:
        start_ts = None
        end_ts = None

    # 3) Fetch histories
    yes_hist, yes_err = fetch_prices_history_token(cfg, yes_token_id, start_ts, end_ts)
    if yes_err or yes_hist is None:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "clob_prices_history_failed_yes",
            "yes_token_id": yes_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "error": yes_err,
        }

    no_hist, no_err = fetch_prices_history_token(cfg, no_token_id, start_ts, end_ts)
    if no_err or no_hist is None:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "clob_prices_history_failed_no",
            "no_token_id": no_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "error": no_err,
        }

    # 4) Build series + compute OBSERVED window
    yes_ts, yes_ps = build_series(yes_hist)
    no_ts, no_ps = build_series(no_hist)

    start_candidates: List[int] = []
    end_candidates: List[int] = []

    if yes_ts:
        start_candidates.append(yes_ts[0])
        end_candidates.append(yes_ts[-1])
    if no_ts:
        start_candidates.append(no_ts[0])
        end_candidates.append(no_ts[-1])

    if not end_candidates:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "empty_histories_after_parse",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
        }

    observed_start_ts = min(start_candidates) if start_candidates else None
    observed_end_ts = max(end_candidates)
    observed_span_hours = None
    if observed_start_ts is not None:
        observed_span_hours = round((observed_end_ts - observed_start_ts) / 3600.0, 6)

    # Anchor snapshots to observed end (last discovered price)
    anchor_end_ts = observed_end_ts

    # 5) Compute snapshots
    prices_yes: Dict[str, Optional[float]] = {}
    prices_no: Dict[str, Optional[float]] = {}
    missing_yes: List[str] = []
    missing_no: List[str] = []

    snapshot_targets_ts: Dict[str, int] = {}
    snapshot_source_ts_yes: Dict[str, Optional[int]] = {}
    snapshot_source_ts_no: Dict[str, Optional[int]] = {}

    for label, off in SNAPSHOTS:
        target_ts = int(anchor_end_ts - off)
        snapshot_targets_ts[label] = target_ts

        py, y_src = pick_price_from_series(yes_ts, yes_ps, target_ts)
        prices_yes[label] = py
        snapshot_source_ts_yes[label] = y_src
        if py is None:
            missing_yes.append(label)

        pn, n_src = pick_price_from_series(no_ts, no_ps, target_ts)
        prices_no[label] = pn
        snapshot_source_ts_no[label] = n_src
        if pn is None:
            missing_no.append(label)

    if not any_price_present(prices_yes) and not any_price_present(prices_no):
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "no_snapshot_prices_found",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "observed_end_ts": observed_end_ts,
        }

    # Complement diagnostics
    complement_violations: List[Dict[str, Any]] = []
    for label, _off in SNAPSHOTS:
        y = prices_yes.get(label)
        n = prices_no.get(label)
        if y is None or n is None:
            continue
        s = y + n
        if abs(s - 1.0) > cfg.complement_tolerance:
            complement_violations.append({"label": label, "yes": y, "no": n, "sum": s})

    # Debug timestamp formatting
    obs_start_dt = ts_to_dt(observed_start_ts)
    obs_end_dt = ts_to_dt(observed_end_ts)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "market_id": mid,
        "slug": slug,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "generated_utc": fmt_dt_utc(datetime.now(timezone.utc)),

        # Query metadata (for reproducibility)
        "gamma_detail_enableOrderBook": enable_ob,
        "query_start_ts": start_ts,
        "query_end_ts": end_ts,

        # Observed window (authoritative for snapshots)
        "observed_start_ts": observed_start_ts,
        "observed_end_ts": observed_end_ts,
        "observed_start_utc": fmt_dt_utc(obs_start_dt),
        "observed_end_utc": fmt_dt_utc(obs_end_dt),
        "observed_span_hours": observed_span_hours,

        # Snapshot timing audit
        "snapshot_anchor_end_ts": anchor_end_ts,
        "snapshot_targets_ts": snapshot_targets_ts,
        "snapshot_source_ts_yes": snapshot_source_ts_yes,
        "snapshot_source_ts_no": snapshot_source_ts_no,

        # Prices
        "prices_yes": prices_yes,
        "prices_no": prices_no,
        "missing_yes": missing_yes,
        "missing_no": missing_no,

        # Quality checks
        "complement_tolerance": cfg.complement_tolerance,
        "complement_violations": complement_violations,
    }

    # Optional local-time debug fields (off by default)
    if cfg.include_local_debug_fields:
        zi = _get_zoneinfo(cfg.local_tz_name)
        if zi is not None:
            def _fmt_local(dt: Optional[datetime]) -> str:
                if dt is None:
                    return "N/A"
                return dt.astimezone(zi).strftime("%Y-%m-%d %H:%M:%S%z")

            record["observed_start_local"] = _fmt_local(obs_start_dt)
            record["observed_end_local"] = _fmt_local(obs_end_dt)
            record["local_tz"] = cfg.local_tz_name

    return record, None


# -------------------------
# CLI / Main
# -------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Polymarket YES/NO historical snapshot prices (JSONL outputs + TXT summary).")

    p.add_argument("--input", type=str, default=None, help="Input markets JSONL path.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for JSONL + summary.txt")
    p.add_argument("--data-root", type=str, default=None, help="Project data root. If omitted, tries a Windows default then ./data")

    p.add_argument("--gamma-base", type=str, default=os.getenv("POLY_GAMMA_URL", "https://gamma-api.polymarket.com"))
    p.add_argument("--clob-base", type=str, default=os.getenv("POLY_CLOB_URL", "https://clob.polymarket.com"))

    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--http-timeout", type=float, default=25.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=0.8)

    p.add_argument("--price-fidelity-min", type=int, default=5, help="Default fidelity in minutes.")
    p.add_argument("--min-fidelity-closed-min", type=int, default=60 * 12, help="Fallback fidelity in minutes (closed markets).")
    p.add_argument("--buffer-seconds", type=int, default=2 * 3600, help="Extra buffer around snapshot window (seconds).")

    p.add_argument("--complement-tolerance", type=float, default=0.05)

    p.add_argument("--include-local-debug-fields", action="store_true", help="Include local time debug fields (not recommended for final datasets).")
    p.add_argument("--local-tz", type=str, default="Europe/Stockholm")

    p.add_argument("--test", action="store_true", help="Limit number of markets (quick dev runs).")
    p.add_argument("--test-max-markets", type=int, default=15)

    p.add_argument("--user-agent", type=str, default="polymarket-historical-prices/2.0")

    return p.parse_args(argv)

def _default_data_root(script_dir: Path) -> Path:
    """
    Default data-root behavior:
      1) If the common Windows thesis directory exists, use it.
      2) Else use ./data next to this script.
    """
    win_root = Path(r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data")
    if win_root.exists():
        return win_root
    return script_dir / "data"

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent

    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        data_root = _default_data_root(script_dir)

    data_root.mkdir(parents=True, exist_ok=True)

    default_input = data_root / "markets" / "markets.jsonl"
    default_out_dir = data_root / "poly_prices"

    input_path = Path(args.input).expanduser().resolve() if args.input else default_input
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prices_jsonl = out_dir / "poly_prices.jsonl"
    out_failed_jsonl = out_dir / "failed_poly_markets.jsonl"
    out_summary_txt = out_dir / "summary.txt"

    cfg = Config(
        gamma_base=str(args.gamma_base).rstrip("/"),
        clob_base=str(args.clob_base).rstrip("/"),
        max_workers=int(args.max_workers),
        http_timeout=float(args.http_timeout),
        retries=int(args.retries),
        retry_sleep_s=float(args.retry_sleep),
        price_fidelity_min=int(args.price_fidelity_min),
        min_fidelity_closed_min=int(args.min_fidelity_closed_min),
        buffer_seconds=int(args.buffer_seconds),
        complement_tolerance=float(args.complement_tolerance),
        include_local_debug_fields=bool(args.include_local_debug_fields),
        local_tz_name=str(args.local_tz),
        test_mode=bool(args.test),
        test_max_markets=int(args.test_max_markets),
        user_agent=str(args.user_agent),
    )

    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%SZ")

    markets = load_jsonl(input_path)
    total_before = len(markets)

    if cfg.test_mode:
        markets = markets[: max(0, cfg.test_max_markets)]

    tqdm.write(f"[{fmt_dt_utc(run_started)}] Run {run_id} starting")
    tqdm.write(f"- Input:  {input_path} ({total_before} rows; processing {len(markets)})")
    tqdm.write(f"- Output: {out_dir}")
    tqdm.write(f"- Workers: {cfg.max_workers} | timeout={cfg.http_timeout}s | retries={cfg.retries}")
    tqdm.write(f"- Fidelity: {cfg.price_fidelity_min}m (fallback {cfg.min_fidelity_closed_min}m)")
    tqdm.write(f"- Complement tolerance: {cfg.complement_tolerance}")

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    fail_reason_counts = Counter()
    missing_yes_counts = Counter()
    missing_no_counts = Counter()
    complement_violation_markets = 0
    partial_missing_markets = 0

    observed_starts: List[int] = []
    observed_ends: List[int] = []

    if cfg.max_workers <= 1:
        it = tqdm(markets, desc="Fetching snapshots", unit="market", dynamic_ncols=True)
        for m in it:
            ok, fail = process_market(cfg, m, run_id)
            if ok is not None:
                successes.append(ok)
            if fail is not None:
                failures.append(fail)
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = [ex.submit(process_market, cfg, m, run_id) for m in markets]
            with tqdm(total=len(futures), desc="Fetching snapshots", unit="market", dynamic_ncols=True) as pbar:
                for fut in as_completed(futures):
                    ok, fail = fut.result()
                    pbar.update(1)

                    if ok is not None:
                        successes.append(ok)

                        if isinstance(ok.get("observed_start_ts"), int):
                            observed_starts.append(int(ok["observed_start_ts"]))
                        if isinstance(ok.get("observed_end_ts"), int):
                            observed_ends.append(int(ok["observed_end_ts"]))

                        my = ok.get("missing_yes") or []
                        mn = ok.get("missing_no") or []
                        if my or mn:
                            partial_missing_markets += 1
                            for lab in my:
                                missing_yes_counts[lab] += 1
                            for lab in mn:
                                missing_no_counts[lab] += 1

                        cv = ok.get("complement_violations") or []
                        if cv:
                            complement_violation_markets += 1

                    if fail is not None:
                        failures.append(fail)
                        fail_reason_counts[fail.get("reason", "unknown")] += 1

                    pbar.set_postfix(
                        {
                            "ok": len(successes),
                            "fail": len(failures),
                            "partial": partial_missing_markets,
                            "comp_viols": complement_violation_markets,
                        }
                    )

    # Deterministic ordering for reproducibility
    successes.sort(key=lambda r: (str(r.get("market_id", "")), str(r.get("slug", ""))))
    failures.sort(key=lambda r: (str(r.get("market_id", "")), str(r.get("slug", "")), str(r.get("reason", ""))))

    atomic_write_jsonl(out_prices_jsonl, successes)
    atomic_write_jsonl(out_failed_jsonl, failures)

    run_finished = datetime.now(timezone.utc)
    elapsed_s = round((run_finished - run_started).total_seconds(), 3)

    obs_earliest_dt = ts_to_dt(min(observed_starts)) if observed_starts else None
    obs_latest_dt = ts_to_dt(max(observed_ends)) if observed_ends else None

    # -------------------------
    # TXT Summary (human-readable)
    # -------------------------
    lines: List[str] = []
    lines.append("Polymarket Earnings — Historical Prices Fetch Summary")
    lines.append("=" * 56)
    lines.append(f"Run ID:            {run_id}")
    lines.append(f"Generated (UTC):   {fmt_dt_utc(run_finished)}")
    lines.append(f"Elapsed seconds:   {elapsed_s}")
    lines.append("")
    lines.append("Mode")
    lines.append(f"- TEST:            {cfg.test_mode}")
    if cfg.test_mode:
        lines.append(f"- TEST_MAX_MARKETS:{cfg.test_max_markets}")
    lines.append("")
    lines.append("Inputs")
    lines.append(f"- Markets JSONL:   {input_path}")
    lines.append(f"- Markets in file: {total_before}")
    lines.append(f"- Markets processed:{len(markets)}")
    lines.append("")
    lines.append("Resolution window (based on OBSERVED first/last price timestamps)")
    lines.append(f"- Earliest first price (UTC): {fmt_dt_utc(obs_earliest_dt)}")
    lines.append(f"- Latest last price (UTC):    {fmt_dt_utc(obs_latest_dt)}")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"- Historical prices JSONL: {out_prices_jsonl}")
    lines.append(f"- Failed markets JSONL:    {out_failed_jsonl}")
    lines.append(f"- Summary TXT:             {out_summary_txt}")
    lines.append("")
    lines.append("Results")
    lines.append(f"- Successful markets written: {len(successes)}")
    lines.append(f"- Failed markets:             {len(failures)}")
    lines.append("")
    lines.append("Failure reasons (hard failures only)")
    if fail_reason_counts:
        for k, v in fail_reason_counts.most_common():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Quality stats (not failures)")
    lines.append(f"- Markets with partial missing snapshots: {partial_missing_markets}")
    lines.append(f"- Markets with YES+NO complement violations: {complement_violation_markets}")
    lines.append(f"- YES+NO complement tolerance: {cfg.complement_tolerance}")
    lines.append("")
    lines.append("Missing snapshot counts (not failures)")
    lines.append("YES missing:")
    for lab, _ in SNAPSHOTS:
        lines.append(f"- {lab}: {int(missing_yes_counts.get(lab, 0))}")
    lines.append("")
    lines.append("NO missing:")
    for lab, _ in SNAPSHOTS:
        lines.append(f"- {lab}: {int(missing_no_counts.get(lab, 0))}")
    lines.append("")
    lines.append("Config (selected)")
    lines.append(f"- gamma_base:      {cfg.gamma_base}")
    lines.append(f"- clob_base:       {cfg.clob_base}")
    lines.append(f"- max_workers:     {cfg.max_workers}")
    lines.append(f"- http_timeout:    {cfg.http_timeout}")
    lines.append(f"- retries:         {cfg.retries}")
    lines.append(f"- retry_sleep_s:   {cfg.retry_sleep_s}")
    lines.append(f"- fidelity_min:    {cfg.price_fidelity_min} minutes")
    lines.append(f"- fidelity_fallback:{cfg.min_fidelity_closed_min} minutes")
    lines.append(f"- buffer_seconds:  {cfg.buffer_seconds}")
    lines.append("")

    summary_txt = "\n".join(lines)
    atomic_write_text(out_summary_txt, summary_txt)

    tqdm.write(f"[{fmt_dt_utc(run_finished)}] DONE")
    tqdm.write(f"- Success: {len(successes)} | Fail: {len(failures)} | Elapsed: {elapsed_s}s")
    tqdm.write(f"- Summary TXT: {out_summary_txt}")


if __name__ == "__main__":
    main()
