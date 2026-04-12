#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_fetch_poly_prices.py

Polymarket Earnings — Historical Snapshot Prices (YES + NO)

What this script does
---------------------
For each Polymarket orderbook market in an input JSONL file, this script fetches
historical price series for the YES and NO outcome tokens from the Polymarket
CLOB API and produces snapshot prices at fixed offsets before the company's
earnings release time.

Key behavior
------------
- Snapshot anchor uses `earnings_release_datetime` from:
    data/corporate_info/corporate_info_by_market.jsonl
- Snapshot targets are:
    (earnings_release_ts - offset_seconds)
- Snapshot price is the last price with:
    ts <= target_ts
- Complement check flags labels where:
    |YES + NO - 1| > tolerance
- Query window is chosen to fully cover the snapshot horizon relative to the
  earnings release anchor, while still allowing the fetch to extend through the
  market's known close/resolution time when available.

Outputs
-------
- poly_prices.jsonl           (success records)
- failed_poly_markets.jsonl   (failure records)
- poly_prices_wide.csv        (1 row per market)
- poly_prices_long.csv        (1 row per market x snapshot)
- failed_poly_markets.csv     (1 row per failure)
- summary.txt                 (human-readable run summary)

Notes for thesis/review
-----------------------
- Per-market outputs are JSONL + CSV.
- Summary remains TXT.
- All timestamps in JSON/CSV outputs are UTC.
- Optional local debug fields are off by default.

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
# Default filenames
# -------------------------
DEFAULT_PRICES_WIDE_CSV = "poly_prices_wide.csv"
DEFAULT_PRICES_LONG_CSV = "poly_prices_long.csv"
DEFAULT_FAILED_CSV = "failed_poly_markets.csv"


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

    Important:
    - If the input has no timezone, it is interpreted as UTC.
    - This matches the corporate_info example where
      earnings_release_datetime is stored without an explicit 'Z'
      but represents UTC time.
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


def _to_json_str(x: Any) -> Optional[str]:
    """
    For CSV: store lists/dicts as JSON strings to avoid losing info.
    """
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)


def write_csv_outputs(
    out_prices_wide_csv: Path,
    out_prices_long_csv: Path,
    out_failed_csv: Path,
    successes: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    """
    Writes:
      1) WIDE prices: 1 row per market, columns yes_4w, no_4w, ...
      2) LONG prices: 1 row per (market, snapshot_label) with yes/no/target/source_ts
      3) FAILED:      1 row per failed market
    """
    import pandas as pd  # local import to keep dependencies minimal

    out_prices_wide_csv.parent.mkdir(parents=True, exist_ok=True)

    snapshot_labels = [lab for lab, _ in SNAPSHOTS]

    # -------------------------
    # 1) WIDE (market-level)
    # -------------------------
    wide_rows: List[Dict[str, Any]] = []
    for r in successes:
        row: Dict[str, Any] = {}

        # Basic identifiers
        row["run_id"] = r.get("run_id")
        row["market_id"] = r.get("market_id")
        row["slug"] = r.get("slug")
        row["yes_token_id"] = r.get("yes_token_id")
        row["no_token_id"] = r.get("no_token_id")
        row["generated_utc"] = r.get("generated_utc")

        # Earnings anchor metadata
        row["snapshot_anchor_type"] = r.get("snapshot_anchor_type")
        row["earnings_release_datetime_raw"] = r.get("earnings_release_datetime_raw")
        row["earnings_release_ts"] = r.get("earnings_release_ts")
        row["earnings_release_utc"] = r.get("earnings_release_utc")
        row["corporate_info_join_method"] = r.get("corporate_info_join_method")

        # Observed window in fetched histories
        row["observed_start_ts"] = r.get("observed_start_ts")
        row["observed_end_ts"] = r.get("observed_end_ts")
        row["observed_start_utc"] = r.get("observed_start_utc")
        row["observed_end_utc"] = r.get("observed_end_utc")
        row["observed_span_hours"] = r.get("observed_span_hours")

        # Query window
        row["query_start_ts"] = r.get("query_start_ts")
        row["query_end_ts"] = r.get("query_end_ts")
        row["market_query_end_ts"] = r.get("market_query_end_ts")

        # Quality checks
        row["complement_tolerance"] = r.get("complement_tolerance")
        row["complement_violations"] = _to_json_str(r.get("complement_violations"))
        row["missing_yes"] = _to_json_str(r.get("missing_yes"))
        row["missing_no"] = _to_json_str(r.get("missing_no"))

        row["snapshot_anchor_source"] = r.get("snapshot_anchor_source")

        prices_yes = r.get("prices_yes") or {}
        prices_no = r.get("prices_no") or {}
        targets = r.get("snapshot_targets_ts") or {}
        src_yes = r.get("snapshot_source_ts_yes") or {}
        src_no = r.get("snapshot_source_ts_no") or {}

        for lab in snapshot_labels:
            row[f"yes_{lab}"] = prices_yes.get(lab)
            row[f"no_{lab}"] = prices_no.get(lab)
            row[f"target_ts_{lab}"] = targets.get(lab)
            row[f"src_yes_ts_{lab}"] = src_yes.get(lab)
            row[f"src_no_ts_{lab}"] = src_no.get(lab)

        wide_rows.append(row)

    pd.DataFrame(wide_rows).to_csv(out_prices_wide_csv, index=False, encoding="utf-8")

    # -------------------------
    # 2) LONG (snapshot-level)
    # -------------------------
    long_rows: List[Dict[str, Any]] = []
    for r in successes:
        prices_yes = r.get("prices_yes") or {}
        prices_no = r.get("prices_no") or {}
        targets = r.get("snapshot_targets_ts") or {}
        src_yes = r.get("snapshot_source_ts_yes") or {}
        src_no = r.get("snapshot_source_ts_no") or {}

        base = {
            "run_id": r.get("run_id"),
            "market_id": r.get("market_id"),
            "slug": r.get("slug"),
            "generated_utc": r.get("generated_utc"),
            "snapshot_anchor_type": r.get("snapshot_anchor_type"),
            "earnings_release_datetime_raw": r.get("earnings_release_datetime_raw"),
            "earnings_release_ts": r.get("earnings_release_ts"),
            "earnings_release_utc": r.get("earnings_release_utc"),
            "corporate_info_join_method": r.get("corporate_info_join_method"),
            "observed_end_ts": r.get("observed_end_ts"),
            "observed_end_utc": r.get("observed_end_utc"),
            "complement_tolerance": r.get("complement_tolerance"),
            "snapshot_anchor_source": r.get("snapshot_anchor_source"),
        }

        for lab, off in SNAPSHOTS:
            row = dict(base)
            row["snapshot_label"] = lab
            row["snapshot_offset_seconds"] = int(off)

            row["target_ts"] = targets.get(lab)
            row["src_yes_ts"] = src_yes.get(lab)
            row["src_no_ts"] = src_no.get(lab)

            row["price_yes"] = prices_yes.get(lab)
            row["price_no"] = prices_no.get(lab)

            y = row["price_yes"]
            n = row["price_no"]
            row["yes_plus_no"] = (y + n) if (isinstance(y, (int, float)) and isinstance(n, (int, float))) else None
            row["abs_complement_error"] = (
                abs((y + n) - 1.0)
                if (isinstance(y, (int, float)) and isinstance(n, (int, float)))
                else None
            )

            long_rows.append(row)

    pd.DataFrame(long_rows).to_csv(out_prices_long_csv, index=False, encoding="utf-8")

    # -------------------------
    # 3) FAILED markets CSV
    # -------------------------
    failed_rows: List[Dict[str, Any]] = []
    for f in failures:
        row = dict(f)
        for k, v in list(row.items()):
            if isinstance(v, (dict, list)):
                row[k] = _to_json_str(v)
        failed_rows.append(row)

    pd.DataFrame(failed_rows).to_csv(out_failed_csv, index=False, encoding="utf-8")


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
# Corporate info helpers
# -------------------------
def build_corporate_info_lookup(
    corporate_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Build lookups for corporate info rows.

    Primary key:
      - market_id
    Fallback key:
      - slug
    """
    by_market_id: Dict[str, Dict[str, Any]] = {}
    by_slug: Dict[str, Dict[str, Any]] = {}

    for row in corporate_rows:
        mid = str(row.get("market_id", "")).strip()
        slug = str(row.get("slug", "")).strip()

        if mid:
            by_market_id[mid] = row
        if slug:
            by_slug[slug] = row

    return by_market_id, by_slug


def resolve_corporate_info_row(
    market_id: str,
    slug: str,
    corporate_by_market_id: Dict[str, Dict[str, Any]],
    corporate_by_slug: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Resolve the corporate info row for a market.

    Join order:
      1) market_id
      2) slug
    """
    if market_id and market_id in corporate_by_market_id:
        return corporate_by_market_id[market_id], "market_id"
    if slug and slug in corporate_by_slug:
        return corporate_by_slug[slug], "slug"
    return None, None


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


def gamma_get(
    cfg: Config,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    headers = {"Accept": "application/json", "User-Agent": cfg.user_agent}
    return _request_json("GET", f"{cfg.gamma_base}{path}", params, headers, cfg.http_timeout, cfg.retries, cfg.retry_sleep_s)


def clob_get(
    cfg: Config,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
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

def resolve_anchor_dt(
    corp_row: Optional[Dict[str, Any]],
    input_market: Dict[str, Any],
    gamma_detail: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[datetime], Optional[str], Optional[str]]:
    """
    Resolve the snapshot anchor datetime.

    Priority:
      1) corporate_info.earnings_release_datetime
      2) input market endDate
      3) gamma detail endDate
    """
    candidates = [
        ("corporate_info.earnings_release_datetime", (corp_row or {}).get("earnings_release_datetime")),
        ("input_market.endDate", input_market.get("endDate")),
        ("gamma_detail.endDate", (gamma_detail or {}).get("endDate")),
    ]

    for source, raw in candidates:
        dt = parse_iso_dt(raw)
        if dt is not None:
            return dt, raw, source

    return None, None, None


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
    if t_raw > 10_000_000_000:
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


def pick_price_from_series(
    ts_list: List[int],
    p_list: List[float],
    target_ts: int,
) -> Tuple[Optional[float], Optional[int]]:
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
    Choose a conservative market end timestamp (UTC epoch seconds) for querying
    CLOB history.

    We prefer the latest timestamp among known close/resolution/end fields to
    reduce the risk of querying too early and missing late trading.
    """
    candidates: List[datetime] = []

    candidates += [
        parse_iso_dt(input_market.get("endDate")),
        parse_iso_dt(input_market.get("closedTime")),
        parse_iso_dt(input_market.get("resolvedTime")),
        parse_iso_dt(input_market.get("resolutionTime")),
    ]

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


def clamp_start_ts(anchor_ts: int, created_ts: Optional[int], buffer_seconds: int) -> int:
    """
    Compute a start_ts that covers all snapshot offsets relative to the
    earnings-release anchor, while not going before market creation time
    when creation time is known.
    """
    start_ts = anchor_ts - MAX_OFFSET_SECONDS - buffer_seconds
    if created_ts is not None:
        start_ts = max(start_ts, created_ts)
    return max(0, start_ts)


def choose_fetch_end_ts(anchor_ts: int, market_query_end_ts: Optional[int], buffer_seconds: int) -> int:
    """
    Compute the end_ts used in the CLOB history request.

    We always ensure the fetch extends at least a bit beyond the earnings
    release time, and when the market's close/resolution time is known we
    extend to that later timestamp.
    """
    min_end_ts = anchor_ts + buffer_seconds
    if market_query_end_ts is None:
        return min_end_ts
    return max(min_end_ts, market_query_end_ts)


def fetch_prices_history_token(
    cfg: Config,
    token_id: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Fetch price history for a CLOB token.

    Try several variants, then choose the non-empty history whose first point is earliest.
    If tied, prefer the one with more points.
    """
    attempts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    def try_call(params: Dict[str, Any], tag: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        payload, err = clob_get(cfg, "/prices-history", params=params)
        rec: Dict[str, Any] = {"tag": tag, "params": params, "err": err}

        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            hist = payload["history"]
            rec["history_len"] = len(hist)
            attempts.append(rec)

            if hist:
                ts_list, _ = build_series(hist)
                if ts_list:
                    candidates.append(
                        {
                            "tag": tag,
                            "params": params,
                            "hist": hist,
                            "first_ts": ts_list[0],
                            "last_ts": ts_list[-1],
                            "count": len(ts_list),
                        }
                    )
            return hist, None

        rec["payload_type"] = type(payload).__name__ if payload is not None else None
        attempts.append(rec)
        return None, err or {"error": "unexpected_payload", "payload_preview": _preview_payload(payload, 500)}

    fids: List[int] = [int(cfg.price_fidelity_min)]
    if cfg.price_fidelity_min < cfg.min_fidelity_closed_min:
        fids.append(int(cfg.min_fidelity_closed_min))

    last_http_err: Optional[Dict[str, Any]] = None

    if start_ts is not None and end_ts is not None:
        for fid in fids:
            _, e = try_call(
                {"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts), "fidelity": int(fid)},
                f"range_fid_{fid}",
            )
            if e is not None:
                last_http_err = e

    for fid in fids:
        _, e = try_call({"market": token_id, "interval": "all", "fidelity": int(fid)}, f"all_fid_{fid}")
        if e is not None:
            last_http_err = e

    for fid in fids:
        _, e = try_call({"market": token_id, "interval": "max", "fidelity": int(fid)}, f"max_fid_{fid}")
        if e is not None:
            last_http_err = e

    if start_ts is not None and end_ts is not None:
        _, e = try_call({"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts)}, "range_no_fid")
        if e is not None:
            last_http_err = e

    _, e = try_call({"market": token_id, "interval": "all"}, "all_no_fid")
    if e is not None:
        last_http_err = e

    _, e = try_call({"market": token_id, "interval": "max"}, "max_no_fid")
    if e is not None:
        last_http_err = e

    if candidates:
        candidates.sort(key=lambda x: (x["first_ts"], -x["count"]))
        return candidates[0]["hist"], None

    for a in attempts:
        if a.get("err") is None and isinstance(a.get("history_len"), int):
            return [], None

    return None, {"last_http_error": last_http_err, "attempts": attempts}

# -------------------------
# Worker
# -------------------------
def process_market(
    cfg: Config,
    m: Dict[str, Any],
    run_id: str,
    corporate_by_market_id: Dict[str, Dict[str, Any]],
    corporate_by_slug: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single market record.

    Returns:
      (success_record, None) or (None, failure_record)
    """
    mid = str(m.get("id", "")).strip()
    slug = str(m.get("slug", "")).strip()

    if not mid or not slug:
        return None, {
            "run_id": run_id,
            "market_id": mid or None,
            "slug": slug or None,
            "reason": "missing_id_or_slug",
        }

    # 0) Resolve earnings release anchor from corporate info
    corp_row, join_method = resolve_corporate_info_row(mid, slug, corporate_by_market_id, corporate_by_slug)
    if corp_row is None:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "missing_corporate_info_match",
        }

    # 1) Gamma market detail (required for token ids, and can also provide endDate fallback)
    detail, derr = gamma_get(cfg, f"/markets/{mid}")
    if derr or not isinstance(detail, dict):
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "gamma_market_detail_failed",
            "error": derr,
            "corporate_info_join_method": join_method,
        }

    earnings_release_dt, earnings_release_raw, anchor_source = resolve_anchor_dt(corp_row, m, detail)
    if earnings_release_dt is None:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "missing_or_invalid_earnings_release_datetime",
            "earnings_release_datetime_raw": (corp_row or {}).get("earnings_release_datetime"),
            "corporate_info_join_method": join_method,
        }

    enable_ob = detail.get("enableOrderBook")
    if enable_ob is not True:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "not_orderbook_market",
            "enableOrderBook": enable_ob,
            "earnings_release_datetime_raw": earnings_release_raw,
            "earnings_release_ts": anchor_end_ts,
        }

    yes_token_id, no_token_id = get_yes_no_token_ids(detail)
    if not yes_token_id or not no_token_id:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "missing_yes_or_no_token_id",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "earnings_release_datetime_raw": earnings_release_raw,
            "earnings_release_ts": anchor_end_ts,
        }

    # 2) Choose query window for CLOB fetch
    market_query_end_ts = choose_query_end_ts(m, detail)
    created_ts = choose_created_ts(m, detail)

    start_ts = clamp_start_ts(anchor_end_ts, created_ts, cfg.buffer_seconds)
    end_ts = choose_fetch_end_ts(anchor_end_ts, market_query_end_ts, cfg.buffer_seconds)

    if start_ts > end_ts:
        return None, {
            "run_id": run_id,
            "market_id": mid,
            "slug": slug,
            "reason": "invalid_query_window",
            "startTs": start_ts,
            "endTs": end_ts,
            "created_ts": created_ts,
            "market_query_end_ts": market_query_end_ts,
            "earnings_release_ts": anchor_end_ts,
        }

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
            "earnings_release_ts": anchor_end_ts,
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
            "earnings_release_ts": anchor_end_ts,
            "error": no_err,
        }

    # 4) Build series + compute observed fetched window
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
            "earnings_release_ts": anchor_end_ts,
        }

    observed_start_ts = min(start_candidates) if start_candidates else None
    observed_end_ts = max(end_candidates)
    observed_span_hours = None
    if observed_start_ts is not None:
        observed_span_hours = round((observed_end_ts - observed_start_ts) / 3600.0, 6)

    # 5) Compute snapshots relative to earnings release anchor
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
            "earnings_release_ts": anchor_end_ts,
            "observed_end_ts": observed_end_ts,
            "snapshot_anchor_source": anchor_source,
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

    obs_start_dt = ts_to_dt(observed_start_ts)
    obs_end_dt = ts_to_dt(observed_end_ts)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "market_id": mid,
        "slug": slug,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "generated_utc": fmt_dt_utc(datetime.now(timezone.utc)),

        # Earnings anchor metadata
        "snapshot_anchor_type": "resolved_anchor_datetime",
        "earnings_release_datetime_raw": earnings_release_raw,
        "earnings_release_ts": anchor_end_ts,
        "earnings_release_utc": fmt_dt_utc(earnings_release_dt),
        "corporate_info_join_method": join_method,

        # Query metadata
        "gamma_detail_enableOrderBook": enable_ob,
        "query_start_ts": start_ts,
        "query_end_ts": end_ts,
        "market_query_end_ts": market_query_end_ts,

        # Observed fetched window
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
        "snapshot_anchor_source": anchor_source,
    }

    if cfg.include_local_debug_fields:
        zi = _get_zoneinfo(cfg.local_tz_name)
        if zi is not None:

            def _fmt_local(dt: Optional[datetime]) -> str:
                if dt is None:
                    return "N/A"
                return dt.astimezone(zi).strftime("%Y-%m-%d %H:%M:%S%z")

            record["earnings_release_local"] = _fmt_local(earnings_release_dt)
            record["observed_start_local"] = _fmt_local(obs_start_dt)
            record["observed_end_local"] = _fmt_local(obs_end_dt)
            record["local_tz"] = cfg.local_tz_name

    return record, None


# -------------------------
# CLI / Main
# -------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch Polymarket YES/NO historical snapshot prices. "
            "Snapshots are anchored to earnings_release_datetime from corporate_info_by_market.jsonl."
        )
    )

    p.add_argument("--input", type=str, default=None, help="Input markets JSONL path.")
    p.add_argument("--corporate-info", type=str, default=None, help="Corporate info JSONL path.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for JSONL + CSV + summary.txt")
    p.add_argument("--data-root", type=str, default=None, help="Project data root. Defaults to ../data relative to this script.")
    p.add_argument("--out-prices-wide-csv", type=str, default=DEFAULT_PRICES_WIDE_CSV)
    p.add_argument("--out-prices-long-csv", type=str, default=DEFAULT_PRICES_LONG_CSV)
    p.add_argument("--out-failed-csv", type=str, default=DEFAULT_FAILED_CSV)

    p.add_argument("--gamma-base", type=str, default=os.getenv("POLY_GAMMA_URL", "https://gamma-api.polymarket.com"))
    p.add_argument("--clob-base", type=str, default=os.getenv("POLY_CLOB_URL", "https://clob.polymarket.com"))

    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--http-timeout", type=float, default=25.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=0.8)

    p.add_argument("--price-fidelity-min", type=int, default=5, help="Default fidelity in minutes.")
    p.add_argument(
        "--min-fidelity-closed-min",
        type=int,
        default=60 * 12,
        help="Fallback fidelity in minutes (closed markets).",
    )
    p.add_argument(
        "--buffer-seconds",
        type=int,
        default=2 * 3600,
        help="Extra buffer around the earnings-anchor snapshot window (seconds).",
    )

    p.add_argument("--complement-tolerance", type=float, default=0.05)

    p.add_argument(
        "--include-local-debug-fields",
        action="store_true",
        help="Include local time debug fields (not recommended for final datasets).",
    )
    p.add_argument("--local-tz", type=str, default="Europe/Stockholm")

    p.add_argument("--test", action="store_true", help="Limit number of markets (quick dev runs).")
    p.add_argument("--test-max-markets", type=int, default=15)

    p.add_argument("--user-agent", type=str, default="polymarket-historical-prices/3.0")

    return p.parse_args(argv)


def _default_data_root(script_dir: Path) -> Path:
    """
    Default data-root behavior (relative only):
      - Script location: Corporate_Earnings/python/01_fetch_poly_prices.py
      - Data root:       Corporate_Earnings/data
    """
    return (script_dir.parent / "data").resolve()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent

    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        data_root = _default_data_root(script_dir)

    data_root.mkdir(parents=True, exist_ok=True)

    default_input = data_root / "markets" / "markets.jsonl"
    default_corporate_info = data_root / "corporate_info" / "corporate_info_by_market.jsonl"
    default_out_dir = data_root / "poly_prices"

    input_path = Path(args.input).expanduser().resolve() if args.input else default_input
    corporate_info_path = Path(args.corporate_info).expanduser().resolve() if args.corporate_info else default_corporate_info
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prices_jsonl = out_dir / "poly_prices.jsonl"
    out_failed_jsonl = out_dir / "failed_poly_markets.jsonl"
    out_summary_txt = out_dir / "summary.txt"

    out_prices_wide_csv = out_dir / str(args.out_prices_wide_csv)
    out_prices_long_csv = out_dir / str(args.out_prices_long_csv)
    out_failed_csv = out_dir / str(args.out_failed_csv)

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
    corporate_rows = load_jsonl(corporate_info_path)
    corporate_by_market_id, corporate_by_slug = build_corporate_info_lookup(corporate_rows)

    total_before = len(markets)

    if cfg.test_mode:
        markets = markets[: max(0, cfg.test_max_markets)]

    tqdm.write(f"[{fmt_dt_utc(run_started)}] Run {run_id} starting")
    tqdm.write(f"- Markets input:    {input_path} ({total_before} rows; processing {len(markets)})")
    tqdm.write(f"- Corporate input:  {corporate_info_path} ({len(corporate_rows)} rows)")
    tqdm.write(f"- Output:           {out_dir}")
    tqdm.write(f"- Workers:          {cfg.max_workers} | timeout={cfg.http_timeout}s | retries={cfg.retries}")
    tqdm.write(f"- Fidelity:         {cfg.price_fidelity_min}m (fallback {cfg.min_fidelity_closed_min}m)")
    tqdm.write(f"- Complement tol.:  {cfg.complement_tolerance}")

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    fail_reason_counts = Counter()
    missing_yes_counts = Counter()
    missing_no_counts = Counter()
    complement_violation_markets = 0
    partial_missing_markets = 0
    corporate_join_counts = Counter()

    observed_starts: List[int] = []
    observed_ends: List[int] = []
    earnings_release_anchors: List[int] = []

    def register_result(ok: Optional[Dict[str, Any]], fail: Optional[Dict[str, Any]]) -> None:
        nonlocal complement_violation_markets, partial_missing_markets

        if ok is not None:
            successes.append(ok)

            if isinstance(ok.get("observed_start_ts"), int):
                observed_starts.append(int(ok["observed_start_ts"]))
            if isinstance(ok.get("observed_end_ts"), int):
                observed_ends.append(int(ok["observed_end_ts"]))
            if isinstance(ok.get("earnings_release_ts"), int):
                earnings_release_anchors.append(int(ok["earnings_release_ts"]))

            join_method = str(ok.get("corporate_info_join_method", "")).strip()
            if join_method:
                corporate_join_counts[join_method] += 1

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

    if cfg.max_workers <= 1:
        it = tqdm(markets, desc="Fetching snapshots", unit="market", dynamic_ncols=True)
        for m in it:
            ok, fail = process_market(cfg, m, run_id, corporate_by_market_id, corporate_by_slug)
            register_result(ok, fail)
            it.set_postfix(
                {
                    "ok": len(successes),
                    "fail": len(failures),
                    "partial": partial_missing_markets,
                    "comp_viols": complement_violation_markets,
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = [
                ex.submit(process_market, cfg, m, run_id, corporate_by_market_id, corporate_by_slug)
                for m in markets
            ]
            with tqdm(total=len(futures), desc="Fetching snapshots", unit="market", dynamic_ncols=True) as pbar:
                for fut in as_completed(futures):
                    ok, fail = fut.result()
                    pbar.update(1)
                    register_result(ok, fail)
                    pbar.set_postfix(
                        {
                            "ok": len(successes),
                            "fail": len(failures),
                            "partial": partial_missing_markets,
                            "comp_viols": complement_violation_markets,
                        }
                    )

    successes.sort(key=lambda r: (str(r.get("market_id", "")), str(r.get("slug", ""))))
    failures.sort(key=lambda r: (str(r.get("market_id", "")), str(r.get("slug", "")), str(r.get("reason", ""))))

    atomic_write_jsonl(out_prices_jsonl, successes)
    atomic_write_jsonl(out_failed_jsonl, failures)

    write_csv_outputs(
        out_prices_wide_csv=out_prices_wide_csv,
        out_prices_long_csv=out_prices_long_csv,
        out_failed_csv=out_failed_csv,
        successes=successes,
        failures=failures,
    )

    run_finished = datetime.now(timezone.utc)
    elapsed_s = round((run_finished - run_started).total_seconds(), 3)

    obs_earliest_dt = ts_to_dt(min(observed_starts)) if observed_starts else None
    obs_latest_dt = ts_to_dt(max(observed_ends)) if observed_ends else None

    anchor_earliest_dt = ts_to_dt(min(earnings_release_anchors)) if earnings_release_anchors else None
    anchor_latest_dt = ts_to_dt(max(earnings_release_anchors)) if earnings_release_anchors else None

    # -------------------------
    # TXT Summary
    # -------------------------
    lines: List[str] = []
    lines.append("Polymarket Earnings — Historical Prices Fetch Summary")
    lines.append("=" * 56)
    lines.append(f"Run ID:              {run_id}")
    lines.append(f"Generated (UTC):     {fmt_dt_utc(run_finished)}")
    lines.append(f"Elapsed seconds:     {elapsed_s}")
    lines.append("")
    lines.append("Mode")
    lines.append(f"- TEST:              {cfg.test_mode}")
    if cfg.test_mode:
        lines.append(f"- TEST_MAX_MARKETS:  {cfg.test_max_markets}")
    lines.append("")
    lines.append("Inputs")
    lines.append(f"- Markets JSONL:     {input_path}")
    lines.append(f"- Corporate JSONL:   {corporate_info_path}")
    lines.append(f"- Markets in file:   {total_before}")
    lines.append(f"- Markets processed: {len(markets)}")
    lines.append(f"- Corporate rows:    {len(corporate_rows)}")
    lines.append("")
    lines.append("Snapshot anchor window (based on earnings_release_datetime)")
    lines.append(f"- Earliest anchor (UTC): {fmt_dt_utc(anchor_earliest_dt)}")
    lines.append(f"- Latest anchor (UTC):   {fmt_dt_utc(anchor_latest_dt)}")
    lines.append("")
    lines.append("Fetched history window (based on observed first/last fetched price timestamps)")
    lines.append(f"- Earliest first price (UTC): {fmt_dt_utc(obs_earliest_dt)}")
    lines.append(f"- Latest last price (UTC):    {fmt_dt_utc(obs_latest_dt)}")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"- Historical prices JSONL: {out_prices_jsonl}")
    lines.append(f"- Failed markets JSONL:    {out_failed_jsonl}")
    lines.append(f"- Prices WIDE CSV:         {out_prices_wide_csv}")
    lines.append(f"- Prices LONG CSV:         {out_prices_long_csv}")
    lines.append(f"- Failed markets CSV:      {out_failed_csv}")
    lines.append(f"- Summary TXT:             {out_summary_txt}")
    lines.append("")
    lines.append("Results")
    lines.append(f"- Successful markets written: {len(successes)}")
    lines.append(f"- Failed markets:             {len(failures)}")
    lines.append("")
    lines.append("Corporate info joins")
    if corporate_join_counts:
        for k, v in corporate_join_counts.most_common():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")
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
    lines.append(f"- gamma_base:         {cfg.gamma_base}")
    lines.append(f"- clob_base:          {cfg.clob_base}")
    lines.append(f"- max_workers:        {cfg.max_workers}")
    lines.append(f"- http_timeout:       {cfg.http_timeout}")
    lines.append(f"- retries:            {cfg.retries}")
    lines.append(f"- retry_sleep_s:      {cfg.retry_sleep_s}")
    lines.append(f"- fidelity_min:       {cfg.price_fidelity_min} minutes")
    lines.append(f"- fidelity_fallback:  {cfg.min_fidelity_closed_min} minutes")
    lines.append(f"- buffer_seconds:     {cfg.buffer_seconds}")
    lines.append(f"- anchor_field:       earnings_release_datetime")
    lines.append("")

    summary_txt = "\n".join(lines)
    atomic_write_text(out_summary_txt, summary_txt)

    tqdm.write(f"[{fmt_dt_utc(run_finished)}] DONE")
    tqdm.write(f"- Success: {len(successes)} | Fail: {len(failures)} | Elapsed: {elapsed_s}s")
    tqdm.write(f"- Summary TXT: {out_summary_txt}")
    tqdm.write(f"- Prices WIDE CSV: {out_prices_wide_csv}")
    tqdm.write(f"- Prices LONG CSV: {out_prices_long_csv}")
    tqdm.write(f"- Failed markets CSV: {out_failed_csv}")


if __name__ == "__main__":
    main()