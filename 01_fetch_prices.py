#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket Earnings — Historical Snapshot Prices (YES + NO)

What this script does
---------------------
Given an input JSONL of *resolved* (closed) earnings markets (one JSON object per line),
this script:

1) For each market ID, fetches the full market detail from the Gamma API
   to obtain the YES/NO CLOB token IDs.

2) For each token (YES and NO), fetches historical prices from the CLOB API
   using /prices-history.

3) For each market, computes a set of “snapshot” prices at fixed offsets
   before the market endDate (e.g., 1d before, 12h before, etc.),
   taking the last known price at or before each target timestamp.

4) Writes:
   - historical_prices.jsonl: one output record per market (successful markets only)
   - failed_markets.jsonl: only “hard failures” where we could not obtain any snapshot price
     (across YES and NO) for the market.
   - summary.txt: a human-readable summary of what happened.

Important behavior / definitions
--------------------------------
- “Success”:
    A market is considered successful if at least ONE snapshot price exists
    on either YES or NO side.

- “Hard failure”:
    A market is considered failed ONLY if:
      - we could fetch detail & history successfully, but NONE of the snapshot
        timestamps had prices on either side, OR
      - we couldn’t fetch required IDs/history at all.

- Fidelity:
    CLOB /prices-history supports a “fidelity” parameter (minutes). Lower is more granular.
    Resolved markets sometimes return empty history for small fidelity values, so we retry with
    a coarser fidelity (default 12h).

Progress / UX
-------------
This version uses tqdm progress bars for:
- Processing markets concurrently (completed / total)
It also uses tqdm.write() for log lines so they don't break the progress bar.

Requirements
------------
pip install requests tqdm
"""

from __future__ import annotations

import json
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm  # NEW

# =========================
# TEST MODE
# =========================
TEST = False
TEST_MAX_MARKETS = 15  # only used if TEST=True

# =========================
# CONFIG
# =========================
MAX_WORKERS = 10
HTTP_TIMEOUT = 25
RETRIES = 3
RETRY_SLEEP_S = 0.8

# Price history resolution in minutes (fine granularity attempt)
PRICE_FIDELITY_MIN = 5

# For resolved/closed markets, /prices-history can return [] for small fidelity.
# Retry at 12h to get *some* history.
MIN_FIDELITY_CLOSED_MIN = 60 * 12  # 720 minutes

# Extra lookback padding before earliest snapshot (helps ensure we have history points)
BUFFER_SECONDS = 2 * 3600

# YES+NO complement tolerance:
# If YES and NO are both present at a snapshot, we check y+n ~= 1.
COMPLEMENT_TOLERANCE = 0.05

# APIs
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "polymarket-historical-prices/1.6",
}

# =========================
# Paths
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

# Prefer your Windows data location if it exists, otherwise fall back to ./data next to the script.
WIN_DATA_ROOT = Path(r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data")
DATA_DIR = WIN_DATA_ROOT if WIN_DATA_ROOT.exists() else (SCRIPT_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input file: output from the market-fetcher script
INPUT_DIR = DATA_DIR / "markets"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
IN_RESOLVED_JSONL = INPUT_DIR / "markets.jsonl"

# Output directory for prices
OUT_DIR = DATA_DIR / "prices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRICES_JSONL = OUT_DIR / "historical_prices.jsonl"
OUT_FAILED_JSONL = OUT_DIR / "failed_markets.jsonl"
OUT_SUMMARY_TXT = OUT_DIR / "summary.txt"

# =========================
# Snapshot spec
# =========================
# Each tuple is: (label, seconds_before_endDate)
SNAPSHOTS: List[Tuple[str, int]] = [
    ("4w", 4 * 7 * 24 * 3600),
    ("3w", 3 * 7 * 24 * 3600),
    ("2w", 2 * 7 * 24 * 3600),
    ("1w", 1 * 7 * 24 * 3600),
    ("7d", 7 * 24 * 3600),
    ("6d", 6 * 24 * 3600),
    ("5d", 5 * 24 * 3600),
    ("4d", 4 * 24 * 3600),
    ("3d", 3 * 24 * 3600),
    ("2d", 2 * 24 * 3600),
    ("1d", 1 * 24 * 3600),
    ("24h", 24 * 3600),
    ("12h", 12 * 3600),
    ("6h", 6 * 3600),
]
MAX_OFFSET_SECONDS = max(s for _, s in SNAPSHOTS)

# =========================
# Helpers (I/O, parsing, HTTP)
# =========================
def log(msg: str) -> None:
    """
    Logging helper. Uses tqdm.write() to avoid breaking progress bars.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[{ts}] {msg}")

def atomic_write_text(path: Path, text: str) -> None:
    """
    Writes a file atomically:
      - write to temp file
      - rename/replace to target
    This prevents partially-written outputs if the script is interrupted.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def parse_iso_dt(s: Any) -> Optional[datetime]:
    """
    Parse ISO datetime strings (with 'Z' supported) into UTC datetime objects.
    Returns None on failure.
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

def fmt_dt(dt: Optional[datetime]) -> str:
    """
    Format datetime in UTC for summaries/logs.
    """
    if dt is None:
        return "N/A"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def parse_json_list_maybe(v: Any) -> Optional[List[Any]]:
    """
    Some Gamma fields are lists, but occasionally appear as JSON-encoded strings.
    This helper accepts:
      - a list -> returns as-is
      - a "[...]" string -> json.loads(...) -> list
      - else -> None
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

def _request_json(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Make an HTTP request with retries and return (payload, error).

    - On success (HTTP 2xx): returns (parsed_json_or_text, None)
    - On failure: returns (None, error_dict)

    Retries on:
      - HTTP 429, 500, 502, 503, 504
      - network exceptions

    Note:
      This function does NOT raise; it returns structured errors.
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
                "response": payload if isinstance(payload, (dict, list)) else str(payload)[:1200],
            }

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
                continue

            return None, last_err

        except Exception as e:
            last_err = {"status_code": None, "url": url, "params": params, "exception": repr(e)}
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_S * (attempt + 1))
                continue
            return None, last_err

    return None, last_err

def gamma_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Convenience wrapper for Gamma API GET.
    """
    return _request_json("GET", f"{GAMMA}{path}", params=params)

def clob_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Convenience wrapper for CLOB API GET.
    """
    return _request_json("GET", f"{CLOB}{path}", params=params)

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file of dicts (one JSON object per line).
    Invalid lines are skipped.
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

def get_yes_no_token_ids(detail: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    From a Gamma market detail payload, locate YES/NO token IDs
    using outcomes[] alignment with clobTokenIds[].

    Returns: (yes_token_id, no_token_id)
    """
    outcomes = parse_json_list_maybe(detail.get("outcomes")) or []
    token_ids = parse_json_list_maybe(detail.get("clobTokenIds")) or []

    if not isinstance(outcomes, list) or not isinstance(token_ids, list):
        return None, None
    if len(outcomes) < 2 or len(token_ids) < 2:
        return None, None

    outs_lower = [str(x).strip().lower() for x in outcomes]

    yes_id = None
    no_id = None

    if "yes" in outs_lower:
        i = outs_lower.index("yes")
        try:
            yes_id = str(token_ids[i]).strip() if token_ids[i] is not None else None
        except Exception:
            yes_id = None

    if "no" in outs_lower:
        i = outs_lower.index("no")
        try:
            no_id = str(token_ids[i]).strip() if token_ids[i] is not None else None
        except Exception:
            no_id = None

    return yes_id, no_id

def clamp_start_ts(end_ts: int, created_ts: Optional[int]) -> int:
    """
    Choose the start timestamp for prices-history:
    - we need history far enough back to cover the largest snapshot offset
    - we add an additional BUFFER_SECONDS cushion
    - we don't want to query before market creation time (if known)
    """
    start_ts = end_ts - MAX_OFFSET_SECONDS - BUFFER_SECONDS
    if created_ts is not None:
        start_ts = max(start_ts, created_ts)
    return max(0, start_ts)

def _normalize_epoch_seconds(t_raw: int) -> int:
    """
    Defensive conversion:
    - If timestamps are accidentally returned in milliseconds, convert to seconds.
    """
    if t_raw > 10_000_000_000:
        return int(t_raw // 1000)
    return int(t_raw)

def fetch_prices_history_token(
    token_id: str,
    start_ts: int,
    end_ts: int
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Fetch price history for a token via CLOB /prices-history.

    Returns (history_list, error_dict).

    Behavior:
    - If endpoint returns {"history": []}, that is considered valid -> returns ([], None)
    - error_dict is only returned when all attempts fail at HTTP/payload level.

    Strategy:
    - Try range with fidelity=PRICE_FIDELITY_MIN
    - If empty, try range with fidelity=MIN_FIDELITY_CLOSED_MIN
    - Then same with interval=max
    - Then try without specifying fidelity (some backends differ)
    """
    attempts: List[Dict[str, Any]] = []

    def try_call(params: Dict[str, Any], tag: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        payload, err = clob_get("/prices-history", params=params)
        rec: Dict[str, Any] = {"tag": tag, "params": params, "err": err}
        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            rec["history_len"] = len(payload["history"])
            attempts.append(rec)
            return payload["history"], None
        rec["payload_type"] = type(payload).__name__ if payload is not None else None
        attempts.append(rec)
        return None, err or {"error": "unexpected_payload", "payload_preview": str(payload)[:500]}

    fids: List[int] = [int(PRICE_FIDELITY_MIN)]
    if PRICE_FIDELITY_MIN < MIN_FIDELITY_CLOSED_MIN:
        fids.append(int(MIN_FIDELITY_CLOSED_MIN))

    last_http_err: Optional[Dict[str, Any]] = None

    # 1) range with fidelity
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

    # 2) max interval with fidelity
    for fid in fids:
        hist, e = try_call(
            {"market": token_id, "interval": "max", "fidelity": int(fid)},
            f"max_fid_{fid}",
        )
        if e is not None:
            last_http_err = e
            continue
        if hist is not None and len(hist) > 0:
            return hist, None

    # 3) range without fidelity
    hist3, e3 = try_call({"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts)}, "range_no_fid")
    if e3 is None and hist3 is not None and len(hist3) > 0:
        return hist3, None

    # 4) max without fidelity
    hist4, e4 = try_call({"market": token_id, "interval": "max"}, "max_no_fid")
    if e4 is None and hist4 is not None and len(hist4) > 0:
        return hist4, None

    # Prefer returning [] if any attempt produced a valid list (even empty)
    for a in attempts:
        if a.get("err") is None and isinstance(a.get("history_len"), int):
            return [], None

    return None, {"last_http_error": last_http_err, "attempts": attempts}

def pick_price_at_or_before(history: List[Dict[str, Any]], target_ts: int) -> Tuple[Optional[float], Optional[int]]:
    """
    Given a CLOB history list [{t, p}, ...], find the last price p
    whose timestamp t <= target_ts.

    Returns:
      (price, source_timestamp_used)
    """
    ts_list: List[int] = []
    p_list: List[float] = []
    for pt in history:
        try:
            t_raw = int(pt.get("t"))
            t = _normalize_epoch_seconds(t_raw)
            p = float(pt.get("p"))
        except Exception:
            continue
        ts_list.append(t)
        p_list.append(p)

    if not ts_list:
        return None, None

    # Ensure sorted by time (CLOB usually returns sorted, but we defend anyway)
    if any(ts_list[i] > ts_list[i + 1] for i in range(len(ts_list) - 1)):
        pairs = sorted(zip(ts_list, p_list), key=lambda x: x[0])
        ts_list = [x[0] for x in pairs]
        p_list = [x[1] for x in pairs]

    idx = bisect_right(ts_list, int(target_ts)) - 1
    if idx < 0:
        return None, None
    return p_list[idx], ts_list[idx]

def any_price_present(prices: Dict[str, Optional[float]]) -> bool:
    """
    True if at least one snapshot price is non-null.
    """
    return any(v is not None for v in prices.values())

# =========================
# Worker (one market)
# =========================
def process_market(m: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single market from the input list.

    Returns:
      (success_record, failure_record)

    success_record:
      Contains snapshot prices for YES and NO (with missing labels listed),
      plus complement-violation diagnostics.

    failure_record:
      Only returned when we cannot get any usable snapshot price (or critical
      API steps fail).

    Notes:
    - This function is safe to run in threads (no shared mutation).
    """
    mid = str(m.get("id", "")).strip()
    slug = str(m.get("slug", "")).strip()
    end_dt = parse_iso_dt(m.get("endDate"))
    created_dt = parse_iso_dt(m.get("createdAt"))

    if not mid or not slug or end_dt is None:
        return None, {"market_id": mid or None, "slug": slug or None, "reason": "missing_id_slug_or_endDate"}

    end_ts = int(end_dt.timestamp())
    created_ts = int(created_dt.timestamp()) if created_dt else None
    start_ts = clamp_start_ts(end_ts, created_ts)

    # --- Step 1: Fetch market detail to get token IDs
    detail, derr = gamma_get(f"/markets/{mid}")
    if derr or not isinstance(detail, dict):
        return None, {"market_id": mid, "slug": slug, "reason": "gamma_market_detail_failed", "error": derr}

    # --- Step 2: Ensure this is an order-book market (has CLOB tokens)
    enable_ob = detail.get("enableOrderBook")
    if enable_ob is not True:
        return None, {"market_id": mid, "slug": slug, "reason": "not_orderbook_market", "enableOrderBook": enable_ob}

    # --- Step 3: Pull YES/NO token IDs aligned by outcomes
    yes_token_id, no_token_id = get_yes_no_token_ids(detail)
    if not yes_token_id or not no_token_id:
        return None, {
            "market_id": mid,
            "slug": slug,
            "reason": "missing_yes_or_no_token_id",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
        }

    # --- Step 4: Fetch price histories for both tokens
    yes_hist, yes_err = fetch_prices_history_token(yes_token_id, start_ts, end_ts)
    if yes_err or yes_hist is None:
        return None, {
            "market_id": mid,
            "slug": slug,
            "reason": "clob_prices_history_failed_yes",
            "yes_token_id": yes_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "error": yes_err,
        }

    no_hist, no_err = fetch_prices_history_token(no_token_id, start_ts, end_ts)
    if no_err or no_hist is None:
        return None, {
            "market_id": mid,
            "slug": slug,
            "reason": "clob_prices_history_failed_no",
            "no_token_id": no_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "error": no_err,
        }

    # --- Step 5: Compute snapshot prices
    prices_yes: Dict[str, Optional[float]] = {}
    prices_no: Dict[str, Optional[float]] = {}
    missing_yes: List[str] = []
    missing_no: List[str] = []

    for label, off in SNAPSHOTS:
        target_ts = end_ts - off

        py, _ = pick_price_at_or_before(yes_hist, target_ts)
        prices_yes[label] = py
        if py is None:
            missing_yes.append(label)

        pn, _ = pick_price_at_or_before(no_hist, target_ts)
        prices_no[label] = pn
        if pn is None:
            missing_no.append(label)

    # --- Step 6: Determine hard-fail vs success
    if not any_price_present(prices_yes) and not any_price_present(prices_no):
        return None, {
            "market_id": mid,
            "slug": slug,
            "reason": "no_snapshot_prices_found",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
        }

    # --- Step 7: Complement diagnostics (not a failure)
    complement_violations: List[Dict[str, Any]] = []
    for label, _off in SNAPSHOTS:
        y = prices_yes.get(label)
        n = prices_no.get(label)
        if y is None or n is None:
            continue
        s = y + n
        if abs(s - 1.0) > COMPLEMENT_TOLERANCE:
            complement_violations.append({"label": label, "yes": y, "no": n, "sum": s})

    success = {
        "slug": slug,
        "market_id": mid,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "endDate": end_dt.isoformat().replace("+00:00", "Z"),
        "prices_yes": prices_yes,
        "prices_no": prices_no,
        "missing_yes": missing_yes,
        "missing_no": missing_no,
        "complement_tolerance": COMPLEMENT_TOLERANCE,
        "complement_violations": complement_violations,
    }

    return success, None

# =========================
# Main
# =========================
def main() -> None:
    """
    Orchestrates the full run:
      1) Load input markets
      2) Optionally cap list for TEST mode
      3) Concurrently process markets with a progress bar
      4) Write outputs (prices, failures, summary)
    """
    started = time.time()

    # --- Load input markets list
    markets = load_jsonl(IN_RESOLVED_JSONL)
    total_before = len(markets)

    # --- TEST mode cap
    if TEST:
        markets = markets[: max(0, int(TEST_MAX_MARKETS))]
        log(f"TEST=True: limiting markets from {total_before} -> {len(markets)} (first {len(markets)})")

    log(f"Loaded {len(markets)} markets from {IN_RESOLVED_JSONL}")

    # --- Compute resolution window for summary
    end_dts = [parse_iso_dt(m.get("endDate")) for m in markets]
    end_dts2 = [d for d in end_dts if d is not None]
    earliest = min(end_dts2) if end_dts2 else None
    latest = max(end_dts2) if end_dts2 else None

    log(f"Resolution window (endDate): {fmt_dt(earliest)} -> {fmt_dt(latest)}")
    log(f"Output prices file: {OUT_PRICES_JSONL}")
    log(f"Output failed file: {OUT_FAILED_JSONL}")
    log(f"Output summary file: {OUT_SUMMARY_TXT}")
    log(
        f"TEST={TEST} TEST_MAX_MARKETS={TEST_MAX_MARKETS} | "
        f"fidelity={PRICE_FIDELITY_MIN}m (fallback {MIN_FIDELITY_CLOSED_MIN}m) | workers={MAX_WORKERS}"
    )

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    fail_reason_counts = Counter()
    missing_yes_counts = Counter()
    missing_no_counts = Counter()
    complement_violation_markets = 0
    partial_missing_markets = 0

    # --- Concurrent processing with tqdm progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_market, m) for m in markets]

        with tqdm(total=len(futures), desc="Fetching snapshots", unit="market", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                ok, fail = fut.result()
                pbar.update(1)

                if ok is not None:
                    successes.append(ok)

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

                # Update postfix for quick live stats
                pbar.set_postfix({
                    "ok": len(successes),
                    "fail": len(failures),
                    "partial": partial_missing_markets,
                    "comp_viols": complement_violation_markets,
                })

    # --- Write outputs
    prices_jsonl = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in successes)
    atomic_write_text(OUT_PRICES_JSONL, prices_jsonl)

    failed_jsonl = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failures)
    atomic_write_text(OUT_FAILED_JSONL, failed_jsonl)

    # --- Summary
    finished = datetime.now(timezone.utc)
    elapsed = round(time.time() - started, 3)

    lines: List[str] = []
    lines.append("Polymarket Earnings — Historical Prices Fetch Summary")
    lines.append("=" * 56)
    lines.append(f"Generated (UTC): {fmt_dt(finished)}")
    lines.append("")
    lines.append("Mode")
    lines.append(f"- TEST: {TEST}")
    if TEST:
        lines.append(f"- TEST_MAX_MARKETS: {TEST_MAX_MARKETS}")
    lines.append("")
    lines.append("Inputs")
    lines.append(f"- Markets JSONL: {IN_RESOLVED_JSONL}")
    lines.append(f"- Markets processed: {len(markets)}")
    lines.append("")
    lines.append("Resolution window (based on endDate) for processed set")
    lines.append(f"- Earliest resolved: {fmt_dt(earliest)}")
    lines.append(f"- Latest resolved:   {fmt_dt(latest)}")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"- Historical prices JSONL: {OUT_PRICES_JSONL}")
    lines.append(f"- Failed markets JSONL:    {OUT_FAILED_JSONL}")
    lines.append(f"- Summary TXT:             {OUT_SUMMARY_TXT}")
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
    lines.append("")
    lines.append("Missing snapshot counts (not failures)")
    lines.append("YES missing:")
    for lab, _ in SNAPSHOTS:
        lines.append(f"- {lab}: {missing_yes_counts.get(lab, 0)}")
    lines.append("")
    lines.append("NO missing:")
    for lab, _ in SNAPSHOTS:
        lines.append(f"- {lab}: {missing_no_counts.get(lab, 0)}")
    lines.append("")
    lines.append(f"YES+NO complement tolerance: {COMPLEMENT_TOLERANCE}")
    lines.append(f"Elapsed seconds: {elapsed}")
    lines.append("")

    summary = "\n".join(lines)
    atomic_write_text(OUT_SUMMARY_TXT, summary)

    log("DONE. Summary:")
    print(summary, flush=True)

if __name__ == "__main__":
    main()
