#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corporate Earnings — Brier Scores (DETAILED) by Market × Horizon
===============================================================

This script computes Brier losses from Polymarket YES token snapshot prices,
and writes ONE ROW PER (market × horizon), including rich diagnostics.

For each (market, horizon), we record:
- Market metadata (id/slug/ticker, close time, outcome y)
- Polymarket join method (by market_id, fallback slug)
- Snapshot timestamp (source preferred, else target)
- p_polymarket_yes, p_dice_0p5, p_hist_asof_end_minus_1d (leakage-safe)
- Per-row losses: (p - y)^2 for polymarket, dice, historical
- Flags + reasons when data is missing/invalid

We ALSO write an aggregated "by horizon" summary computed from rows where
Polymarket is usable (to keep baselines comparable to Polymarket on the
same sample, as in the original script).

INPUTS
------
1) Market outcomes + market close datetime:
   Corporate_Earnings/data/markets/markets.jsonl

2) Polymarket snapshot YES prices by horizon:
   Corporate_Earnings/data/poly_prices/poly_prices.jsonl

OUTPUTS
-------
Written to: Corporate_Earnings/data/brier_scores/

Detailed (market × horizon):
- brier_scores_market_horizon.csv
- brier_scores_market_horizon.json
- brier_scores_market_horizon.jsonl

Aggregated (by horizon, derived from usable polymarket rows):
- brier_scores_by_horizon.csv
- brier_scores_by_horizon.json
- brier_scores_by_horizon.jsonl

USAGE
-----
python Corporate_Earnings/06_brier_scores_detailed.py

Optional arguments:
  --exclude-horizons "4w,3w,2w"

Notes
-----
- All timestamps are treated as UTC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from bisect import bisect_right

UTC = timezone.utc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

def find_project_root(explicit: Optional[str] = None) -> Path:
    """
    Locate the project root in a portable way.

    A valid project root is a directory that contains:
      - data/markets/markets.jsonl
      - data/poly_prices/poly_prices.jsonl

    Backward compatible:
      - If a directory contains Corporate_Earnings/data/... it returns that Corporate_Earnings folder.

    You can also force it with --project-root.
    """
    def is_valid_root(p: Path) -> bool:
        return (
            (p / "data" / "markets" / "markets.jsonl").exists()
            and (p / "data" / "poly_prices" / "poly_prices.jsonl").exists()
        )

    if explicit:
        p = Path(explicit).expanduser().resolve()
        if is_valid_root(p):
            return p
        # Backward compatible structure: explicit points to parent of Corporate_Earnings
        ce = p / "Corporate_Earnings"
        if is_valid_root(ce):
            return ce
        raise FileNotFoundError(
            f"--project-root was provided but is not a valid project root: {p}\n"
            "Expected to find:\n"
            "  data/markets/markets.jsonl\n"
            "  data/poly_prices/poly_prices.jsonl"
        )

    # Search upward from this script's location
    start = Path(__file__).resolve()
    start_dir = start if start.is_dir() else start.parent

    for p in [start_dir] + list(start_dir.parents):
        if is_valid_root(p):
            return p
        ce = p / "Corporate_Earnings"
        if is_valid_root(ce):
            return ce

    # Also try current working directory upward
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if is_valid_root(p):
            return p
        ce = p / "Corporate_Earnings"
        if is_valid_root(ce):
            return ce

    raise FileNotFoundError(
        "Could not locate project root.\n"
        "Looked for either:\n"
        "  <root>/data/markets/markets.jsonl and <root>/data/poly_prices/poly_prices.jsonl\n"
        "or:\n"
        "  <root>/Corporate_Earnings/data/markets/markets.jsonl and <root>/Corporate_Earnings/data/poly_prices/poly_prices.jsonl\n"
        "Fix: run from the repo root, move files to match the expected structure, or pass --project-root <path>."
    )


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def parse_iso_utc(ts: str) -> datetime:
    """Parse ISO8601 with optional trailing 'Z' into timezone-aware UTC."""
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(UTC)


def parse_generated_utc(ts: Any) -> Optional[datetime]:
    """
    Parse poly_prices generated_utc.
    Accepts common formats:
      - 'YYYY-mm-dd HH:MM:SSZ'
      - ISO8601 variants (with 'Z' or offset)
    Returns None if parsing fails.
    """
    if ts is None:
        return None
    s = str(ts).strip()
    try:
        if s.endswith("Z") and "T" not in s:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=UTC)
        return parse_iso_utc(s)
    except Exception:
        return None


def parse_datetime_any_utc(x: Any) -> Optional[datetime]:
    """
    Parse a datetime-like field into UTC.

    Handles:
    - ISO strings (with Z/offset)
    - UNIX seconds (int/float)
    - UNIX milliseconds (int/float, detected if large)
    """
    if x is None:
        return None

    # Numeric epoch
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        v = float(x)
        if v >= 1e12:  # ms
            v = v / 1000.0
        try:
            return datetime.fromtimestamp(v, tz=UTC)
        except Exception:
            return None

    s = str(x).strip()
    if not s:
        return None

    # Numeric string epoch
    try:
        v = float(s)
        if math.isfinite(v):
            if v >= 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=UTC)
    except Exception:
        pass

    # ISO
    try:
        return parse_iso_utc(s)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def brier_loss(p: float, y: int) -> float:
    return (p - float(y)) ** 2


def mean_sum_count(s: float, n: int) -> float:
    return (s / n) if n > 0 else float("nan")


def sanitize_for_json(obj: Any) -> Any:
    """Convert NaN/inf to None so JSON is strict."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def dt_iso(dt: Optional[datetime]) -> Optional[str]:
    return None if dt is None else dt.astimezone(UTC).isoformat()


_H_RE = re.compile(r"^\s*(\d+)\s*([wdh])\s*$", re.IGNORECASE)

def horizon_to_seconds(h: str) -> Optional[int]:
    """
    Parse horizons like '4w', '6d', '12h' to seconds.
    Returns None if not parseable.
    """
    m = _H_RE.match(str(h))
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "h":
        return n * 3600
    if unit == "d":
        return n * 86400
    if unit == "w":
        return n * 7 * 86400
    return None


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MarketMeta:
    id: str
    slug: str
    ticker: str
    uma_end_dt: datetime  # market close datetime (UTC)
    y: int               # 1 if YES, 0 if NO


@dataclass(frozen=True)
class PolyPricesRecord:
    market_id: str
    slug: str
    generated_dt: Optional[datetime]
    prices_yes: Dict[str, Optional[float]]
    snapshot_source_ts_yes: Dict[str, Optional[int]]
    snapshot_targets_ts: Dict[str, Optional[int]]


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_markets_meta(path_jsonl: Path) -> Dict[str, MarketMeta]:
    """
    Load resolved markets from markets.jsonl.

    Uses umaEndDate as the market close time (UTC). If umaEndDate is missing, falls back to endDate.
    """
    out: Dict[str, MarketMeta] = {}

    with path_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            resolved = rec.get("resolvedOutcome")
            if resolved is None:
                continue

            r = str(resolved).strip().lower()
            if r == "yes":
                y = 1
            elif r == "no":
                y = 0
            else:
                continue

            end_raw = rec.get("umaEndDate")
            if end_raw is None:
                end_raw = rec.get("endDate")
            end_dt = parse_datetime_any_utc(end_raw)
            if end_dt is None:
                continue

            mid = str(rec.get("id", "")).strip()
            slug = str(rec.get("slug", "")).strip()
            ticker = str(rec.get("ticker", "")).strip()

            if not mid or not slug:
                continue

            out[mid] = MarketMeta(id=mid, slug=slug, ticker=ticker, uma_end_dt=end_dt, y=y)

    return out


def load_poly_prices_latest(path_jsonl: Path) -> Tuple[Dict[str, PolyPricesRecord], Dict[str, PolyPricesRecord], List[str]]:
    """
    Load poly_prices.jsonl and keep the latest record per market_id (and per slug fallback),
    based on generated_utc.

    Returns:
      - by_market_id: latest record per market_id
      - by_slug: latest record per slug
      - horizons: horizons found in prices_yes, ordered by preferred order when possible
    """
    by_mid: Dict[str, PolyPricesRecord] = {}
    by_slug: Dict[str, PolyPricesRecord] = {}
    horizons_set = set()

    def is_newer(a: Optional[datetime], b: Optional[datetime]) -> bool:
        if a is None and b is None:
            return False
        if a is None:
            return False
        if b is None:
            return True
        return a > b

    with path_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            market_id = str(rec.get("market_id", "")).strip()
            slug = str(rec.get("slug", "")).strip()
            if not market_id and not slug:
                continue

            gen_dt = parse_generated_utc(rec.get("generated_utc"))

            prices_yes_raw = rec.get("prices_yes") if isinstance(rec.get("prices_yes"), dict) else {}
            ssrc_yes_raw = rec.get("snapshot_source_ts_yes") if isinstance(rec.get("snapshot_source_ts_yes"), dict) else {}
            stgt_raw = rec.get("snapshot_targets_ts") if isinstance(rec.get("snapshot_targets_ts"), dict) else {}

            prices_yes: Dict[str, Optional[float]] = {}
            for k, v in prices_yes_raw.items():
                prices_yes[str(k)] = safe_float(v)
            horizons_set.update(prices_yes.keys())

            ssrc_yes: Dict[str, Optional[int]] = {}
            for k, v in ssrc_yes_raw.items():
                try:
                    ssrc_yes[str(k)] = None if v is None else int(v)
                except Exception:
                    ssrc_yes[str(k)] = None

            stgt: Dict[str, Optional[int]] = {}
            for k, v in stgt_raw.items():
                try:
                    stgt[str(k)] = None if v is None else int(v)
                except Exception:
                    stgt[str(k)] = None

            record = PolyPricesRecord(
                market_id=market_id,
                slug=slug,
                generated_dt=gen_dt,
                prices_yes=prices_yes,
                snapshot_source_ts_yes=ssrc_yes,
                snapshot_targets_ts=stgt,
            )

            if market_id:
                prev = by_mid.get(market_id)
                if prev is None or is_newer(record.generated_dt, prev.generated_dt):
                    by_mid[market_id] = record

            if slug:
                prev = by_slug.get(slug)
                if prev is None or is_newer(record.generated_dt, prev.generated_dt):
                    by_slug[slug] = record

    preferred = ["4w", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h"]
    horizons = [h for h in preferred if h in horizons_set] + sorted([h for h in horizons_set if h not in preferred])

    return by_mid, by_slug, horizons


def get_snapshot_dt(poly: PolyPricesRecord, horizon: str) -> Tuple[Optional[datetime], Optional[int], Optional[int]]:
    """
    Determine snapshot time for a market/horizon.

    Preference:
      1) snapshot_source_ts_yes[horizon] (actual observed timestamp used)
      2) snapshot_targets_ts[horizon] (target timestamp)

    Returns: (snapshot_dt, snapshot_source_ts_yes, snapshot_targets_ts)
    """
    src_ts = poly.snapshot_source_ts_yes.get(horizon)
    tgt_ts = poly.snapshot_targets_ts.get(horizon)

    ts = src_ts if src_ts is not None else tgt_ts
    if ts is None:
        return None, src_ts, tgt_ts
    try:
        return datetime.fromtimestamp(int(ts), tz=UTC), src_ts, tgt_ts
    except Exception:
        return None, src_ts, tgt_ts


# ---------------------------------------------------------------------
# Historical baseline (as-of umaEndDate - 1 day), with counts
# ---------------------------------------------------------------------

def compute_hist_stats_by_market(markets: Dict[str, MarketMeta]) -> Dict[str, Dict[str, Any]]:
    """
    Compute p_hist_i for each market i using ONLY outcomes available as of:

        cutoff_i = umaEndDate_i - 1 day

    p_hist_i = (# YES among markets with umaEndDate <= cutoff_i) / (count markets with umaEndDate <= cutoff_i)

    If count == 0, use p_hist_i = 0.5.

    Returns dict:
      mid -> { "p_hist": float, "hist_yes": int, "hist_total": int, "cutoff_dt": datetime }
    """
    items = sorted(((m.uma_end_dt, m.y, mid) for mid, m in markets.items()), key=lambda t: t[0])
    end_list = [t[0] for t in items]

    prefix_yes: List[int] = []
    s = 0
    for end_dt, y, _mid in items:
        s += int(y)
        prefix_yes.append(s)

    def hist_up_to(cutoff: datetime) -> Tuple[float, int, int]:
        pos = bisect_right(end_list, cutoff)  # number of markets with end_dt <= cutoff
        if pos <= 0:
            return 0.5, 0, 0
        total = pos
        yes = prefix_yes[pos - 1]
        return yes / total, yes, total

    out: Dict[str, Dict[str, Any]] = {}
    for mid, m in markets.items():
        cutoff = m.uma_end_dt - timedelta(days=1)
        p_hist, yes, total = hist_up_to(cutoff)
        out[mid] = {
            "p_hist": float(p_hist),
            "hist_yes": int(yes),
            "hist_total": int(total),
            "cutoff_dt": cutoff,
        }

    return out


# ---------------------------------------------------------------------
# Detailed rows + summary by horizon
# ---------------------------------------------------------------------

def compute_brier_rows_market_horizon(
    markets: Dict[str, MarketMeta],
    poly_by_mid: Dict[str, PolyPricesRecord],
    poly_by_slug: Dict[str, PolyPricesRecord],
    horizons: List[str],
    exclude_horizons: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      - detail_rows: one row per (market × horizon)
      - extra: meta + diagnostics_by_horizon
      - summary_rows: one row per horizon (means over usable_polymarket rows)
    """
    exclude = {h.strip() for h in (exclude_horizons or []) if h and str(h).strip()}
    horizons_use = [h for h in horizons if h not in exclude]

    hist_stats = compute_hist_stats_by_market(markets)

    # Diagnostics accumulators (by horizon)
    diag: Dict[str, Dict[str, Any]] = {}
    for h in horizons_use:
        diag[h] = {
            "total_markets_considered": 0,
            "matched_by_id": 0,
            "matched_by_slug": 0,
            "missing_poly_record": 0,
            "missing_price_at_horizon": 0,
            "invalid_price_range": 0,
            "missing_snapshot_time": 0,
            "usable_polymarket_n": 0,
            "sum_loss_polymarket": 0.0,
            "sum_loss_hist_on_pm_sample": 0.0,
            "sum_loss_dice_on_pm_sample": 0.0,
        }

    detail_rows: List[Dict[str, Any]] = []

    for mid, m in markets.items():
        hstat = hist_stats.get(mid, {"p_hist": 0.5, "hist_yes": 0, "hist_total": 0, "cutoff_dt": (m.uma_end_dt - timedelta(days=1))})
        p_hist = float(hstat.get("p_hist", 0.5))
        hist_yes = int(hstat.get("hist_yes", 0))
        hist_total = int(hstat.get("hist_total", 0))
        cutoff_dt = hstat.get("cutoff_dt")
        if not isinstance(cutoff_dt, datetime):
            cutoff_dt = m.uma_end_dt - timedelta(days=1)

        for h in horizons_use:
            d = diag[h]
            d["total_markets_considered"] += 1

            poly = poly_by_mid.get(mid)
            join_method = None
            if poly is not None:
                join_method = "market_id"
                d["matched_by_id"] += 1
            else:
                poly = poly_by_slug.get(m.slug)
                if poly is not None:
                    join_method = "slug"
                    d["matched_by_slug"] += 1
                else:
                    join_method = "none"
                    d["missing_poly_record"] += 1

            # defaults
            p_pm = None
            snap_dt = None
            src_ts = None
            tgt_ts = None
            pm_ok = False
            status = "ok"
            reason = ""

            if poly is None:
                status = "missing"
                reason = "no_poly_record"
            else:
                p_pm = poly.prices_yes.get(h)
                if p_pm is None:
                    status = "missing"
                    reason = "missing_price_at_horizon"
                    d["missing_price_at_horizon"] += 1
                else:
                    try:
                        p_pm_f = float(p_pm)
                    except Exception:
                        p_pm_f = float("nan")

                    if not math.isfinite(p_pm_f):
                        status = "invalid"
                        reason = "non_finite_price"
                        d["invalid_price_range"] += 1
                        p_pm = None
                    elif not (0.0 <= p_pm_f <= 1.0):
                        status = "invalid"
                        reason = "price_out_of_[0,1]"
                        d["invalid_price_range"] += 1
                        p_pm = p_pm_f  # keep for debugging
                    else:
                        p_pm = p_pm_f
                        snap_dt, src_ts, tgt_ts = get_snapshot_dt(poly, h)
                        if snap_dt is None:
                            status = "missing"
                            reason = "missing_snapshot_time"
                            d["missing_snapshot_time"] += 1
                        else:
                            pm_ok = True

            y = int(m.y)
            resolved_outcome = "YES" if y == 1 else "NO"

            # Always computable baselines
            p_dice = 0.5
            loss_dice = brier_loss(p_dice, y)
            loss_hist = brier_loss(p_hist, y)

            loss_pm = None
            seconds_before_close = None
            if pm_ok and p_pm is not None and snap_dt is not None:
                loss_pm = brier_loss(float(p_pm), y)
                seconds_before_close = (m.uma_end_dt - snap_dt).total_seconds()

                d["usable_polymarket_n"] += 1
                d["sum_loss_polymarket"] += float(loss_pm)
                d["sum_loss_hist_on_pm_sample"] += float(loss_hist)
                d["sum_loss_dice_on_pm_sample"] += float(loss_dice)

            detail_rows.append({
                # market identity
                "market_id": mid,
                "slug": m.slug,
                "ticker": m.ticker,
                # outcome + times
                "y": y,
                "resolved_outcome": resolved_outcome,
                "uma_end_dt_utc": dt_iso(m.uma_end_dt),
                # horizon
                "horizon": h,
                "horizon_seconds": horizon_to_seconds(h),
                # historical baseline internals
                "hist_cutoff_dt_utc": dt_iso(cutoff_dt),
                "hist_total": hist_total,
                "hist_yes": hist_yes,
                "p_hist_asof_end_minus_1d": p_hist,
                # polymarket join & timestamps
                "poly_join_method": join_method,
                "poly_generated_dt_utc": dt_iso(poly.generated_dt) if poly is not None else None,
                "snapshot_dt_utc": dt_iso(snap_dt),
                "snapshot_source_ts_yes": src_ts,
                "snapshot_targets_ts": tgt_ts,
                # probabilities
                "p_polymarket_yes": p_pm,
                "p_dice_0p5": p_dice,
                # losses
                "loss_polymarket": loss_pm,
                "loss_dice": loss_dice,
                "loss_hist": loss_hist,
                # sanity check timing
                "seconds_before_close": seconds_before_close,
                # usability
                "usable_polymarket": pm_ok,
                "status": status,
                "reason": reason,
            })

    # Build summary rows by horizon from diagnostics sums
    summary_rows: List[Dict[str, Any]] = []
    for h in horizons_use:
        d = diag[h]
        n = int(d["usable_polymarket_n"])
        summary_rows.append({
            "horizon": h,
            "n": n,
            "brier_polymarket": mean_sum_count(float(d["sum_loss_polymarket"]), n),
            "brier_dice_50_50": mean_sum_count(float(d["sum_loss_dice_on_pm_sample"]), n),
            "brier_historical_asof_end_minus_1d": mean_sum_count(float(d["sum_loss_hist_on_pm_sample"]), n),
        })

    meta = {
        "horizons_available": horizons,
        "horizons_excluded": sorted(exclude),
        "horizons_analyzed": horizons_use,
        "n_markets_resolved": len(markets),
        "detail_rows": "one row per (market × horizon), including missing/invalid rows with status/reason",
        "historical_baseline_rule": "p_hist_i computed using markets with umaEndDate <= (umaEndDate_i - 1 day); if none, p_hist_i=0.5",
        "join_rule": "market_id primary; slug fallback",
        "snapshot_time_required_for_polymarket": "snapshot_source_ts_yes[h] else snapshot_targets_ts[h] must exist",
        "summary_rule": "by-horizon summary computed over rows with usable_polymarket=True (baselines on same sample)",
    }

    return detail_rows, {"meta": meta, "diagnostics_by_horizon": diag}, summary_rows


# ---------------------------------------------------------------------
# Writers (CSV + JSON + JSONL)
# ---------------------------------------------------------------------

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(sanitize_for_json(r), ensure_ascii=False, allow_nan=False, default=str) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # stable-ish column order: union of keys in first-seen order
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    def _csv_val(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return "" if v is None else v

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k)) for k in fieldnames})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute detailed Brier rows per (market × horizon) + horizon summary.")
    ap.add_argument(
        "--exclude-horizons",
        default="",
        help="Comma-separated horizons to exclude (default: none). Example: '4w,3w,2w'",
    )
    args = ap.parse_args()

    root = find_project_root()

    markets_path = root / "data" / "markets" / "markets.jsonl"
    poly_path = root / "data" / "poly_prices" / "poly_prices.jsonl"
    out_dir = root / "data" / "brier_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    markets = load_markets_meta(markets_path)
    poly_by_mid, poly_by_slug, horizons_all = load_poly_prices_latest(poly_path)

    exclude = [h.strip() for h in str(args.exclude_horizons).split(",") if h.strip()]

    detail_rows, extra, summary_rows = compute_brier_rows_market_horizon(
        markets=markets,
        poly_by_mid=poly_by_mid,
        poly_by_slug=poly_by_slug,
        horizons=horizons_all,
        exclude_horizons=exclude,
    )

    # Console output: summary (compact)
    print("Brier summary by horizon (computed over usable_polymarket rows):")
    for r in summary_rows:
        h = r["horizon"]
        n = r["n"]
        pm = r["brier_polymarket"]
        di = r["brier_dice_50_50"]
        hi = r["brier_historical_asof_end_minus_1d"]
        pm_s = "NA" if (not isinstance(pm, float) or not math.isfinite(pm)) else f"{pm:.6f}"
        di_s = "NA" if (not isinstance(di, float) or not math.isfinite(di)) else f"{di:.6f}"
        hi_s = "NA" if (not isinstance(hi, float) or not math.isfinite(hi)) else f"{hi:.6f}"
        print(f"  {h:>4} | n={n:>5} | PM={pm_s} | DICE={di_s} | HIST={hi_s}")

    # Write detailed outputs
    detail_csv = out_dir / "brier_scores_market_horizon.csv"
    detail_json = out_dir / "brier_scores_market_horizon.json"
    detail_jsonl = out_dir / "brier_scores_market_horizon.jsonl"

    write_csv(detail_csv, detail_rows)
    write_json(detail_json, {
        "meta": extra["meta"],
        "diagnostics_by_horizon": extra["diagnostics_by_horizon"],
        "n_detail_rows": len(detail_rows),
        "results_market_horizon": detail_rows,
    })
    write_jsonl(detail_jsonl, detail_rows)

    # Write summary outputs (by horizon)
    summ_csv = out_dir / "brier_scores_by_horizon.csv"
    summ_json = out_dir / "brier_scores_by_horizon.json"
    summ_jsonl = out_dir / "brier_scores_by_horizon.jsonl"

    write_csv(summ_csv, summary_rows)
    write_json(summ_json, {
        "meta": extra["meta"],
        "diagnostics_by_horizon": extra["diagnostics_by_horizon"],
        "results_by_horizon": summary_rows,
    })
    write_jsonl(summ_jsonl, summary_rows)

    print("\nDone. Wrote:")
    print(f"  {detail_csv}")
    print(f"  {detail_json}")
    print(f"  {detail_jsonl}")
    print(f"  {summ_csv}")
    print(f"  {summ_json}")
    print(f"  {summ_jsonl}")


if __name__ == "__main__":
    main()
