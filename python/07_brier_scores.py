#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corporate Earnings — Brier Scores (ONLY) by Snapshot Horizon
===========================================================

This script computes Brier scores for prediction probabilities taken from Polymarket YES token prices.

For each snapshot horizon (e.g., 1w, 6d, 1d, 12h, 6h), we compute Brier scores for:

1) **Polymarket**:
   - p = Polymarket YES token snapshot price (interpreted as probability of YES)
   - Brier = mean( (p - y)^2 )

2) **Dice (50/50) baseline**:
   - p = 0.5 for every market at every horizon
   - Brier = mean( (0.5 - y)^2 )  (this will equal 0.25 for any binary sample)

3) **Historical average baseline** (leakage-safe cutoff at 1 day before close):
   - For each market i, define:
       cutoff_i = umaEndDate_i - 1 day
     and compute:
       p_hist_i = (# YES among markets with umaEndDate <= cutoff_i) / (total markets with umaEndDate <= cutoff_i)
   - IMPORTANT:
       The historical baseline uses the average **as of 1 day before the market close**,
       and does **not** use later outcomes.
   - If no markets exist before cutoff_i, we use p_hist_i = 0.5.

INPUTS
------
1) Market outcomes + market close datetime:
   Corporate_Earnings/data/markets/markets.jsonl
   Required fields per row:
     - id
     - slug
     - ticker (optional but used if present)
     - resolvedOutcome  ("YES" / "NO")
     - umaEndDate  (market close datetime; preferred)
       (fallback: endDate if umaEndDate missing)

2) Polymarket snapshot YES prices by horizon:
   Corporate_Earnings/data/poly_prices/poly_prices.jsonl
   Required fields per row (latest per market is used):
     - market_id (or slug as fallback join key)
     - slug
     - generated_utc (to identify the latest record per market)
     - prices_yes (dict: horizon -> YES price)
     - snapshot_source_ts_yes and/or snapshot_targets_ts (to validate snapshot timing exists)

OUTPUTS
-------
Written to: Corporate_Earnings/data/brier_scores/

- brier_scores_by_horizon.csv
- brier_scores_by_horizon.json
- brier_scores_by_horizon.jsonl   (included to satisfy the project convention: CSV + JSONL)

USAGE
-----
python Corporate_Earnings/06_brier_scores.py

Optional arguments:
  --exclude-horizons "4w,3w,2w"   # exclude specific horizons if desired

Notes
-----
- All timestamps are treated as UTC.
- This script is designed to be run standalone or imported; main work is done in functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from bisect import bisect_right

UTC = timezone.utc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Locate the 'Corporate_Earnings' directory by walking upward from this script location.
    This keeps paths relative and portable across machines.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if p.name == "Corporate_Earnings":
            return p

    cwd = Path.cwd().resolve()
    if cwd.name == "Corporate_Earnings":
        return cwd
    if (cwd / "Corporate_Earnings").exists():
        return (cwd / "Corporate_Earnings").resolve()

    raise FileNotFoundError(
        "Could not locate project root folder named 'Corporate_Earnings'. "
        "Place this script somewhere inside Corporate_Earnings/ or run from a directory that contains it."
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
        # Heuristic: milliseconds if >= 1e12
        if v >= 1e12:
            v = v / 1000.0
        try:
            return datetime.fromtimestamp(v, tz=UTC)
        except Exception:
            return None

    # String (ISO or numeric string)
    s = str(x).strip()
    if not s:
        return None

    # Try numeric string epoch
    try:
        v = float(s)
        if math.isfinite(v):
            if v >= 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=UTC)
    except Exception:
        pass

    # Try ISO
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


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sanitize_for_json(obj: Any) -> Any:
    """Convert NaN/inf to None so JSON is strict."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


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

            # Prefer umaEndDate (per your instruction); fallback to endDate
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


def get_snapshot_dt(poly: PolyPricesRecord, horizon: str) -> Optional[datetime]:
    """
    Determine snapshot time for a market/horizon.

    Preference:
      1) snapshot_source_ts_yes[horizon] (actual observed timestamp used)
      2) snapshot_targets_ts[horizon] (target timestamp)
    """
    ts = poly.snapshot_source_ts_yes.get(horizon)
    if ts is None:
        ts = poly.snapshot_targets_ts.get(horizon)
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=UTC)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Historical baseline (as-of umaEndDate - 1 day)
# ---------------------------------------------------------------------

def compute_hist_prob_by_market(markets: Dict[str, MarketMeta]) -> Dict[str, float]:
    """
    Compute p_hist_i for each market i using ONLY outcomes available as of:

        cutoff_i = umaEndDate_i - 1 day

    p_hist_i = (# YES among markets with umaEndDate <= cutoff_i) / (count markets with umaEndDate <= cutoff_i)

    If count == 0, use p_hist_i = 0.5.
    """
    # Sort all markets by their uma_end_dt to build cumulative YES counts.
    items = sorted(((m.uma_end_dt, m.y, mid) for mid, m in markets.items()), key=lambda t: t[0])
    end_list = [t[0] for t in items]

    prefix_yes: List[int] = []
    s = 0
    for end_dt, y, _mid in items:
        s += int(y)
        prefix_yes.append(s)

    def hist_mean_up_to(cutoff: datetime) -> float:
        # position = number of markets with end_dt <= cutoff
        pos = bisect_right(end_list, cutoff)  # returns index in [0..n]
        if pos <= 0:
            return 0.5
        total = pos
        yes = prefix_yes[pos - 1]
        return yes / total

    out: Dict[str, float] = {}
    for mid, m in markets.items():
        cutoff = m.uma_end_dt - timedelta(days=1)
        out[mid] = float(hist_mean_up_to(cutoff))

    return out


# ---------------------------------------------------------------------
# Brier score computation per horizon
# ---------------------------------------------------------------------

def compute_brier_scores_by_horizon(
    markets: Dict[str, MarketMeta],
    poly_by_mid: Dict[str, PolyPricesRecord],
    poly_by_slug: Dict[str, PolyPricesRecord],
    horizons: List[str],
    exclude_horizons: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    For each horizon, join markets to poly_prices and compute Brier scores for:
      - polymarket (YES price)
      - dice (0.5)
      - historical (as-of umaEndDate-1d baseline)

    Returns:
      - rows: list of dict rows (one per horizon)
      - meta: diagnostics / inputs summary
    """
    exclude = {h.strip() for h in (exclude_horizons or []) if h and str(h).strip()}
    horizons_use = [h for h in horizons if h not in exclude]

    hist_prob = compute_hist_prob_by_market(markets)

    rows: List[Dict[str, Any]] = []
    diagnostics_by_h: Dict[str, Any] = {}

    for h in horizons_use:
        matched_id = 0
        matched_slug = 0
        missing_poly = 0
        missing_price = 0
        missing_snapshot_time = 0

        losses_pm: List[float] = []
        losses_hist: List[float] = []
        losses_dice: List[float] = []

        for mid, m in markets.items():
            poly = poly_by_mid.get(mid)
            if poly is not None:
                matched_id += 1
            else:
                poly = poly_by_slug.get(m.slug)
                if poly is not None:
                    matched_slug += 1
                else:
                    missing_poly += 1
                    continue

            p = poly.prices_yes.get(h)
            if p is None or not (0.0 <= float(p) <= 1.0):
                missing_price += 1
                continue

            # Keep the requirement that a snapshot time exists for the horizon
            snap_dt = get_snapshot_dt(poly, h)
            if snap_dt is None:
                missing_snapshot_time += 1
                continue

            y = m.y
            p_pm = float(p)
            p_h = float(hist_prob.get(mid, 0.5))  # should exist; fallback 0.5

            losses_pm.append(brier_loss(p_pm, y))
            losses_hist.append(brier_loss(p_h, y))
            losses_dice.append(brier_loss(0.5, y))  # always 0.25

        n = len(losses_pm)
        bs_pm = mean(losses_pm)
        bs_hist = mean(losses_hist)
        bs_dice = mean(losses_dice) if n > 0 else float("nan")

        diagnostics_by_h[h] = {
            "matched_by_id": matched_id,
            "matched_by_slug": matched_slug,
            "missing_poly_record": missing_poly,
            "missing_price_at_horizon": missing_price,
            "missing_snapshot_time": missing_snapshot_time,
            "usable_n": n,
        }

        rows.append({
            "horizon": h,
            "n": n,
            "brier_polymarket": bs_pm,
            "brier_dice_50_50": bs_dice,
            "brier_historical_asof_end_minus_1d": bs_hist,
        })

    meta = {
        "horizons_available": horizons,
        "horizons_excluded": sorted(exclude),
        "horizons_analyzed": [r["horizon"] for r in rows],
        "n_markets_resolved": len(markets),
        "historical_baseline_rule": "p_hist_i computed using markets with umaEndDate <= (umaEndDate_i - 1 day); if none, p_hist_i=0.5",
        "join_rule": "market_id primary; slug fallback",
        "snapshot_time_required": "snapshot_source_ts_yes[h] else snapshot_targets_ts[h] must exist",
    }

    return rows, {"meta": meta, "diagnostics_by_horizon": diagnostics_by_h}


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
        # Still create an empty file with headers if possible? Here we skip.
        return

    # stable column order
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    def _csv_val(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k, "")) for k in fieldnames})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ONLY Brier scores by horizon (Polymarket vs baselines).")
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

    markets = load_markets_meta(markets_path)
    poly_by_mid, poly_by_slug, horizons_all = load_poly_prices_latest(poly_path)

    exclude = [h.strip() for h in str(args.exclude_horizons).split(",") if h.strip()]

    rows, extra = compute_brier_scores_by_horizon(
        markets=markets,
        poly_by_mid=poly_by_mid,
        poly_by_slug=poly_by_slug,
        horizons=horizons_all,
        exclude_horizons=exclude,
    )

    # Console output (simple)
    print("Brier scores by horizon:")
    for r in rows:
        h = r["horizon"]
        n = r["n"]
        pm = r["brier_polymarket"]
        di = r["brier_dice_50_50"]
        hi = r["brier_historical_asof_end_minus_1d"]
        pm_s = "NA" if (not isinstance(pm, float) or not math.isfinite(pm)) else f"{pm:.6f}"
        di_s = "NA" if (not isinstance(di, float) or not math.isfinite(di)) else f"{di:.6f}"
        hi_s = "NA" if (not isinstance(hi, float) or not math.isfinite(hi)) else f"{hi:.6f}"
        print(f"  {h:>4} | n={n:>5} | PM={pm_s} | DICE={di_s} | HIST={hi_s}")

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "brier_scores_by_horizon.csv"
    json_path = out_dir / "brier_scores_by_horizon.json"
    jsonl_path = out_dir / "brier_scores_by_horizon.jsonl"

    write_csv(csv_path, rows)
    write_json(json_path, {"meta": extra["meta"], "diagnostics_by_horizon": extra["diagnostics_by_horizon"], "results": rows})
    write_jsonl(jsonl_path, rows)

    print(f"\nDone. Wrote:\n  {csv_path}\n  {json_path}\n  {jsonl_path}")


if __name__ == "__main__":
    main()
