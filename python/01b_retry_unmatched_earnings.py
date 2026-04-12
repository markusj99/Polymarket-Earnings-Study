#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry only the earnings markets that were left unmatched by 01_check_consistency.py.

Why this exists:
- The main script fetches all markets again, which is slow.
- Some unmatched cases appear to be caused by the EPS event fetch / date-column parsing,
  not by missing markets.
- This script reads data/validation/unmatched.jsonl and retries only those rows.

Default behavior:
- Read data/validation/unmatched.jsonl
- Retry records with skip reasons like no_event_match / no_events_returned / no_actual_eps / no_estimate
- Write:
    data/validation/unmatched_retry_matched.jsonl
    data/validation/unmatched_retry_matched.csv
    data/validation/unmatched_retry_still_unmatched.jsonl
    data/validation/unmatched_retry_still_unmatched.csv
    data/validation/unmatched_retry_summary.txt

Optional:
- --write-back updates markets.jsonl and markets.csv in place with the retried val_* fields.

Recommended placement:
- Put this file next to 01_check_consistency.py inside the project's python/ folder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RETRYABLE_SKIP_REASONS = {
    "no_event_match",
    "no_events_returned",
    "no_actual_eps",
    "no_estimate",
}

DEFAULT_EVENT_PRE_DAYS = 10
DEFAULT_EVENT_POST_DAYS = 120
DEFAULT_MAX_EVENT_DISTANCE_DAYS = 120
DEFAULT_LOOKBACK_YEARS = 12
DEFAULT_EPS_CHUNK_SIZE = 10


def load_base_module(base_script: Path):
    spec = importlib.util.spec_from_file_location("check_consistency_base", str(base_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script: {base_script}")

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.13 dataclasses during dynamic imports
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module

def normalize_col_name(col: Any) -> str:
    return str(col).strip().lower().replace("_", " ")


def choose_col(
    cols: Iterable[Any],
    *,
    exact: Iterable[str] = (),
    contains: Iterable[str] = (),
    exclude_contains: Iterable[str] = (),
) -> Optional[Any]:
    exact_norm = {normalize_col_name(x) for x in exact}
    contains_norm = [normalize_col_name(x) for x in contains]
    exclude_norm = [normalize_col_name(x) for x in exclude_contains]

    best: Optional[Tuple[int, Any]] = None
    for col in cols:
        c = normalize_col_name(col)
        if any(x in c for x in exclude_norm):
            continue

        score = None
        if c in exact_norm:
            score = 100
        else:
            matched = [x for x in contains_norm if x and x in c]
            if matched:
                score = max(10 + len(x) for x in matched)

        if score is None:
            continue

        if best is None or score > best[0]:
            best = (score, col)

    return None if best is None else best[1]


def dedupe_events(events: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for ev in events:
        key = (
            getattr(ev, "announce_date", None),
            getattr(ev, "fperiod", None),
            getattr(ev, "period_end_date", None),
            getattr(ev, "actual_eps", None),
            getattr(ev, "mean_estimate", None),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    out.sort(key=lambda e: (getattr(e, "announce_date", date.min), getattr(e, "period_end_date", date.min) or date.min))
    return out


def fetch_eps_events_retry_for_ric(
    mod,
    ric: str,
    *,
    anchor_dt: date,
    fail_fast: bool,
    lookback_years: int,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Retry a single RIC with stricter column selection than the original script.

    Key difference vs the original:
    - announce_date deliberately avoids generic matching to "Period End Date"
    - if announce_date is unavailable but Period End Date exists, that is only used as a
      last-resort fallback so the case is still inspectable
    """
    fields = [
        "TR.EPSActValue",
        "TR.EPSActValue.date",
        "TR.EPSActValue.fperiod",
        "TR.EPSActValue.PeriodEndDate",
        "TR.EPSMean",
        "TR.EPSHigh",
        "TR.EPSLow",
        "TR.EPSStdDev",
    ]

    # Try the original parameterization first, then a looser quarterly fetch.
    today = datetime.now(timezone.utc).date()
    start = min(today - timedelta(days=lookback_years * 365), anchor_dt - timedelta(days=450))
    end = max(today + timedelta(days=365), anchor_dt + timedelta(days=120))
    param_variants = [
        {"SDate": start.isoformat(), "EDate": end.isoformat(), "Period": "FQ0", "Frq": "FQ"},
        {"SDate": start.isoformat(), "EDate": end.isoformat(), "Frq": "FQ"},
    ]

    attempts: List[Dict[str, Any]] = []
    all_events: List[Any] = []

    for params in param_variants:
        df, err = mod.safe_get_data([ric], fields, params, retries=mod.EIKON_RETRIES, fail_fast=fail_fast)
        attempt_info = {
            "ric": ric,
            "params": params,
            "had_df": df is not None,
            "rows": 0 if df is None else int(len(df)),
            "columns": [] if df is None else [str(c) for c in df.columns],
            "err": None if err is None else str(err),
        }
        attempts.append(attempt_info)

        if df is None or getattr(df, "empty", True):
            continue

        cols = list(df.columns)
        inst_col = "Instrument" if "Instrument" in cols else (cols[0] if cols else None)
        if inst_col is None:
            continue

        actual_col = choose_col(
            cols,
            exact=["TR.EPSActValue"],
            contains=["tr.epsactvalue", "eps actual", "earnings per share - actual"],
        )
        announce_date_col = choose_col(
            cols,
            exact=["TR.EPSActValue.date", "Date", "Announcement Date", "Report Date"],
            contains=["tr.epsactvalue.date", "announcement date", "report date"],
            exclude_contains=["period end"],
        )
        fperiod_col = choose_col(
            cols,
            exact=["TR.EPSActValue.fperiod"],
            contains=["tr.epsactvalue.fperiod", "financial period absolute", "fperiod"],
        )
        period_end_col = choose_col(
            cols,
            exact=["TR.EPSActValue.PeriodEndDate", "Period End Date"],
            contains=["tr.epsactvalue.periodenddate", "period end date", "periodenddate"],
        )
        mean_col = choose_col(cols, exact=["TR.EPSMean"], contains=["tr.epsmean", "eps mean", "earnings per share - mean"])
        high_col = choose_col(cols, exact=["TR.EPSHigh"], contains=["tr.epshigh", "eps high", "earnings per share - high"])
        low_col = choose_col(cols, exact=["TR.EPSLow"], contains=["tr.epslow", "eps low", "earnings per share - low"])
        std_col = choose_col(cols, exact=["TR.EPSStdDev"], contains=["tr.epsstddev", "stdev", "std dev", "standard deviation"])

        parsed_events = []
        for _, row in df.iterrows():
            try:
                inst = row.get(inst_col)
                if inst is None or str(inst).strip() != ric:
                    continue

                announce_dt = mod.parse_any_datetime_to_date(row.get(announce_date_col)) if announce_date_col else None
                period_end_dt = mod.parse_any_datetime_to_date(row.get(period_end_col)) if period_end_col else None

                # Last resort fallback: if the API did not expose an announce date at all,
                # keep the row using the period-end date so the market can still be reviewed.
                chosen_dt = announce_dt or period_end_dt
                if not chosen_dt:
                    continue

                actual = mod._parse_float(row.get(actual_col)) if actual_col else None
                fperiod = mod._safe_str(row.get(fperiod_col)) if fperiod_col else None
                mean = mod._parse_float(row.get(mean_col)) if mean_col else None
                high = mod._parse_float(row.get(high_col)) if high_col else None
                low = mod._parse_float(row.get(low_col)) if low_col else None
                std = mod._parse_float(row.get(std_col)) if std_col else None

                parsed_events.append(
                    mod.EarningsEvent(
                        announce_date=chosen_dt,
                        fperiod=fperiod,
                        period_end_date=period_end_dt,
                        actual_eps=actual,
                        mean_estimate=mean,
                        high_estimate=high,
                        low_estimate=low,
                        stddev_estimate=std,
                    )
                )
            except Exception:
                continue

        all_events.extend(parsed_events)

    return dedupe_events(all_events), attempts


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def update_markets_inplace(mod, markets_jsonl_path: Path, markets_csv_path: Path, retried_rows: List[Dict[str, Any]]) -> int:
    all_markets = read_jsonl(markets_jsonl_path)
    by_line_no = {int(r["line_no"]): r for r in retried_rows if r.get("line_no") is not None}
    updated = 0
    updated_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for idx, market in enumerate(all_markets, start=1):
        r = by_line_no.get(idx)
        if r is None:
            continue
        for k, v in r.items():
            market[f"{mod.VAL_PREFIX}{k}"] = v
        market[f"{mod.VAL_PREFIX}updated_utc"] = updated_utc
        market[f"{mod.VAL_PREFIX}script"] = Path(__file__).name
        updated += 1

    mod.write_markets_jsonl_inplace(markets_jsonl_path, all_markets)
    mod.write_markets_csv_inplace(markets_csv_path, all_markets)
    return updated


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve()
    project_root = here.parent.parent
    p = argparse.ArgumentParser(description="Retry only unmatched earnings rows from validation/unmatched.jsonl")
    p.add_argument("--base-script", type=str, default=str(here.with_name("01_check_consistency.py")))
    p.add_argument("--unmatched-jsonl", type=str, default=str(project_root / "data" / "validation" / "unmatched.jsonl"))
    p.add_argument("--markets-jsonl", type=str, default=str(project_root / "data" / "markets" / "markets.jsonl"))
    p.add_argument("--markets-csv", type=str, default=str(project_root / "data" / "markets" / "markets.csv"))
    p.add_argument("--validation-dir", type=str, default=str(project_root / "data" / "validation"))
    p.add_argument("--app-key", nargs="?", const="__ENV__", default=None,
                   help="Eikon app key. If passed without value, use EIKON_APP_KEY from env.")
    p.add_argument("--eikon-port", type=int, default=None)
    p.add_argument("--skip-proxy-check", action="store_true")
    p.add_argument("--no-fail-fast", action="store_true")
    p.add_argument("--retry-all-unmatched", action="store_true",
                   help="Retry every unmatched row, not just the retryable skip reasons.")
    p.add_argument("--event-pre-days", type=int, default=DEFAULT_EVENT_PRE_DAYS)
    p.add_argument("--event-post-days", type=int, default=DEFAULT_EVENT_POST_DAYS)
    p.add_argument("--max-event-distance-days", type=int, default=DEFAULT_MAX_EVENT_DISTANCE_DAYS)
    p.add_argument("--lookback-years", type=int, default=DEFAULT_LOOKBACK_YEARS)
    p.add_argument("--write-back", action="store_true",
                   help="Also update markets.jsonl and markets.csv with retried val_* fields.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.app_key is None:
        print("Missing --app-key. Provide it or pass --app-key with no value to use EIKON_APP_KEY.", file=sys.stderr)
        return 2

    if args.app_key == "__ENV__":
        app_key = os.getenv("EIKON_APP_KEY") or os.getenv("APP_KEY") or ""
        if not app_key:
            print("EIKON_APP_KEY (or APP_KEY) not found in environment.", file=sys.stderr)
            return 2
    else:
        app_key = args.app_key

    base_script = Path(args.base_script)
    unmatched_jsonl = Path(args.unmatched_jsonl)
    markets_jsonl = Path(args.markets_jsonl)
    markets_csv = Path(args.markets_csv)
    validation_dir = Path(args.validation_dir)
    fail_fast = not args.no_fail_fast

    if not base_script.exists():
        print(f"Base script not found: {base_script}", file=sys.stderr)
        return 2
    if not unmatched_jsonl.exists():
        print(f"Unmatched file not found: {unmatched_jsonl}", file=sys.stderr)
        return 2

    mod = load_base_module(base_script)
    mod.setup_logging()
    mod.setup_warnings_suppression()
    mod.init_eikon(app_key, eikon_port=args.eikon_port, require_proxy=(not args.skip_proxy_check))

    rows = read_jsonl(unmatched_jsonl)
    if not args.retry_all_unmatched:
        rows = [r for r in rows if r.get("skip_reason") in RETRYABLE_SKIP_REASONS]

    if not rows:
        print("No eligible unmatched rows to retry.")
        return 0

    matched_rows: List[Dict[str, Any]] = []
    still_unmatched_rows: List[Dict[str, Any]] = []

    tickers_needing_rics = sorted({
        str(r.get("ticker", "")).upper()
        for r in rows
        if r.get("ticker") and not r.get("ric")
    })
    ticker_to_ric = mod.resolve_tickers_to_rics_batched(
        tickers_needing_rics,
        fail_fast=fail_fast,
        symbology_chunk_size=mod.DEFAULT_SYMBOLOGY_CHUNK_SIZE,
        validate_chunk_size=mod.DEFAULT_VALIDATE_CHUNK_SIZE,
        show_progress=True,
    ) if tickers_needing_rics else {}

    for row in rows:
        out = dict(row)
        out["retry_script"] = Path(__file__).name
        out["retry_timestamp_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        ticker = str(out.get("ticker") or "").upper().strip() or None
        ric = out.get("ric") or (ticker_to_ric.get(ticker) if ticker else None)
        if not ric:
            out["retry_status"] = "STILL_UNMATCHED"
            out["retry_skip_reason"] = "no_ric"
            still_unmatched_rows.append(out)
            continue

        out["ric"] = ric

        anchor_dt = mod.parse_any_datetime_to_date(out.get("anchor_date"))
        if not anchor_dt:
            out["retry_status"] = "STILL_UNMATCHED"
            out["retry_skip_reason"] = "no_anchor_date"
            still_unmatched_rows.append(out)
            continue

        events, attempts = fetch_eps_events_retry_for_ric(
            mod,
            str(ric),
            anchor_dt=anchor_dt,
            fail_fast=fail_fast,
            lookback_years=args.lookback_years,
        )
        out["retry_attempt_count"] = len(attempts)
        out["retry_attempts"] = attempts

        if not events:
            out["retry_status"] = "STILL_UNMATCHED"
            out["retry_skip_reason"] = "no_events_returned"
            still_unmatched_rows.append(out)
            continue

        ev, method = mod.match_event_by_anchor_date(
            events,
            anchor_dt,
            pre_days=args.event_pre_days,
            post_days=args.event_post_days,
            max_distance_days=args.max_event_distance_days,
        )
        if not ev:
            out["retry_status"] = "STILL_UNMATCHED"
            out["retry_skip_reason"] = "no_event_match"
            still_unmatched_rows.append(out)
            continue

        out["matched_announce_date"] = ev.announce_date.isoformat() if ev.announce_date else None
        out["matched_fperiod"] = ev.fperiod
        out["matched_period_end_date"] = ev.period_end_date.isoformat() if ev.period_end_date else None
        out["eikon_actual_eps"] = ev.actual_eps
        out["eikon_eps_mean_estimate"] = ev.mean_estimate
        out["eikon_eps_high_estimate"] = ev.high_estimate
        out["eikon_eps_low_estimate"] = ev.low_estimate
        out["eikon_eps_stddev_estimate"] = ev.stddev_estimate
        out["match_method"] = f"{method}|retry"

        if ev.actual_eps is None:
            out["retry_status"] = "STILL_UNMATCHED"
            out["retry_skip_reason"] = "no_actual_eps"
            still_unmatched_rows.append(out)
            continue

        estimate_used = None
        estimate_source = None
        if out.get("polymarket_estimate") is not None:
            estimate_used = float(out["polymarket_estimate"])
            estimate_source = out.get("polymarket_estimate_source")
        elif ev.mean_estimate is not None:
            estimate_used = float(ev.mean_estimate)
            estimate_source = "eikon_mean"

        if estimate_used is None:
            out["retry_status"] = "STILL_UNMATCHED"
            out["retry_skip_reason"] = "no_estimate"
            still_unmatched_rows.append(out)
            continue

        out["estimate_used"] = estimate_used
        out["estimate_used_source"] = estimate_source
        out["surprise"] = float(ev.actual_eps) - estimate_used
        out["label"] = mod.decide_label(float(ev.actual_eps), estimate_used)
        out["expected_resolution"] = mod.expected_resolution_from_label(
            out["label"],
            out.get("yes_semantics") or "YES_MEANS_BEAT",
            out.get("inline_counts_as") or "NO",
        )

        if out.get("expected_resolution") == out.get("polymarket_resolved_outcome"):
            out["status"] = "MATCHED_CORRECT"
        else:
            out["status"] = "MATCHED_INCORRECT"

        out["retry_status"] = "MATCHED_ON_RETRY"
        out["skip_reason"] = None
        matched_rows.append(out)

    matched_jsonl = validation_dir / "unmatched_retry_matched.jsonl"
    matched_csv = validation_dir / "unmatched_retry_matched.csv"
    still_jsonl = validation_dir / "unmatched_retry_still_unmatched.jsonl"
    still_csv = validation_dir / "unmatched_retry_still_unmatched.csv"
    summary_txt = validation_dir / "unmatched_retry_summary.txt"

    write_jsonl(matched_jsonl, matched_rows)
    write_csv(matched_csv, matched_rows)
    write_jsonl(still_jsonl, still_unmatched_rows)
    write_csv(still_csv, still_unmatched_rows)

    updated_count = 0
    if args.write_back and matched_rows:
        updated_count = update_markets_inplace(mod, markets_jsonl, markets_csv, matched_rows)

    summary = (
        "==================== UNMATCHED RETRY SUMMARY ====================\n"
        f"Timestamp (UTC):                  {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n"
        f"Base script:                      {base_script}\n"
        f"Unmatched input:                  {unmatched_jsonl}\n"
        f"Rows retried:                     {len(rows)}\n"
        f"Matched on retry:                 {len(matched_rows)}\n"
        f"Still unmatched:                  {len(still_unmatched_rows)}\n"
        f"Write-back enabled:               {bool(args.write_back)}\n"
        f"Markets updated in place:         {updated_count}\n"
        "\n"
        f"Matched JSONL:                    {matched_jsonl}\n"
        f"Matched CSV:                      {matched_csv}\n"
        f"Still unmatched JSONL:            {still_jsonl}\n"
        f"Still unmatched CSV:              {still_csv}\n"
        "===============================================================\n"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
