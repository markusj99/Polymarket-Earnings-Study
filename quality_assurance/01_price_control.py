r"""
random_market_price_table.py

Select N random markets from a historical Polymarket JSONL file and produce a
timeline table of YES prices (decimal -> percent).

Run modes
---------
1) Standalone:
   python random_market_price_table.py

2) Imported/called from another script:
   import random_market_price_table as rpt
   rpt.run(input_jsonl_path=Path(r"...\other.jsonl"), n_random_markets=10)

Windows note
------------
Use raw strings for Windows paths: r"C:\Users\..."
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# DEFAULT CONFIG (kept inside this script; used when running standalone)
# =============================================================================

DEFAULT_N_RANDOM_MARKETS: int = 15
DEFAULT_RANDOM_SEED: Optional[int] = None # Set to an int for reproducible sampling

DEFAULT_INPUT_JSONL_PATH: Path = Path(
    r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\prices\historical_prices.jsonl"
)

DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\quality_assurance\results"
)

DEFAULT_PRINT_TO_CONSOLE: bool = True
DEFAULT_WRITE_COMBINED_CSV: bool = True
DEFAULT_COMBINED_CSV_NAME: str = "random_prices_sample.csv"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class TimelineRow:
    """One datapoint for a market/run/bucket where YES price exists."""
    market_id: str
    slug: str
    run_id: str
    generated_utc: str
    bucket: str

    snapshot_target_ts: Optional[int]
    snapshot_target_utc: Optional[str]

    snapshot_source_ts_yes: Optional[int]
    snapshot_source_utc_yes: Optional[str]

    yes_price_decimal: float
    yes_price_pct: float


@dataclass(frozen=True)
class RunResult:
    """Returned by run() so other scripts can reuse outputs programmatically."""
    sampled_market_ids: List[str]
    rows: List[TimelineRow]
    combined_csv_path: Optional[Path]


# =============================================================================
# Helpers
# =============================================================================

def _utc_from_ts(ts: Optional[int]) -> Optional[str]:
    """Convert unix seconds -> UTC 'YYYY-MM-DD HH:MM:SSZ', or None if ts is None."""
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _safe_float(x) -> Optional[float]:
    """Convert x to float if possible, else None (handles nulls)."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_int(x) -> Optional[int]:
    """Convert x to int if possible, else None."""
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _reservoir_sample_unique_market_ids(
    jsonl_path: Path,
    sample_size: int,
    rng: random.Random,
) -> List[str]:
    """
    Reservoir sample *unique* market_ids from a JSONL file.

    Markets can appear multiple times (multiple runs). We only count each market_id once
    for sampling so no market is oversampled.
    """
    if sample_size <= 0:
        return []

    seen: Set[str] = set()
    reservoir: List[str] = []
    unique_count = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            market_id = str(obj.get("market_id", "")).strip()
            if not market_id or market_id in seen:
                continue

            seen.add(market_id)
            unique_count += 1

            if len(reservoir) < sample_size:
                reservoir.append(market_id)
            else:
                j = rng.randint(1, unique_count)
                if j <= sample_size:
                    reservoir[j - 1] = market_id

    if unique_count == 0:
        raise RuntimeError("No valid market_id values found in the JSONL file.")

    if sample_size > unique_count:
        return list(seen)

    return reservoir


def _expand_market_line_to_rows(obj: dict) -> List[TimelineRow]:
    """
    Expand one JSONL line into TimelineRow rows, but ONLY for buckets where YES price exists.

    Skipping logic:
    - If prices_yes is missing OR prices_yes[bucket] is null/unparseable -> skip bucket.
    - We do NOT output "empty" snapshots to CSV/console.
    """
    market_id = str(obj.get("market_id", "")).strip()
    slug = str(obj.get("slug", "")).strip()
    run_id = str(obj.get("run_id", "")).strip()
    generated_utc = str(obj.get("generated_utc", "")).strip()

    prices_yes: Dict[str, object] = obj.get("prices_yes") or {}
    targets: Dict[str, object] = obj.get("snapshot_targets_ts") or {}
    sources_yes: Dict[str, object] = obj.get("snapshot_source_ts_yes") or {}

    # Only consider buckets that appear in prices_yes; other metadata-only buckets are irrelevant.
    buckets = sorted(prices_yes.keys())

    rows: List[TimelineRow] = []
    for bucket in buckets:
        yes_dec_opt = _safe_float(prices_yes.get(bucket))
        if yes_dec_opt is None:
            # KEY CHANGE: skip missing YES prices completely
            continue

        yes_pct = round(yes_dec_opt * 100.0, 4)

        tgt_ts = _safe_int(targets.get(bucket))
        src_ts = _safe_int(sources_yes.get(bucket))

        rows.append(
            TimelineRow(
                market_id=market_id,
                slug=slug,
                run_id=run_id,
                generated_utc=generated_utc,
                bucket=bucket,
                snapshot_target_ts=tgt_ts,
                snapshot_target_utc=_utc_from_ts(tgt_ts),
                snapshot_source_ts_yes=src_ts,
                snapshot_source_utc_yes=_utc_from_ts(src_ts),
                yes_price_decimal=yes_dec_opt,
                yes_price_pct=yes_pct,
            )
        )

    # Order by target timestamp when possible (timeline feel)
    rows.sort(key=lambda r: (r.snapshot_target_ts is None, r.snapshot_target_ts or 0, r.bucket))
    return rows


def _format_table(rows: List[TimelineRow]) -> str:
    """Create a simple fixed-width table for console printing (stdlib only)."""
    headers = ["bucket", "target_utc", "source_utc_yes", "YES(%)", "YES(dec)"]
    data: List[List[str]] = []

    for r in rows:
        data.append(
            [
                r.bucket,
                r.snapshot_target_utc or "",
                r.snapshot_source_utc_yes or "",
                f"{r.yes_price_pct:.4f}",
                f"{r.yes_price_decimal:.6f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)

    out_lines = [fmt_row(headers), sep]
    out_lines.extend(fmt_row(r) for r in data)
    return "\n".join(out_lines)


# =============================================================================
# Public API
# =============================================================================

def run(
    input_jsonl_path: Path = DEFAULT_INPUT_JSONL_PATH,
    n_random_markets: int = DEFAULT_N_RANDOM_MARKETS,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    print_to_console: bool = DEFAULT_PRINT_TO_CONSOLE,
    write_combined_csv: bool = DEFAULT_WRITE_COMBINED_CSV,
    combined_csv_name: str = DEFAULT_COMBINED_CSV_NAME,
) -> RunResult:
    """
    Run the sampling + timeline-table generation.

    This version only includes rows where YES price exists.
    """
    if not input_jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    sampled_market_ids = _reservoir_sample_unique_market_ids(
        jsonl_path=input_jsonl_path,
        sample_size=n_random_markets,
        rng=rng,
    )
    sampled_set = set(sampled_market_ids)

    # Collect rows for sampled markets (includes all runs/lines for those markets).
    rows: List[TimelineRow] = []
    with input_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            market_id = str(obj.get("market_id", "")).strip()
            if market_id in sampled_set:
                rows.extend(_expand_market_line_to_rows(obj))

    # It's possible a sampled market has zero usable prices_yes entries in this file.
    # We keep going; result.rows might be empty for that market.
    rows.sort(
        key=lambda r: (
            r.market_id,
            r.slug,
            r.run_id,
            r.generated_utc,
            r.snapshot_target_ts is None,
            r.snapshot_target_ts or 0,
            r.bucket,
        )
    )

    combined_csv_path: Optional[Path] = None
    if write_combined_csv:
        combined_csv_path = output_dir / combined_csv_name
        with combined_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "market_id",
                    "slug",
                    "run_id",
                    "generated_utc",
                    "bucket",
                    "snapshot_target_ts",
                    "snapshot_target_utc",
                    "snapshot_source_ts_yes",
                    "snapshot_source_utc_yes",
                    "yes_price_decimal",
                    "yes_price_pct",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r.market_id,
                        r.slug,
                        r.run_id,
                        r.generated_utc,
                        r.bucket,
                        r.snapshot_target_ts,
                        r.snapshot_target_utc,
                        r.snapshot_source_ts_yes,
                        r.snapshot_source_utc_yes,
                        r.yes_price_decimal,
                        r.yes_price_pct,
                    ]
                )

    if print_to_console:
        if combined_csv_path is not None:
            print(f"\nWrote combined CSV:\n  {combined_csv_path}")

        print("\nSampled market_ids:")
        for mid in sampled_market_ids:
            print(f"  - {mid}")

        # Group rows by (market_id, slug, run_id, generated_utc)
        current_key: Optional[Tuple[str, str, str, str]] = None
        group: List[TimelineRow] = []

        def flush() -> None:
            nonlocal group, current_key
            if current_key is None:
                return
            market_id, slug, run_id, generated_utc = current_key
            print("\n" + "=" * 90)
            print(f"market_id : {market_id}")
            print(f"slug      : {slug}")
            print(f"run_id    : {run_id}")
            print(f"generated : {generated_utc}")
            print("-" * 90)
            if group:
                print(_format_table(group))
            else:
                print("(No YES prices available for this market/run in the JSONL file.)")
            group = []

        for r in rows:
            key = (r.market_id, r.slug, r.run_id, r.generated_utc)
            if current_key is None:
                current_key = key
            if key != current_key:
                flush()
                current_key = key
            group.append(r)

        flush()
        print("\nDone.\n")

    return RunResult(
        sampled_market_ids=sampled_market_ids,
        rows=rows,
        combined_csv_path=combined_csv_path,
    )


def main() -> None:
    """Standalone entry point (uses DEFAULT_* variables above)."""
    run()


if __name__ == "__main__":
    main()
