#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
random_markets_table_from_correct_jsonl.py

Purpose
-------
Select N random markets from:
    C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\validation\correct.jsonl

…and output a table/CSV including (among others) these requested variables:
- Polymarket estimate to beat (polymarket_estimate)
- Polymarket resolved outcome (polymarket_resolved_outcome)
- Eikon estimate (eikon_eps_mean_estimate)
- Eikon actual result (eikon_actual_eps)

Works both:
1) Standalone:
   python random_markets_table_from_correct_jsonl.py

2) Imported/called from another script:
   import random_markets_table_from_correct_jsonl as rmt
   result = rmt.run(input_jsonl_path=Path(r"...\correct.jsonl"), n_random_markets=10)
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set


# =============================================================================
# DEFAULT CONFIG (kept inside this script; used when running standalone)
# =============================================================================

DEFAULT_N_RANDOM_MARKETS: int = 15
DEFAULT_RANDOM_SEED: Optional[int] = None  # Set to an int for reproducible sampling

DEFAULT_INPUT_JSONL_PATH: Path = Path(
    r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\validation\correct.jsonl"
)

DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\quality_assurance\results"
)

DEFAULT_PRINT_TO_CONSOLE: bool = True
DEFAULT_WRITE_CSV: bool = True
DEFAULT_CSV_NAME: str = "random_correct_markets_sample.csv"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class CorrectRow:
    ticker: str
    slug: str
    question: str
    anchor_date: str

    # Requested additions (from correct.jsonl)
    polymarket_estimate: str                 # "estimate to beat as specified by Polymarket"
    polymarket_resolved_outcome: str         # Polymarket resolved outcome
    eikon_eps_mean_estimate: str             # Eikon estimate
    eikon_actual_eps: str                    # Eikon actual result

    # Existing useful context fields
    ric: str
    estimate_used: str
    estimate_used_source: str


@dataclass(frozen=True)
class RunResult:
    sampled_market_ids: List[str]
    rows: List[CorrectRow]
    csv_path: Optional[Path]


# =============================================================================
# Helpers
# =============================================================================

def _safe_str(x: Any) -> str:
    """Convert to a readable string, using empty string for None."""
    if x is None:
        return ""
    return str(x)


def _reservoir_sample_unique_market_ids(
    jsonl_path: Path,
    sample_size: int,
    rng: random.Random,
) -> List[str]:
    """
    Reservoir sample unique market IDs from correct.jsonl.
    Uses 'market_id' field.
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

            mid = _safe_str(obj.get("market_id", "")).strip()
            if not mid or mid in seen:
                continue

            seen.add(mid)
            unique_count += 1

            if len(reservoir) < sample_size:
                reservoir.append(mid)
            else:
                j = rng.randint(1, unique_count)
                if j <= sample_size:
                    reservoir[j - 1] = mid

    if unique_count == 0:
        raise RuntimeError("No valid 'market_id' values found in correct.jsonl.")

    if sample_size > unique_count:
        return list(seen)

    return reservoir


def _format_console_table(rows: List[CorrectRow]) -> str:
    """
    Console-friendly table. Truncates long text fields for readability.
    Full text is preserved in CSV.
    """
    def trunc(s: str, n: int) -> str:
        s = s.replace("\n", " ").strip()
        if len(s) <= n:
            return s
        return s[: n - 1] + "…"

    headers = [
        "ticker",
        "slug",
        "anchor_date",
        "pm_estimate",
        "pm_outcome",
        "eikon_est",
        "eikon_actual",
        "ric",
        "estimate_used",
        "estimate_src",
        "question",
    ]

    data: List[List[str]] = []
    for r in rows:
        data.append(
            [
                trunc(r.ticker, 10),
                trunc(r.slug, 44),
                trunc(r.anchor_date, 12),
                trunc(r.polymarket_estimate, 12),
                trunc(r.polymarket_resolved_outcome, 10),
                trunc(r.eikon_eps_mean_estimate, 12),
                trunc(r.eikon_actual_eps, 12),
                trunc(r.ric, 16),
                trunc(r.estimate_used, 14),
                trunc(r.estimate_used_source, 18),
                trunc(r.question, 70),
            ]
        )

    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)

    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in data)
    return "\n".join(out)


# =============================================================================
# Public API
# =============================================================================

def run(
    input_jsonl_path: Path = DEFAULT_INPUT_JSONL_PATH,
    n_random_markets: int = DEFAULT_N_RANDOM_MARKETS,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    print_to_console: bool = DEFAULT_PRINT_TO_CONSOLE,
    write_csv: bool = DEFAULT_WRITE_CSV,
    csv_name: str = DEFAULT_CSV_NAME,
) -> RunResult:
    """
    Sample N random markets from correct.jsonl and output a CSV + console table.

    CSV columns include:
    - polymarket_estimate (Polymarket threshold)
    - polymarket_resolved_outcome
    - eikon_eps_mean_estimate
    - eikon_actual_eps
    """
    if not input_jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    sampled_market_ids = _reservoir_sample_unique_market_ids(input_jsonl_path, n_random_markets, rng)
    sampled_set = set(sampled_market_ids)

    rows: List[CorrectRow] = []
    with input_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            mid = _safe_str(obj.get("market_id", "")).strip()
            if mid not in sampled_set:
                continue

            rows.append(
                CorrectRow(
                    ticker=_safe_str(obj.get("ticker")),
                    slug=_safe_str(obj.get("slug")),
                    question=_safe_str(obj.get("question")),
                    anchor_date=_safe_str(obj.get("anchor_date")),

                    polymarket_estimate=_safe_str(obj.get("polymarket_estimate")),
                    polymarket_resolved_outcome=_safe_str(obj.get("polymarket_resolved_outcome")),
                    eikon_eps_mean_estimate=_safe_str(obj.get("eikon_eps_mean_estimate")),
                    eikon_actual_eps=_safe_str(obj.get("eikon_actual_eps")),

                    ric=_safe_str(obj.get("ric")),
                    estimate_used=_safe_str(obj.get("estimate_used")),
                    estimate_used_source=_safe_str(obj.get("estimate_used_source")),
                )
            )

    # Stable sort for readability
    rows.sort(key=lambda r: (r.ticker, r.slug, r.anchor_date))

    csv_path: Optional[Path] = None
    if write_csv:
        csv_path = output_dir / csv_name
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ticker",
                    "slug",
                    "question",
                    "anchor_date",
                    "polymarket_estimate",
                    "polymarket_resolved_outcome",
                    "eikon_eps_mean_estimate",
                    "eikon_actual_eps",
                    "ric",
                    "estimate_used",
                    "estimate_used_source",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r.ticker,
                        r.slug,
                        r.question,
                        r.anchor_date,
                        r.polymarket_estimate,
                        r.polymarket_resolved_outcome,
                        r.eikon_eps_mean_estimate,
                        r.eikon_actual_eps,
                        r.ric,
                        r.estimate_used,
                        r.estimate_used_source,
                    ]
                )

    if print_to_console:
        if csv_path is not None:
            print(f"\nWrote CSV:\n  {csv_path}")
        print("\nSampled market_ids:")
        for mid in sampled_market_ids:
            print(f"  - {mid}")
        print("\n" + _format_console_table(rows) + "\n")

    return RunResult(sampled_market_ids=sampled_market_ids, rows=rows, csv_path=csv_path)


def main() -> None:
    """Standalone entry point (uses DEFAULT_* variables above)."""
    run()


if __name__ == "__main__":
    main()
