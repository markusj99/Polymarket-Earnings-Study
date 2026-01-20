r"""
random_markets_table_from_markets_jsonl.py

Purpose
-------
Select N random markets from:
    C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\markets\markets.jsonl

…and output a table/CSV with these fields IN THIS ORDER:
    ticker, slug, question, endDate, resolvedOutcome, outcomePrices, volumeNum

Works both:
1) Standalone:
   python random_markets_table_from_markets_jsonl.py

2) Imported/called from another script:
   import random_markets_table_from_markets_jsonl as rmt
   result = rmt.run(input_jsonl_path=Path(r"...\markets.jsonl"), n_random_markets=10)

Notes
-----
- This script does NOT use CLI. Defaults are defined in-script.
- outcomePrices in markets.jsonl may be:
    * missing
    * a list like ["0.63","0.37"]
    * a JSON-encoded string like "[\"1\", \"0\"]"
  We normalize it to a compact string:
    "Yes=<yes>, No=<no>" when possible, otherwise the raw value as string.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# DEFAULT CONFIG (kept inside this script; used when running standalone)
# =============================================================================

DEFAULT_N_RANDOM_MARKETS: int = 15
DEFAULT_RANDOM_SEED: Optional[int] = None # Set to an int for reproducible sampling

DEFAULT_INPUT_JSONL_PATH: Path = Path(
    r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\data\markets\markets.jsonl"
)

DEFAULT_OUTPUT_DIR: Path = Path(
    r"C:\Users\lasts\Desktop\Polymarket\Corporate_Earnings\quality_assurance\results"
)

DEFAULT_PRINT_TO_CONSOLE: bool = True
DEFAULT_WRITE_CSV: bool = True
DEFAULT_CSV_NAME: str = "random_markets_sample.csv"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class MarketRow:
    ticker: str
    slug: str
    question: str
    endDate: str
    resolvedOutcome: str
    outcomePrices: str
    volumeNum: str


@dataclass(frozen=True)
class RunResult:
    sampled_ids: List[str]
    rows: List[MarketRow]
    csv_path: Optional[Path]


# =============================================================================
# Helpers
# =============================================================================

def _safe_str(x: Any) -> str:
    """Convert to a readable string, using empty string for None."""
    if x is None:
        return ""
    return str(x)


def _normalize_outcome_prices(obj: Dict[str, Any]) -> str:
    """
    Normalize outcomePrices to a readable single string.

    outcomePrices formats seen:
    - missing -> ""
    - list -> ["0.63","0.37"] or [0.63, 0.37]
    - string containing JSON list -> "[\"1\", \"0\"]"
    - something else -> str(...)
    """
    raw = obj.get("outcomePrices", None)
    if raw is None:
        return ""

    # If it's already a list/tuple, use directly
    if isinstance(raw, (list, tuple)):
        vals = list(raw)
        if len(vals) >= 2:
            return f"Yes={vals[0]}, No={vals[1]}"
        return _safe_str(raw)

    # If it's a string, it might be JSON-encoded list
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ""
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) >= 2:
                return f"Yes={parsed[0]}, No={parsed[1]}"
            # If parsed but not list, fall back to stringified parsed
            return _safe_str(parsed)
        except json.JSONDecodeError:
            # Not JSON; keep as-is
            return s

    # Fallback
    return _safe_str(raw)


def _reservoir_sample_unique_ids(
    jsonl_path: Path,
    sample_size: int,
    rng: random.Random,
) -> List[str]:
    """
    Reservoir sample unique market IDs from markets.jsonl.
    Uses 'id' field.
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

            mid = _safe_str(obj.get("id", "")).strip()
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
        raise RuntimeError("No valid 'id' values found in markets.jsonl.")

    if sample_size > unique_count:
        return list(seen)

    return reservoir


def _format_console_table(rows: List[MarketRow]) -> str:
    """
    Console-friendly table. Truncates long text fields for readability.
    Full text is preserved in CSV.
    """
    def trunc(s: str, n: int) -> str:
        s = s.replace("\n", " ").strip()
        if len(s) <= n:
            return s
        return s[: n - 1] + "…"

    headers = ["ticker", "slug", "endDate", "resolvedOutcome", "outcomePrices", "volumeNum", "question"]
    data: List[List[str]] = []
    for r in rows:
        data.append(
            [
                trunc(r.ticker, 10),
                trunc(r.slug, 42),
                trunc(r.endDate, 20),
                trunc(r.resolvedOutcome, 12),
                trunc(r.outcomePrices, 24),
                trunc(r.volumeNum, 14),
                trunc(r.question, 60),
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
    Sample N random markets from markets.jsonl and output a CSV + console table.

    Output column order (exactly as requested):
        ticker, slug, question, endDate, resolvedOutcome, outcomePrices, volumeNum
    """
    if not input_jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    sampled_ids = _reservoir_sample_unique_ids(input_jsonl_path, n_random_markets, rng)
    sampled_set = set(sampled_ids)

    rows: List[MarketRow] = []
    with input_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            mid = _safe_str(obj.get("id", "")).strip()
            if mid not in sampled_set:
                continue

            rows.append(
                MarketRow(
                    ticker=_safe_str(obj.get("ticker")),
                    slug=_safe_str(obj.get("slug")),
                    question=_safe_str(obj.get("question")),
                    endDate=_safe_str(obj.get("endDate")),
                    resolvedOutcome=_safe_str(obj.get("resolvedOutcome")),
                    outcomePrices=_normalize_outcome_prices(obj),
                    volumeNum=_safe_str(obj.get("volumeNum")),
                )
            )

    # Stable sort for readability
    rows.sort(key=lambda r: (r.ticker, r.slug))

    csv_path: Optional[Path] = None
    if write_csv:
        csv_path = output_dir / csv_name
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Exact order requested:
            w.writerow(["ticker", "slug", "question", "endDate", "resolvedOutcome", "outcomePrices", "volumeNum"])
            for r in rows:
                w.writerow([r.ticker, r.slug, r.question, r.endDate, r.resolvedOutcome, r.outcomePrices, r.volumeNum])

    if print_to_console:
        if csv_path is not None:
            print(f"\nWrote CSV:\n  {csv_path}")
        print("\nSampled market ids:")
        for mid in sampled_ids:
            print(f"  - {mid}")

        print("\n" + _format_console_table(rows) + "\n")

    return RunResult(sampled_ids=sampled_ids, rows=rows, csv_path=csv_path)


def main() -> None:
    """Standalone entry point (uses DEFAULT_* variables above)."""
    run()


if __name__ == "__main__":
    main()
