import json
from collections import Counter
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DATA_PATH = Path("data/complete_dataset_wide.jsonl")
NY = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def parse_as_utc(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def classify(dt_utc: datetime) -> str:
    dt_ny = dt_utc.astimezone(NY)
    t = dt_ny.time().replace(tzinfo=None)

    if dt_ny.weekday() >= 5:
        return "weekend"

    if t < MARKET_OPEN:
        return "pre_market"
    if MARKET_OPEN <= t < MARKET_CLOSE:
        return "intraday"
    return "after_hours"

total = 0
valid = 0
counts = Counter()
examples = {
    "pre_market": [],
    "intraday": [],
    "after_hours": [],
    "weekend": [],
}

with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        raw = row.get("earnings_release_datetime")
        if not raw:
            continue

        try:
            dt_utc = parse_as_utc(raw)
        except Exception:
            continue

        valid += 1
        bucket = classify(dt_utc)
        counts[bucket] += 1

        if len(examples[bucket]) < 5:
            examples[bucket].append({
                "ticker": row.get("ticker"),
                "raw": raw,
                "ny_time": dt_utc.astimezone(NY).isoformat(),
            })

print(f"Total rows: {total}")
print(f"Valid rows: {valid}")
print()

for bucket in ["pre_market", "intraday", "after_hours", "weekend"]:
    n = counts[bucket]
    pct = 100 * n / valid if valid else 0
    print(f"{bucket}: {n} ({pct:.2f}%)")
    for ex in examples[bucket]:
        print(f"  {ex['ticker']}: raw={ex['raw']} -> NY={ex['ny_time']}")
    print()