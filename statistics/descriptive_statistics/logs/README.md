# Descriptive Statistics Output

- Run timestamp: **20260329T185431**
- Generated at: **2026-03-29 18:54:43.977084**
- Script: `R/00_descriptive_statistics.R`
- Output directory: `statistics/descriptive_statistics/`

## Inputs

The script reads the following input files (relative to project root):

- `data/complete_dataset_long.csv`
- `data/stock_prices/stock_prices_daily.csv`
- `data/heckman_selection_model/heckman_universe_events.csv`

## Key definitions / filters used

- **Active trading hours** = `abs(difftime(umaEndDate, acceptingOrdersTimestamp, units='hours'))` (best available proxy given reduced inputs).
- **Valid snapshot probabilities** require:
  - `p_polymarket_yes` present and in [0,1]
  - if `snapshot_dt_utc` exists in the file: it must be non-missing
- **Hours from earnings release to UMA end** = `difftime(umaEndDate, earnings_release_datetime, units='hours')`.

- **Sample markets** are restricted to resolved outcome in {YES, NO}.

## Output files

Tables are written as both **CSV** and **JSONL**. Plots are **PNG**.

See `00_output_manifest.csv` for a complete list of outputs and descriptions.

## Notes on calibration plots

Calibration plots show **observed YES rate** vs **implied probability** (Polymarket `p_polymarket_yes`).
The dashed 45-degree line represents perfect calibration.
