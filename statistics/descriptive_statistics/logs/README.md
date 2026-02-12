# Descriptive Statistics Output

- Run timestamp: **20260212T124834**
- Generated at: **2026-02-12 12:48:53.998541**
- Script: `Corporate_Earnings/R/scripts/descriptive_stats.R`
- Output directory: `Corporate_Earnings/statistics/descriptive_statistics/`

## Inputs

The script reads the following input files (relative to project root):

- `data/markets/markets.csv`
- `data/brier_scores/brier_scores_market_horizon.csv`
- `data/poly_prices/poly_prices_long.csv`
- `data/stock_prices/stock_prices_daily.csv`
- `data/corporate_info/corporate_info.csv`
- `data/heckman_selection_model/heckman_universe_companies.csv`
- `data/heckman_selection_model/heckman_universe_events.csv`

## Key definitions / filters used

- **Active trading hours** = `abs(difftime(umaEndDate, startDate, units='hours'))`.
- **Valid (non-stale) snapshot prices** require:
  - `price_yes` and `price_no` not missing
  - `src_yes_ts` and `src_no_ts` not missing
  - `abs(price_yes + price_no - 1) <= complement_tolerance` (default 0.05 if tolerance missing)

- **Sample markets** are restricted to:
  - Resolved outcome in {YES, NO}
  - Non-missing `val_ric` and `val_anchor_date` (matched to corporate events)
  - If `val_status` exists: `val_status` starts with `MATCHED`

## Output files

Tables are written as both **CSV** and **JSONL**. Plots are **PNG**.

See `00_output_manifest.csv` for a complete list of outputs and descriptions.

## Notes on calibration plots

Calibration plots show **observed YES rate** vs **implied probability** (Polymarket `price_yes`).
The dashed 45-degree line represents perfect calibration (e.g., p=0.5 should resolve YES 50% of the time).
