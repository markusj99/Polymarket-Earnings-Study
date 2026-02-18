# Brier score statistical analysis outputs

- Run timestamp: **20260216T150502**
- Generated at: **2026-02-16 15:05:06.392662**
- Script: `R/BS_BrierScore_Analysis.R`

## What this script does
- Uses precomputed Brier scores from `data/brier_scores/brier_scores_market_horizon.csv`.
- Filters to **non-stale/usable** observations via `usable_polymarket == TRUE` (and `status == ok/usable` if present).
- Excludes horizons: 4w, 3w, 2w.
- Produces Brier score tables (mean ± 95% CI) overall and by horizon.
- Computes Brier Skill Score (BSS) vs coinflip and historical base-rate benchmarks.
- Builds a market-level correlation matrix (Pearson) including p-values.
- Runs OLS regressions for Polymarket Brier loss (panel + market-level) with cluster-robust SE by market_id.
- Runs Logit + Probit models for probability Polymarket prediction is correct (cluster-robust SE).
- Computes 5-bin (width 0.2) empirical `P(YES | price bin)` tables and a calibration-style plot for one horizon.
- Prints paper-style regression tables to console/log (modelsummary), including R^2 for OLS and McFadden R^2 for GLMs.

## Outputs
- Tables: CSV + JSONL + JSON (see `BA_00_output_manifest.csv`).
- Plots: PNG.
- Logs: this README + sessionInfo in `logs/`.
