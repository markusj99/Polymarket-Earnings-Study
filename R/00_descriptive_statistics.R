#!/usr/bin/env Rscript
# =============================================================================
# File:    Corporate_Earnings/R/scripts/descriptive_stats.R
# Purpose: Create descriptive statistics tables + plots for the Polymarket
#          Corporate Earnings Study.
#
# Inputs (relative to project root):
#   Polymarket-Earnings-Study/data/markets/markets.csv
#   Polymarket-Earnings-Study/data/brier_scores/brier_scores_market_horizon.csv
#   Polymarket-Earnings-Study/data/poly_prices/poly_prices_long.csv
#   Polymarket-Earnings-Study/data/stock_prices/stock_prices_daily.csv
#   Polymarket-Earnings-Study/data/corporate_info/corporate_info.csv
#   Polymarket-Earnings-Study/data/heckman_selection_model/heckman_universe_companies.csv
#   Polymarket-Earnings-Study/data/heckman_selection_model/heckman_universe_events.csv
#
# Outputs (relative to project root):
#   Corporate_Earnings/statistics/descriptive_statistics/
#     - Tables: CSV + JSONL
#     - Plots:  PNG
#     - README.md manifest and run log
#
# Notes:
#   - The script uses ONLY relative paths (root is inferred from this script's
#     location: root/Corporate_Earnings/R/scripts/descriptive_stats.R).
#   - “Stale prices” are excluded by requiring non-missing price_yes/price_no AND
#     non-missing src_yes_ts/src_no_ts AND complement error within tolerance.
#   - Time snapshot summaries are produced per snapshot_label (e.g., 4w, 3w, ...).
# =============================================================================

options(stringsAsFactors = FALSE, scipen = 999)

# ---------- Project root finder (base R, cross-platform) ----------
find_project_root <- function(start = getwd()) {
  dir <- normalizePath(start, winslash = "/", mustWork = FALSE)
  for (i in 1:100) {
    has_lock <- file.exists(file.path(dir, "renv.lock"))
    has_rproj <- length(list.files(dir, pattern = "\\.Rproj$", full.names = TRUE)) > 0
    if (has_lock || has_rproj) return(dir)
    
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  stop("Could not find project root (renv.lock or .Rproj). Run from inside the project.")
}
get_start_dir <- function() {
  # 1) Works for Rscript --file=...
  cmd <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", cmd)
  if (length(m) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", cmd[m[1]]), winslash = "/", mustWork = FALSE)))
  }
  
  # 2) Works in RStudio when you "Source" the file or run interactively
  if (interactive() &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) {
      return(dirname(normalizePath(p, winslash = "/", mustWork = FALSE)))
    }
  }
  
  # 3) Fallback: current working directory
  getwd()
}

ROOT <- find_project_root(get_start_dir())

# --- Canonical project paths (project root is Polymarket-Earnings-Study/) ---
root_dir <- ROOT
data_dir <- file.path(root_dir, "data")
out_dir  <- file.path(root_dir, "statistics", "descriptive_statistics")

# Make sure output folders exist BEFORE sink()/writes
fs::dir_create(out_dir)
fs::dir_create(file.path(out_dir, "logs"))

# ---------- renv activation + restore ----------
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
renv::load(project = ROOT)
renv::restore(project = ROOT, prompt = FALSE)

# ---------- Packages ----------
REQUIRED_PKGS <- c("tidyverse","lubridate","janitor","scales","jsonlite","glue","fs")

missing <- REQUIRED_PKGS[!vapply(REQUIRED_PKGS, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
if (length(missing)) {
  stop(
    "Missing packages even after renv::restore(): ",
    paste(missing, collapse = ", "),
    "\nRun: renv::restore() or ask the maintainer to update renv.lock via renv::snapshot().",
    call. = FALSE
  )
}

invisible(lapply(REQUIRED_PKGS, library, character.only = TRUE))

# ------------------------------ Color palette --------------------------------
COL_GREY_1    <- "#808080"
COL_GREY_2    <- "#A9A9A9"
COL_RED       <- "#E3170A"
COL_DARKBLUE  <- "#00008B"
COL_LIGHTBLUE <- "#008dd5"
COL_BLUE      <- "#0000FF"
COL_GREEN_YES <- "#3EC300"   # explicitly requested for YES markets in the diverging plot
DATA_COL <- COL_LIGHTBLUE
BORDER_COL <- "white"

theme_corporate <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      axis.title = ggplot2::element_text(face = "bold"),
      legend.position = "bottom"
    )
}

# ------------------------------ Path helpers ---------------------------------
get_script_path <- function() {
  # Works when run via: Rscript descriptive_stats.R  (uses --file=...)
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
  }
  
  # Works in RStudio if file is open and RStudio API is available
  if (interactive() &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) {
      return(normalizePath(p, winslash = "/", mustWork = FALSE))
    }
  }
  
  # Fallback: unknown script path
  return(NA_character_)
}

script_path <- get_script_path()
script_dir  <- if (!is.na(script_path)) dirname(script_path) else getwd()

# ------------------------------- IO helpers ----------------------------------
read_csv_required <- function(path) {
  if (!file.exists(path)) {
    stop(glue("Input file not found: {path}"), call. = FALSE)
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

write_table_dual <- function(df, stem, out_dir) {
  table_dir <- file.path(out_dir, "tables")
  fs::dir_create(table_dir)
  
  csv_path   <- file.path(table_dir, paste0(stem, ".csv"))
  jsonl_path <- file.path(table_dir, paste0(stem, ".jsonl"))
  
  readr::write_csv(df, csv_path, na = "")
  
  con <- file(jsonl_path, open = "wt")
  on.exit(close(con), add = TRUE)
  jsonlite::stream_out(df, con = con, verbose = FALSE)
  
  list(csv = csv_path, jsonl = jsonl_path)
}

save_plot_png <- function(p, stem, out_dir, width = 10, height = 6, dpi = 300) {
  plot_dir <- file.path(out_dir, "plots")
  fs::dir_create(plot_dir)
  
  png_path <- file.path(plot_dir, paste0(stem, ".png"))
  ggplot2::ggsave(filename = png_path, plot = p, width = width, height = height, dpi = dpi)
  png_path
}

safe_quantile <- function(x, probs) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(rep(NA_real_, length(probs)))
  as.numeric(stats::quantile(x, probs = probs, na.rm = TRUE, type = 7))
}

safe_min <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  min(x)
}

safe_max <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  max(x)
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}

safe_median <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  median(x)
}

parse_ts_utc <- function(x) {
  if (inherits(x, "POSIXct")) return(with_tz(x, tzone = "UTC"))
  if (is.numeric(x)) return(as_datetime(x, tz = "UTC"))
  x_chr <- as.character(x)
  
  suppressWarnings(
    parse_date_time(
      x_chr,
      orders = c(
        "ymdHMSOSz", "ymdHMSz", "ymdHMSOS", "ymdHMS",
        "ymdTz", "ymd"
      ),
      tz = "UTC",
      exact = FALSE
    )
  )
}

parse_date_utc <- function(x) {
  suppressWarnings(ymd(as.character(x), tz = "UTC"))
}

normalize_ric <- function(x) {
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_to_upper(x)
  x <- dplyr::na_if(x, "")
  x
}

# ------------------------------- Run logging ---------------------------------
run_ts <- format(Sys.time(), "%Y%m%dT%H%M%S")
log_path <- file.path(out_dir, "logs", paste0("descriptive_stats_run_", run_ts, ".log.txt"))
sink(log_path, split = TRUE)
on.exit(sink(), add = TRUE)

cat(glue("Descriptive statistics run started: {Sys.time()}\n"))
cat(glue("Script directory: {script_dir}\n"))
cat(glue("Inferred project root: {root_dir}\n"))
cat(glue("Output directory: {out_dir}\n\n"))

# ------------------------------ Read input data (central loader) -------------
source(file.path(root_dir, "R", "utils", "load_data.R"))

cat("Reading input files via load_project_data()...\n")
D <- load_project_data(root_dir)

# ---------------------------------------------------------------------------
# Paths used for README (define here so fs::path_rel() never sees NULL)
# ---------------------------------------------------------------------------
paths <- list(
  markets      = file.path(root_dir, "data", "markets", "markets.csv"),
  poly_prices  = file.path(root_dir, "data", "poly_prices", "poly_prices_long.csv"),
  stock_prices = file.path(root_dir, "data", "stock_prices", "stock_prices_daily.csv"),
  brier        = file.path(root_dir, "data", "brier_scores", "brier_scores_market_horizon.csv"),
  corporate    = file.path(root_dir, "data", "corporate_info", "corporate_info.csv"),
  heck_events  = file.path(root_dir, "data", "heckman_selection_model", "heckman_universe_events.csv"),
  heck_comp    = file.path(root_dir, "data", "heckman_selection_model", "heckman_selection_companies.csv")
)

# Keep the same *_raw object names so the rest of the script is unchanged
markets_raw      <- D$markets
poly_prices_raw  <- D$poly_prices
stock_prices_raw <- D$stock_prices
brier_raw        <- D$brier_scores
corporate_raw    <- D$corporate_info

# Update names to match your new loader outputs
heck_events_raw  <- D$heckman_universe_events
heck_comp_raw    <- D$heckman_universe_companies


cat("Cleaning column names...\n")
markets      <- janitor::clean_names(markets_raw)
brier        <- janitor::clean_names(brier_raw)
poly_prices  <- janitor::clean_names(poly_prices_raw)
stock_prices <- janitor::clean_names(stock_prices_raw)
corporate    <- janitor::clean_names(corporate_raw)
heck_comp    <- janitor::clean_names(heck_comp_raw)
heck_events  <- janitor::clean_names(heck_events_raw)

# ------------------------------ Prepare markets ------------------------------
cat("Preparing markets dataset...\n")

markets <- markets %>%
  mutate(
    id = as.character(id),
    start_date_utc = if ("start_date" %in% names(.)) parse_ts_utc(start_date) else as.POSIXct(NA),
    uma_end_date_utc = if ("uma_end_date" %in% names(.)) parse_ts_utc(uma_end_date) else as.POSIXct(NA),
    end_date_utc = if ("end_date" %in% names(.)) parse_ts_utc(end_date) else as.POSIXct(NA),
    val_anchor_date = if ("val_anchor_date" %in% names(.)) parse_date_utc(val_anchor_date) else as.Date(NA),
    
    resolved_outcome_std = case_when(
      "resolved_outcome" %in% names(.) & str_to_upper(resolved_outcome) %in% c("YES", "Y") ~ "YES",
      "resolved_outcome" %in% names(.) & str_to_upper(resolved_outcome) %in% c("NO", "N") ~ "NO",
      TRUE ~ NA_character_
    ),
    
    volume_num    = suppressWarnings(as.numeric(volume_num)),
    liquidity_num = suppressWarnings(as.numeric(liquidity_num)),
    
    active_trading_hours = as.numeric(difftime(uma_end_date_utc, start_date_utc, units = "hours")),
    active_trading_hours = if_else(!is.na(active_trading_hours), abs(active_trading_hours), NA_real_),
    
    # EPS fields (if present)
    val_eikon_eps_mean_estimate = if ("val_eikon_eps_mean_estimate" %in% names(.)) suppressWarnings(as.numeric(val_eikon_eps_mean_estimate)) else NA_real_,
    val_polymarket_estimate     = if ("val_polymarket_estimate" %in% names(.)) suppressWarnings(as.numeric(val_polymarket_estimate)) else NA_real_,
    eps_estimate_diff           = val_polymarket_estimate - val_eikon_eps_mean_estimate,
    
    val_surprise = if ("val_surprise" %in% names(.)) suppressWarnings(as.numeric(val_surprise)) else NA_real_,
    
    ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
    val_ric = if ("val_ric" %in% names(.)) normalize_ric(val_ric) else NA_character_
  )

# Define the analysis sample:
# - keep resolved YES/NO
# - keep matched corporate events if val_ric & val_anchor_date exist
# - if val_status exists, keep those starting with MATCHED
markets_sample <- markets %>%
  filter(resolved_outcome_std %in% c("YES", "NO")) %>%
  {
    if ("val_ric" %in% names(.) && "val_anchor_date" %in% names(.)) {
      filter(., !is.na(val_ric), !is.na(val_anchor_date))
    } else {
      .
    }
  } %>%
  {
    if ("val_status" %in% names(.)) {
      filter(., !is.na(val_status), str_detect(val_status, "^MATCHED"))
    } else {
      .
    }
  }

cat(glue("Markets (all):    {nrow(markets)} rows\n"))
cat(glue("Markets (sample): {nrow(markets_sample)} rows\n\n"))

# ---------------------------- Prepare corporate info --------------------------
cat("Preparing corporate_info dataset...\n")

corporate <- corporate %>%
  mutate(
    ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
    ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
    market_cap_usd = if ("market_cap_usd" %in% names(.)) suppressWarnings(as.numeric(market_cap_usd)) else NA_real_,
    analysts_covering_latest = if ("analysts_covering_latest" %in% names(.)) suppressWarnings(as.numeric(analysts_covering_latest)) else NA_real_,
    analysts_covering_sample_mean = if ("analysts_covering_sample_mean" %in% names(.)) suppressWarnings(as.numeric(analysts_covering_sample_mean)) else NA_real_,
    analysts_covering_sample_median = if ("analysts_covering_sample_median" %in% names(.)) suppressWarnings(as.numeric(analysts_covering_sample_median)) else NA_real_
  )

# Sample firms = firms that appear in markets_sample
sample_firms <- markets_sample %>%
  distinct(val_ric) %>%
  rename(ric = val_ric) %>%
  left_join(corporate, by = "ric")

cat(glue("Sample firms (unique RIC): {n_distinct(markets_sample$val_ric)}\n\n"))

# ----------------------------- Prepare poly prices ----------------------------
cat("Preparing poly_prices_long dataset (and excluding stale prices)...\n")

poly_prices <- poly_prices %>%
  mutate(
    market_id = as.character(market_id),
    snapshot_label = as.character(snapshot_label),
    snapshot_offset_seconds = suppressWarnings(as.numeric(snapshot_offset_seconds)),
    price_yes = suppressWarnings(as.numeric(price_yes)),
    price_no  = suppressWarnings(as.numeric(price_no)),
    src_yes_ts = suppressWarnings(as.numeric(src_yes_ts)),
    src_no_ts  = suppressWarnings(as.numeric(src_no_ts)),
    complement_tolerance = suppressWarnings(as.numeric(complement_tolerance)),
    complement_error = abs((price_yes + price_no) - 1)
  )

# Snapshot ordering by offset seconds (largest offset first, e.g., 4w then 3w ...)
snapshot_levels <- poly_prices %>%
  distinct(snapshot_label, snapshot_offset_seconds) %>%
  arrange(desc(snapshot_offset_seconds)) %>%
  pull(snapshot_label) %>%
  unique()

# Valid (non-stale) price definition:
# - non-missing prices
# - non-missing src timestamps (so we know it wasn't imputed/missing)
# - complement error within tolerance (or default tolerance if missing)
prices_valid <- poly_prices %>%
  mutate(
    snapshot_label = factor(snapshot_label, levels = snapshot_levels),
    complement_tolerance = if_else(is.na(complement_tolerance), 0.05, complement_tolerance)
  ) %>%
  filter(
    !is.na(price_yes), !is.na(price_no),
    !is.na(src_yes_ts), !is.na(src_no_ts),
    is.finite(complement_error),
    complement_error <= complement_tolerance
  )

cat(glue("Poly price rows (all):   {nrow(poly_prices)}\n"))
cat(glue("Poly price rows (valid): {nrow(prices_valid)}\n\n"))

# Join prices to sample markets and firm attributes
prices_sample <- prices_valid %>%
  inner_join(
    markets_sample %>%
      select(
        id, ticker, slug, resolved_outcome_std, volume_num, liquidity_num,
        start_date_utc, uma_end_date_utc, active_trading_hours, val_ric, val_anchor_date,
        eps_estimate_diff, val_surprise
      ),
    by = c("market_id" = "id")
  ) %>%
  left_join(
    sample_firms %>%
      select(
        ric, market_cap_usd, gics_sector, gics_industry, trbc_industry,
        hq_country, primary_exchange, analysts_covering_latest,
        analysts_covering_sample_mean, analysts_covering_sample_median
      ),
    by = c("val_ric" = "ric")
  ) %>%
  mutate(outcome_yes = as.integer(resolved_outcome_std == "YES"))

cat(glue("Joined snapshot sample rows: {nrow(prices_sample)}\n"))
cat(glue("Distinct markets with valid snapshot prices: {n_distinct(prices_sample$market_id)}\n\n"))

# ----------------------------- Output manifest --------------------------------
manifest <- tibble(
  type = character(),
  file = character(),
  description = character()
)

add_manifest <- function(type, path, description) {
  manifest <<- bind_rows(
    manifest,
    tibble(type = type, file = basename(path), description = description)
  )
}

# =============================================================================
# 1) Number of observations per snapshot
# =============================================================================
cat("1) Observations per snapshot...\n")

obs_per_snapshot <- prices_sample %>%
  count(snapshot_label, name = "n_observations") %>%
  arrange(snapshot_label)

out_paths <- write_table_dual(obs_per_snapshot, "01_obs_per_snapshot", out_dir)
add_manifest("table", out_paths$csv,   "Number of (valid-price) observations per snapshot_label (CSV).")
add_manifest("table", out_paths$jsonl, "Number of (valid-price) observations per snapshot_label (JSONL).")

p_obs <- ggplot(obs_per_snapshot, aes(x = snapshot_label, y = n_observations)) +
  geom_col(fill = DATA_COL, color = BORDER_COL) +
  labs(
    title = "Number of observations per time snapshot",
    x = "Snapshot label",
    y = "Number of market-snapshot observations (valid prices only)"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_obs, "01_obs_per_snapshot", out_dir, width = 9, height = 5)
add_manifest("plot", plot_path, "Bar chart of valid-price observations per snapshot_label.")

# Also: availability vs total sample markets
total_sample_markets <- n_distinct(markets_sample$id)
availability_per_snapshot <- obs_per_snapshot %>%
  mutate(
    n_total_sample_markets = total_sample_markets,
    share_with_valid_price = n_observations / n_total_sample_markets
  )

out_paths <- write_table_dual(availability_per_snapshot, "01b_price_availability_per_snapshot", out_dir)
add_manifest("table", out_paths$csv,   "Share of sample markets with valid prices per snapshot (CSV).")
add_manifest("table", out_paths$jsonl, "Share of sample markets with valid prices per snapshot (JSONL).")

p_avail <- ggplot(availability_per_snapshot, aes(x = snapshot_label, y = share_with_valid_price)) +
  geom_col(fill = DATA_COL, color = BORDER_COL, alpha = 0.85) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Price availability by time snapshot",
    x = "Snapshot label",
    y = "Share of sample markets with valid snapshot prices"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_avail, "01b_price_availability_per_snapshot", out_dir, width = 9, height = 5)
add_manifest("plot", plot_path, "Bar chart: share of sample markets with valid snapshot prices per snapshot_label.")

# =============================================================================
# 2) Active trading hours (startDate -> umaEndDate)
# =============================================================================
cat("2) Active trading hours...\n")

active_hours_summary <- markets_sample %>%
  transmute(active_trading_hours) %>%
  summarise(
    n = sum(!is.na(active_trading_hours)),
    min = safe_min(active_trading_hours),
    p25 = safe_quantile(active_trading_hours, 0.25),
    mean = safe_mean(active_trading_hours),
    median = safe_median(active_trading_hours),
    p75 = safe_quantile(active_trading_hours, 0.75),
    max = safe_max(active_trading_hours)
  )

out_paths <- write_table_dual(active_hours_summary, "02_active_trading_hours_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for active_trading_hours (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for active_trading_hours (JSONL).")

p_hours <- ggplot(markets_sample, aes(x = active_trading_hours)) +
  geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  labs(
    title = "Distribution of active trading hours (UMA end - start)",
    x = "Active trading hours",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_hours, "02_active_trading_hours_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: active_trading_hours across markets.")

# =============================================================================
# 3) YES vs NO resolved markets over time (diverging stacked bars)
# =============================================================================
cat("3) Resolved YES vs NO over time...\n")

resolved_counts_by_date <- markets_sample %>%
  filter(!is.na(uma_end_date_utc)) %>%
  mutate(uma_end_date = as.Date(uma_end_date_utc)) %>%
  count(uma_end_date, resolved_outcome_std, name = "n_markets") %>%
  mutate(n_signed = if_else(resolved_outcome_std == "NO", -n_markets, n_markets)) %>%
  arrange(uma_end_date, resolved_outcome_std)

out_paths <- write_table_dual(resolved_counts_by_date, "03_resolved_counts_by_uma_end_date", out_dir)
add_manifest("table", out_paths$csv,   "Counts of resolved markets per UMA end date (NO shown negative in plot; CSV).")
add_manifest("table", out_paths$jsonl, "Counts of resolved markets per UMA end date (JSONL).")

p_resolved_time <- ggplot(resolved_counts_by_date, aes(x = uma_end_date, y = n_signed, fill = resolved_outcome_std)) +
  geom_col() +
  geom_hline(yintercept = 0, color = BORDER_COL, linewidth = 0.3) +
  scale_fill_manual(
    values = c("YES" = COL_GREEN_YES, "NO" = COL_RED),
    name = "Resolved outcome"
  ) +
  scale_x_date(date_labels = "%Y-%m-%d") +
  labs(
    title = "Markets resolved to YES vs NO over time (by UMA end date)",
    x = "UMA end date",
    y = "Number of markets (NO below zero)"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_resolved_time, "03_resolved_yes_no_over_time", out_dir, width = 12, height = 6)
add_manifest("plot", plot_path, "Diverging bar chart: resolved YES (positive) and NO (negative) over time.")

# =============================================================================
# 4) Share of markets resolved to YES vs NO
# =============================================================================
cat("4) Share of resolved YES vs NO...\n")

resolved_share <- markets_sample %>%
  count(resolved_outcome_std, name = "n_markets") %>%
  mutate(share = n_markets / sum(n_markets)) %>%
  arrange(desc(n_markets))

out_paths <- write_table_dual(resolved_share, "04_resolved_share_yes_no", out_dir)
add_manifest("table", out_paths$csv,   "Overall share of resolved YES vs NO (CSV).")
add_manifest("table", out_paths$jsonl, "Overall share of resolved YES vs NO (JSONL).")

p_share <- ggplot(resolved_share, aes(x = resolved_outcome_std, y = share, fill = resolved_outcome_std)) +
  geom_col(color = BORDER_COL) +
  geom_text(aes(label = percent(share, accuracy = 0.1)), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("YES" = COL_GREEN_YES, "NO" = COL_RED), guide = "none") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, max(resolved_share$share, na.rm = TRUE) * 1.15)) +
  labs(
    title = "Share of markets resolved to YES vs NO",
    x = NULL,
    y = "Share of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_share, "04_resolved_share_yes_no", out_dir, width = 7, height = 5)
add_manifest("plot", plot_path, "Bar chart: share of markets resolved to YES vs NO.")

# =============================================================================
# 5) Calibration: implied probability vs realized YES frequency (one plot per snapshot)
# =============================================================================
cat("5) Calibration plots per snapshot...\n")

calibration_bins <- prices_sample %>%
  filter(!is.na(price_yes), price_yes >= 0, price_yes <= 1, !is.na(outcome_yes)) %>%
  mutate(prob_bin = cut(price_yes, breaks = seq(0, 1, by = 0.2), include.lowest = TRUE, right = TRUE)) %>%
  group_by(snapshot_label, prob_bin) %>%
  summarise(
    n = n(),
    mean_prob = mean(price_yes, na.rm = TRUE),
    observed_yes_rate = mean(outcome_yes, na.rm = TRUE),
    observed_yes_n = sum(outcome_yes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(snapshot_label, prob_bin)

out_paths <- write_table_dual(calibration_bins, "05_calibration_bins_by_snapshot", out_dir)
add_manifest("table", out_paths$csv,   "Calibration bins: mean implied prob vs observed YES rate (and counts) by snapshot (CSV).")
add_manifest("table", out_paths$jsonl, "Calibration bins: mean implied prob vs observed YES rate (and counts) by snapshot (JSONL).")

# Single calibration plot with all snapshots (faceted)
snapshot_list <- sort(unique(as.character(prices_sample$snapshot_label)))
p_cal_facet <- ggplot(calibration_bins, aes(x = mean_prob, y = observed_yes_rate, group = 1)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = COL_GREY_1) +
  geom_line(color = DATA_COL, linewidth = 0.9) +
  geom_point(color = DATA_COL, size = 2.5) +
  geom_text(aes(label = n), vjust = -0.8, color = BORDER_COL, size = 3) +
  facet_wrap(~ snapshot_label) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2), labels = number_format(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2), labels = number_format(accuracy = 0.1)) +
  labs(
    title = "Calibration (binned): observed YES rate vs implied probability, by snapshot",
    x = "Mean implied probability (price_yes) within bin",
    y = "Observed YES rate (mean outcome_yes)"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_cal_facet, "05_calibration_all_snapshots_faceted", out_dir, width = 12, height = 8)
add_manifest("plot", plot_path, "Faceted calibration plot: mean implied prob vs observed YES rate for all snapshots.")


# =============================================================================
# 6) Volume distribution (log scale but readable in USD)
# =============================================================================
cat("6) Volume distribution...\n")

volume_df <- markets_sample %>% filter(!is.na(volume_num), volume_num > 0)

volume_summary <- volume_df %>%
  summarise(
    n = n(),
    min = safe_min(volume_num),
    p25 = safe_quantile(volume_num, 0.25),
    mean = safe_mean(volume_num),
    median = safe_median(volume_num),
    p75 = safe_quantile(volume_num, 0.75),
    max = safe_max(volume_num)
  )

out_paths <- write_table_dual(volume_summary, "06_volume_summary_usd", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for Polymarket volume (USD) (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for Polymarket volume (USD) (JSONL).")

p_vol <- ggplot(volume_df, aes(x = volume_num)) +
  geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  scale_x_log10(labels = dollar_format(accuracy = 1)) +
  labs(
    title = "Distribution of Polymarket market volume (log scale, labeled in USD)",
    x = "Volume (USD, log10 scale)",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_vol, "06_volume_distribution_log_usd", out_dir, width = 11, height = 6)
add_manifest("plot", plot_path, "Histogram: volume distribution with log10 x-axis labeled in USD.")

# =============================================================================
# 7) Difference: Eikon EPS estimate vs Polymarket EPS estimate
# =============================================================================
cat("7) EPS estimate difference...\n")

eps_diff_df <- markets_sample %>%
  filter(!is.na(eps_estimate_diff), is.finite(eps_estimate_diff))

eps_diff_summary <- eps_diff_df %>%
  summarise(
    n = n(),
    min = safe_min(eps_estimate_diff),
    p25 = safe_quantile(eps_estimate_diff, 0.25),
    mean = safe_mean(eps_estimate_diff),
    median = safe_median(eps_estimate_diff),
    p75 = safe_quantile(eps_estimate_diff, 0.75),
    max = safe_max(eps_estimate_diff)
  )

out_paths <- write_table_dual(eps_diff_summary, "07_eps_estimate_diff_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for (Polymarket EPS estimate - Eikon mean EPS estimate) (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for (Polymarket EPS estimate - Eikon mean EPS estimate) (JSONL).")

p_eps <- ggplot(eps_diff_df, aes(x = eps_estimate_diff)) +
  geom_histogram(bins = 40, fill = DATA_COL, alpha = 0.75, color = BORDER_COL) +
  labs(
    title = "Difference between estimated Polymarket EPS and Eikon EPS (mean estimate)",
    x = "Polymarket estimate - Eikon mean estimate (EPS units)",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_eps, "07_eps_estimate_diff_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: EPS estimate differences (Polymarket - Eikon).")

# =============================================================================
# 8) Distribution of surprise
# =============================================================================
cat("8) Surprise distribution...\n")

surprise_df <- markets_sample %>% filter(!is.na(val_surprise), is.finite(val_surprise))

surprise_summary <- surprise_df %>%
  summarise(
    n = n(),
    min = safe_min(val_surprise),
    p25 = safe_quantile(val_surprise, 0.25),
    mean = safe_mean(val_surprise),
    median = safe_median(val_surprise),
    p75 = safe_quantile(val_surprise, 0.75),
    max = safe_max(val_surprise)
  )

out_paths <- write_table_dual(surprise_summary, "08_surprise_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for surprise (val_surprise) (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for surprise (val_surprise) (JSONL).")

p_surprise <- ggplot(surprise_df, aes(x = val_surprise)) +
  geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  labs(
    title = "Distribution of surprise",
    x = "Surprise (val_surprise)",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_surprise, "08_surprise_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: surprise distribution (val_surprise).")

# =============================================================================
# 9) Scatter: active trading hours (y) vs volume (x)
# =============================================================================
cat("9) Active trading hours vs volume scatter...\n")

hours_vs_volume_df <- markets_sample %>%
  filter(!is.na(active_trading_hours), is.finite(active_trading_hours),
         !is.na(volume_num), volume_num > 0)

p_hours_vol <- ggplot(hours_vs_volume_df, aes(x = volume_num, y = active_trading_hours)) +
  geom_point(alpha = 0.6, color = DATA_COL) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.8, color = COL_RED) +
  scale_x_log10(labels = dollar_format(accuracy = 1)) +
  labs(
    title = "Active trading hours vs Polymarket market volume",
    x = "Volume (USD, log10 scale)",
    y = "Active trading hours"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_hours_vol, "09_active_hours_vs_volume", out_dir, width = 11, height = 6)
add_manifest("plot", plot_path, "Scatter + LOESS: active trading hours vs volume (log USD).")

# =============================================================================
# 10) Firms observed more than once
# =============================================================================
cat("10) Firms observed more than once...\n")

firm_counts <- markets_sample %>%
  mutate(val_ric = normalize_ric(val_ric)) %>%
  filter(!is.na(val_ric)) %>%
  group_by(val_ric) %>%
  summarise(n_markets = n_distinct(id), .groups = "drop") %>%
  arrange(desc(n_markets))

firm_multi_summary <- firm_counts %>%
  summarise(
    n_firms_total = n(),
    n_firms_gt1 = sum(n_markets > 1),
    share_firms_gt1 = n_firms_gt1 / n_firms_total,
    max_markets_per_firm = max(n_markets, na.rm = TRUE)
  )

# Distribution table (how many firms have 1 market, 2 markets, 3 markets, ...)
firm_count_dist <- firm_counts %>%
  count(n_markets, name = "n_firms") %>%
  arrange(n_markets)

out_paths <- write_table_dual(firm_counts, "10_firm_market_counts", out_dir)
add_manifest("table", out_paths$csv,   "Count of markets per firm (RIC) in sample (CSV).")
add_manifest("table", out_paths$jsonl, "Count of markets per firm (RIC) in sample (JSONL).")

out_paths <- write_table_dual(firm_multi_summary, "10b_firms_observed_more_than_once_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary: how many firms appear >1 time in sample (CSV).")
add_manifest("table", out_paths$jsonl, "Summary: how many firms appear >1 time in sample (JSONL).")

out_paths <- write_table_dual(firm_count_dist, "10c_markets_per_firm_distribution_table", out_dir)
add_manifest("table", out_paths$csv,   "Distribution table: number of firms by markets-per-firm (CSV).")
add_manifest("table", out_paths$jsonl, "Distribution table: number of firms by markets-per-firm (JSONL).")

p_firm_counts <- ggplot(firm_count_dist, aes(x = factor(n_markets), y = n_firms)) +
  geom_col(fill = DATA_COL, color = BORDER_COL) +
  labs(
    title = "Distribution: number of markets per firm in the sample",
    x = "Number of markets per firm (RIC)",
    y = "Number of firms"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_firm_counts, "10c_markets_per_firm_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Bar chart: distribution of markets per firm (RIC).")


# =============================================================================
# 11) Market cap distribution (log scale, readable in USD)
# =============================================================================
cat("11) Market cap distribution...\n")

mcap_df <- sample_firms %>% filter(!is.na(market_cap_usd), market_cap_usd > 0)

mcap_summary <- mcap_df %>%
  summarise(
    n = n(),
    min = safe_min(market_cap_usd),
    p25 = safe_quantile(market_cap_usd, 0.25),
    mean = safe_mean(market_cap_usd),
    median = safe_median(market_cap_usd),
    p75 = safe_quantile(market_cap_usd, 0.75),
    max = safe_max(market_cap_usd)
  )

out_paths <- write_table_dual(mcap_summary, "11_market_cap_summary_usd", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for firm market cap (USD) in sample (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for firm market cap (USD) in sample (JSONL).")

p_mcap <- ggplot(mcap_df, aes(x = market_cap_usd)) +
  geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  scale_x_log10(labels = dollar_format(accuracy = 1)) +
  labs(
    title = "Distribution of firm market capitalization (log scale, labeled in USD)",
    x = "Market cap (USD, log10 scale)",
    y = "Count of firms"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_mcap, "11_market_cap_distribution_log_usd", out_dir, width = 11, height = 6)
add_manifest("plot", plot_path, "Histogram: market cap distribution (log USD).")

# =============================================================================
# 12) Scatter: Polymarket volume (y) vs firm market cap (x)
# =============================================================================
cat("12) Volume vs market cap scatter...\n")

vol_mcap_df <- markets_sample %>%
  left_join(sample_firms %>% select(ric, market_cap_usd), by = c("val_ric" = "ric")) %>%
  filter(!is.na(volume_num), volume_num > 0, !is.na(market_cap_usd), market_cap_usd > 0)

p_vol_mcap <- ggplot(vol_mcap_df, aes(x = market_cap_usd, y = volume_num)) +
  geom_point(alpha = 0.6, color = DATA_COL) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.8, color = COL_RED) +
  scale_x_log10(labels = dollar_format(accuracy = 1)) +
  scale_y_log10(labels = dollar_format(accuracy = 1)) +
  labs(
    title = "Polymarket market volume vs firm market cap",
    x = "Market cap (USD, log10 scale)",
    y = "Polymarket volume (USD, log10 scale)"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_vol_mcap, "12_volume_vs_market_cap", out_dir, width = 11, height = 6)
add_manifest("plot", plot_path, "Scatter + LOESS: Polymarket volume vs market cap (log-log).")

# =============================================================================
# 13) HQ country + main exchange distributions (tables + plots)
# =============================================================================
cat("13) HQ country and exchange distributions...\n")

make_dist_table <- function(df, col_name, top_n = 25) {
  if (!col_name %in% names(df)) return(tibble())
  df %>%
    mutate(val = as.character(.data[[col_name]])) %>%
    mutate(val = if_else(is.na(val) | !nzchar(val), "Unknown", val)) %>%
    count(val, name = "n") %>%
    mutate(share = n / sum(n)) %>%
    arrange(desc(n)) %>%
    { if (nrow(.) > top_n) {
      top <- slice_head(., n = top_n)
      other <- tibble(val = "Other", n = sum(.$n[-(1:top_n)]), share = sum(.$share[-(1:top_n)]))
      bind_rows(top, other)
    } else .
    }
}

hq_country_dist <- make_dist_table(sample_firms, "hq_country", top_n = 25)
if (nrow(hq_country_dist) > 0) {
  out_paths <- write_table_dual(hq_country_dist, "13_hq_country_distribution", out_dir)
  add_manifest("table", out_paths$csv,   "Distribution of HQ country for sample firms (CSV).")
  add_manifest("table", out_paths$jsonl, "Distribution of HQ country for sample firms (JSONL).")
  
  p_hq <- ggplot(hq_country_dist, aes(x = reorder(val, n), y = n)) +
    geom_col(fill = DATA_COL, color = BORDER_COL) +
    coord_flip() +
    labs(
      title = "Distribution of firm HQ country (sample firms)",
      x = NULL,
      y = "Number of firms"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_hq, "13_hq_country_distribution", out_dir, width = 10, height = 8)
  add_manifest("plot", plot_path, "Bar chart: HQ country distribution (top categories + Other).")
}

exchange_dist <- make_dist_table(sample_firms, "primary_exchange", top_n = 25)
if (nrow(exchange_dist) > 0) {
  out_paths <- write_table_dual(exchange_dist, "13b_primary_exchange_distribution", out_dir)
  add_manifest("table", out_paths$csv,   "Distribution of primary exchange for sample firms (CSV).")
  add_manifest("table", out_paths$jsonl, "Distribution of primary exchange for sample firms (JSONL).")
  
  p_ex <- ggplot(exchange_dist, aes(x = reorder(val, n), y = n)) +
    geom_col(fill = DATA_COL, alpha = 0.85) +
    coord_flip() +
    labs(
      title = "Distribution of primary exchange (sample firms)",
      x = NULL,
      y = "Number of firms"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_ex, "13b_primary_exchange_distribution", out_dir, width = 10, height = 8)
  add_manifest("plot", plot_path, "Bar chart: primary exchange distribution (top categories + Other).")
}

# =============================================================================
# 14) GICS sector + industry distributions
# =============================================================================
cat("14) GICS sector and industry distributions...\n")

gics_sector_dist <- make_dist_table(sample_firms, "gics_sector", top_n = 50)
if (nrow(gics_sector_dist) > 0) {
  out_paths <- write_table_dual(gics_sector_dist, "14_gics_sector_distribution", out_dir)
  add_manifest("table", out_paths$csv,   "Distribution of GICS sector for sample firms (CSV).")
  add_manifest("table", out_paths$jsonl, "Distribution of GICS sector for sample firms (JSONL).")
  
  p_sector <- ggplot(gics_sector_dist, aes(x = reorder(val, n), y = n)) +
    geom_col(fill = DATA_COL, color = BORDER_COL) +
    coord_flip() +
    labs(
      title = "Distribution of GICS sector (sample firms)",
      x = NULL,
      y = "Number of firms"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_sector, "14_gics_sector_distribution", out_dir, width = 10, height = 7)
  add_manifest("plot", plot_path, "Bar chart: GICS sector distribution.")
}

gics_industry_dist <- make_dist_table(sample_firms, "gics_industry", top_n = 25)
if (nrow(gics_industry_dist) > 0) {
  out_paths <- write_table_dual(gics_industry_dist, "14b_gics_industry_distribution", out_dir)
  add_manifest("table", out_paths$csv,   "Distribution of GICS industry (top categories + Other) for sample firms (CSV).")
  add_manifest("table", out_paths$jsonl, "Distribution of GICS industry (top categories + Other) for sample firms (JSONL).")
  
  p_ind <- ggplot(gics_industry_dist, aes(x = reorder(val, n), y = n)) +
    geom_col(fill = DATA_COL, alpha = 0.85) +
    coord_flip() +
    labs(
      title = "Distribution of GICS industry (sample firms; top categories)",
      x = NULL,
      y = "Number of firms"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_ind, "14b_gics_industry_distribution", out_dir, width = 10, height = 8)
  add_manifest("plot", plot_path, "Bar chart: GICS industry distribution (top categories + Other).")
}

# =============================================================================
# 15) Distribution of analysts covering the firm
# =============================================================================
cat("15) Analysts covering distribution...\n")

analyst_df <- sample_firms %>%
  mutate(analysts_covering = coalesce(analysts_covering_sample_mean, analysts_covering_latest)) %>%
  filter(!is.na(analysts_covering), is.finite(analysts_covering), analysts_covering >= 0)

analyst_summary <- analyst_df %>%
  summarise(
    n = n(),
    min = safe_min(analysts_covering),
    p25 = safe_quantile(analysts_covering, 0.25),
    mean = safe_mean(analysts_covering),
    median = safe_median(analysts_covering),
    p75 = safe_quantile(analysts_covering, 0.75),
    max = safe_max(analysts_covering)
  )

out_paths <- write_table_dual(analyst_summary, "15_analysts_covering_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for analysts covering (sample_mean preferred; else latest) (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for analysts covering (JSONL).")

p_analysts <- ggplot(analyst_df, aes(x = analysts_covering)) +
  geom_histogram(bins = 30, fill = DATA_COL, color = BORDER_COL) +
  labs(
    title = "Distribution of analyst coverage (sample firms)",
    x = "Number of analysts covering the firm",
    y = "Count of firms"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_analysts, "15_analysts_covering_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: analysts covering distribution (sample firms).")

# =============================================================================
# 16) Share of events included in our sample (vs Heckman universe)
# =============================================================================
cat("16) Event inclusion share (sample vs universe)...\n")

# (Safety) If normalize_ric() wasn't defined earlier for some reason, define it here.
if (!exists("normalize_ric", mode = "function")) {
  normalize_ric <- function(x) {
    x <- as.character(x)
    x <- stringr::str_trim(x)
    x <- stringr::str_to_upper(x)
    x <- dplyr::na_if(x, "")
    x
  }
}

# Prepare universe events
heck_events <- heck_events %>%
  mutate(
    ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
    event_date = if ("event_date" %in% names(.)) parse_date_utc(event_date) else as.Date(NA)
  )

universe_events <- heck_events %>%
  filter(!is.na(ric), !is.na(event_date)) %>%
  distinct(ric, event_date, .keep_all = FALSE)

# Prepare sample events
sample_events <- markets_sample %>%
  transmute(
    ric = if ("val_ric" %in% names(.)) normalize_ric(val_ric) else NA_character_,
    event_date = val_anchor_date
  ) %>%
  filter(!is.na(ric), !is.na(event_date)) %>%
  distinct()

# --- Plot 16: sample events vs universe events (counts) ---
n_universe <- nrow(universe_events)
n_sample   <- nrow(sample_events)

event_counts <- tibble(
  dataset = c("Sample events", "Universe events"),
  n = c(n_sample, n_universe)
) %>%
  mutate(dataset = factor(dataset, levels = c("Sample events", "Universe events")))

# Save table
out_paths <- write_table_dual(event_counts, "16_event_counts_sample_vs_universe", out_dir)
add_manifest("table", out_paths$csv,   "Counts: sample events vs universe events (CSV).")
add_manifest("table", out_paths$jsonl, "Counts: sample events vs universe events (JSONL).")

# Plot counts (optionally show sample share in subtitle)
share_txt <- if (is.finite(n_universe) && n_universe > 0) {
  scales::percent(n_sample / n_universe, accuracy = 0.1)
} else {
  NA_character_
}

p_events_counts <- ggplot(event_counts, aes(x = dataset, y = n, fill = dataset)) +
  geom_col(color = BORDER_COL) +
  geom_text(aes(label = paste0("n=", n)), vjust = -0.3, size = 4, color = COL_GREY_1) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_fill_manual(values = c("Sample events" = DATA_COL, "Universe events" = COL_GREY_2), guide = "none") +
  labs(
    title = "Number of events: sample vs universe",
    subtitle = if (!is.na(share_txt)) paste0("Sample as share of universe: ", share_txt) else NULL,
    x = NULL,
    y = "Number of events"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_events_counts, "16_event_counts_sample_vs_universe", out_dir, width = 8, height = 5)
add_manifest("plot", plot_path, "Bar chart: number of sample events vs universe events.")


# Optional: inclusion by sector if available in universe events
if ("gics_sector" %in% names(heck_events)) {
  universe_by_sector <- heck_events %>%
    filter(!is.na(ric), !is.na(event_date)) %>%
    distinct(ric, event_date, gics_sector) %>%
    mutate(in_sample = paste(ric, event_date) %in% paste(sample_events$ric, sample_events$event_date)) %>%
    group_by(gics_sector) %>%
    summarise(
      n_universe = n(),
      n_in_sample = sum(in_sample),
      share_in_sample = n_in_sample / n_universe,
      .groups = "drop"
    ) %>%
    arrange(desc(n_universe))
  
  out_paths <- write_table_dual(universe_by_sector, "16b_event_inclusion_by_sector", out_dir)
  add_manifest("table", out_paths$csv,   "Event inclusion share by GICS sector (CSV).")
  add_manifest("table", out_paths$jsonl, "Event inclusion share by GICS sector (JSONL).")
  
  p_inc_sector <- ggplot(universe_by_sector, aes(x = reorder(gics_sector, share_in_sample), y = share_in_sample)) +
    geom_col(fill = DATA_COL, alpha = 0.85) +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title = "Share of universe events included in sample, by GICS sector",
      x = NULL,
      y = "Share of events included"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_inc_sector, "16b_event_inclusion_by_sector", out_dir, width = 10, height = 7)
  add_manifest("plot", plot_path, "Bar chart: share of universe events included in sample by GICS sector.")
}

# =============================================================================
# 17) Summary table for each time snapshot: min, p25, mean, median, p75, max
# =============================================================================
cat("17) Snapshot-level numeric summaries...\n")

# Build a numeric-only summary per snapshot for the joined snapshot dataset.
# (Includes snapshot-specific variables like price_yes, complement_error, etc.)
numeric_cols <- prices_sample %>%
  select(where(is.numeric)) %>%
  names()

# Ensure snapshot_label is kept
snapshot_numeric <- prices_sample %>%
  select(snapshot_label, all_of(numeric_cols))

snapshot_summary_all <- snapshot_numeric %>%
  pivot_longer(cols = -snapshot_label, names_to = "variable", values_to = "value") %>%
  group_by(snapshot_label, variable) %>%
  summarise(
    n = sum(is.finite(value)),
    min = safe_min(value),
    p25 = safe_quantile(value, 0.25),
    mean = safe_mean(value),
    median = safe_median(value),
    p75 = safe_quantile(value, 0.75),
    max = safe_max(value),
    .groups = "drop"
  ) %>%
  arrange(snapshot_label, variable)

out_paths <- write_table_dual(snapshot_summary_all, "17_snapshot_numeric_summary_all", out_dir)
add_manifest("table", out_paths$csv,   "Combined snapshot summary table for all numeric variables (CSV).")
add_manifest("table", out_paths$jsonl, "Combined snapshot summary table for all numeric variables (JSONL).")

# Write separate table per snapshot_label (explicitly requested)
for (sl in snapshot_list) {
  safe_label <- gsub("[^A-Za-z0-9_-]", "_", sl)
  df_sl <- snapshot_summary_all %>% filter(as.character(snapshot_label) == sl)
  
  out_paths <- write_table_dual(df_sl, glue("17_snapshot_numeric_summary_{safe_label}"), out_dir)
  add_manifest("table", out_paths$csv,   glue("Snapshot numeric summary (CSV) for snapshot {sl}."))
  add_manifest("table", out_paths$jsonl, glue("Snapshot numeric summary (JSONL) for snapshot {sl}."))
}

# =============================================================================
# 18) Extra helpful descriptives for a scientific article (recommended)
# =============================================================================
cat("18) Additional helpful descriptives...\n")

# 18a) Distribution of implied probabilities by snapshot (faceted histogram)
prob_df <- prices_sample %>%
  filter(!is.na(price_yes), price_yes >= 0, price_yes <= 1)

p_prob_facet <- ggplot(prob_df, aes(x = price_yes)) +
  geom_histogram(bins = 25, fill = DATA_COL, color = BORDER_COL) +
  facet_wrap(~ snapshot_label) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  labs(
    title = "Distribution of implied probabilities (price_yes) by snapshot",
    x = "Implied probability (price_yes)",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_prob_facet, "18a_implied_probability_distribution_by_snapshot", out_dir, width = 12, height = 8)
add_manifest("plot", plot_path, "Faceted histograms: implied probabilities by snapshot_label.")

# 18b) Complement error distribution (quality check)
p_comp_err <- ggplot(prices_sample, aes(x = complement_error)) +
  geom_histogram(bins = 30, fill = DATA_COL, alpha = 0.75, color = BORDER_COL) +
  labs(
    title = "Complement error distribution (|price_yes + price_no - 1|)",
    x = "Complement error",
    y = "Count of market-snapshot observations"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_comp_err, "18b_complement_error_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: complement error (price consistency) for snapshot observations.")

# 18c) Optional: event-day stock return distribution (if feasible)
stock_prices <- stock_prices %>%
  mutate(
    market_id = if ("market_id" %in% names(.)) as.character(market_id) else NA_character_,
    offset_td = if ("offset_td" %in% names(.)) suppressWarnings(as.integer(offset_td)) else NA_integer_,
    close = if ("close" %in% names(.)) suppressWarnings(as.numeric(close)) else NA_real_
  )

if (all(c("market_id", "offset_td", "close") %in% names(stock_prices))) {
  event_ret <- stock_prices %>%
    filter(offset_td %in% c(-1, 0)) %>%
    select(market_id, offset_td, close) %>%
    mutate(offset_label = paste0("td_", if_else(offset_td < 0, paste0("m", abs(offset_td)), as.character(offset_td)))) %>%
    select(-offset_td) %>%
    pivot_wider(names_from = offset_label, values_from = close) %>%
    mutate(event_day_return = (td_0 / td_m1) - 1) %>%
    filter(is.finite(event_day_return))
  
  if (nrow(event_ret) > 0) {
    event_ret_summary <- event_ret %>%
      summarise(
        n = n(),
        min = safe_min(event_day_return),
        p25 = safe_quantile(event_day_return, 0.25),
        mean = safe_mean(event_day_return),
        median = safe_median(event_day_return),
        p75 = safe_quantile(event_day_return, 0.75),
        max = safe_max(event_day_return)
      )
    
    out_paths <- write_table_dual(event_ret_summary, "18c_event_day_return_summary", out_dir)
    add_manifest("table", out_paths$csv,   "Event-day return summary (close[0]/close[-1]-1) if available (CSV).")
    add_manifest("table", out_paths$jsonl, "Event-day return summary if available (JSONL).")
    
    p_ret <- ggplot(event_ret, aes(x = event_day_return)) +
      geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
      scale_x_continuous(labels = percent_format(accuracy = 0.1)) +
      labs(
        title = "Distribution of event-day stock return (if available)",
        x = "Event-day return (close[0]/close[-1] - 1)",
        y = "Count of markets"
      ) +
      theme_corporate()
    
    plot_path <- save_plot_png(p_ret, "18c_event_day_return_distribution", out_dir, width = 10, height = 6)
    add_manifest("plot", plot_path, "Histogram: event-day stock return distribution (if data available).")
  }
} else {
  cat("  (Info) stock_prices_daily.csv missing required columns for event-day return. Skipping.\n")
}

# =============================================================================
# Write manifest + README instructions
# =============================================================================
cat("Writing manifest and README...\n")

manifest <- manifest %>%
  arrange(match(type, c("table", "plot")), file)

out_paths <- write_table_dual(manifest, "00_output_manifest", out_dir)
add_manifest("table", out_paths$csv,   "Manifest listing all outputs (CSV).")
add_manifest("table", out_paths$jsonl, "Manifest listing all outputs (JSONL).")

readme_path <- file.path(out_dir, "logs", "README.md")

readme_lines <- c(
  "# Descriptive Statistics Output",
  "",
  glue("- Run timestamp: **{run_ts}**"),
  glue("- Generated at: **{Sys.time()}**"),
  glue("- Script: `Corporate_Earnings/R/scripts/descriptive_stats.R`"),
  glue("- Output directory: `Corporate_Earnings/statistics/descriptive_statistics/`"),
  "",
  "## Inputs",
  "",
  "The script reads the following input files (relative to project root):",
  "",
  paste0("- `", fs::path_rel(paths$markets, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$brier, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$poly_prices, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$stock_prices, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$corporate, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$heck_comp, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$heck_events, start = root_dir), "`"),
  "",
  "## Key definitions / filters used",
  "",
  "- **Active trading hours** = `abs(difftime(umaEndDate, startDate, units='hours'))`.",
  "- **Valid (non-stale) snapshot prices** require:",
  "  - `price_yes` and `price_no` not missing",
  "  - `src_yes_ts` and `src_no_ts` not missing",
  "  - `abs(price_yes + price_no - 1) <= complement_tolerance` (default 0.05 if tolerance missing)",
  "",
  "- **Sample markets** are restricted to:",
  "  - Resolved outcome in {YES, NO}",
  "  - Non-missing `val_ric` and `val_anchor_date` (matched to corporate events)",
  "  - If `val_status` exists: `val_status` starts with `MATCHED`",
  "",
  "## Output files",
  "",
  "Tables are written as both **CSV** and **JSONL**. Plots are **PNG**.",
  "",
  "See `00_output_manifest.csv` for a complete list of outputs and descriptions.",
  "",
  "## Notes on calibration plots",
  "",
  "Calibration plots show **observed YES rate** vs **implied probability** (Polymarket `price_yes`).",
  "The dashed 45-degree line represents perfect calibration (e.g., p=0.5 should resolve YES 50% of the time)."
)

writeLines(readme_lines, con = readme_path)
add_manifest("doc", readme_path, "README with instructions and key definitions.")

# Also write a short run summary table
run_summary <- tibble(
  run_ts = run_ts,
  n_markets_all = nrow(markets),
  n_markets_sample = nrow(markets_sample),
  n_sample_firms = n_distinct(markets_sample$val_ric),
  n_snapshot_rows_valid = nrow(prices_sample),
  n_markets_with_valid_prices = n_distinct(prices_sample$market_id),
  output_dir = fs::path_rel(out_dir, start = root_dir)
)

out_paths <- write_table_dual(run_summary, "00_run_summary", out_dir)
add_manifest("table", out_paths$csv,   "Run summary (counts and locations) (CSV).")
add_manifest("table", out_paths$jsonl, "Run summary (counts and locations) (JSONL).")

cat("\n==================== RUN COMPLETE ====================\n")
cat(glue("Run log saved to: {log_path}\n"))
cat(glue("README saved to:  {readme_path}\n"))
cat(glue("Outputs saved in: {out_dir}\n"))
cat("======================================================\n\n")
