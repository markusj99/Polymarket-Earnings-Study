#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/00_descriptive_statistics.R
# Purpose: Create descriptive statistics tables + plots for the Polymarket
#          Corporate Earnings Study (using ONLY the unified long dataset).
#
# Inputs (relative to project root):
#   - data/complete_dataset_long.csv
#   - data/stock_prices/stock_prices_daily.csv
#   - data/heckman_selection_model/heckman_universe_events.csv
#
# Loader:
#   - R/utils/load_data.R  (load_project_data())
#
# Outputs (relative to project root):
#   - statistics/descriptive_statistics/
#       - tables/*.csv + *.jsonl
#       - plots/*.png
#       - logs/*.log.txt + README.md
#
# Notes:
#   - Uses only relative paths (project root inferred from renv.lock or .Rproj).
#   - “Stale” Polymarket probabilities are excluded by requiring non-missing
#     p_polymarket_yes and a valid snapshot timestamp (if present).
#   - Snapshot summaries are produced per horizon (e.g., 4w, 3w, ...).
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
REQUIRED_PKGS <- c(
  "tidyverse","lubridate","janitor","scales","jsonlite","glue","fs"
)

missing <- REQUIRED_PKGS[!vapply(REQUIRED_PKGS, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
if (length(missing)) {
  stop(
    "Missing packages even after renv::restore(): ",
    paste(missing, collapse = ", "),
    "\nRun: renv::restore() or update renv.lock via renv::snapshot().",
    call. = FALSE
  )
}
invisible(lapply(REQUIRED_PKGS, library, character.only = TRUE))

# ------------------------------ Color palette --------------------------------
# Required project scheme:
COL_GREY_1    <- "#808080"
COL_GREY_2    <- "#A9A9A9"
COL_RED       <- "#E3170A"
COL_DARKBLUE  <- "#00008B"
COL_BLUE      <- "#0000FF"

DATA_COL   <- COL_BLUE
BORDER_COL <- COL_GREY_2

theme_corporate <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      axis.title = ggplot2::element_text(face = "bold"),
      legend.position = "bottom"
    )
}

# ------------------------------- IO helpers ----------------------------------
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
safe_min <- function(x) { x <- x[is.finite(x)]; if (!length(x)) NA_real_ else min(x) }
safe_max <- function(x) { x <- x[is.finite(x)]; if (!length(x)) NA_real_ else max(x) }
safe_mean <- function(x) { x <- x[is.finite(x)]; if (!length(x)) NA_real_ else mean(x) }
safe_median <- function(x) { x <- x[is.finite(x)]; if (!length(x)) NA_real_ else median(x) }
wilson_ci <- function(k, n, conf = 0.95) {
  if (is.na(k) || is.na(n) || n <= 0) return(c(NA_real_, NA_real_))
  z <- stats::qnorm(1 - (1 - conf) / 2)
  phat <- k / n
  denom <- 1 + (z^2) / n
  center <- (phat + (z^2) / (2 * n)) / denom
  half <- (z * sqrt((phat * (1 - phat) + (z^2) / (4 * n)) / n)) / denom
  c(max(0, center - half), min(1, center + half))
}


parse_ts_utc <- function(x) {
  if (inherits(x, "POSIXct")) return(lubridate::with_tz(x, tzone = "UTC"))
  if (is.numeric(x)) return(lubridate::as_datetime(x, tz = "UTC"))
  x_chr <- as.character(x)
  suppressWarnings(
    lubridate::parse_date_time(
      x_chr,
      orders = c("ymdHMSOSz","ymdHMSz","ymdHMSOS","ymdHMS","ymdTz","ymd"),
      tz = "UTC",
      exact = FALSE
    )
  )
}

parse_date_utc <- function(x) {
  suppressWarnings(lubridate::ymd(as.character(x), tz = "UTC"))
}

normalize_ric <- function(x) {
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_to_upper(x)
  dplyr::na_if(x, "")
}

# ------------------------------- Run logging ---------------------------------
run_ts <- format(Sys.time(), "%Y%m%dT%H%M%S")
log_path <- file.path(out_dir, "logs", paste0("descriptive_stats_run_", run_ts, ".log.txt"))
sink(log_path, split = TRUE)
on.exit(sink(), add = TRUE)

cat(glue::glue("Descriptive statistics run started: {Sys.time()}\n"))
cat(glue::glue("Inferred project root: {root_dir}\n"))
cat(glue::glue("Output directory: {out_dir}\n\n"))

# ------------------------------ Read input data (central loader) -------------
source(file.path(root_dir, "R", "utils", "load_data.R"))

cat("Reading input files via load_project_data()...\n")
D <- load_project_data(root_dir)

# Paths used for README
paths <- list(
  dataset_long  = file.path(root_dir, "data", "complete_dataset_long.csv"),
  stock_prices  = file.path(root_dir, "data", "stock_prices", "stock_prices_daily.csv"),
  heck_events   = file.path(root_dir, "data", "heckman_selection_model", "heckman_universe_events.csv")
)

dataset_long_raw <- D$dataset_long
stock_prices_raw <- D$stock_prices
heck_events_raw  <- D$heckman_universe_events

cat("Cleaning column names...\n")
dataset_long <- janitor::clean_names(dataset_long_raw)
stock_prices <- janitor::clean_names(stock_prices_raw)
heck_events  <- janitor::clean_names(heck_events_raw)

# ------------------------------ Prepare dataset_long --------------------------
cat("Preparing unified long dataset...\n")

dataset_long <- dataset_long %>%
  mutate(
    market_id = as.character(.data$id),
    ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
    slug   = if ("slug" %in% names(.)) as.character(slug) else NA_character_,
    
    resolved_outcome_std = dplyr::case_when(
      "resolved_outcome" %in% names(.) & stringr::str_to_upper(resolved_outcome) %in% c("YES","Y") ~ "YES",
      "resolved_outcome" %in% names(.) & stringr::str_to_upper(resolved_outcome) %in% c("NO","N")  ~ "NO",
      TRUE ~ NA_character_
    ),
    
    uma_end_date_utc = if ("uma_end_date" %in% names(.)) parse_ts_utc(uma_end_date) else as.POSIXct(NA),
    earnings_release_datetime_utc = if ("earnings_release_datetime" %in% names(.)) parse_ts_utc(earnings_release_datetime) else as.POSIXct(NA, tz = "UTC"),
    accepting_orders_ts_utc = if ("accepting_orders_timestamp" %in% names(.)) parse_ts_utc(accepting_orders_timestamp) else as.POSIXct(NA),
    snapshot_dt_utc = if ("snapshot_dt_utc" %in% names(.)) parse_ts_utc(snapshot_dt_utc) else as.POSIXct(NA),
    
    # Best available proxy for "active trading hours" given the reduced inputs:
    # acceptingOrdersTimestamp -> umaEndDate
    active_trading_hours = as.numeric(difftime(uma_end_date_utc, accepting_orders_ts_utc, units = "hours")),
    active_trading_hours = dplyr::if_else(is.finite(active_trading_hours), abs(active_trading_hours), NA_real_),
    
    # Snapshot label is now "horizon" from the unified dataset
    snapshot_label = if ("horizon" %in% names(.)) as.character(horizon) else NA_character_,
    snapshot_offset_seconds = if ("horizon_seconds" %in% names(.)) suppressWarnings(as.numeric(horizon_seconds)) else NA_real_,
    
    seconds_before_close = if ("seconds_before_close" %in% names(.)) suppressWarnings(as.numeric(seconds_before_close)) else NA_real_,
    
    # Key probability used for calibration
    price_yes = if ("p_polymarket_yes" %in% names(.)) suppressWarnings(as.numeric(p_polymarket_yes)) else NA_real_,
    
    # Other probabilities (if present)
    p_hist = if ("p_hist_asof_end_minus_1d" %in% names(.)) suppressWarnings(as.numeric(p_hist_asof_end_minus_1d)) else NA_real_,
    p_dice = if ("p_dice_0p5" %in% names(.)) suppressWarnings(as.numeric(p_dice_0p5)) else NA_real_,
    
    # Losses (if present)
    loss_polymarket = if ("loss_polymarket" %in% names(.)) suppressWarnings(as.numeric(loss_polymarket)) else NA_real_,
    loss_hist       = if ("loss_hist" %in% names(.)) suppressWarnings(as.numeric(loss_hist)) else NA_real_,
    loss_dice       = if ("loss_dice" %in% names(.)) suppressWarnings(as.numeric(loss_dice)) else NA_real_,
    
    # Market/fundamental controls (if present)
    volume_num = if ("volume_num" %in% names(.)) suppressWarnings(as.numeric(volume_num)) else NA_real_,
    val_surprise = if ("val_surprise" %in% names(.)) suppressWarnings(as.numeric(val_surprise)) else NA_real_,
    val_eikon_eps_stddev_estimate = if ("val_eikon_eps_stddev_estimate" %in% names(.)) suppressWarnings(as.numeric(val_eikon_eps_stddev_estimate)) else NA_real_,
    
    ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
    asof_date = if ("asof_date" %in% names(.)) parse_date_utc(asof_date) else as.Date(NA),
    
    market_cap_usd = if ("market_cap_usd_asof" %in% names(.)) suppressWarnings(as.numeric(market_cap_usd_asof)) else NA_real_,
    analysts_covering = if ("analysts_covering_asof" %in% names(.)) suppressWarnings(as.numeric(analysts_covering_asof)) else NA_real_,
    gics_sector = if ("gics_sector" %in% names(.)) as.character(gics_sector) else NA_character_,
    
    turnover_6m_sum_volume = if ("turnover_6m_sum_volume" %in% names(.)) suppressWarnings(as.numeric(turnover_6m_sum_volume)) else NA_real_,
    turnover_6m_avg_daily_volume = if ("turnover_6m_avg_daily_volume" %in% names(.)) suppressWarnings(as.numeric(turnover_6m_avg_daily_volume)) else NA_real_,
    volatility_6m = if ("volatility_6m" %in% names(.)) suppressWarnings(as.numeric(volatility_6m)) else NA_real_,
    
    outcome_yes = as.integer(resolved_outcome_std == "YES")
  )

# Snapshot ordering by offset seconds (largest offset first, e.g., 4w then 3w ...)
snapshot_levels <- dataset_long %>%
  distinct(snapshot_label, snapshot_offset_seconds) %>%
  arrange(desc(snapshot_offset_seconds), snapshot_label) %>%
  pull(snapshot_label) %>%
  unique()

# Valid (non-stale) definition under reduced inputs:
# - resolved outcome is YES/NO
# - price_yes exists and is within [0,1]
# - if snapshot_dt_utc exists in file, require it (prevents "stale"/unanchored)
prices_sample <- dataset_long %>%
  filter(resolved_outcome_std %in% c("YES", "NO")) %>%
  filter(!is.na(price_yes), is.finite(price_yes), price_yes >= 0, price_yes <= 1) %>%
  {
    if ("snapshot_dt_utc" %in% names(.) && any(!is.na(.$snapshot_dt_utc))) {
      filter(., !is.na(snapshot_dt_utc))
    } else .
  } %>%
  mutate(snapshot_label = factor(snapshot_label, levels = snapshot_levels))

# Exclude long-horizon snapshots
prices_sample <- prices_sample %>%
  dplyr::filter(!(as.character(snapshot_label) %in% c("4w", "3w", "2w"))) %>%
  dplyr::mutate(snapshot_label = droplevels(snapshot_label))

# Market-level sample (one row per market_id)
markets_sample <- prices_sample %>%
  group_by(market_id) %>%
  summarise(
    ticker = dplyr::first(ticker),
    slug   = dplyr::first(slug),
    ric    = dplyr::first(ric),
    asof_date = dplyr::first(asof_date),
    resolved_outcome_std = dplyr::first(resolved_outcome_std),
    uma_end_date_utc = dplyr::first(uma_end_date_utc),
    earnings_release_datetime_utc = dplyr::first(earnings_release_datetime_utc),
    accepting_orders_ts_utc = dplyr::first(accepting_orders_ts_utc),
    active_trading_hours = dplyr::first(active_trading_hours),
    volume_num = dplyr::first(volume_num),
    val_surprise = dplyr::first(val_surprise),
    val_eikon_eps_stddev_estimate = dplyr::first(val_eikon_eps_stddev_estimate),
    market_cap_usd = dplyr::first(market_cap_usd),
    analysts_covering = dplyr::first(analysts_covering),
    gics_sector = dplyr::first(gics_sector),
    turnover_6m_sum_volume = dplyr::first(turnover_6m_sum_volume),
    turnover_6m_avg_daily_volume = dplyr::first(turnover_6m_avg_daily_volume),
    volatility_6m = dplyr::first(volatility_6m),
    .groups = "drop"
  )

# Firm-level sample (one row per RIC; keep first non-missing where possible)
sample_firms <- markets_sample %>%
  filter(!is.na(ric)) %>%
  group_by(ric) %>%
  summarise(
    gics_sector = dplyr::first(na.omit(gics_sector)),
    market_cap_usd = dplyr::first(na.omit(market_cap_usd)),
    analysts_covering = dplyr::first(na.omit(analysts_covering)),
    .groups = "drop"
  )

cat(glue::glue("Rows (prices_sample): {nrow(prices_sample)}\n"))
cat(glue::glue("Distinct markets (sample): {nrow(markets_sample)}\n"))
cat(glue::glue("Distinct firms (sample): {n_distinct(markets_sample$ric)}\n\n"))

# ----------------------------- Output manifest --------------------------------
manifest <- tibble::tibble(type = character(), file = character(), description = character())
add_manifest <- function(type, path, description) {
  manifest <<- dplyr::bind_rows(
    manifest,
    tibble::tibble(type = type, file = basename(path), description = description)
  )
}

# =============================================================================
# 1) Number of observations per snapshot
# =============================================================================
cat("1) Observations per snapshot...\n")

obs_per_snapshot <- prices_sample %>%
  dplyr::group_by(snapshot_label) %>%
  dplyr::summarise(
    n_observations = dplyr::n(),
    n_markets = dplyr::n_distinct(market_id),
    .groups = "drop"
  ) %>%
  dplyr::arrange(snapshot_label)

out_paths <- write_table_dual(obs_per_snapshot, "01_obs_per_snapshot", out_dir)
add_manifest("table", out_paths$csv,   "Number of observations and distinct markets per snapshot_label (CSV).")
add_manifest("table", out_paths$jsonl, "Number of observations and distinct markets per snapshot_label (JSONL).")

p_obs <- ggplot2::ggplot(obs_per_snapshot, ggplot2::aes(x = snapshot_label, y = n_observations)) +
  ggplot2::geom_col(fill = DATA_COL, color = BORDER_COL) +
  ggplot2::labs(
    title = "Number of observations per time snapshot",
    x = "Snapshot label (horizon)",
    y = "Number of market-snapshot observations"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_obs, "01_obs_per_snapshot", out_dir, width = 9, height = 5)
add_manifest("plot", plot_path, "Bar chart of observations per snapshot_label.")

total_sample_markets <- nrow(markets_sample)
availability_per_snapshot <- obs_per_snapshot %>%
  dplyr::mutate(
    n_total_sample_markets = total_sample_markets,
    share_markets_with_price = n_markets / n_total_sample_markets
  )

out_paths <- write_table_dual(availability_per_snapshot, "01b_price_availability_per_snapshot", out_dir)
add_manifest("table", out_paths$csv,   "Share of sample markets with prices per snapshot (CSV).")
add_manifest("table", out_paths$jsonl, "Share of sample markets with prices per snapshot (JSONL).")

p_avail <- ggplot2::ggplot(availability_per_snapshot, ggplot2::aes(x = snapshot_label, y = share_markets_with_price)) +
  ggplot2::geom_col(fill = DATA_COL, color = BORDER_COL) +
  ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  ggplot2::labs(
    title = "Price availability by time snapshot",
    x = "Snapshot label (horizon)",
    y = "Share of sample markets with snapshot probability"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_avail, "01b_price_availability_per_snapshot", out_dir, width = 9, height = 5)
add_manifest("plot", plot_path, "Bar chart: share of markets with probabilities per snapshot_label.")

# =============================================================================
# 2) Active trading hours (acceptingOrdersTimestamp -> umaEndDate)
# =============================================================================
cat("2) Active trading hours...\n")

active_hours_summary <- markets_sample %>%
  dplyr::transmute(active_trading_hours) %>%
  dplyr::summarise(
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

p_hours <- ggplot2::ggplot(markets_sample, ggplot2::aes(x = active_trading_hours)) +
  ggplot2::geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::labs(
    title = "Distribution of active trading hours (UMA end - accepting orders)",
    x = "Active trading hours",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_hours, "02_active_trading_hours_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: active_trading_hours across markets.")

# =============================================================================
# 2b) Time between earnings release and UMA end
# =============================================================================
cat("2b) Time between earnings release and UMA end...\n")

release_gap_df <- markets_sample %>%
  dplyr::mutate(
    earnings_to_uma_end_hours = as.numeric(
      difftime(uma_end_date_utc, earnings_release_datetime_utc, units = "hours")
    ),
    earnings_to_uma_end_days = earnings_to_uma_end_hours / 24
  ) %>%
  dplyr::filter(
    !is.na(earnings_release_datetime_utc),
    !is.na(uma_end_date_utc),
    is.finite(earnings_to_uma_end_hours)
  )

release_gap_summary <- release_gap_df %>%
  dplyr::summarise(
    n = dplyr::n(),
    min_hours = safe_min(earnings_to_uma_end_hours),
    p25_hours = safe_quantile(earnings_to_uma_end_hours, 0.25),
    mean_hours = safe_mean(earnings_to_uma_end_hours),
    median_hours = safe_median(earnings_to_uma_end_hours),
    p75_hours = safe_quantile(earnings_to_uma_end_hours, 0.75),
    max_hours = safe_max(earnings_to_uma_end_hours),
    min_days = safe_min(earnings_to_uma_end_days),
    p25_days = safe_quantile(earnings_to_uma_end_days, 0.25),
    mean_days = safe_mean(earnings_to_uma_end_days),
    median_days = safe_median(earnings_to_uma_end_days),
    p75_days = safe_quantile(earnings_to_uma_end_days, 0.75),
    max_days = safe_max(earnings_to_uma_end_days),
    n_negative_hours = sum(earnings_to_uma_end_hours < 0, na.rm = TRUE)
  )

out_paths <- write_table_dual(release_gap_summary, "02b_earnings_release_to_uma_end_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for time between earnings_release_datetime and uma_end_date_utc (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for time between earnings_release_datetime and uma_end_date_utc (JSONL).")

x99 <- as.numeric(stats::quantile(release_gap_df$earnings_to_uma_end_hours, probs = 0.99, na.rm = TRUE))
x01 <- as.numeric(stats::quantile(release_gap_df$earnings_to_uma_end_hours, probs = 0.01, na.rm = TRUE))

release_gap_zoom_df <- release_gap_df %>%
  dplyr::filter(
    earnings_to_uma_end_hours >= x01,
    earnings_to_uma_end_hours <= x99
  )

x_breaks <- seq(
  from = floor(min(release_gap_zoom_df$earnings_to_uma_end_hours, na.rm = TRUE) / 5) * 5,
  to   = ceiling(max(release_gap_zoom_df$earnings_to_uma_end_hours, na.rm = TRUE) / 5) * 5,
  by   = 5
)

p_release_gap_zoom <- ggplot2::ggplot(release_gap_zoom_df, ggplot2::aes(x = earnings_to_uma_end_hours)) +
  ggplot2::geom_histogram(bins = 30, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::geom_vline(xintercept = 0, color = COL_RED, linewidth = 0.7, linetype = "dashed") +
  ggplot2::scale_x_continuous(breaks = x_breaks) +
  ggplot2::labs(
    title = "Distribution of time between earnings release and UMA end",
    subtitle = "Histogram shown for the 1st-99th percentile range; positive values mean release before market close",
    x = "Hours from earnings release to UMA end",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_release_gap_zoom, "02b_earnings_release_to_uma_end_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: time between earnings_release_datetime and uma_end_date_utc in hours, zoomed to the 1st-99th percentile range.")

# =============================================================================
# 3) YES vs NO resolved markets over time (diverging stacked bars)
# =============================================================================
cat("3) Resolved YES vs NO over time...\n")

resolved_counts_by_date <- markets_sample %>%
  dplyr::filter(!is.na(uma_end_date_utc)) %>%
  dplyr::mutate(uma_end_date = as.Date(uma_end_date_utc)) %>%
  dplyr::count(uma_end_date, resolved_outcome_std, name = "n_markets") %>%
  dplyr::mutate(n_signed = dplyr::if_else(resolved_outcome_std == "NO", -n_markets, n_markets)) %>%
  dplyr::arrange(uma_end_date, resolved_outcome_std)

out_paths <- write_table_dual(resolved_counts_by_date, "03_resolved_counts_by_uma_end_date", out_dir)
add_manifest("table", out_paths$csv,   "Counts of resolved markets per UMA end date (NO shown negative in plot; CSV).")
add_manifest("table", out_paths$jsonl, "Counts of resolved markets per UMA end date (JSONL).")

p_resolved_time <- ggplot2::ggplot(resolved_counts_by_date, ggplot2::aes(x = uma_end_date, y = n_signed, fill = resolved_outcome_std)) +
  ggplot2::geom_col() +
  ggplot2::geom_hline(yintercept = 0, color = BORDER_COL, linewidth = 0.3) +
  ggplot2::scale_fill_manual(
    values = c("YES" = COL_BLUE, "NO" = COL_RED),
    name = "Resolved outcome"
  ) +
  ggplot2::scale_x_date(date_labels = "%Y-%m-%d") +
  ggplot2::labs(
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
  dplyr::count(resolved_outcome_std, name = "n_markets") %>%
  dplyr::mutate(share = n_markets / sum(n_markets)) %>%
  dplyr::arrange(desc(n_markets))

out_paths <- write_table_dual(resolved_share, "04_resolved_share_yes_no", out_dir)
add_manifest("table", out_paths$csv,   "Overall share of resolved YES vs NO (CSV).")
add_manifest("table", out_paths$jsonl, "Overall share of resolved YES vs NO (JSONL).")

p_share <- ggplot2::ggplot(resolved_share, ggplot2::aes(x = resolved_outcome_std, y = share, fill = resolved_outcome_std)) +
  ggplot2::geom_col(color = BORDER_COL) +
  ggplot2::geom_text(ggplot2::aes(label = scales::percent(share, accuracy = 0.1)), vjust = -0.4, size = 4, color = COL_GREY_1) +
  ggplot2::scale_fill_manual(values = c("YES" = COL_BLUE, "NO" = COL_RED), guide = "none") +
  ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, max(resolved_share$share, na.rm = TRUE) * 1.15)) +
  ggplot2::labs(
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
  dplyr::filter(!is.na(price_yes), price_yes >= 0, price_yes <= 1, !is.na(outcome_yes)) %>%
  dplyr::mutate(prob_bin = cut(price_yes, breaks = seq(0, 1, by = 0.2), include.lowest = TRUE, right = TRUE)) %>%
  dplyr::group_by(snapshot_label, prob_bin) %>%
  dplyr::summarise(
    n = dplyr::n(),
    mean_prob = mean(price_yes, na.rm = TRUE),
    observed_yes_rate = mean(outcome_yes, na.rm = TRUE),
    observed_yes_n = sum(outcome_yes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    ci_low_95  = mapply(function(k, n) wilson_ci(k, n)[1], observed_yes_n, n),
    ci_high_95 = mapply(function(k, n) wilson_ci(k, n)[2], observed_yes_n, n)
  ) %>%
  dplyr::arrange(snapshot_label, prob_bin)


out_paths <- write_table_dual(calibration_bins, "05_calibration_bins_by_snapshot", out_dir)
add_manifest("table", out_paths$csv,   "Calibration bins: mean implied prob vs observed YES rate (and counts) by snapshot (CSV).")
add_manifest("table", out_paths$jsonl, "Calibration bins: mean implied prob vs observed YES rate (and counts) by snapshot (JSONL).")

snapshot_list <- sort(unique(as.character(prices_sample$snapshot_label)))

p_cal_facet <- ggplot2::ggplot(calibration_bins, ggplot2::aes(x = mean_prob, y = observed_yes_rate, group = 1)) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = COL_GREY_1) +
  ggplot2::geom_errorbar(
    ggplot2::aes(ymin = ci_low_95, ymax = ci_high_95),
    width = 0.02,
    color = COL_GREY_2,
    linewidth = 0.5
  ) +
  ggplot2::geom_line(color = DATA_COL, linewidth = 0.9) +
  ggplot2::geom_point(color = DATA_COL, size = 2.5) +
  ggplot2::geom_text(ggplot2::aes(label = n), vjust = -0.8, color = COL_GREY_1, size = 3) +
  ggplot2::facet_wrap(~ snapshot_label) +
  ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2), labels = scales::number_format(accuracy = 0.1)) +
  ggplot2::scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2), labels = scales::number_format(accuracy = 0.1)) +
  ggplot2::labs(
    title = "Calibration (binned): observed YES rate vs implied probability, by snapshot",
    x = "Mean implied probability (p_polymarket_yes) within bin",
    y = "Observed YES rate"
  ) +
  theme_corporate()


plot_path <- save_plot_png(p_cal_facet, "05_calibration_all_snapshots_faceted", out_dir, width = 12, height = 8)
add_manifest("plot", plot_path, "Faceted calibration plot: mean implied prob vs observed YES rate for all snapshots.")

# =============================================================================
# 6) Volume distribution (log scale but readable in USD)
# =============================================================================
cat("6) Volume distribution...\n")

volume_df <- markets_sample %>% dplyr::filter(!is.na(volume_num), is.finite(volume_num), volume_num > 0)

volume_summary <- volume_df %>%
  dplyr::summarise(
    n = dplyr::n(),
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

p_vol <- ggplot2::ggplot(volume_df, ggplot2::aes(x = volume_num)) +
  ggplot2::geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::scale_x_log10(labels = scales::dollar_format(accuracy = 1)) +
  ggplot2::labs(
    title = "Distribution of Polymarket market volume (log scale, labeled in USD)",
    x = "Volume (USD, log10 scale)",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_vol, "06_volume_distribution_log_usd", out_dir, width = 11, height = 6)
add_manifest("plot", plot_path, "Histogram: volume distribution with log10 x-axis labeled in USD.")

# =============================================================================
# 7) Analyst forecast dispersion proxy: Eikon EPS stddev estimate
# =============================================================================
cat("7) Eikon EPS stddev estimate distribution...\n")

eps_sd_df <- markets_sample %>%
  dplyr::filter(!is.na(val_eikon_eps_stddev_estimate), is.finite(val_eikon_eps_stddev_estimate), val_eikon_eps_stddev_estimate >= 0)

eps_sd_summary <- eps_sd_df %>%
  dplyr::summarise(
    n = dplyr::n(),
    min = safe_min(val_eikon_eps_stddev_estimate),
    p25 = safe_quantile(val_eikon_eps_stddev_estimate, 0.25),
    mean = safe_mean(val_eikon_eps_stddev_estimate),
    median = safe_median(val_eikon_eps_stddev_estimate),
    p75 = safe_quantile(val_eikon_eps_stddev_estimate, 0.75),
    max = safe_max(val_eikon_eps_stddev_estimate)
  )

out_paths <- write_table_dual(eps_sd_summary, "07_eikon_eps_stddev_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for Eikon EPS stddev estimate (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for Eikon EPS stddev estimate (JSONL).")

p_eps_sd <- ggplot2::ggplot(eps_sd_df, ggplot2::aes(x = val_eikon_eps_stddev_estimate)) +
  ggplot2::geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::labs(
    title = "Distribution of Eikon EPS estimate dispersion (stddev)",
    x = "Eikon EPS stddev estimate",
    y = "Count of markets"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_eps_sd, "07_eikon_eps_stddev_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: Eikon EPS stddev estimate distribution.")

# =============================================================================
# 8) Distribution of surprise
# =============================================================================
cat("8) Surprise distribution...\n")

surprise_df <- markets_sample %>% dplyr::filter(!is.na(val_surprise), is.finite(val_surprise))

surprise_summary <- surprise_df %>%
  dplyr::summarise(
    n = dplyr::n(),
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

p_surprise <- ggplot2::ggplot(surprise_df, ggplot2::aes(x = val_surprise)) +
  ggplot2::geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::labs(
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
  dplyr::filter(!is.na(active_trading_hours), is.finite(active_trading_hours),
                !is.na(volume_num), is.finite(volume_num), volume_num > 0)

p_hours_vol <- ggplot2::ggplot(hours_vs_volume_df, ggplot2::aes(x = volume_num, y = active_trading_hours)) +
  ggplot2::geom_point(alpha = 0.6, color = DATA_COL) +
  ggplot2::geom_smooth(method = "loess", se = FALSE, linewidth = 0.8, color = COL_RED) +
  ggplot2::scale_x_log10(labels = scales::dollar_format(accuracy = 1)) +
  ggplot2::labs(
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
  dplyr::mutate(ric = normalize_ric(ric)) %>%
  dplyr::filter(!is.na(ric)) %>%
  dplyr::group_by(ric) %>%
  dplyr::summarise(n_markets = dplyr::n_distinct(market_id), .groups = "drop") %>%
  dplyr::arrange(desc(n_markets))

firm_multi_summary <- firm_counts %>%
  dplyr::summarise(
    n_firms_total = dplyr::n(),
    n_firms_gt1 = sum(n_markets > 1),
    share_firms_gt1 = n_firms_gt1 / n_firms_total,
    max_markets_per_firm = max(n_markets, na.rm = TRUE)
  )

firm_count_dist <- firm_counts %>%
  dplyr::count(n_markets, name = "n_firms") %>%
  dplyr::arrange(n_markets)

out_paths <- write_table_dual(firm_counts, "10_firm_market_counts", out_dir)
add_manifest("table", out_paths$csv,   "Count of markets per firm (RIC) in sample (CSV).")
add_manifest("table", out_paths$jsonl, "Count of markets per firm (RIC) in sample (JSONL).")

out_paths <- write_table_dual(firm_multi_summary, "10b_firms_observed_more_than_once_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary: how many firms appear >1 time in sample (CSV).")
add_manifest("table", out_paths$jsonl, "Summary: how many firms appear >1 time in sample (JSONL).")

out_paths <- write_table_dual(firm_count_dist, "10c_markets_per_firm_distribution_table", out_dir)
add_manifest("table", out_paths$csv,   "Distribution table: number of firms by markets-per-firm (CSV).")
add_manifest("table", out_paths$jsonl, "Distribution table: number of firms by markets-per-firm (JSONL).")

p_firm_counts <- ggplot2::ggplot(firm_count_dist, ggplot2::aes(x = factor(n_markets), y = n_firms)) +
  ggplot2::geom_col(fill = DATA_COL, color = BORDER_COL) +
  ggplot2::labs(
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

mcap_df <- sample_firms %>% dplyr::filter(!is.na(market_cap_usd), is.finite(market_cap_usd), market_cap_usd > 0)

mcap_summary <- mcap_df %>%
  dplyr::summarise(
    n = dplyr::n(),
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

p_mcap <- ggplot2::ggplot(mcap_df, ggplot2::aes(x = market_cap_usd)) +
  ggplot2::geom_histogram(bins = 40, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::scale_x_log10(labels = scales::dollar_format(accuracy = 1)) +
  ggplot2::labs(
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
  dplyr::filter(!is.na(volume_num), is.finite(volume_num), volume_num > 0,
                !is.na(market_cap_usd), is.finite(market_cap_usd), market_cap_usd > 0)

p_vol_mcap <- ggplot2::ggplot(vol_mcap_df, ggplot2::aes(x = market_cap_usd, y = volume_num)) +
  ggplot2::geom_point(alpha = 0.6, color = DATA_COL) +
  ggplot2::geom_smooth(method = "loess", se = FALSE, linewidth = 0.8, color = COL_RED) +
  ggplot2::scale_x_log10(labels = scales::dollar_format(accuracy = 1)) +
  ggplot2::scale_y_log10(labels = scales::dollar_format(accuracy = 1)) +
  ggplot2::labs(
    title = "Polymarket market volume vs firm market cap",
    x = "Market cap (USD, log10 scale)",
    y = "Polymarket volume (USD, log10 scale)"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_vol_mcap, "12_volume_vs_market_cap", out_dir, width = 11, height = 6)
add_manifest("plot", plot_path, "Scatter + LOESS: Polymarket volume vs market cap (log-log).")

# =============================================================================
# 13) Optional: generic distributions if columns exist
# =============================================================================
cat("13) Optional distributions (only if columns exist)...\n")

make_dist_table <- function(df, col_name, top_n = 25) {
  if (!col_name %in% names(df)) return(tibble::tibble())
  df %>%
    dplyr::mutate(val = as.character(.data[[col_name]])) %>%
    dplyr::mutate(val = dplyr::if_else(is.na(val) | !nzchar(val), "Unknown", val)) %>%
    dplyr::count(val, name = "n") %>%
    dplyr::mutate(share = n / sum(n)) %>%
    dplyr::arrange(desc(n)) %>%
    { if (nrow(.) > top_n) {
      top <- dplyr::slice_head(., n = top_n)
      other <- tibble::tibble(val = "Other", n = sum(.$n[-(1:top_n)]), share = sum(.$share[-(1:top_n)]))
      dplyr::bind_rows(top, other)
    } else . }
}

# GICS sector distribution (this should exist)
gics_sector_dist <- make_dist_table(sample_firms, "gics_sector", top_n = 50)
if (nrow(gics_sector_dist) > 0) {
  out_paths <- write_table_dual(gics_sector_dist, "13_gics_sector_distribution", out_dir)
  add_manifest("table", out_paths$csv,   "Distribution of GICS sector for sample firms (CSV).")
  add_manifest("table", out_paths$jsonl, "Distribution of GICS sector for sample firms (JSONL).")
  
  p_sector <- ggplot2::ggplot(gics_sector_dist, ggplot2::aes(x = reorder(val, n), y = n)) +
    ggplot2::geom_col(fill = DATA_COL, color = BORDER_COL) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Distribution of GICS sector (sample firms)",
      x = NULL,
      y = "Number of firms"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_sector, "13_gics_sector_distribution", out_dir, width = 10, height = 7)
  add_manifest("plot", plot_path, "Bar chart: GICS sector distribution.")
}

# =============================================================================
# 14) Distribution of analysts covering the firm
# =============================================================================
cat("14) Analysts covering distribution...\n")

analyst_df <- sample_firms %>%
  dplyr::filter(!is.na(analysts_covering), is.finite(analysts_covering), analysts_covering >= 0)

analyst_summary <- analyst_df %>%
  dplyr::summarise(
    n = dplyr::n(),
    min = safe_min(analysts_covering),
    p25 = safe_quantile(analysts_covering, 0.25),
    mean = safe_mean(analysts_covering),
    median = safe_median(analysts_covering),
    p75 = safe_quantile(analysts_covering, 0.75),
    max = safe_max(analysts_covering)
  )

out_paths <- write_table_dual(analyst_summary, "14_analysts_covering_summary", out_dir)
add_manifest("table", out_paths$csv,   "Summary stats for analysts covering (CSV).")
add_manifest("table", out_paths$jsonl, "Summary stats for analysts covering (JSONL).")

p_analysts <- ggplot2::ggplot(analyst_df, ggplot2::aes(x = analysts_covering)) +
  ggplot2::geom_histogram(bins = 30, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::labs(
    title = "Distribution of analyst coverage (sample firms)",
    x = "Number of analysts covering the firm",
    y = "Count of firms"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_analysts, "14_analysts_covering_distribution", out_dir, width = 10, height = 6)
add_manifest("plot", plot_path, "Histogram: analysts covering distribution (sample firms).")

# =============================================================================
# 15) Share of events included in our sample (vs Heckman universe)
# =============================================================================
cat("15) Event inclusion share (sample vs universe)...\n")

heck_events <- heck_events %>%
  dplyr::mutate(
    ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
    event_date = if ("event_date" %in% names(.)) parse_date_utc(event_date) else as.Date(NA)
  )

universe_events <- heck_events %>%
  dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
  dplyr::distinct(ric, event_date, .keep_all = FALSE)

# Sample events:
# Prefer asof_date if present, else use UMA end date as fallback.
sample_events <- markets_sample %>%
  dplyr::transmute(
    ric = normalize_ric(ric),
    event_date = dplyr::if_else(!is.na(asof_date), asof_date, as.Date(uma_end_date_utc))
  ) %>%
  dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
  dplyr::distinct()

n_universe <- nrow(universe_events)
n_sample   <- nrow(sample_events)

event_counts <- tibble::tibble(
  dataset = c("Sample events", "Universe events"),
  n = c(n_sample, n_universe)
) %>%
  dplyr::mutate(dataset = factor(dataset, levels = c("Sample events", "Universe events")))

out_paths <- write_table_dual(event_counts, "15_event_counts_sample_vs_universe", out_dir)
add_manifest("table", out_paths$csv,   "Counts: sample events vs universe events (CSV).")
add_manifest("table", out_paths$jsonl, "Counts: sample events vs universe events (JSONL).")

share_txt <- if (is.finite(n_universe) && n_universe > 0) {
  scales::percent(n_sample / n_universe, accuracy = 0.1)
} else {
  NA_character_
}

p_events_counts <- ggplot2::ggplot(event_counts, ggplot2::aes(x = dataset, y = n, fill = dataset)) +
  ggplot2::geom_col(color = BORDER_COL) +
  ggplot2::geom_text(ggplot2::aes(label = paste0("n=", n)), vjust = -0.3, size = 4, color = COL_GREY_1) +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.1))) +
  ggplot2::scale_fill_manual(values = c("Sample events" = DATA_COL, "Universe events" = COL_GREY_2), guide = "none") +
  ggplot2::labs(
    title = "Number of events: sample vs universe",
    subtitle = if (!is.na(share_txt)) paste0("Sample as share of universe: ", share_txt) else NULL,
    x = NULL,
    y = "Number of events"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_events_counts, "15_event_counts_sample_vs_universe", out_dir, width = 8, height = 5)
add_manifest("plot", plot_path, "Bar chart: number of sample events vs universe events.")

# =============================================================================
# 16) Summary table for each time snapshot: min, p25, mean, median, p75, max
# =============================================================================
cat("16) Snapshot-level numeric summaries...\n")

numeric_cols <- prices_sample %>%
  dplyr::select(where(is.numeric)) %>%
  names()

snapshot_numeric <- prices_sample %>%
  dplyr::select(snapshot_label, dplyr::all_of(numeric_cols))

snapshot_summary_all <- snapshot_numeric %>%
  tidyr::pivot_longer(cols = -snapshot_label, names_to = "variable", values_to = "value") %>%
  dplyr::group_by(snapshot_label, variable) %>%
  dplyr::summarise(
    n = sum(is.finite(value)),
    min = safe_min(value),
    p25 = safe_quantile(value, 0.25),
    mean = safe_mean(value),
    median = safe_median(value),
    p75 = safe_quantile(value, 0.75),
    max = safe_max(value),
    .groups = "drop"
  ) %>%
  dplyr::arrange(snapshot_label, variable)

out_paths <- write_table_dual(snapshot_summary_all, "16_snapshot_numeric_summary_all", out_dir)
add_manifest("table", out_paths$csv,   "Combined snapshot summary table for all numeric variables (CSV).")
add_manifest("table", out_paths$jsonl, "Combined snapshot summary table for all numeric variables (JSONL).")

for (sl in snapshot_list) {
  safe_label <- gsub("[^A-Za-z0-9_-]", "_", sl)
  df_sl <- snapshot_summary_all %>% dplyr::filter(as.character(snapshot_label) == sl)
  
  out_paths <- write_table_dual(df_sl, glue::glue("16_snapshot_numeric_summary_{safe_label}"), out_dir)
  add_manifest("table", out_paths$csv,   glue::glue("Snapshot numeric summary (CSV) for snapshot {sl}."))
  add_manifest("table", out_paths$jsonl, glue::glue("Snapshot numeric summary (JSONL) for snapshot {sl}."))
}

# =============================================================================
# 17) Extra helpful descriptives (recommended)
# =============================================================================
cat("17) Additional helpful descriptives...\n")

# 17a) Distribution of implied probabilities by snapshot (faceted histogram)
prob_df <- prices_sample %>%
  dplyr::filter(!is.na(price_yes), price_yes >= 0, price_yes <= 1)

p_prob_facet <- ggplot2::ggplot(prob_df, ggplot2::aes(x = price_yes)) +
  ggplot2::geom_histogram(bins = 25, fill = DATA_COL, color = BORDER_COL) +
  ggplot2::facet_wrap(~ snapshot_label) +
  ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  ggplot2::labs(
    title = "Distribution of implied probabilities (p_polymarket_yes) by snapshot",
    x = "Implied probability (p_polymarket_yes)",
    y = "Count of market-snapshot observations"
  ) +
  theme_corporate()

plot_path <- save_plot_png(p_prob_facet, "17a_implied_probability_distribution_by_snapshot", out_dir, width = 12, height = 8)
add_manifest("plot", plot_path, "Faceted histograms: implied probabilities by snapshot_label.")

# =============================================================================
# Write manifest + README
# =============================================================================
cat("Writing manifest and README...\n")

manifest <- manifest %>%
  dplyr::arrange(match(type, c("table", "plot", "doc")), file)

out_paths <- write_table_dual(manifest, "00_output_manifest", out_dir)
add_manifest("table", out_paths$csv,   "Manifest listing all outputs (CSV).")
add_manifest("table", out_paths$jsonl, "Manifest listing all outputs (JSONL).")

readme_path <- file.path(out_dir, "logs", "README.md")

readme_lines <- c(
  "# Descriptive Statistics Output",
  "",
  glue::glue("- Run timestamp: **{run_ts}**"),
  glue::glue("- Generated at: **{Sys.time()}**"),
  "- Script: `R/00_descriptive_statistics.R`",
  "- Output directory: `statistics/descriptive_statistics/`",
  "",
  "## Inputs",
  "",
  "The script reads the following input files (relative to project root):",
  "",
  paste0("- `", fs::path_rel(paths$dataset_long, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$stock_prices, start = root_dir), "`"),
  paste0("- `", fs::path_rel(paths$heck_events, start = root_dir), "`"),
  "",
  "## Key definitions / filters used",
  "",
  "- **Active trading hours** = `abs(difftime(umaEndDate, acceptingOrdersTimestamp, units='hours'))` (best available proxy given reduced inputs).",
  "- **Valid snapshot probabilities** require:",
  "  - `p_polymarket_yes` present and in [0,1]",
  "  - if `snapshot_dt_utc` exists in the file: it must be non-missing",
  "- **Hours from earnings release to UMA end** = `difftime(umaEndDate, earnings_release_datetime, units='hours')`.",
  "",
  "- **Sample markets** are restricted to resolved outcome in {YES, NO}.",
  "",
  "## Output files",
  "",
  "Tables are written as both **CSV** and **JSONL**. Plots are **PNG**.",
  "",
  "See `00_output_manifest.csv` for a complete list of outputs and descriptions.",
  "",
  "## Notes on calibration plots",
  "",
  "Calibration plots show **observed YES rate** vs **implied probability** (Polymarket `p_polymarket_yes`).",
  "The dashed 45-degree line represents perfect calibration."
)

writeLines(readme_lines, con = readme_path)
add_manifest("doc", readme_path, "README with instructions and key definitions.")

run_summary <- tibble::tibble(
  run_ts = run_ts,
  n_markets_sample = nrow(markets_sample),
  n_sample_firms = dplyr::n_distinct(markets_sample$ric),
  n_snapshot_rows = nrow(prices_sample),
  n_markets_with_prices = dplyr::n_distinct(prices_sample$market_id),
  output_dir = fs::path_rel(out_dir, start = root_dir)
)

out_paths <- write_table_dual(run_summary, "00_run_summary", out_dir)
add_manifest("table", out_paths$csv,   "Run summary (counts and locations) (CSV).")
add_manifest("table", out_paths$jsonl, "Run summary (counts and locations) (JSONL).")

cat("\n==================== RUN COMPLETE ====================\n")
cat(glue::glue("Run log saved to: {log_path}\n"))
cat(glue::glue("README saved to:  {readme_path}\n"))
cat(glue::glue("Outputs saved in: {out_dir}\n"))
cat("======================================================\n\n")
