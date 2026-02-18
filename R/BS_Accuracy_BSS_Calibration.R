#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/BS_Accuracy_BSS_Calibration.R
# Author:  (generated/updated by ChatGPT)
# -----------------------------------------------------------------------------
# PURPOSE
#   This script produces scientifically-oriented accuracy statistics for
#   Polymarket earnings markets and links those accuracy measures to:
#     (i)   benchmark forecasters (coinflip and historical base-rate),
#     (ii)  regression-based determinants of Brier loss,
#     (iii) Heckman (two-step) selection correction (universe events),
#     (iv)  an event-study style trading analysis:
#           "What happens if we buy the stock when Polymarket probability is x?"
#           where x is binned into five probability bins of width 0.2.
#
#   The script is designed for reproducibility (paper-compatible output tables):
#     - clean and explicit sample filters,
#     - non-stale Polymarket prices only,
#     - robust / clustered inference where appropriate,
#     - outputs saved as CSV + JSONL + JSON (rows).
#
# -----------------------------------------------------------------------------
# INPUTS (relative to project root "Polymarket-Earnings-Study/")
#   data/markets/markets.csv
#   data/poly_prices/poly_prices_long.csv
#   data/stock_prices/stock_prices_daily.csv
#   data/corporate_info/corporate_info.csv
#   data/heckman_selection_model/heckman_universe_companies.csv
#   data/heckman_selection_model/heckman_universe_events.csv
#
#   (Optional; read if present)
#   data/brier_scores/brier_scores_market_horizon.csv
#
# -----------------------------------------------------------------------------
# OUTPUTS (relative to project root)
#   statistics/test_statistics/
#     tables/ : CSV + JSONL + JSON
#     plots/  : PNG
#     logs/   : run log + README + sessionInfo + manifest
#
# -----------------------------------------------------------------------------
# KEY SAMPLE RULES / SCIENTIFIC CHOICES
#   1) Resolution cutoff:
#      We keep only markets that are resolved (YES/NO) AND have UMA end date
#      (or a fallback resolution timestamp) at least 1 day before "now".
#
#   2) Matched corporate events only:
#      Requires val_ric and val_anchor_date, and (if available) val_status that
#      starts with "MATCHED".
#
#   3) Non-stale Polymarket snapshots only (per your project rule):
#      - price_yes and price_no not missing
#      - src_yes_ts and src_no_ts not missing
#      - |price_yes + price_no - 1| <= complement_tolerance (default 0.05)
#      - for duplicated pulls, we keep the latest generated_utc per
#        (market_id, snapshot_label).
#
#   4) Event definition:
#      We score forecasts on the event "BEAT" (earnings beat) whenever possible.
#      If we can infer that YES corresponds to BEAT, we use price_yes and y_yes;
#      if YES corresponds to MISS, we flip the mapping so that p_mkt always means
#      P(BEAT) and y always equals 1{BEAT}.
#
#      This makes the event-study interpretation ("buy when p= x") coherent.
#
# -----------------------------------------------------------------------------
# NOTES
#   - This script is intentionally defensive: it checks for required columns and
#     handles missing optional inputs gracefully.
#   - Your supervisors can audit every transformation because each step is
#     explicit and saved outputs are self-describing.
# =============================================================================

options(stringsAsFactors = FALSE, scipen = 999)

# =============================================================================
# 0) Project root discovery (cross-platform, base-R)
# =============================================================================
find_project_root <- function(start = getwd()) {
  dir <- normalizePath(start, winslash = "/", mustWork = FALSE)

  for (i in 1:100) {
    has_lock  <- file.exists(file.path(dir, "renv.lock"))
    has_rproj <- length(list.files(dir, pattern = "\\.Rproj$", full.names = TRUE)) > 0

    # Fallback heuristic (in case renv.lock/.Rproj is missing)
    has_data  <- dir.exists(file.path(dir, "data"))
    has_stats <- dir.exists(file.path(dir, "statistics"))
    looks_like_project <- (basename(dir) == "Polymarket-Earnings-Study") || (has_data && has_stats)

    if (has_lock || has_rproj || looks_like_project) return(dir)

    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }

  stop(
    "Could not find project root.\n",
    "Searched upward from: ", start, "\n",
    "Expected to find renv.lock, an .Rproj file, or (data/ and statistics/ folders).",
    call. = FALSE
  )
}

get_script_path <- function() {
  # Works when run via: Rscript path/to/script.R  (uses --file=...)
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
  }

  # Works in RStudio if file is open and rstudioapi is available
  if (interactive() &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) {
      return(normalizePath(p, winslash = "/", mustWork = FALSE))
    }
  }

  NA_character_
}

script_path <- get_script_path()
start_dir   <- if (!is.na(script_path)) dirname(script_path) else getwd()
ROOT        <- find_project_root(start_dir)
root_dir    <- ROOT

# =============================================================================
# 0b) renv activation (best-effort)
# =============================================================================
has_renv_lock <- file.exists(file.path(root_dir, "renv.lock"))
if (has_renv_lock) {
  if (!requireNamespace("renv", quietly = TRUE)) {
    install.packages("renv", repos = "https://cloud.r-project.org")
  }
  # Load/restore inside tryCatch so the script fails with clear messages later
  # if packages are missing, rather than failing abruptly here.
  tryCatch({
    renv::load(project = root_dir)
    # Restore ensures missing packages are installed. If offline, this may fail.
    renv::restore(project = root_dir, prompt = FALSE)
  }, error = function(e) {
    message("WARNING: renv::restore() failed. You may need to run it manually.\n",
            "Error: ", e$message)
  })
}

# =============================================================================
# 0c) Packages
# =============================================================================
REQUIRED_PKGS <- c(
  "tidyverse", "lubridate", "janitor", "scales", "jsonlite", "glue", "fs",
  "broom", "sandwich", "lmtest"
)

missing <- REQUIRED_PKGS[!vapply(REQUIRED_PKGS, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
if (length(missing)) {
  stop(
    "Missing packages: ", paste(missing, collapse = ", "), "\n",
    "If you use renv, run: renv::restore(). Otherwise, install.packages().",
    call. = FALSE
  )
}
invisible(lapply(REQUIRED_PKGS, library, character.only = TRUE))

# =============================================================================
# 0d) Color palette (project requirement)
# =============================================================================
COL_GREY_1   <- "#808080"
COL_GREY_2   <- "#A9A9A9"
COL_RED      <- "#E3170A"
COL_DARKBLUE <- "#00008B"
COL_BLUE     <- "#0000FF"

theme_corporate <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      axis.title = ggplot2::element_text(face = "bold"),
      legend.position = "bottom"
    )
}

# =============================================================================
# 0e) IO helpers
# =============================================================================
read_csv_required <- function(path) {
  if (!file.exists(path)) stop(glue::glue("Input file not found: {path}"), call. = FALSE)
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

read_csv_optional <- function(path) {
  if (!file.exists(path)) {
    message(glue::glue("NOTE: Optional input missing (skipping): {path}"))
    return(tibble::tibble())
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

write_table_triple <- function(df, stem, out_dir) {
  table_dir <- file.path(out_dir, "tables")
  fs::dir_create(table_dir)

  csv_path   <- file.path(table_dir, paste0(stem, ".csv"))
  jsonl_path <- file.path(table_dir, paste0(stem, ".jsonl"))
  json_path  <- file.path(table_dir, paste0(stem, ".json"))

  # CSV
  readr::write_csv(df, csv_path, na = "")

  # JSONL (streaming rows)
  con <- file(jsonl_path, open = "wt")
  on.exit(close(con), add = TRUE)
  jsonlite::stream_out(df, con = con, verbose = FALSE)

  # JSON (array of rows) - often easier for downstream tooling
  jsonlite::write_json(
    x = df,
    path = json_path,
    dataframe = "rows",
    auto_unbox = TRUE,
    pretty = TRUE,
    na = "null"
  )

  list(csv = csv_path, jsonl = jsonl_path, json = json_path)
}

save_plot_png <- function(p, stem, out_dir, width = 10, height = 6, dpi = 300) {
  plot_dir <- file.path(out_dir, "plots")
  fs::dir_create(plot_dir)
  png_path <- file.path(plot_dir, paste0(stem, ".png"))
  ggplot2::ggsave(filename = png_path, plot = p, width = width, height = height, dpi = dpi)
  png_path
}

# =============================================================================
# 0f) Parsing + safety helpers
# =============================================================================
parse_ts_utc <- function(x) {
  # Parse timestamps into POSIXct (UTC). Handles:
  # - POSIXct already
  # - numeric epoch seconds
  # - strings like "2025-06-03T21:34:19.809928Z" or "2026-02-02 15:55:56Z"
  #
  # We defensively normalize common ISO-8601 variants:
  #   - Replace "T" with space
  #   - Replace trailing "Z" with "+00:00"
  if (inherits(x, "POSIXct")) return(lubridate::with_tz(x, tzone = "UTC"))
  if (is.numeric(x)) return(lubridate::as_datetime(x, tz = "UTC"))

  x_chr <- as.character(x)
  x_chr <- stringr::str_trim(x_chr)
  x_chr <- dplyr::na_if(x_chr, "")

  # normalize ISO-8601
  x_chr <- stringr::str_replace_all(x_chr, "T", " ")
  x_chr <- stringr::str_replace(x_chr, "Z$", "+00:00")

  suppressWarnings(
    lubridate::parse_date_time(
      x_chr,
      orders = c(
        "ymd HMSOSz", "ymd HMSz", "ymd HMSOS", "ymd HMS",
        "ymdHMSOSz", "ymdHMSz", "ymdHMSOS", "ymdHMS",
        "ymd"
      ),
      tz = "UTC",
      exact = FALSE
    )
  )
}

parse_date_utc <- function(x) {
  # Always return Date
  if (inherits(x, "Date")) return(x)
  if (inherits(x, "POSIXct")) return(as.Date(lubridate::with_tz(x, "UTC")))
  x_chr <- as.character(x)
  suppressWarnings(lubridate::ymd(x_chr))
}

normalize_ric <- function(x) {
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_to_upper(x)
  dplyr::na_if(x, "")
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

safe_log <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  log(pmax(x, 1))
}

# Infer whether YES corresponds to BEAT. This allows us to define p_mkt := P(BEAT)
# and y := 1{BEAT} consistently even if some questions are "Will the company miss?"
yes_means_beat <- function(val_yes_semantics = NA_character_, question = NA_character_) {
  s <- stringr::str_to_upper(as.character(val_yes_semantics))
  q <- stringr::str_to_upper(as.character(question))

  # If an explicit semantics field exists, prefer it.
  if (!is.na(s) && nzchar(s)) {
    if (stringr::str_detect(s, "BEAT") && !stringr::str_detect(s, "MISS")) return(TRUE)
    if (stringr::str_detect(s, "MISS") && !stringr::str_detect(s, "BEAT")) return(FALSE)
  }

  # Fallback: infer from the question text.
  if (!is.na(q) && nzchar(q)) {
    if (stringr::str_detect(q, "\\bBEAT\\b")) return(TRUE)
    if (stringr::str_detect(q, "\\bMISS\\b")) return(FALSE)
  }

  # Conservative fallback: assume YES corresponds to BEAT (common in your dataset).
  TRUE
}

prob_bin_20pct <- function(p) {
  cut(
    p,
    breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
    include.lowest = TRUE,
    right = TRUE,
    labels = c("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
  )
}


# Factor safety helper:
# Some regressions include factor fixed effects (e.g., gics_sector, year, quarter).
# R will error if a factor included in the model has <2 observed levels:
#   "contrasts can be applied only to factors with 2 or more levels"
# We defensively test factor terms before adding them to formulas.
factor_has_2plus <- function(x) {
  f <- droplevels(as.factor(x))
  nlevels(f) >= 2
}

# Robust/cluster variance helper (defensive)
vcov_cluster_or_hc <- function(model, cluster = NULL, type = "HC1") {
  if (!is.null(cluster)) {
    cl <- as.factor(cluster)
    n_cl <- length(unique(cl[!is.na(cl)]))
    # Use cluster-robust only if we have enough clusters; otherwise HC1.
    if (n_cl >= 30) {
      return(sandwich::vcovCL(model, cluster = cl, type = type))
    }
  }
  sandwich::vcovHC(model, type = type)
}

tidy_coeftest <- function(ct, model_name) {
  # Convert an lmtest::coeftest object into a clean, publication-friendly tibble.
  #
  # coeftest() returns a matrix with (at least) 4 columns in this order:
  #   1) estimate, 2) std. error, 3) test statistic (t or z), 4) p-value
  mat <- as.matrix(ct)
  if (ncol(mat) < 4) stop("coeftest object has unexpected shape.", call. = FALSE)

  out <- tibble::tibble(
    term      = rownames(mat),
    estimate  = as.numeric(mat[, 1]),
    std_error = as.numeric(mat[, 2]),
    statistic = as.numeric(mat[, 3]),
    p_value   = as.numeric(mat[, 4]),
    model     = model_name
  ) %>%
    dplyr::mutate(
      conf_low_95  = estimate - 1.96 * std_error,
      conf_high_95 = estimate + 1.96 * std_error
    )

  out
}

one_sample_mean_test <- function(x) {
  # Simple one-sample t test summary (mean different from 0).
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3) {
    return(tibble::tibble(
      N = n, mean = mean(x), sd = sd(x), se = NA_real_,
      t_stat = NA_real_, p_value = NA_real_,
      conf_low_95 = NA_real_, conf_high_95 = NA_real_
    ))
  }
  m <- mean(x)
  s <- stats::sd(x)
  se <- s / sqrt(n)
  t <- m / se
  p <- 2 * stats::pt(abs(t), df = n - 1, lower.tail = FALSE)
  ci_half <- stats::qt(0.975, df = n - 1) * se
  tibble::tibble(
    N = n,
    mean = m,
    sd = s,
    se = se,
    t_stat = t,
    p_value = p,
    conf_low_95 = m - ci_half,
    conf_high_95 = m + ci_half
  )
}

# =============================================================================
# 0g) Output directories + run logging
# =============================================================================
data_dir <- file.path(root_dir, "data")
out_dir  <- file.path(root_dir, "statistics", "test_statistics")

fs::dir_create(out_dir)
fs::dir_create(file.path(out_dir, "logs"))

run_ts   <- format(Sys.time(), "%Y%m%dT%H%M%S")
log_path <- file.path(out_dir, "logs", paste0("BS_accuracy_run_", run_ts, ".log.txt"))

sink(log_path, split = TRUE)
on.exit(sink(), add = TRUE)

cat(glue::glue("Accuracy/BSS/Calibration/EventStudy run started: {Sys.time()}\n"))
cat(glue::glue("Inferred project root: {root_dir}\n"))
cat(glue::glue("Output directory:      {out_dir}\n\n"))

manifest <- tibble::tibble(
  type = character(),
  file = character(),
  rel_path = character(),
  description = character()
)

record_output <- function(type, path, description) {
  manifest <<- dplyr::bind_rows(
    manifest,
    tibble::tibble(
      type = type,
      file = fs::path_file(path),
      rel_path = fs::path_rel(path, start = root_dir),
      description = description
    )
  )
}

# =============================================================================
# 1) Read input data
# =============================================================================
paths <- list(
  markets      = file.path(data_dir, "markets", "markets.csv"),
  brier        = file.path(data_dir, "brier_scores", "brier_scores_market_horizon.csv"), # optional
  poly_prices  = file.path(data_dir, "poly_prices", "poly_prices_long.csv"),
  stock_prices = file.path(data_dir, "stock_prices", "stock_prices_daily.csv"),
  corporate    = file.path(data_dir, "corporate_info", "corporate_info.csv"),
  heck_comp    = file.path(data_dir, "heckman_selection_model", "heckman_universe_companies.csv"),
  heck_events  = file.path(data_dir, "heckman_selection_model", "heckman_universe_events.csv")
)

cat("Reading input files...\n")
markets_raw      <- read_csv_required(paths$markets)
brier_raw        <- read_csv_optional(paths$brier)
poly_prices_raw  <- read_csv_required(paths$poly_prices)
stock_prices_raw <- read_csv_required(paths$stock_prices)
corporate_raw    <- read_csv_required(paths$corporate)

# Heckman inputs are required for that section, but we read them now and handle
# missingness later (so other outputs can still be produced).
heck_comp_raw   <- read_csv_optional(paths$heck_comp)
heck_events_raw <- read_csv_optional(paths$heck_events)

cat("Cleaning column names...\n")
markets      <- janitor::clean_names(markets_raw)
brier_file   <- janitor::clean_names(brier_raw)
poly_prices  <- janitor::clean_names(poly_prices_raw)
stock_prices <- janitor::clean_names(stock_prices_raw)
corporate    <- janitor::clean_names(corporate_raw)
heck_comp    <- janitor::clean_names(heck_comp_raw)
heck_events  <- janitor::clean_names(heck_events_raw)

# =============================================================================
# 2) Build analysis sample of markets (resolved YES/NO, matched events, cutoff)
# =============================================================================
cat("\n[2] Preparing market-level analysis sample (resolution cutoff + matched events)...\n")

if (!"id" %in% names(markets)) stop("markets.csv must contain column 'id'.", call. = FALSE)

markets <- markets %>%
  dplyr::mutate(
    id = as.character(id),
    ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
    slug   = if ("slug" %in% names(.)) as.character(slug) else NA_character_,
    question = if ("question" %in% names(.)) as.character(question) else NA_character_,

    val_ric = if ("val_ric" %in% names(.)) normalize_ric(val_ric) else NA_character_,
    val_anchor_date = if ("val_anchor_date" %in% names(.)) parse_date_utc(val_anchor_date) else as.Date(NA),

    # timestamps
    uma_end_date_utc = if ("uma_end_date" %in% names(.)) parse_ts_utc(uma_end_date) else as.POSIXct(NA),
    closed_time_utc  = if ("closed_time" %in% names(.)) parse_ts_utc(closed_time) else as.POSIXct(NA),
    updated_at_utc   = if ("updated_at" %in% names(.)) parse_ts_utc(updated_at) else as.POSIXct(NA),
    start_date_utc   = if ("start_date" %in% names(.)) parse_ts_utc(start_date) else as.POSIXct(NA),

    # resolution outcome standardization
    resolved_outcome_std = dplyr::case_when(
      "resolved_outcome" %in% names(.) & stringr::str_to_upper(resolved_outcome) %in% c("YES", "Y") ~ "YES",
      "resolved_outcome" %in% names(.) & stringr::str_to_upper(resolved_outcome) %in% c("NO", "N")  ~ "NO",
      TRUE ~ NA_character_
    ),

    # Polymarket market activity measures
    volume_num    = if ("volume_num" %in% names(.)) safe_numeric(volume_num) else NA_real_,
    liquidity_num = if ("liquidity_num" %in% names(.)) safe_numeric(liquidity_num) else NA_real_,

    log_poly_volume = safe_log(volume_num),
    log_liquidity   = safe_log(liquidity_num),

    # active trading time (hours) - use uma_end_date if available, otherwise closed/updated
    resolution_ts_utc = dplyr::coalesce(uma_end_date_utc, closed_time_utc, updated_at_utc),
    active_trading_hours = as.numeric(difftime(resolution_ts_utc, start_date_utc, units = "hours")),
    active_trading_hours = dplyr::if_else(is.finite(active_trading_hours), abs(active_trading_hours), NA_real_),

    # Extra potential covariates (forecast dispersion proxies)
    val_eikon_eps_stddev_estimate = if ("val_eikon_eps_stddev_estimate" %in% names(.)) safe_numeric(val_eikon_eps_stddev_estimate) else NA_real_,
    val_eikon_eps_high_estimate   = if ("val_eikon_eps_high_estimate" %in% names(.)) safe_numeric(val_eikon_eps_high_estimate) else NA_real_,
    val_eikon_eps_low_estimate    = if ("val_eikon_eps_low_estimate" %in% names(.)) safe_numeric(val_eikon_eps_low_estimate) else NA_real_,
    eps_estimate_range = val_eikon_eps_high_estimate - val_eikon_eps_low_estimate,

    # Semantics (optional)
    val_yes_semantics = if ("val_yes_semantics" %in% names(.)) as.character(val_yes_semantics) else NA_character_
  )

# Resolution cutoff: ensure event was resolved before we run this script.
cutoff_ts <- lubridate::with_tz(Sys.time(), tzone = "UTC") - lubridate::days(1)

markets_sample <- markets %>%
  dplyr::filter(resolved_outcome_std %in% c("YES", "NO")) %>%

  # keep matched corporate events if required fields exist
  { if (all(c("val_ric", "val_anchor_date") %in% names(.))) dplyr::filter(., !is.na(val_ric), !is.na(val_anchor_date)) else . } %>%

  # if val_status exists, keep those starting with MATCHED
  { if ("val_status" %in% names(.)) dplyr::filter(., !is.na(val_status), stringr::str_detect(val_status, "^MATCHED")) else . } %>%

  # apply cutoff on available resolution timestamp
  dplyr::filter(!is.na(resolution_ts_utc), resolution_ts_utc <= cutoff_ts) %>%

  # rename for clarity downstream
  dplyr::rename(market_id = id)

cat(glue::glue("Markets (all):                         {nrow(markets)}\n"))
cat(glue::glue("Markets (sample, resolved + cutoff):    {nrow(markets_sample)}\n"))
cat(glue::glue("Distinct markets in sample:             {dplyr::n_distinct(markets_sample$market_id)}\n"))
cat(glue::glue("Cutoff timestamp (UTC):                 {cutoff_ts}\n\n"))

if (nrow(markets_sample) == 0) {
  stop("No markets remain after sample filters. Check inputs and cutoff.", call. = FALSE)
}

# =============================================================================
# 3) Corporate info (firm covariates)
# =============================================================================
cat("[3] Preparing corporate info covariates...\n")

corporate <- corporate %>%
  dplyr::mutate(
    ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,

    market_cap_usd = if ("market_cap_usd" %in% names(.)) safe_numeric(market_cap_usd) else NA_real_,

    analysts_covering_latest = if ("analysts_covering_latest" %in% names(.)) safe_numeric(analysts_covering_latest) else NA_real_,
    analysts_covering_sample_mean = if ("analysts_covering_sample_mean" %in% names(.)) safe_numeric(analysts_covering_sample_mean) else NA_real_,

    turnover_6m_sum_volume_mean = if ("turnover_6m_sum_volume_mean" %in% names(.)) safe_numeric(turnover_6m_sum_volume_mean) else NA_real_,

    volatility_6m_mean = if ("volatility_6m_mean" %in% names(.)) safe_numeric(volatility_6m_mean) else NA_real_,
    volatility_6m_median = if ("volatility_6m_median" %in% names(.)) safe_numeric(volatility_6m_median) else NA_real_,

    gics_sector = if ("gics_sector" %in% names(.)) as.character(gics_sector) else NA_character_
  ) %>%
  dplyr::mutate(
    analysts_covering = dplyr::coalesce(analysts_covering_sample_mean, analysts_covering_latest),
    volatility_6m     = dplyr::coalesce(volatility_6m_mean, volatility_6m_median),

    log_mcap     = safe_log(market_cap_usd),
    log_turnover = safe_log(turnover_6m_sum_volume_mean)
  ) %>%
  dplyr::select(
    ric, company_name, ticker, gics_sector,
    market_cap_usd, log_mcap,
    analysts_covering, analysts_covering_latest, analysts_covering_sample_mean,
    turnover_6m_sum_volume_mean, log_turnover,
    volatility_6m
  ) %>%
  dplyr::distinct(ric, .keep_all = TRUE)

cat(glue::glue("Corporate rows: {nrow(corporate)}\n\n"))

# =============================================================================
# 4) Polymarket prices: filter non-stale + keep latest per (market_id, snapshot)
# =============================================================================
cat("[4] Preparing Polymarket prices and excluding stale snapshots...\n")

if (!all(c("market_id", "snapshot_label") %in% names(poly_prices))) {
  stop("poly_prices_long.csv must contain at least: market_id, snapshot_label.", call. = FALSE)
}

poly_prices <- poly_prices %>%
  dplyr::mutate(
    market_id = as.character(market_id),
    snapshot_label = as.character(snapshot_label),

    generated_utc = if ("generated_utc" %in% names(.)) parse_ts_utc(generated_utc) else as.POSIXct(NA),
    run_id = if ("run_id" %in% names(.)) as.character(run_id) else NA_character_,

    snapshot_offset_seconds = if ("snapshot_offset_seconds" %in% names(.)) safe_numeric(snapshot_offset_seconds) else NA_real_,
    price_yes = if ("price_yes" %in% names(.)) safe_numeric(price_yes) else NA_real_,
    price_no  = if ("price_no" %in% names(.)) safe_numeric(price_no) else NA_real_,
    src_yes_ts = if ("src_yes_ts" %in% names(.)) safe_numeric(src_yes_ts) else NA_real_,
    src_no_ts  = if ("src_no_ts" %in% names(.)) safe_numeric(src_no_ts) else NA_real_,

    complement_tolerance = if ("complement_tolerance" %in% names(.)) safe_numeric(complement_tolerance) else NA_real_,
    complement_tolerance = dplyr::if_else(is.na(complement_tolerance), 0.05, complement_tolerance),

    complement_error = abs((price_yes + price_no) - 1)
  )

# Filter to non-stale rows (mirrors your descriptive_statistics rule)
prices_valid <- poly_prices %>%
  dplyr::filter(
    !is.na(market_id), !is.na(snapshot_label),
    !is.na(price_yes), !is.na(price_no),
    !is.na(src_yes_ts), !is.na(src_no_ts),
    is.finite(complement_error),
    complement_error <= complement_tolerance
  )

# Keep the latest snapshot pull per market_id x snapshot_label
prices_latest <- prices_valid %>%
  dplyr::arrange(dplyr::desc(generated_utc), dplyr::desc(run_id)) %>%
  dplyr::group_by(market_id, snapshot_label) %>%
  dplyr::slice(1) %>%
  dplyr::ungroup()

cat(glue::glue("Poly price rows (all):              {nrow(poly_prices)}\n"))
cat(glue::glue("Poly price rows (valid non-stale):  {nrow(prices_valid)}\n"))
cat(glue::glue("Poly price rows (latest per cell):  {nrow(prices_latest)}\n\n"))

if (nrow(prices_latest) == 0) {
  stop("No valid (non-stale) Polymarket prices remain after filtering.", call. = FALSE)
}

# Snapshot ordering (largest offset first, if available)
snapshot_levels <- prices_latest %>%
  dplyr::distinct(snapshot_label, snapshot_offset_seconds) %>%
  dplyr::arrange(dplyr::desc(snapshot_offset_seconds), snapshot_label) %>%
  dplyr::pull(snapshot_label) %>%
  unique()

# =============================================================================
# 5) Join: markets_sample + prices_latest + corporate covariates
# =============================================================================
cat("[5] Joining markets to price snapshots + constructing event outcome (BEAT) + Brier loss...\n")

prices_sample <- prices_latest %>%
  dplyr::mutate(snapshot_label = factor(snapshot_label, levels = snapshot_levels)) %>%
  dplyr::inner_join(markets_sample, by = "market_id") %>%
  dplyr::left_join(corporate, by = c("val_ric" = "ric")) %>%
  dplyr::mutate(
    # Outcome on Polymarket (YES=1, NO=0)
    y_yes = dplyr::if_else(resolved_outcome_std == "YES", 1, 0),

    # Semantics: does YES mean BEAT? (so we can define y and p as BEAT)
    yes_is_beat = purrr::pmap_lgl(
      list(val_yes_semantics, question),
      ~ yes_means_beat(..1, ..2)
    ),

    # Define event as "BEAT" consistently
    y = dplyr::if_else(yes_is_beat, y_yes, 1 - y_yes),

    # p_mkt = P(BEAT) from Polymarket snapshot
    p_yes = price_yes,
    p_no  = price_no,
    p_mkt = dplyr::if_else(yes_is_beat, p_yes, p_no),

    # Sanity bounds (some rare rows may be slightly outside [0,1] due to rounding)
    p_mkt = dplyr::if_else(p_mkt < 0, 0, dplyr::if_else(p_mkt > 1, 1, p_mkt)),

    # Brier loss for Polymarket
    loss_mkt = (p_mkt - y)^2,

    # event date helpers
    event_year = lubridate::year(val_anchor_date),
    event_qtr  = lubridate::quarter(val_anchor_date)
  )

cat(glue::glue("Joined rows (market x snapshot):        {nrow(prices_sample)}\n"))
cat(glue::glue("Distinct markets with valid prices:     {dplyr::n_distinct(prices_sample$market_id)}\n\n"))

if (nrow(prices_sample) == 0) {
  stop("Join produced 0 rows. Check that market_id values match between markets.csv and poly_prices_long.csv.", call. = FALSE)
}

# =============================================================================
# 6) Benchmarks (three models): Polymarket vs coinflip vs historical base-rate
# =============================================================================
cat("[6] Computing benchmark probabilities and Brier losses...\n")

# Event-level outcomes (one row per market_id) to compute base-rates without double-counting snapshots.
event_outcomes <- prices_sample %>%
  dplyr::distinct(market_id, val_anchor_date, y) %>%
  dplyr::arrange(val_anchor_date, market_id)

p_bar_in_sample <- mean(event_outcomes$y, na.rm = TRUE)

# Expanding base-rate (no look-ahead across event time):
# For event i, use mean(y) over events 1..i-1, ordered by anchor_date.
event_outcomes <- event_outcomes %>%
  dplyr::mutate(
    p_bar_expanding = dplyr::lag(dplyr::cummean(y)),
    # initialization for the first event: use 0.5 (no look-ahead).
    p_bar_expanding = dplyr::if_else(is.na(p_bar_expanding), 0.5, p_bar_expanding)
  ) %>%
  dplyr::select(market_id, p_bar_expanding)

prices_sample <- prices_sample %>%
  dplyr::left_join(event_outcomes, by = "market_id") %>%
  dplyr::mutate(
    p_05 = 0.5,
    p_bar_in_sample = p_bar_in_sample,

    loss_05            = (p_05 - y)^2,
    loss_bar_in_sample = (p_bar_in_sample - y)^2,
    loss_bar_expanding = (p_bar_expanding - y)^2
  )

cat(glue::glue("Base rate (in-sample mean BEAT): {round(p_bar_in_sample, 4)}\n\n"))

# =============================================================================
# 7) Brier score tables (overall + by snapshot) and BSS
# =============================================================================
cat("[7] Brier score comparisons and Brier Skill Scores...\n")

# IMPORTANT: Brier score is the mean of per-event Brier losses.
# In this script, each row is a market_id x snapshot_label cell (latest pull only),
# so means are consistent within each snapshot.

brier_overall <- prices_sample %>%
  dplyr::summarise(
    N = dplyr::n(),
    n_markets = dplyr::n_distinct(market_id),
    brier_polymarket      = mean(loss_mkt, na.rm = TRUE),
    brier_coinflip_05     = mean(loss_05, na.rm = TRUE),
    brier_base_expanding  = mean(loss_bar_expanding, na.rm = TRUE),
    # in-sample base-rate is reported as an additional robustness benchmark
    brier_base_in_sample  = mean(loss_bar_in_sample, na.rm = TRUE)
  )

brier_by_snapshot <- prices_sample %>%
  dplyr::group_by(snapshot_label) %>%
  dplyr::summarise(
    N = dplyr::n(),
    n_markets = dplyr::n_distinct(market_id),
    brier_polymarket      = mean(loss_mkt, na.rm = TRUE),
    brier_coinflip_05     = mean(loss_05, na.rm = TRUE),
    brier_base_expanding  = mean(loss_bar_expanding, na.rm = TRUE),
    brier_base_in_sample  = mean(loss_bar_in_sample, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(snapshot_label)

out_paths <- write_table_triple(brier_overall, "01_brier_overall_three_models", out_dir)
record_output("table", out_paths$csv,  "Overall Brier scores: Polymarket vs coinflip vs expanding base-rate (CSV).")
record_output("table", out_paths$jsonl,"Overall Brier scores: Polymarket vs coinflip vs expanding base-rate (JSONL).")
record_output("table", out_paths$json, "Overall Brier scores: Polymarket vs coinflip vs expanding base-rate (JSON).")

out_paths <- write_table_triple(brier_by_snapshot, "02_brier_by_snapshot_three_models", out_dir)
record_output("table", out_paths$csv,  "Brier scores by snapshot_label (CSV).")
record_output("table", out_paths$jsonl,"Brier scores by snapshot_label (JSONL).")
record_output("table", out_paths$json, "Brier scores by snapshot_label (JSON).")

# Brier Skill Score (BSS) vs reference: BSS = 1 - (Brier_model / Brier_reference)
bss_overall <- tibble::tibble(
  benchmark = c("coinflip_0.5", "base_rate_expanding", "base_rate_in_sample"),
  brier_model = brier_overall$brier_polymarket,
  brier_ref   = c(brier_overall$brier_coinflip_05, brier_overall$brier_base_expanding, brier_overall$brier_base_in_sample)
) %>%
  dplyr::mutate(bss = 1 - (brier_model / brier_ref))

bss_by_snapshot <- brier_by_snapshot %>%
  dplyr::transmute(
    snapshot_label, N, n_markets,
    bss_vs_05              = 1 - (brier_polymarket / brier_coinflip_05),
    bss_vs_base_expanding  = 1 - (brier_polymarket / brier_base_expanding),
    bss_vs_base_in_sample  = 1 - (brier_polymarket / brier_base_in_sample)
  )

out_paths <- write_table_triple(bss_overall, "03_bss_overall", out_dir)
record_output("table", out_paths$csv,  "Overall Brier Skill Scores vs benchmarks (CSV).")
record_output("table", out_paths$jsonl,"Overall Brier Skill Scores vs benchmarks (JSONL).")
record_output("table", out_paths$json, "Overall Brier Skill Scores vs benchmarks (JSON).")

out_paths <- write_table_triple(bss_by_snapshot, "04_bss_by_snapshot", out_dir)
record_output("table", out_paths$csv,  "Brier Skill Scores by snapshot_label (CSV).")
record_output("table", out_paths$jsonl,"Brier Skill Scores by snapshot_label (JSONL).")
record_output("table", out_paths$json, "Brier Skill Scores by snapshot_label (JSON).")

# =============================================================================
# 8) Statistical tests: are Brier scores significantly different?
#    We test mean differences in per-event Brier losses (paired by market_id).
# =============================================================================
cat("[8] Statistical tests of Brier loss differences (paired, by snapshot; primary = 1d)...\n")

# Function to compute paired diff tests for a given data frame (already filtered)
brier_diff_tests <- function(df) {
  # Differences in Brier losses (model A minus model B)
  d_pm_vs_cf <- df$loss_mkt - df$loss_05
  d_pm_vs_br <- df$loss_mkt - df$loss_bar_expanding
  d_br_vs_cf <- df$loss_bar_expanding - df$loss_05

  out <- tibble::tibble(
    comparison = c("Polymarket - Coinflip(0.5)", "Polymarket - BaseRate(Expanding)", "BaseRate(Expanding) - Coinflip(0.5)"),
    dplyr::bind_rows(
      one_sample_mean_test(d_pm_vs_cf),
      one_sample_mean_test(d_pm_vs_br),
      one_sample_mean_test(d_br_vs_cf)
    )
  )
  out
}

brier_tests_by_snapshot <- prices_sample %>%
  dplyr::group_by(snapshot_label) %>%
  dplyr::group_modify(~ brier_diff_tests(.x)) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(snapshot_label, comparison)

out_paths <- write_table_triple(brier_tests_by_snapshot, "05_brier_loss_diff_tests_by_snapshot", out_dir)
record_output("table", out_paths$csv,  "Paired t-tests of mean Brier loss differences by snapshot_label (CSV).")
record_output("table", out_paths$jsonl,"Paired t-tests of mean Brier loss differences by snapshot_label (JSONL).")
record_output("table", out_paths$json, "Paired t-tests of mean Brier loss differences by snapshot_label (JSON).")

# Primary horizon for paper-style reporting (1d if available)
if (any(as.character(prices_sample$snapshot_label) == "1d")) {
  brier_tests_1d <- brier_diff_tests(prices_sample %>% dplyr::filter(as.character(snapshot_label) == "1d"))
  out_paths <- write_table_triple(brier_tests_1d, "05b_brier_loss_diff_tests_1d_primary", out_dir)
  record_output("table", out_paths$csv,  "Paired t-tests of mean Brier loss differences (primary horizon: 1d) (CSV).")
  record_output("table", out_paths$jsonl,"Paired t-tests of mean Brier loss differences (primary horizon: 1d) (JSONL).")
  record_output("table", out_paths$json, "Paired t-tests of mean Brier loss differences (primary horizon: 1d) (JSON).")


  # Paper-friendly single-row table: Brier scores (three models) + paired loss-difference tests (1d)
  brier_1d_row <- brier_by_snapshot %>%
    dplyr::filter(as.character(snapshot_label) == "1d") %>%
    dplyr::mutate(snapshot_label = "1d") %>%
    dplyr::select(
      snapshot_label, N, n_markets,
      brier_polymarket, brier_coinflip_05, brier_base_expanding
    )

  tests_1d_wide <- brier_tests_1d %>%
    dplyr::mutate(
      comp_key = dplyr::case_when(
        comparison == "Polymarket - Coinflip(0.5)" ~ "pm_minus_cf",
        comparison == "Polymarket - BaseRate(Expanding)" ~ "pm_minus_br",
        comparison == "BaseRate(Expanding) - Coinflip(0.5)" ~ "br_minus_cf",
        TRUE ~ "other"
      )
    ) %>%
    dplyr::select(
      comp_key,
      N_test = N,
      mean_diff = mean,
      t_stat,
      p_value,
      conf_low_95,
      conf_high_95
    ) %>%
    tidyr::pivot_wider(
      names_from = comp_key,
      values_from = c(N_test, mean_diff, t_stat, p_value, conf_low_95, conf_high_95)
    )

  paper_brier_1d <- dplyr::bind_cols(brier_1d_row, tests_1d_wide)

  out_paths <- write_table_triple(paper_brier_1d, "05c_paper_table_brier_and_tests_1d", out_dir)
  record_output("table", out_paths$csv,  "Paper table: Brier scores + paired loss-difference tests (1d) (CSV).")
  record_output("table", out_paths$jsonl,"Paper table: Brier scores + paired loss-difference tests (1d) (JSONL).")
  record_output("table", out_paths$json, "Paper table: Brier scores + paired loss-difference tests (1d) (JSON).")
} else {
  cat("NOTE: No snapshot_label == '1d' found in valid prices. Skipping primary 1d-only tests.\n")
}

# =============================================================================
# 9) Classification accuracy for Polymarket (predict BEAT if p>=0.5)
# =============================================================================
cat("[9] Computing classification accuracy (Polymarket; predict BEAT if p>=0.5)...\n")

acc_by_snapshot <- prices_sample %>%
  dplyr::mutate(pred_beat = as.integer(p_mkt >= 0.5), correct = (pred_beat == y)) %>%
  dplyr::group_by(snapshot_label) %>%
  dplyr::summarise(
    N = dplyr::n(),
    n_markets = dplyr::n_distinct(market_id),
    accuracy = mean(correct, na.rm = TRUE),
    accuracy_pct = round(100 * accuracy, 1),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(accuracy))

out_paths <- write_table_triple(acc_by_snapshot, "06_accuracy_by_snapshot_polymarket", out_dir)
record_output("table", out_paths$csv,  "Accuracy by snapshot_label (predict BEAT if p>=0.5) (CSV).")
record_output("table", out_paths$jsonl,"Accuracy by snapshot_label (predict BEAT if p>=0.5) (JSONL).")
record_output("table", out_paths$json, "Accuracy by snapshot_label (predict BEAT if p>=0.5) (JSON).")

# =============================================================================
# 10) Calibration (primary horizon = 1d): bins + plot
# =============================================================================
cat("[10] Calibration (1d) bins + plot...\n")

if (any(as.character(prices_sample$snapshot_label) == "1d")) {
  cal_1d <- prices_sample %>%
    dplyr::filter(as.character(snapshot_label) == "1d", is.finite(p_mkt), p_mkt >= 0, p_mkt <= 1) %>%
    dplyr::mutate(p_bin = prob_bin_20pct(p_mkt)) %>%
    dplyr::group_by(p_bin) %>%
    dplyr::summarise(
      N = dplyr::n(),
      n_markets = dplyr::n_distinct(market_id),
      avg_p = mean(p_mkt, na.rm = TRUE),
      realized = mean(y, na.rm = TRUE),
      .groups = "drop"
    )

  out_paths <- write_table_triple(cal_1d, "07_calibration_bins_1d", out_dir)
  record_output("table", out_paths$csv,  "Calibration bins for 1d horizon (CSV).")
  record_output("table", out_paths$jsonl,"Calibration bins for 1d horizon (JSONL).")
  record_output("table", out_paths$json, "Calibration bins for 1d horizon (JSON).")

  p_cal_1d <- ggplot2::ggplot(cal_1d, ggplot2::aes(x = avg_p, y = realized)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = COL_GREY_1) +
    ggplot2::geom_point(ggplot2::aes(size = N), color = COL_DARKBLUE) +
    ggplot2::geom_text(
      ggplot2::aes(label = paste0(p_bin, "\nN=", N)),
      nudge_y = 0.03, show.legend = FALSE, color = COL_GREY_1, size = 3
    ) +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::labs(
      title = "Calibration (1d before earnings): Polymarket P(BEAT) vs realized BEAT frequency",
      x = "Average implied probability in bin",
      y = "Realized BEAT frequency"
    ) +
    theme_corporate()

  plot_path <- save_plot_png(p_cal_1d, "07_calibration_plot_1d", out_dir, width = 9, height = 5)
  record_output("plot", plot_path, "Calibration plot for 1d horizon (PNG).")
} else {
  cat("NOTE: No 1d snapshot available; skipping calibration section.\n")
}

# =============================================================================
# 11) Determinants of Brier loss (primary horizon = 1d)
#     OLS with robust/clustered inference (cluster by firm RIC if available)
# =============================================================================
cat("[11] Determinants of Brier loss (primary horizon = 1d)...\n")

if (any(as.character(prices_sample$snapshot_label) == "1d")) {

  df_1d <- prices_sample %>%
    dplyr::filter(as.character(snapshot_label) == "1d") %>%
    dplyr::mutate(
      # ensure covariates exist even if corporate join failed
      gics_sector = dplyr::coalesce(as.character(gics_sector), "Unknown"),
      analysts_covering = safe_numeric(analysts_covering),
      log_mcap = safe_numeric(log_mcap),
      log_turnover = safe_numeric(log_turnover),
      volatility_6m = safe_numeric(volatility_6m),
      eps_estimate_range = safe_numeric(eps_estimate_range),
      val_eikon_eps_stddev_estimate = safe_numeric(val_eikon_eps_stddev_estimate)
    ) %>%
    # Ensure one row per market_id (defensive; should already hold)
    dplyr::group_by(market_id) %>%
    dplyr::summarise(
      dplyr::across(dplyr::where(is.numeric), ~ mean(.x, na.rm = TRUE)),
      dplyr::across(dplyr::where(~ !is.numeric(.x)), ~ dplyr::first(.x)),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      gics_sector = as.factor(gics_sector),
      event_year = as.factor(as.integer(round(event_year))),
      event_qtr  = as.factor(as.integer(round(event_qtr)))
    )

  # Model dataset: drop missing key columns
  df_1d_model <- df_1d %>%
    tidyr::drop_na(loss_mkt, p_mkt, y, log_mcap, analysts_covering, log_poly_volume, log_liquidity)

  # Add additional covariates if available:
  # - volatility_6m (uncertainty / noisier environment)
  # - eps_estimate_range and/or eps stddev (forecast dispersion)
  rhs <- c("log_mcap", "analysts_covering", "log_poly_volume", "log_liquidity", "active_trading_hours")

  if ("volatility_6m" %in% names(df_1d_model) && any(is.finite(df_1d_model$volatility_6m))) {
    rhs <- c(rhs, "volatility_6m")
  }
  if ("val_eikon_eps_stddev_estimate" %in% names(df_1d_model) && any(is.finite(df_1d_model$val_eikon_eps_stddev_estimate))) {
    rhs <- c(rhs, "val_eikon_eps_stddev_estimate")
  } else if ("eps_estimate_range" %in% names(df_1d_model) && any(is.finite(df_1d_model$eps_estimate_range))) {
    rhs <- c(rhs, "eps_estimate_range")
  }

  # Sector fixed effects (common in finance/event study regressions)
  if ("gics_sector" %in% names(df_1d_model)) {
    if (factor_has_2plus(df_1d_model$gics_sector)) {
      rhs <- c(rhs, "gics_sector")
    } else {
      cat("NOTE: Dropping gics_sector FE in determinants regression (only 0/1 observed level in this sample).\n")
    }
  }

  # Optional time controls (year-quarter) — include only if there is variation.
  if ("event_year" %in% names(df_1d_model)) {
    if (factor_has_2plus(df_1d_model$event_year)) {
      rhs <- c(rhs, "event_year")
    } else {
      cat("NOTE: Dropping event_year FE in determinants regression (only one observed year).\n")
    }
  }
  if ("event_qtr" %in% names(df_1d_model)) {
    if (factor_has_2plus(df_1d_model$event_qtr)) {
      rhs <- c(rhs, "event_qtr")
    } else {
      cat("NOTE: Dropping event_qtr FE in determinants regression (only one observed quarter).\n")
    }
  }

  f_loss <- stats::as.formula(paste("loss_mkt ~", paste(rhs, collapse = " + ")))
  m_loss_1d <- stats::lm(f_loss, data = df_1d_model)

  # Cluster by firm (RIC) if we have enough clusters; else HC1.
  cluster_vec <- if ("val_ric" %in% names(df_1d_model)) df_1d_model$val_ric else NULL
  vc <- vcov_cluster_or_hc(m_loss_1d, cluster = cluster_vec, type = "HC1")
  ct <- lmtest::coeftest(m_loss_1d, vcov. = vc)

  coef_table_1d <- tidy_coeftest(ct, model_name = "loss_1d_robust_or_clustered")

  fit_stats_1d <- tibble::tibble(
    model = "loss_1d_robust_or_clustered",
    N = nrow(df_1d_model),
    n_markets = dplyr::n_distinct(df_1d_model$market_id),
    n_firms = if ("val_ric" %in% names(df_1d_model)) dplyr::n_distinct(df_1d_model$val_ric) else NA_integer_,
    r2 = summary(m_loss_1d)$r.squared,
    adj_r2 = summary(m_loss_1d)$adj.r.squared
  )

  out_paths <- write_table_triple(coef_table_1d, "08_loss_determinants_1d_coefficients", out_dir)
  record_output("table", out_paths$csv,  "Regression: determinants of 1d Brier loss (robust/clustered) coefficients (CSV).")
  record_output("table", out_paths$jsonl,"Regression: determinants of 1d Brier loss (robust/clustered) coefficients (JSONL).")
  record_output("table", out_paths$json, "Regression: determinants of 1d Brier loss (robust/clustered) coefficients (JSON).")

  out_paths <- write_table_triple(fit_stats_1d, "08b_loss_determinants_1d_fitstats", out_dir)
  record_output("table", out_paths$csv,  "Regression: determinants of 1d Brier loss fit stats (CSV).")
  record_output("table", out_paths$jsonl,"Regression: determinants of 1d Brier loss fit stats (JSONL).")
  record_output("table", out_paths$json, "Regression: determinants of 1d Brier loss fit stats (JSON).")

} else {
  cat("NOTE: No 1d snapshot available; skipping determinants regression.\n")
}

# =============================================================================
# 12) Heckman selection model (two-step, universe events)
#     Selection: 1{event appears in Polymarket matched sample and has 1d loss}
# =============================================================================
cat("[12] Heckman two-step selection model...\n")

run_heckman <- (nrow(heck_comp) > 0 && nrow(heck_events) > 0 && any(as.character(prices_sample$snapshot_label) == "1d"))

if (!run_heckman) {
  cat("NOTE: Skipping Heckman section (missing heckman input files OR missing 1d snapshot data).\n")
} else {

  # Universe events
  heck_events2 <- heck_events %>%
    dplyr::mutate(
      ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
      event_date = if ("event_date" %in% names(.)) parse_date_utc(event_date) else as.Date(NA)
    ) %>%
    dplyr::select(-dplyr::any_of("gics_sector")) %>%
    dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
    dplyr::distinct(ric, event_date, .keep_all = TRUE)

  # Universe companies covariates
  heck_comp2 <- heck_comp %>%
    dplyr::mutate(
      ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
      market_cap_usd = if ("market_cap_usd" %in% names(.)) safe_numeric(market_cap_usd) else NA_real_,
      analysts_covering_latest = if ("analysts_covering_latest" %in% names(.)) safe_numeric(analysts_covering_latest) else NA_real_,
      turnover_6m_sum_volume_mean = if ("turnover_6m_sum_volume_mean" %in% names(.)) safe_numeric(turnover_6m_sum_volume_mean) else NA_real_,
      volatility_6m_mean = if ("volatility_6m_mean" %in% names(.)) safe_numeric(volatility_6m_mean) else NA_real_,
      volatility_6m_median = if ("volatility_6m_median" %in% names(.)) safe_numeric(volatility_6m_median) else NA_real_,
      gics_sector = if ("gics_sector" %in% names(.)) as.character(gics_sector) else NA_character_
    ) %>%
    dplyr::mutate(
      log_mcap = safe_log(market_cap_usd),
      log_turnover = safe_log(turnover_6m_sum_volume_mean),
      volatility_6m = dplyr::coalesce(volatility_6m_mean, volatility_6m_median),
      gics_sector = dplyr::coalesce(gics_sector, "Unknown")
    ) %>%
    dplyr::select(ric, log_mcap, analysts_covering_latest, log_turnover, volatility_6m, gics_sector) %>%
    dplyr::distinct(ric, .keep_all = TRUE)

  # Sample events (from markets_sample)
  sample_events <- markets_sample %>%
    dplyr::transmute(
      ric = normalize_ric(val_ric),
      event_date = val_anchor_date
    ) %>%
    dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
    dplyr::distinct()

  # Outcome variable at event level: 1d Brier loss (BEAT) per event
  df_1d_event <- prices_sample %>%
    dplyr::filter(as.character(snapshot_label) == "1d") %>%
    dplyr::transmute(
      market_id,
      ric = normalize_ric(val_ric),
      event_date = val_anchor_date,
      loss_1d = loss_mkt,
      log_poly_volume = log_poly_volume,
      log_liquidity = log_liquidity,
      active_trading_hours = active_trading_hours
    ) %>%
    dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
    # If there are multiple markets per (ric,event_date), average (paper-friendly aggregation).
    dplyr::group_by(ric, event_date) %>%
    dplyr::summarise(
      loss_1d = mean(loss_1d, na.rm = TRUE),
      log_poly_volume = mean(log_poly_volume, na.rm = TRUE),
      log_liquidity   = mean(log_liquidity, na.rm = TRUE),
      active_trading_hours = mean(active_trading_hours, na.rm = TRUE),
      .groups = "drop"
    )

  # Build selection dataset on universe events
  sel_df <- heck_events2 %>%
    dplyr::left_join(heck_comp2, by = "ric") %>%
    dplyr::mutate(
      in_sample = as.integer(paste(ric, event_date) %in% paste(sample_events$ric, sample_events$event_date))
    ) %>%
    dplyr::left_join(df_1d_event, by = c("ric", "event_date")) %>%
    dplyr::mutate(
      # Selected observations: in_sample == 1 AND outcome observed (finite loss)
      selected = as.integer(in_sample == 1 & is.finite(loss_1d)),
      gics_sector = as.factor(dplyr::coalesce(as.character(gics_sector), "Unknown"))
    )

  cat(glue::glue("Universe events:                         {nrow(sel_df)}\n"))
  cat(glue::glue("In-sample events (market matched):       {sum(sel_df$in_sample, na.rm = TRUE)}\n"))
  cat(glue::glue("Selected events (with finite 1d loss):   {sum(sel_df$selected, na.rm = TRUE)}\n\n"))

  # ---------------------------
  # 12a) First-stage probit selection model (two-step Heckman)
  #
  # Selection definition (paper-friendly):
  #   selected = 1 if a universe (ric, event_date) has:
  #     (i)  a matched Polymarket earnings market (in_sample == 1), AND
  #     (ii) an observed, non-stale 1d snapshot so that a finite Brier loss exists.
  #
  # Exclusion restriction candidate:
  #   log_turnover affects whether the event is covered / tradable but is excluded
  #   from the second-stage outcome regression for loss_1d.
  sel_model_df <- sel_df %>%
    tidyr::drop_na(selected, log_mcap, analysts_covering_latest, log_turnover, gics_sector)

  if (nrow(sel_model_df) < 50) {
    cat("WARNING: Very small first-stage sample for Heckman; results may be unstable.\n")
  }

  if (dplyr::n_distinct(sel_model_df$selected) < 2) {
    cat("NOTE: Heckman selection variable has no variation (all 0 or all 1). Skipping Heckman.\n")
  } else {

    # Optionally include volatility_6m if sufficiently observed
    sel_terms <- c("log_mcap", "analysts_covering_latest", "log_turnover", "gics_sector")
    if ("gics_sector" %in% sel_terms) {
      if (!factor_has_2plus(sel_model_df$gics_sector)) {
        sel_terms <- setdiff(sel_terms, "gics_sector")
        cat("NOTE: Dropping gics_sector in Heckman step 1 (only one observed sector level).\n")
      }
    }
    if ("volatility_6m" %in% names(sel_model_df) && sum(is.finite(sel_model_df$volatility_6m)) >= 30) {
      sel_model_df <- sel_model_df %>% tidyr::drop_na(volatility_6m)
      sel_terms <- c(sel_terms, "volatility_6m")
    }

    sel_formula <- stats::as.formula(paste("selected ~", paste(sel_terms, collapse = " + ")))

    m_sel <- stats::glm(
      sel_formula,
      data = sel_model_df,
      family = binomial(link = "probit")
    )

    # Robust (HC1) SE for the probit coefficients
    vc_sel <- sandwich::vcovHC(m_sel, type = "HC1")
    ct_sel <- lmtest::coeftest(m_sel, vcov. = vc_sel)
    sel_coef <- tidy_coeftest(ct_sel, model_name = "heckman_step1_probit_robust")

    out_paths <- write_table_triple(sel_coef, "09_heckman_step1_selection_probit", out_dir)
    record_output("table", out_paths$csv,  "Heckman step 1: probit selection coefficients (robust SE) (CSV).")
    record_output("table", out_paths$jsonl,"Heckman step 1: probit selection coefficients (robust SE) (JSONL).")
    record_output("table", out_paths$json, "Heckman step 1: probit selection coefficients (robust SE) (JSON).")

    # Predicted index and IMR (for selected observations)
    eta <- stats::predict(m_sel, type = "link")
    Phi <- stats::pnorm(eta)
    phi <- stats::dnorm(eta)

    # Avoid division-by-zero in IMR
    Phi_clip <- pmin(pmax(Phi, 1e-8), 1 - 1e-8)
    imr <- phi / Phi_clip

    sel_model_df <- sel_model_df %>%
      dplyr::mutate(
        eta = eta,
        imr = imr
      )

    # Save readable text summary for appendix
    sel_txt <- file.path(out_dir, "tables", "09b_heckman_step1_summary.txt")
    sink(sel_txt)
    print(summary(m_sel))
    sink()
    record_output("doc", sel_txt, "Heckman step 1 probit summary (TXT).")

    # -------------------------------------------------------------------------
    # 12b) Second-stage outcome regression with IMR (OLS on selected subsample)
    # -------------------------------------------------------------------------
    outcome_df <- sel_model_df %>%
      dplyr::filter(selected == 1, is.finite(loss_1d)) %>%
      tidyr::drop_na(log_mcap, analysts_covering_latest, log_poly_volume, log_liquidity, imr) %>%
      dplyr::mutate(gics_sector = as.factor(gics_sector))

    if (nrow(outcome_df) < 30) {
      cat("WARNING: Very small second-stage sample for Heckman; results may be unstable.\n")
    }

    out_terms <- c("log_mcap", "analysts_covering_latest", "log_poly_volume", "log_liquidity",
                   "active_trading_hours", "gics_sector", "imr")
    if ("gics_sector" %in% out_terms) {
      if (!factor_has_2plus(outcome_df$gics_sector)) {
        out_terms <- setdiff(out_terms, "gics_sector")
        cat("NOTE: Dropping gics_sector in Heckman step 2 (only one observed sector level).\n")
      }
    }

    if ("volatility_6m" %in% names(outcome_df) && sum(is.finite(outcome_df$volatility_6m)) >= 20) {
      outcome_df <- outcome_df %>% tidyr::drop_na(volatility_6m)
      out_terms <- c(out_terms, "volatility_6m")
    }

    out_formula <- stats::as.formula(paste("loss_1d ~", paste(out_terms, collapse = " + ")))

    m_out <- stats::lm(out_formula, data = outcome_df)

    # Robust SE (HC1)
    vc2 <- sandwich::vcovHC(m_out, type = "HC1")
    ct2 <- lmtest::coeftest(m_out, vcov. = vc2)
    out_coef <- tidy_coeftest(ct2, model_name = "heckman_step2_outcome_ols_imr_robust")

    out_paths <- write_table_triple(out_coef, "10_heckman_step2_outcome_with_imr", out_dir)
    record_output("table", out_paths$csv,  "Heckman step 2: outcome regression (loss_1d) incl IMR, robust SE (CSV).")
    record_output("table", out_paths$jsonl,"Heckman step 2: outcome regression (loss_1d) incl IMR, robust SE (JSONL).")
    record_output("table", out_paths$json, "Heckman step 2: outcome regression (loss_1d) incl IMR, robust SE (JSON).")

    out_txt <- file.path(out_dir, "tables", "10b_heckman_step2_summary.txt")
    sink(out_txt)
    print(summary(m_out))
    sink()
    record_output("doc", out_txt, "Heckman step 2 outcome regression summary (TXT).")
  }
}

# =============================================================================
# 13) Event study / trading interpretation:
#     "What happens if we buy the stock when Polymarket probability is x?"
#     - Use 1d snapshot implied probability (P(BEAT)) as x.
#     - Bin x into five 0.2-wide bins.
#     - Compute market-adjusted holding-period returns around earnings:
#         Buy at close t=-1, sell at close t=+1  => log(close_{+1}/close_{-1})
#         Adjust using S&P 500 close: log(spx_{+1}/spx_{-1})
# =============================================================================
cat("[13] Event study: market-adjusted holding returns by Polymarket probability bins...\n")

run_event_study <- any(as.character(prices_sample$snapshot_label) == "1d") &&
  all(c("market_id", "offset_td", "close", "spx_close") %in% names(stock_prices))

if (!run_event_study) {
  cat("NOTE: Skipping event study (requires stock_prices_daily.csv with market_id/offset_td/close/spx_close and 1d prices).\n")
} else {

  # 13a) Polymarket probability at 1d (one row per market_id)
  p_1d <- prices_sample %>%
    dplyr::filter(as.character(snapshot_label) == "1d") %>%
    dplyr::group_by(market_id) %>%
    dplyr::slice(1) %>% # defensive
    dplyr::ungroup() %>%
    dplyr::select(market_id, val_ric, val_anchor_date, p_mkt, y)

  # 13b) Prepare stock prices for offsets -1, 0, +1 (close-to-close holding returns)
  # We create an explicit label so pivoted column names are syntactically safe:
  #   -1 -> "m1", 0 -> "0", +1 -> "p1"
  stock_prices2 <- stock_prices %>%
    dplyr::mutate(
      market_id = as.character(market_id),
      offset_td = safe_numeric(offset_td),
      close     = safe_numeric(close),
      spx_close = safe_numeric(spx_close),
      offset_lab = dplyr::case_when(
        offset_td < 0 ~ paste0("m", abs(offset_td)),
        offset_td == 0 ~ "0",
        offset_td > 0 ~ paste0("p", offset_td),
        TRUE ~ NA_character_
      )
    ) %>%
    dplyr::filter(offset_td %in% c(-1, 0, 1), !is.na(offset_lab)) %>%
    dplyr::select(market_id, offset_td, offset_lab, close, spx_close) %>%
    dplyr::group_by(market_id, offset_lab) %>%
    dplyr::summarise(
      close = mean(close, na.rm = TRUE),
      spx_close = mean(spx_close, na.rm = TRUE),
      .groups = "drop"
    )

  # Pivot to wide for clean holding return computation
  # Resulting columns (if present): close_m1, close_0, close_p1, spx_close_m1, spx_close_0, spx_close_p1
  stock_wide <- stock_prices2 %>%
    tidyr::pivot_wider(
      names_from = offset_lab,
      values_from = c(close, spx_close),
      names_glue = "{.value}_{offset_lab}"
    )

  # Compute holding-period returns (log returns) and market-adjusted versions
  # Buy at t=-1 close and sell at t=+1 close:
  #   ret = log(P_{+1}/P_{-1})
  event_returns <- stock_wide %>%
    dplyr::mutate(
      ret_stock_m1_p1 = dplyr::if_else(
        is.finite(close_m1) & is.finite(close_p1) & close_m1 > 0 & close_p1 > 0,
        log(close_p1 / close_m1),
        NA_real_
      ),
      ret_spx_m1_p1 = dplyr::if_else(
        is.finite(spx_close_m1) & is.finite(spx_close_p1) & spx_close_m1 > 0 & spx_close_p1 > 0,
        log(spx_close_p1 / spx_close_m1),
        NA_real_
      ),
      abret_m1_p1 = ret_stock_m1_p1 - ret_spx_m1_p1,

      # Additional windows (optional, useful for robustness / appendices)
      ret_stock_m1_0 = dplyr::if_else(
        is.finite(close_m1) & is.finite(close_0) & close_m1 > 0 & close_0 > 0,
        log(close_0 / close_m1),
        NA_real_
      ),
      ret_spx_m1_0 = dplyr::if_else(
        is.finite(spx_close_m1) & is.finite(spx_close_0) & spx_close_m1 > 0 & spx_close_0 > 0,
        log(spx_close_0 / spx_close_m1),
        NA_real_
      ),
      abret_m1_0 = ret_stock_m1_0 - ret_spx_m1_0,

      ret_stock_0_1 = dplyr::if_else(
        is.finite(close_0) & is.finite(close_p1) & close_0 > 0 & close_p1 > 0,
        log(close_p1 / close_0),
        NA_real_
      ),
      ret_spx_0_1 = dplyr::if_else(
        is.finite(spx_close_0) & is.finite(spx_close_p1) & spx_close_0 > 0 & spx_close_p1 > 0,
        log(spx_close_p1 / spx_close_0),
        NA_real_
      ),
      abret_0_1 = ret_stock_0_1 - ret_spx_0_1
    ) %>%
    dplyr::select(market_id, abret_m1_p1, abret_m1_0, abret_0_1)

  # 13c) Join probabilities to returns and create bins
  event_study_df <- p_1d %>%
    dplyr::left_join(event_returns, by = "market_id") %>%
    dplyr::filter(is.finite(p_mkt), p_mkt >= 0, p_mkt <= 1) %>%
    dplyr::mutate(
      p_bin = prob_bin_20pct(p_mkt)
    )

  # 13d) Summary table by bin (market-adjusted holding return)
  event_study_by_bin <- event_study_df %>%
    dplyr::group_by(p_bin) %>%
    dplyr::summarise(
      # N counts all events in bin; N_ret counts those with observable holding returns
      N = dplyr::n(),
      N_ret = sum(is.finite(abret_m1_p1)),
      n_markets = dplyr::n_distinct(market_id),

      mean_p = mean(p_mkt, na.rm = TRUE),
      realized_beat_rate = mean(y, na.rm = TRUE),

      mean_abret_m1_p1   = mean(abret_m1_p1, na.rm = TRUE),
      median_abret_m1_p1 = median(abret_m1_p1, na.rm = TRUE),
      sd_abret_m1_p1     = sd(abret_m1_p1, na.rm = TRUE),

      se_abret_m1_p1 = dplyr::if_else(N_ret >= 2, sd_abret_m1_p1 / sqrt(N_ret), NA_real_),
      t_abret_m1_p1  = dplyr::if_else(is.finite(se_abret_m1_p1) & se_abret_m1_p1 > 0, mean_abret_m1_p1 / se_abret_m1_p1, NA_real_),
      pvalue_abret_m1_p1 = dplyr::if_else(
        is.finite(t_abret_m1_p1),
        2 * stats::pt(abs(t_abret_m1_p1), df = pmax(N_ret - 1, 1), lower.tail = FALSE),
        NA_real_
      ),

      .groups = "drop"
    )

  out_paths <- write_table_triple(event_study_by_bin, "11_event_study_abret_by_prob_bin_1d", out_dir)
  record_output("table", out_paths$csv,  "Event study: mean market-adjusted holding return (t=-1 to +1) by 1d probability bin (CSV).")
  record_output("table", out_paths$jsonl,"Event study: mean market-adjusted holding return (t=-1 to +1) by 1d probability bin (JSONL).")
  record_output("table", out_paths$json, "Event study: mean market-adjusted holding return (t=-1 to +1) by 1d probability bin (JSON).")

  # 13e) A simple visualization (mean abnormal return by bin)
  # (This is helpful for papers, but table is the primary output.)
  p_ev <- ggplot2::ggplot(event_study_by_bin, ggplot2::aes(x = p_bin, y = mean_abret_m1_p1)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = COL_GREY_1) +
    ggplot2::geom_point(color = COL_RED, size = 2.5) +
    ggplot2::geom_line(ggplot2::aes(group = 1), color = COL_GREY_2) +
    ggplot2::labs(
      title = "Event study: average market-adjusted holding return (t=-1 close to t=+1 close)\nby Polymarket P(BEAT) bin (1d snapshot)",
      x = "Polymarket implied probability bin (P(BEAT))",
      y = "Mean abnormal log return (stock - S&P 500)"
    ) +
    theme_corporate()

  plot_path <- save_plot_png(p_ev, "11_event_study_plot_abret_by_bin", out_dir, width = 10, height = 5)
  record_output("plot", plot_path, "Event study plot: mean abnormal return by probability bin (PNG).")

  # 13f) Optional: continuous regression (abnormal return ~ p_mkt)
  # This is a compact scientific complement to bin tables.
  ev_reg_df <- event_study_df %>%
    tidyr::drop_na(abret_m1_p1, p_mkt) %>%
    dplyr::mutate(ric = val_ric)

  if (nrow(ev_reg_df) >= 30) {
    m_ev <- stats::lm(abret_m1_p1 ~ p_mkt, data = ev_reg_df)
    vc_ev <- vcov_cluster_or_hc(m_ev, cluster = ev_reg_df$ric, type = "HC1")
    ct_ev <- lmtest::coeftest(m_ev, vcov. = vc_ev)
    ev_coef <- tidy_coeftest(ct_ev, model_name = "event_study_abret_m1_p1_on_p_1d")

    out_paths <- write_table_triple(ev_coef, "11b_event_study_reg_abret_on_prob_1d", out_dir)
    record_output("table", out_paths$csv,  "Event study regression: abnormal return (t=-1 to +1) on Polymarket probability (1d) (CSV).")
    record_output("table", out_paths$jsonl,"Event study regression: abnormal return (t=-1 to +1) on Polymarket probability (1d) (JSONL).")
    record_output("table", out_paths$json, "Event study regression: abnormal return (t=-1 to +1) on Polymarket probability (1d) (JSON).")
  } else {
    cat("NOTE: Too few observations for continuous event-study regression. Skipping.\n")
  }
}

# =============================================================================
# 14) Manifest + README + session info
# =============================================================================
cat("\n[14] Writing manifest + README + session info...\n")

manifest <- manifest %>%
  dplyr::arrange(match(type, c("table", "plot", "doc")), file)

out_paths <- write_table_triple(manifest, "00_output_manifest", out_dir)
record_output("table", out_paths$csv,  "Manifest of all outputs generated in this run (CSV).")
record_output("table", out_paths$jsonl,"Manifest of all outputs generated in this run (JSONL).")
record_output("table", out_paths$json, "Manifest of all outputs generated in this run (JSON).")

readme_path <- file.path(out_dir, "logs", "README.md")
readme_lines <- c(
  "# Accuracy / Brier Score / BSS / Calibration / Heckman / Event Study Outputs",
  "",
  glue::glue("- Run timestamp: **{run_ts}**"),
  glue::glue("- Generated at: **{Sys.time()}**"),
  glue::glue("- Script: `R/BS_Accuracy_BSS_Calibration.R`"),
  glue::glue("- Output directory: `statistics/test_statistics/`"),
  "",
  "## Key filters",
  "",
  glue::glue("- Resolution cutoff (UTC): resolution_ts <= {cutoff_ts}"),
  "- Only resolved YES/NO markets are included.",
  "- Only matched corporate events are included (val_ric + val_anchor_date; and MATCHED status if available).",
  "- Only non-stale Polymarket prices are included (valid yes/no prices, non-missing src timestamps, complement error within tolerance).",
  "- For duplicated price pulls, the latest generated_utc per (market_id, snapshot_label) is used.",
  "",
  "## Brier score models",
  "",
  "- **Polymarket:** p = implied probability snapshot (mapped to P(BEAT) if semantics require flipping)",
  "- **Coin flip:** p = 0.5",
  "- **Historical base rate (expanding):** for each event, p = mean(BEAT) across prior events ordered by anchor date",
  "- (Reported for robustness) **In-sample base rate:** p = mean(BEAT) across the full filtered sample",
  "",
  "## Statistical tests",
  "",
  "- Mean differences in per-event Brier loss are tested using a one-sample t-test on paired differences:",
  "  (loss_A - loss_B).",
  "",
  "## Determinants regression (primary horizon = 1d)",
  "",
  "- OLS: loss_1d ~ size + analysts + Polymarket activity + uncertainty + sector FE + time controls",
  "- Robust / firm-clustered standard errors where feasible.",
  "",
  "## Heckman selection (two-step)",
  "",
  "1) Probit selection: in_sample ~ log_mcap + analysts + log_turnover + volatility + sector FE",
  "2) Outcome regression on selected events: loss_1d ~ ... + IMR (inverse Mills ratio)",
  "",
  "## Event study / trading interpretation",
  "",
  "- Uses 1d Polymarket probability x = P(BEAT).",
  "- Bins x into five 20% bins.",
  "- Computes market-adjusted holding return: buy at close t=-1, sell at close t=+1, adjust using S&P 500 close.",
  "",
  "## Outputs",
  "",
  "- Tables: CSV + JSONL + JSON in `tables/`.",
  "- Figures: PNG in `plots/`.",
  "- Model summaries: TXT in `tables/`.",
  "",
  "See `00_output_manifest.csv` for a full list of outputs."
)
writeLines(readme_lines, readme_path)
record_output("doc", readme_path, "README with definitions, filters, and output list.")

sess_path <- file.path(out_dir, "logs", paste0("sessionInfo_", run_ts, ".txt"))
sink(sess_path)
print(sessionInfo())
sink()
record_output("doc", sess_path, "sessionInfo() snapshot (TXT).")

cat("\n==================== RUN COMPLETE ====================\n")
cat(glue::glue("Run log saved to: {log_path}\n"))
cat(glue::glue("README saved to:  {readme_path}\n"))
cat(glue::glue("Outputs saved in: {out_dir}\n"))
cat("======================================================\n\n")
