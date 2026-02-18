#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/BS_BrierScore_Analysis.R
# Purpose: Statistical analysis of *already computed* Brier scores for Polymarket:
#          (1) Brier score summaries + Brier Skill Score (BSS) vs benchmarks
#          (2) Correlation matrix (market-level)
#          (3) Regression analysis: determinants of Polymarket Brier loss
#          (4) Logit + Probit: probability Polymarket prediction is correct
#          (5) 5-bin (width=0.2) empirical "P(YES | price-bin)" tables + plot
#
# IMPORTANT:
# - This script does NOT recompute Brier scores. It uses:
#   data/brier_scores/brier_scores_market_horizon.csv
# - We exclude stale/unusable observations via usable_polymarket == TRUE
#   (and status == "ok" if present).
# - We exclude horizons/snapshots: "4w", "3w", "2w".
#
# Outputs (relative to project root):
#   statistics/test_statistics/brier_analysis/
#     - Tables (CSV + JSONL + JSON) via write_table_triple()
#     - Plots (PNG)
#     - logs/ (log + README + sessionInfo)
# =============================================================================

# -----------------------------
# 0) Project root discovery
# -----------------------------
find_project_root <- function(start = getwd()) {
  dir <- normalizePath(start, winslash = "/", mustWork = FALSE)
  for (i in 1:100) {
    has_lock  <- file.exists(file.path(dir, "renv.lock"))
    has_rproj <- length(list.files(dir, pattern = "\\.Rproj$", full.names = TRUE)) > 0
    has_data  <- dir.exists(file.path(dir, "data"))
    has_stats <- dir.exists(file.path(dir, "statistics"))
    looks_like_project <- (basename(dir) == "Polymarket-Earnings-Study") || (has_data && has_stats)
    if (has_lock || has_rproj || looks_like_project) return(dir)
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  stop("Could not find project root (expected renv.lock/.Rproj or data/ + statistics/).", call. = FALSE)
}

get_script_path <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
  }
  if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) return(normalizePath(p, winslash = "/", mustWork = FALSE))
  }
  NA_character_
}

# -----------------------------
# 0b) Package helpers
# -----------------------------
ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

vcov_cluster_safe <- function(model, cluster_vec) {
  cl <- as.factor(cluster_vec)
  
  vc <- tryCatch(
    sandwich::vcovCL(model, cluster = cl, type = "HC1"),
    error = function(e) NULL
  )
  
  bad <- TRUE
  if (!is.null(vc)) {
    d <- diag(vc)
    bad <- any(!is.finite(d)) || any(d < 0, na.rm = TRUE)
  }
  
  if (bad && requireNamespace("clubSandwich", quietly = TRUE)) {
    vc <- clubSandwich::vcovCR(model, cluster = cl, type = "CR2")
  }
  
  vc
}

coeftest_from_vcov <- function(model, vc) {
  b <- stats::coef(model)  # may include aliased coefficients as NA
  nm <- names(b)
  
  # Default: NA SE for everything, then fill in by matching names to vcov
  se <- rep(NA_real_, length(b))
  names(se) <- nm
  
  # Defensive: vc might be NULL if vcov computation failed
  if (!is.null(vc) && is.matrix(vc)) {
    rn <- rownames(vc)
    cn <- colnames(vc)
    
    # Ensure dimnames exist (rare but can happen)
    if (is.null(rn) || is.null(cn)) {
      rn <- cn <- nm
      dimnames(vc) <- list(rn, cn)
    }
    
    d <- diag(vc)
    names(d) <- rownames(vc)
    
    # Mark invalid variances as NA (prevents NaN in sqrt)
    d[!is.finite(d) | d < 0] <- NA_real_
    
    common <- intersect(nm, names(d))
    se[common] <- sqrt(d[common])
  }
  
  stat <- b / se
  p <- 2 * stats::pnorm(abs(stat), lower.tail = FALSE)
  
  tibble::tibble(
    term = nm,
    estimate = as.numeric(b),
    std_error = as.numeric(se),
    statistic = as.numeric(stat),
    p_value = as.numeric(p),
    conf_low_95  = estimate - 1.96 * std_error,
    conf_high_95 = estimate + 1.96 * std_error
  )
}

# =============================================================================
# Main runner (callable from other scripts)
# =============================================================================
run_brier_score_analysis <- function(
    root_dir = NULL,
    paths = list(
      markets     = file.path("data", "markets", "markets.csv"),
      corporate   = file.path("data", "corporate_info", "corporate_info.csv"),
      brier       = file.path("data", "brier_scores", "brier_scores_market_horizon.csv")
    ),
    out_subdir = file.path("statistics", "test_statistics", "brier_analysis"),
    calibration_bins = seq(0, 1, by = 0.2),
    calibration_horizon_preference = c("1w", "7d", "6d", "5d", "4d", "3d", "2d", "1d")
) {
  
  # -----------------------------
  # Resolve root dir
  # -----------------------------
  script_path <- get_script_path()
  start_dir   <- if (!is.na(script_path)) dirname(script_path) else getwd()
  if (is.null(root_dir)) root_dir <- find_project_root(start_dir)
  
  # -----------------------------
  # renv (best-effort)
  # -----------------------------
  if (file.exists(file.path(root_dir, "renv.lock"))) {
    ensure_pkg("renv")
    tryCatch({
      renv::load(project = root_dir)
      renv::restore(project = root_dir, prompt = FALSE)
    }, error = function(e) {
      message("WARNING: renv::restore() failed. You may need to run it manually.\nError: ", e$message)
    })
  }
  
  # -----------------------------
  # Shared helpers + packages
  # -----------------------------
  source(file.path(root_dir, "R", "utils", "pm_common.R"))
  pm_load_packages()
  
  # Extra packages for robust inference + tidying + regression tables
  for (p in c("sandwich", "lmtest", "broom", "purrr", "modelsummary", "performance")) ensure_pkg(p)
  
  # -----------------------------
  # Output dirs + logging
  # -----------------------------
  out_dir <- file.path(root_dir, out_subdir)
  fs::dir_create(out_dir)
  fs::dir_create(file.path(out_dir, "logs"))
  
  run_ts   <- format(Sys.time(), "%Y%m%dT%H%M%S")
  log_path <- file.path(out_dir, "logs", paste0("BS_analysis_run_", run_ts, ".log.txt"))
  
  sink(log_path, split = TRUE)
  on.exit(sink(), add = TRUE)
  
  cat(glue::glue("Brier analysis run started: {Sys.time()}\n"))
  cat(glue::glue("Project root: {root_dir}\n"))
  cat(glue::glue("Output dir:   {out_dir}\n\n"))
  
  manifest <- tibble::tibble(type = character(), file = character(), rel_path = character(), description = character())
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
  # 1) Read inputs (NO Brier recomputation)
  # =============================================================================
  cat("Reading inputs...\n")
  
  paths_abs <- list(
    markets   = file.path(root_dir, paths$markets),
    corporate = file.path(root_dir, paths$corporate),
    brier     = file.path(root_dir, paths$brier)
  )
  
  markets_raw   <- read_csv_required(paths_abs$markets)
  corporate_raw <- read_csv_required(paths_abs$corporate)
  brier_raw     <- read_csv_required(paths_abs$brier)
  
  markets   <- janitor::clean_names(markets_raw)
  corporate <- janitor::clean_names(corporate_raw)
  brier     <- janitor::clean_names(brier_raw)
  
  # =============================================================================
  # 2) Clean/construct covariates
  # =============================================================================
  cat("\n[2] Preparing covariates...\n")
  
  # Markets: build market_open_hours, log volume, etc (robust to missing columns)
  markets_cov <- markets %>%
    dplyr::mutate(
      market_id = as.character(dplyr::coalesce(id, val_market_id)),
      ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
      
      # timestamps
      start_dt_utc = if ("start_date" %in% names(.)) parse_ts_utc(start_date) else
        if ("startdate" %in% names(.)) parse_ts_utc(startdate) else as.POSIXct(NA),
      
      uma_end_dt_utc = if ("uma_end_date" %in% names(.)) parse_ts_utc(uma_end_date) else
        if ("umaenddate" %in% names(.)) parse_ts_utc(umaenddate) else as.POSIXct(NA),
      
      closed_dt_utc = if ("closed_time" %in% names(.)) parse_ts_utc(closed_time) else
        if ("closedtime" %in% names(.)) parse_ts_utc(closedtime) else as.POSIXct(NA),
      
      end_dt_utc = if ("end_date" %in% names(.)) parse_ts_utc(end_date) else
        if ("enddate" %in% names(.)) parse_ts_utc(enddate) else as.POSIXct(NA),
      
      updated_dt_utc = if ("updated_at" %in% names(.)) parse_ts_utc(updated_at) else
        if ("updatedat" %in% names(.)) parse_ts_utc(updatedat) else as.POSIXct(NA),
      
      resolution_dt_utc = dplyr::coalesce(uma_end_dt_utc, closed_dt_utc, end_dt_utc, updated_dt_utc),
      
      market_open_hours = as.numeric(difftime(resolution_dt_utc, start_dt_utc, units = "hours")),
      market_open_hours = dplyr::if_else(is.finite(market_open_hours), abs(market_open_hours), NA_real_),
      market_open_days  = market_open_hours / 24,
      
      poly_volume = if ("volume_num" %in% names(.)) safe_numeric(volume_num) else
        if ("volumenum" %in% names(.)) safe_numeric(volumenum) else NA_real_,
      
      poly_liquidity = if ("liquidity_num" %in% names(.)) safe_numeric(liquidity_num) else
        if ("liquiditynum" %in% names(.)) safe_numeric(liquiditynum) else NA_real_,
      
      log_poly_volume   = safe_log(poly_volume),
      log_poly_liquidity = safe_log(poly_liquidity),
      
      val_surprise = if ("val_surprise" %in% names(.)) safe_numeric(val_surprise) else NA_real_,
      abs_surprise = dplyr::if_else(is.finite(val_surprise), abs(val_surprise), NA_real_),
      
      gics_sector_market = if ("gics_sector" %in% names(.)) as.character(gics_sector) else NA_character_
    ) %>%
    dplyr::select(
      market_id, ticker_market = ticker, start_dt_utc, resolution_dt_utc, market_open_hours, market_open_days,
      poly_volume, poly_liquidity, log_poly_volume, log_poly_liquidity,
      val_surprise, abs_surprise, gics_sector_market
    ) %>%
    dplyr::distinct(market_id, .keep_all = TRUE)
  
  # Corporate: create consistent covariates
  corporate_cov <- corporate %>%
    dplyr::mutate(
      ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
      ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
      
      market_cap_usd = if ("market_cap_usd" %in% names(.)) safe_numeric(market_cap_usd) else NA_real_,
      log_mkt_cap = safe_log(market_cap_usd),
      
      analysts = dplyr::coalesce(
        if ("analysts_covering_sample_mean" %in% names(.)) safe_numeric(analysts_covering_sample_mean) else NA_real_,
        if ("analysts_covering_latest" %in% names(.)) safe_numeric(analysts_covering_latest) else NA_real_
      ),
      
      stock_turnover_6m = dplyr::coalesce(
        if ("turnover_6m_sum_volume_mean" %in% names(.)) safe_numeric(turnover_6m_sum_volume_mean) else NA_real_,
        if ("turnover_6m_sum_volume_median" %in% names(.)) safe_numeric(turnover_6m_sum_volume_median) else NA_real_
      ),
      log_stock_turnover_6m = safe_log(stock_turnover_6m),
      
      stock_volatility_6m = dplyr::coalesce(
        if ("volatility_6m_mean" %in% names(.)) safe_numeric(volatility_6m_mean) else NA_real_,
        if ("volatility_6m_median" %in% names(.)) safe_numeric(volatility_6m_median) else NA_real_
      ),
      
      gics_sector = if ("gics_sector" %in% names(.)) as.character(gics_sector) else NA_character_
    ) %>%
    dplyr::select(
      ticker, ric, company_name, gics_sector,
      market_cap_usd, log_mkt_cap, analysts,
      stock_turnover_6m, log_stock_turnover_6m, stock_volatility_6m
    ) %>%
    dplyr::distinct(ticker, .keep_all = TRUE)
  
  # =============================================================================
  # 3) Analysis dataset (panel: market_id x horizon)
  # =============================================================================
  cat("\n[3] Preparing analysis dataset from brier_scores...\n")
  
  brier_panel <- brier %>%
    dplyr::mutate(
      market_id = as.character(market_id),
      ticker = if ("ticker" %in% names(.)) as.character(ticker) else NA_character_,
      horizon = if ("horizon" %in% names(.)) as.character(horizon) else NA_character_,
      horizon_seconds = if ("horizon_seconds" %in% names(.)) safe_numeric(horizon_seconds) else NA_real_,
      
      # Core items from file (already computed)
      y = if ("y" %in% names(.)) safe_numeric(y) else NA_real_,
      p_polymarket_yes = if ("p_polymarket_yes" %in% names(.)) safe_numeric(p_polymarket_yes) else NA_real_,
      p_hist = if ("p_hist_asof_end_minus_1d" %in% names(.)) safe_numeric(p_hist_asof_end_minus_1d) else NA_real_,
      p_dice_0p5 = if ("p_dice_0p5" %in% names(.)) safe_numeric(p_dice_0p5) else 0.5,
      
      loss_polymarket = if ("loss_polymarket" %in% names(.)) safe_numeric(loss_polymarket) else NA_real_,
      loss_hist = if ("loss_hist" %in% names(.)) safe_numeric(loss_hist) else NA_real_,
      loss_dice = if ("loss_dice" %in% names(.)) safe_numeric(loss_dice) else NA_real_,
      
      usable_polymarket = if ("usable_polymarket" %in% names(.)) as.logical(usable_polymarket) else FALSE,
      status = if ("status" %in% names(.)) as.character(status) else NA_character_,
      
      # correctness (Polymarket classification threshold 0.5)
      pred_yes = dplyr::if_else(is.finite(p_polymarket_yes) & p_polymarket_yes >= 0.5, 1, 0),
      correct = dplyr::if_else(is.finite(y) & is.finite(pred_yes) & (pred_yes == y), 1, 0),
      
      # time-to-close (if present)
      seconds_before_close = if ("seconds_before_close" %in% names(.)) safe_numeric(seconds_before_close) else NA_real_
    ) %>%
    # Exclude long-horizon snapshots
    dplyr::filter(!(horizon %in% c("4w", "3w", "2w"))) %>%
    # Non-stale / usable only
    dplyr::filter(usable_polymarket == TRUE) %>%
    { if ("status" %in% names(.)) dplyr::filter(., is.na(status) | status %in% c("ok", "usable")) else . } %>%
    dplyr::filter(is.finite(loss_polymarket), is.finite(loss_hist), is.finite(loss_dice), is.finite(y), is.finite(p_polymarket_yes)) %>%
    dplyr::left_join(markets_cov, by = "market_id") %>%
    dplyr::mutate(
      # If ticker in brier_scores is missing/NA, fill it from markets.csv
      ticker = dplyr::coalesce(ticker, ticker_market)
    ) %>%
    dplyr::select(-ticker_market) %>%
    dplyr::left_join(corporate_cov, by = "ticker") %>%
    dplyr::mutate(
      gics_sector = dplyr::coalesce(gics_sector, gics_sector_market, "Unknown"),
      gics_sector = as.factor(gics_sector)
    )
  
  if (nrow(brier_panel) == 0) stop("No usable rows remain after filtering usable_polymarket/status and finite values.", call. = FALSE)
  
  # Horizon ordering for tables/plots
  horizon_levels <- brier_panel %>%
    dplyr::distinct(horizon, horizon_seconds) %>%
    dplyr::arrange(dplyr::desc(horizon_seconds), horizon) %>%
    dplyr::pull(horizon) %>%
    unique()
  
  brier_panel <- brier_panel %>%
    dplyr::mutate(horizon = factor(horizon, levels = horizon_levels))
  
  cat(glue::glue("Usable observations: {nrow(brier_panel)}\n"))
  cat(glue::glue("Distinct markets:    {dplyr::n_distinct(brier_panel$market_id)}\n\n"))
  
  # Market-level (one row per market) averages for correlation + a cleaner cross-section
  brier_market <- brier_panel %>%
    dplyr::group_by(market_id) %>%
    dplyr::summarise(
      ticker = dplyr::first(ticker),
      gics_sector = dplyr::first(gics_sector),
      
      # dependent variables (averaged across horizons)
      brier_polymarket = mean(loss_polymarket, na.rm = TRUE),
      brier_hist       = mean(loss_hist, na.rm = TRUE),
      brier_dice       = mean(loss_dice, na.rm = TRUE),
      accuracy         = mean(correct, na.rm = TRUE),
      
      # covariates (time-invariant per market)
      log_mkt_cap = dplyr::first(log_mkt_cap),
      analysts = dplyr::first(analysts),
      log_stock_turnover_6m = dplyr::first(log_stock_turnover_6m),
      stock_volatility_6m = dplyr::first(stock_volatility_6m),
      
      log_poly_volume = dplyr::first(log_poly_volume),
      log_poly_liquidity = dplyr::first(log_poly_liquidity),
      market_open_days = dplyr::first(market_open_days),
      abs_surprise = dplyr::first(abs_surprise),
      
      n_horizons = dplyr::n(),
      .groups = "drop"
    )
  
  # =============================================================================
  # 4) Brier score tables + Brier Skill Score
  # =============================================================================
  cat("[4] Brier score summaries + Brier Skill Score...\n")
  
  # Helper: mean + 95% CI (uses pm_common.R mean_ci_95)
  summarise_brier <- function(df, group_var = NULL) {
    df_long <- df %>%
      dplyr::select(dplyr::any_of(c("market_id", "horizon", "loss_polymarket", "loss_hist", "loss_dice"))) %>%
      tidyr::pivot_longer(
        cols = c(loss_polymarket, loss_hist, loss_dice),
        names_to = "model",
        values_to = "loss"
      ) %>%
      dplyr::mutate(
        model = dplyr::recode(
          model,
          loss_polymarket = "Polymarket",
          loss_hist       = "Historical base rate",
          loss_dice       = "Coinflip (0.5)"
        )
      )
    
    if (!is.null(group_var)) {
      df_long <- df_long %>% dplyr::group_by(.data[[group_var]], model)
    } else {
      df_long <- df_long %>% dplyr::group_by(model)
    }
    
    df_long %>%
      dplyr::summarise(
        N = sum(is.finite(loss)),
        n_markets = dplyr::n_distinct(market_id[is.finite(loss)]),
        mean_ci_95(loss),
        .groups = "drop"
      ) %>%
      dplyr::mutate(brier_mean_ci_95 = format_mean_ci(mean, ci_low_95, ci_high_95, digits = 4))
  }
  
  brier_overall <- summarise_brier(brier_panel, group_var = NULL)
  brier_by_horizon <- summarise_brier(brier_panel, group_var = "horizon") %>%
    dplyr::mutate(horizon = as.character(horizon)) %>%
    dplyr::arrange(factor(horizon, levels = horizon_levels), model)
  
  out_paths <- write_table_triple(brier_overall, "BA_01_brier_overall_ci_long", out_dir)
  record_output("table", out_paths$csv, "Overall Brier scores (mean + 95% CI) for Polymarket and benchmarks.")
  
  out_paths <- write_table_triple(brier_by_horizon, "BA_02_brier_by_horizon_ci_long", out_dir)
  record_output("table", out_paths$csv, "Brier scores by horizon (mean + 95% CI), long format.")
  
  cat("\n--- TABLE: BA_01_brier_overall_ci_long ---\n")
  print(brier_overall)
  cat("\n--- TABLE: BA_02_brier_by_horizon_ci_long ---\n")
  print(brier_by_horizon)
  
  # Brier Skill Score: 1 - (BS_model / BS_ref)
  # Overall
  brier_means_overall <- brier_overall %>%
    dplyr::select(model, mean) %>%
    tidyr::pivot_wider(names_from = model, values_from = mean)
  
  bss_overall <- tibble::tibble(
    benchmark = c("Coinflip (0.5)", "Historical base rate"),
    brier_model = brier_means_overall$Polymarket,
    brier_ref   = c(brier_means_overall$`Coinflip (0.5)`, brier_means_overall$`Historical base rate`)
  ) %>%
    dplyr::mutate(bss = 1 - (brier_model / brier_ref))
  
  out_paths <- write_table_triple(bss_overall, "BA_03_bss_overall", out_dir)
  record_output("table", out_paths$csv, "Overall Brier Skill Score (BSS) vs benchmarks.")
  
  # By horizon
  brier_means_by_h <- brier_by_horizon %>%
    dplyr::select(horizon, model, mean) %>%
    tidyr::pivot_wider(names_from = model, values_from = mean)
  
  bss_by_horizon <- brier_means_by_h %>%
    dplyr::mutate(
      bss_vs_coinflip = 1 - (Polymarket / `Coinflip (0.5)`),
      bss_vs_hist     = 1 - (Polymarket / `Historical base rate`)
    ) %>%
    dplyr::select(horizon, bss_vs_coinflip, bss_vs_hist)
  
  out_paths <- write_table_triple(bss_by_horizon, "BA_04_bss_by_horizon", out_dir)
  record_output("table", out_paths$csv, "BSS by horizon vs coinflip and historical base rate.")
  
  cat("\n--- TABLE: BA_03_bss_overall ---\n")
  print(bss_overall)
  cat("\n--- TABLE: BA_04_bss_by_horizon ---\n")
  print(bss_by_horizon)
  
  # Plot: Brier score by horizon (Polymarket + benchmarks)
  col_map <- c(
    "Polymarket" = COL_RED,
    "Coinflip (0.5)" = COL_GREY_1,
    "Historical base rate" = COL_DARKBLUE
  )
  
  plot_df <- brier_by_horizon %>%
    dplyr::mutate(horizon = factor(horizon, levels = horizon_levels))
  
  p_brier_h <- ggplot2::ggplot(plot_df, ggplot2::aes(x = horizon, y = mean, group = model, color = model)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 2) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = ci_low_95, ymax = ci_high_95), width = 0.15) +
    ggplot2::scale_color_manual(values = col_map) +
    ggplot2::labs(
      title = "Brier score by horizon (mean ± 95% CI)",
      x = "Horizon",
      y = "Brier score (mean squared error)"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_brier_h, "BA_05_plot_brier_by_horizon_ci", out_dir, width = 10, height = 5)
  record_output("plot", plot_path, "Plot: Brier score by horizon with 95% CI.")
  
  # =============================================================================
  # 5) Correlation matrix (market-level)
  # =============================================================================
  cat("\n[5] Correlation matrix (market-level averages)...\n")
  
  corr_vars <- c(
    "brier_polymarket",
    "accuracy",
    "log_mkt_cap",
    "analysts",
    "log_stock_turnover_6m",
    "stock_volatility_6m",
    "log_poly_volume",
    "log_poly_liquidity",
    "market_open_days",
    "abs_surprise"
  )
  
  corr_df <- brier_market %>%
    dplyr::select(dplyr::any_of(c("market_id", corr_vars))) %>%
    dplyr::mutate(dplyr::across(dplyr::all_of(corr_vars), ~ dplyr::if_else(is.finite(.x), .x, NA_real_)))
  
  corr_mat_input <- corr_df %>% dplyr::select(dplyr::all_of(corr_vars))
  
  sds <- vapply(corr_mat_input, stats::sd, numeric(1), na.rm = TRUE)
  nonconst_vars <- names(sds)[is.finite(sds) & sds > 0]
  dropped_vars  <- setdiff(corr_vars, nonconst_vars)
  
  if (length(dropped_vars) > 0) {
    cat("Correlation: dropping zero-variance variables:\n")
    cat("  ", paste(dropped_vars, collapse = ", "), "\n\n")
    out_paths <- write_table_triple(
      tibble::tibble(variable = dropped_vars),
      "BA_06a_corr_dropped_zero_variance_vars",
      out_dir
    )
    record_output("table", out_paths$csv, "Variables dropped from correlation matrix due to zero variance.")
  }
  
  corr_mat <- stats::cor(
    corr_mat_input %>% dplyr::select(dplyr::all_of(nonconst_vars)),
    use = "pairwise.complete.obs",
    method = "pearson"
  )
  
  corr_mat_tbl <- as.data.frame(corr_mat) %>%
    tibble::rownames_to_column("var")
  
  out_paths <- write_table_triple(corr_mat_tbl, "BA_06_corr_matrix_pearson_wide", out_dir)
  record_output("table", out_paths$csv, "Pearson correlation matrix (wide), market-level.")
  
  # Long correlations with p-values (pairwise cor.test)
  corr_pairs <- expand.grid(var1 = corr_vars, var2 = corr_vars, stringsAsFactors = FALSE) %>%
    dplyr::filter(var1 <= var2)
  
  corr_long <- purrr::pmap_dfr(
    corr_pairs,
    function(var1, var2) {
      x <- corr_df[[var1]]
      y <- corr_df[[var2]]
      ok <- is.finite(x) & is.finite(y)
      n <- sum(ok)
      if (n < 3) {
        return(tibble::tibble(var1 = var1, var2 = var2, n = n, corr = NA_real_, p_value = NA_real_))
      }
      ct <- suppressWarnings(stats::cor.test(x[ok], y[ok], method = "pearson"))
      tibble::tibble(var1 = var1, var2 = var2, n = n, corr = unname(ct$estimate), p_value = unname(ct$p.value))
    }
  ) %>%
    dplyr::arrange(var1, var2)
  
  out_paths <- write_table_triple(corr_long, "BA_07_corr_matrix_pearson_long_with_pvalues", out_dir)
  record_output("table", out_paths$csv, "Pearson correlations (long) with p-values, market-level.")
  
  cat("\n--- TABLE: BA_06_corr_matrix_pearson_wide ---\n")
  print(corr_mat_tbl)
  cat("\n--- TABLE: BA_07_corr_matrix_pearson_long_with_pvalues ---\n")
  print(corr_long)
  
  # =============================================================================
  # 6) Regressions: determinants of Brier loss
  # =============================================================================
  cat("\n[6] Regression analysis: determinants of Polymarket Brier loss...\n")
  
  # Pretty regression tables (modelsummary)
  print_modelsummary_table <- function(models_named_list, vcov_list, title, add_rows = NULL, gof_omit = NULL) {
    cat("\n", title, "\n", sep = "")
    
    tab <- suppressWarnings(
      modelsummary::modelsummary(
        models_named_list,
        vcov = vcov_list,
        output = "markdown",
        statistic = "({std.error})",
        stars = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
        fmt = 4,
        add_rows = add_rows,
        gof_omit = gof_omit
      )
    )
    
    # modelsummary may return a character string OR a table object (e.g., tinytable)
    if (is.character(tab)) {
      cat(tab, "\n")
    } else {
      print(tab)
      cat("\n")
    }
  }
  
  mcfadden_r2_safe <- function(model) {
    out <- tryCatch(performance::r2_mcfadden(model), error = function(e) NULL)
    if (is.null(out)) return(NA_real_)
    if ("R2" %in% names(out)) return(as.numeric(out$R2[1]))
    NA_real_
  }
  
  # Helper: clustered robust tidy output (existing saved tables)
  tidy_clustered <- function(model, cluster_vec, model_name) {
    vc <- vcov_cluster_safe(model, cluster_vec)
    ct <- coeftest_from_vcov(model, vc)
    
    ct %>%
      dplyr::mutate(
        model = model_name,
        sig = dplyr::case_when(
          is.na(p_value) ~ "",
          p_value < 0.001 ~ "***",
          p_value < 0.01  ~ "**",
          p_value < 0.05  ~ "*",
          p_value < 0.1   ~ ".",
          TRUE ~ ""
        )
      ) %>%
      dplyr::select(model, dplyr::everything())
  }
  
  # PANEL OLS (market_id x horizon), horizon fixed effects, clustered by market_id
  panel_reg_df <- brier_panel %>%
    dplyr::filter(
      is.finite(loss_polymarket),
      is.finite(log_poly_volume) | is.na(log_poly_volume),
      is.finite(log_mkt_cap) | is.na(log_mkt_cap)
    )
  
  # Model specs (incremental)
  m1 <- stats::lm(
    loss_polymarket ~ horizon +
      log_mkt_cap + analysts +
      log_poly_volume + market_open_days,
    data = panel_reg_df
  )
  
  m2 <- stats::lm(
    loss_polymarket ~ horizon +
      log_mkt_cap + analysts +
      log_poly_volume + log_poly_liquidity +
      log_stock_turnover_6m + stock_volatility_6m +
      market_open_days + abs_surprise,
    data = panel_reg_df
  )
  
  m3 <- stats::lm(
    loss_polymarket ~ horizon +
      log_mkt_cap + analysts +
      log_poly_volume + log_poly_liquidity +
      log_stock_turnover_6m + stock_volatility_6m +
      market_open_days + abs_surprise +
      gics_sector,
    data = panel_reg_df
  )
  
  reg_panel_tidy <- dplyr::bind_rows(
    tidy_clustered(m1, panel_reg_df$market_id, "OLS_panel_M1"),
    tidy_clustered(m2, panel_reg_df$market_id, "OLS_panel_M2"),
    tidy_clustered(m3, panel_reg_df$market_id, "OLS_panel_M3")
  )
  
  out_paths <- write_table_triple(reg_panel_tidy, "BA_08_reg_ols_panel_cluster_market", out_dir)
  record_output("table", out_paths$csv, "OLS regressions (panel) for Polymarket loss; cluster-robust SE by market_id.")
  
  cat("\n--- TABLE: BA_08_reg_ols_panel_cluster_market ---\n")
  print(reg_panel_tidy)
  
  # Pretty OLS regression table (panel)
  print_modelsummary_table(
    models_named_list = list("M1" = m1, "M2" = m2, "M3" = m3),
    vcov_list = list(
      "M1" = vcov_cluster_safe(m1, panel_reg_df$market_id),
      "M2" = vcov_cluster_safe(m2, panel_reg_df$market_id),
      "M3" = vcov_cluster_safe(m3, panel_reg_df$market_id)
    ),
    title = "--- REGRESSION TABLE: Panel OLS (clustered SE by market_id) ---",
    gof_omit = "AIC|BIC|Log.Lik|RMSE|Std.Errors|F"
  )
  
  # MARKET-LEVEL OLS (one row per market): average Brier loss
  market_reg_df <- brier_market %>%
    dplyr::mutate(
      gics_sector = as.factor(gics_sector),
      brier_polymarket = dplyr::if_else(is.finite(brier_polymarket), brier_polymarket, NA_real_)
    ) %>%
    dplyr::filter(is.finite(brier_polymarket))
  
  mm1 <- stats::lm(
    brier_polymarket ~ log_mkt_cap + analysts + log_poly_volume + market_open_days,
    data = market_reg_df
  )
  
  mm2 <- stats::lm(
    brier_polymarket ~ log_mkt_cap + analysts +
      log_poly_volume + log_poly_liquidity +
      log_stock_turnover_6m + stock_volatility_6m +
      market_open_days + abs_surprise,
    data = market_reg_df
  )
  
  mm3 <- stats::lm(
    brier_polymarket ~ log_mkt_cap + analysts +
      log_poly_volume + log_poly_liquidity +
      log_stock_turnover_6m + stock_volatility_6m +
      market_open_days + abs_surprise +
      gics_sector,
    data = market_reg_df
  )
  
  reg_market_tidy <- dplyr::bind_rows(
    tidy_clustered(mm1, market_reg_df$market_id, "OLS_market_M1"),
    tidy_clustered(mm2, market_reg_df$market_id, "OLS_market_M2"),
    tidy_clustered(mm3, market_reg_df$market_id, "OLS_market_M3")
  )
  
  out_paths <- write_table_triple(reg_market_tidy, "BA_09_reg_ols_market_cluster_market", out_dir)
  record_output("table", out_paths$csv, "OLS regressions (market-level) for avg Polymarket Brier loss; cluster-robust SE by market_id.")
  
  cat("\n--- TABLE: BA_09_reg_ols_market_cluster_market ---\n")
  print(reg_market_tidy)
  
  # Pretty OLS regression table (market-level)
  print_modelsummary_table(
    models_named_list = list("M1" = mm1, "M2" = mm2, "M3" = mm3),
    vcov_list = list(
      "M1" = vcov_cluster_safe(mm1, market_reg_df$market_id),
      "M2" = vcov_cluster_safe(mm2, market_reg_df$market_id),
      "M3" = vcov_cluster_safe(mm3, market_reg_df$market_id)
    ),
    title = "--- REGRESSION TABLE: Market-level OLS (clustered SE by market_id) ---",
    gof_omit = "AIC|BIC|Log.Lik|RMSE|Std.Errors|F"
  )
  
  # =============================================================================
  # 7) Logit + Probit: P(correct) determinants
  # =============================================================================
  cat("\n[7] Logit + Probit: probability Polymarket prediction is correct...\n")
  
  glm_df <- brier_panel %>%
    dplyr::filter(is.finite(correct)) %>%
    dplyr::mutate(correct = as.integer(correct))
  
  # Logit / Probit specs (pooled with horizon FE)
  logit_m <- stats::glm(
    correct ~ horizon +
      log_mkt_cap + analysts +
      log_poly_volume + log_poly_liquidity +
      log_stock_turnover_6m + stock_volatility_6m +
      market_open_days + abs_surprise +
      gics_sector,
    data = glm_df,
    family = stats::binomial(link = "logit")
  )
  
  probit_m <- stats::glm(
    correct ~ horizon +
      log_mkt_cap + analysts +
      log_poly_volume + log_poly_liquidity +
      log_stock_turnover_6m + stock_volatility_6m +
      market_open_days + abs_surprise +
      gics_sector,
    data = glm_df,
    family = stats::binomial(link = "probit")
  )
  
  tidy_clustered_glm <- function(model, cluster_vec, model_name, exp_coef = FALSE) {
    vc <- vcov_cluster_safe(model, cluster_vec)
    ct <- coeftest_from_vcov(model, vc)
    
    if (exp_coef) {
      ct <- ct %>%
        dplyr::mutate(
          odds_ratio = exp(estimate),
          or_low_95  = exp(conf_low_95),
          or_high_95 = exp(conf_high_95)
        )
    }
    
    ct %>%
      dplyr::mutate(
        model = model_name,
        sig = dplyr::case_when(
          is.na(p_value) ~ "",
          p_value < 0.001 ~ "***",
          p_value < 0.01  ~ "**",
          p_value < 0.05  ~ "*",
          p_value < 0.1   ~ ".",
          TRUE ~ ""
        )
      ) %>%
      dplyr::select(model, dplyr::everything())
  }
  
  glm_tidy <- dplyr::bind_rows(
    tidy_clustered_glm(logit_m, glm_df$market_id, "LOGIT_correct_cluster_market", exp_coef = TRUE),
    tidy_clustered_glm(probit_m, glm_df$market_id, "PROBIT_correct_cluster_market", exp_coef = FALSE)
  )
  
  out_paths <- write_table_triple(glm_tidy, "BA_10_glm_logit_probit_correct_cluster_market", out_dir)
  record_output("table", out_paths$csv, "Logit + Probit models for correctness; cluster-robust SE by market_id (logit includes odds ratios).")
  
  cat("\n--- TABLE: BA_10_glm_logit_probit_correct_cluster_market ---\n")
  print(glm_tidy)
  
  # Pretty Logit/Probit regression table + McFadden R^2
  r2_logit  <- mcfadden_r2_safe(logit_m)
  r2_probit <- mcfadden_r2_safe(probit_m)
  
  add_rows_glm <- tibble::tibble(
    term = "McFadden R2",
    LOGIT  = r2_logit,
    PROBIT = r2_probit
  )
  
  print_modelsummary_table(
    models_named_list = list("LOGIT" = logit_m, "PROBIT" = probit_m),
    vcov_list = list(
      "LOGIT"  = vcov_cluster_safe(logit_m, glm_df$market_id),
      "PROBIT" = vcov_cluster_safe(probit_m, glm_df$market_id)
    ),
    title = "--- REGRESSION TABLE: Logit/Probit P(correct) (clustered SE by market_id) ---",
    add_rows = add_rows_glm,
    gof_omit = "R2|Adj|RMSE|F"
  )
  
  # =============================================================================
  # 8) 5-bin analysis: P(YES | Polymarket price bin), plus calibration-style plot
  # =============================================================================
  cat("\n[8] 5-bin analysis (width 0.2): P(YES | price-bin)...\n")
  
  # Choose a single horizon for a “paper-friendly” table/plot (prefer 1w if present)
  horizons_present <- levels(brier_panel$horizon)
  preferred <- calibration_horizon_preference[calibration_horizon_preference %in% horizons_present]
  horizon_for_cal <- if (length(preferred) > 0) preferred[1] else horizons_present[length(horizons_present)]
  
  # Bin helper (include p=0 and p=1 safely)
  make_bins <- function(p, breaks) {
    p2 <- p
    p2 <- pmin(pmax(p2, 0), 1)
    p2 <- dplyr::if_else(p2 == 1, 1 - 1e-12, p2)
    
    cut(
      p2,
      breaks = breaks,
      include.lowest = TRUE,
      right = FALSE,
      labels = paste0(head(breaks, -1), "-", tail(breaks, -1))
    )
  }
  
  bins_breaks <- calibration_bins
  if (tail(bins_breaks, 1) != 1) stop("calibration_bins must end at 1 (e.g., seq(0,1,0.2)).", call. = FALSE)
  if (length(bins_breaks) != 6) stop("Expected 5 bins => calibration_bins length should be 6 (0,0.2,...,1).", call. = FALSE)
  
  panel_binned <- brier_panel %>%
    dplyr::mutate(
      p_bin = make_bins(p_polymarket_yes, bins_breaks),
      p_bin = factor(p_bin, levels = paste0(head(bins_breaks, -1), "-", tail(bins_breaks, -1)))
    )
  
  bin_table_all_horizons <- panel_binned %>%
    dplyr::group_by(horizon, p_bin) %>%
    dplyr::summarise(
      n = dplyr::n(),
      n_markets = dplyr::n_distinct(market_id),
      mean_p = mean(p_polymarket_yes, na.rm = TRUE),
      p_yes_empirical = mean(y, na.rm = TRUE),
      accuracy = mean(correct, na.rm = TRUE),
      brier = mean(loss_polymarket, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(horizon = as.character(horizon)) %>%
    dplyr::arrange(factor(horizon, levels = horizon_levels), p_bin)
  
  out_paths <- write_table_triple(bin_table_all_horizons, "BA_11_price_bins_p_yes_by_horizon", out_dir)
  record_output("table", out_paths$csv, "5-bin table: empirical P(YES | price-bin) by horizon (width 0.2).")
  
  # Single-horizon “paper table”
  bin_table_one <- bin_table_all_horizons %>%
    dplyr::filter(horizon == horizon_for_cal)
  
  out_paths <- write_table_triple(bin_table_one, "BA_12_price_bins_p_yes_single_horizon", out_dir)
  record_output("table", out_paths$csv, glue::glue("5-bin table for a single horizon used in the plot: {horizon_for_cal}."))
  
  cat("\n--- TABLE: BA_11_price_bins_p_yes_by_horizon ---\n")
  print(bin_table_all_horizons)
  cat("\n--- TABLE: BA_12_price_bins_p_yes_single_horizon ---\n")
  print(bin_table_one)
  
  # Calibration-style plot for chosen horizon
  p_cal <- ggplot2::ggplot(bin_table_one, ggplot2::aes(x = mean_p, y = p_yes_empirical)) +
    ggplot2::geom_abline(intercept = 0, slope = 1, color = COL_GREY_2, linewidth = 0.8) +
    ggplot2::geom_point(color = COL_RED, size = 2) +
    ggplot2::geom_line(color = COL_RED) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(
      title = glue::glue("Empirical P(YES) by Polymarket price bins (horizon: {horizon_for_cal})"),
      x = "Mean Polymarket price in bin",
      y = "Empirical frequency of YES (mean(y))"
    ) +
    theme_corporate()
  
  plot_path <- save_plot_png(p_cal, paste0("BA_13_plot_p_yes_by_price_bins_", horizon_for_cal), out_dir, width = 7, height = 5)
  record_output("plot", plot_path, "Plot: empirical P(YES) by 0.2 price bins with 45° reference line.")
  
  # =============================================================================
  # 9) Optional: “Calibration regression” (scientific standard)
  #    logit(P(Y=1)) = a + b * logit(p)
  # =============================================================================
  cat("\n[9] Calibration regression (optional scientific diagnostic)...\n")
  
  eps <- 1e-6
  cal_df <- brier_panel %>%
    dplyr::mutate(
      p_clip = pmin(pmax(p_polymarket_yes, eps), 1 - eps),
      logit_p = log(p_clip / (1 - p_clip))
    )
  
  cal_logit <- stats::glm(
    y ~ logit_p,
    data = cal_df,
    family = stats::binomial(link = "logit")
  )
  
  cal_tidy <- tidy_clustered_glm(cal_logit, cal_df$market_id, "LOGIT_y_on_logitp_cluster_market", exp_coef = FALSE)
  
  out_paths <- write_table_triple(cal_tidy, "BA_14_calibration_logit_y_on_logitp", out_dir)
  record_output("table", out_paths$csv, "Calibration regression: logit(y) on logit(p), cluster-robust SE by market_id.")
  
  cat("\n--- TABLE: BA_14_calibration_logit_y_on_logitp ---\n")
  print(cal_tidy)
  
  add_rows_cal <- tibble::tibble(
    term = "McFadden R2",
    CAL_LOGIT = mcfadden_r2_safe(cal_logit)
  )
  
  print_modelsummary_table(
    models_named_list = list("CAL_LOGIT" = cal_logit),
    vcov_list = list("CAL_LOGIT" = vcov_cluster_safe(cal_logit, cal_df$market_id)),
    title = "--- REGRESSION TABLE: Calibration logit(y) ~ logit(p) (clustered SE by market_id) ---",
    add_rows = add_rows_cal,
    gof_omit = "R2|Adj|RMSE|F"
  )
  
  # =============================================================================
  # 10) Manifest + README + sessionInfo
  # =============================================================================
  cat("\n[10] Writing manifest + README + session info...\n")
  
  manifest <- manifest %>% dplyr::arrange(match(type, c("table", "plot", "doc")), file)
  out_paths <- write_table_triple(manifest, "BA_00_output_manifest", out_dir)
  record_output("table", out_paths$csv, "Manifest of outputs (Brier analysis script).")
  
  cat("\n--- TABLE: BA_00_output_manifest ---\n")
  print(manifest)
  
  readme_path <- file.path(out_dir, "logs", "README.md")
  readme_lines <- c(
    "# Brier score statistical analysis outputs",
    "",
    glue::glue("- Run timestamp: **{run_ts}**"),
    glue::glue("- Generated at: **{Sys.time()}**"),
    "- Script: `R/BS_BrierScore_Analysis.R`",
    "",
    "## What this script does",
    "- Uses precomputed Brier scores from `data/brier_scores/brier_scores_market_horizon.csv`.",
    "- Filters to **non-stale/usable** observations via `usable_polymarket == TRUE` (and `status == ok/usable` if present).",
    "- Excludes horizons: 4w, 3w, 2w.",
    "- Produces Brier score tables (mean ± 95% CI) overall and by horizon.",
    "- Computes Brier Skill Score (BSS) vs coinflip and historical base-rate benchmarks.",
    "- Builds a market-level correlation matrix (Pearson) including p-values.",
    "- Runs OLS regressions for Polymarket Brier loss (panel + market-level) with cluster-robust SE by market_id.",
    "- Runs Logit + Probit models for probability Polymarket prediction is correct (cluster-robust SE).",
    "- Computes 5-bin (width 0.2) empirical `P(YES | price bin)` tables and a calibration-style plot for one horizon.",
    "- Prints paper-style regression tables to console/log (modelsummary), including R^2 for OLS and McFadden R^2 for GLMs.",
    "",
    "## Outputs",
    "- Tables: CSV + JSONL + JSON (see `BA_00_output_manifest.csv`).",
    "- Plots: PNG.",
    "- Logs: this README + sessionInfo in `logs/`."
  )
  writeLines(readme_lines, readme_path)
  record_output("doc", readme_path, "README for Brier analysis outputs.")
  
  sess_path <- file.path(out_dir, "logs", paste0("sessionInfo_", run_ts, ".txt"))
  sink(sess_path); print(sessionInfo()); sink()
  record_output("doc", sess_path, "sessionInfo() snapshot (Brier analysis script).")
  
  cat("\n==================== BRIER ANALYSIS COMPLETE ====================\n")
  cat(glue::glue("Log: {log_path}\n"))
  cat(glue::glue("Out: {out_dir}\n"))
  cat("=================================================================\n\n")
  
  invisible(list(
    out_dir = out_dir,
    log_path = log_path,
    brier_panel = brier_panel,
    brier_market = brier_market
  ))
}

# =============================================================================
# Auto-run behavior
# - If executed via Rscript: run (sys.nframe() == 0)
# - If sourced in an interactive session (RStudio "Source"): run by default
#   (can be disabled by setting options(pm.run_brier_analysis_on_source = FALSE))
# =============================================================================
if (interactive()) {
  if (isTRUE(getOption("pm.run_brier_analysis_on_source", TRUE))) {
    run_brier_score_analysis()
  }
} else {
  if (sys.nframe() == 0) {
    run_brier_score_analysis()
  }
}
