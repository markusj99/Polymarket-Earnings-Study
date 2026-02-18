#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/Heckman_EventStudy.R
# Purpose: (2) Heckman two-step selection model + Event study/trading analysis
#          with robust inference and publication-style outputs.
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

script_path <- get_script_path()
start_dir   <- if (!is.na(script_path)) dirname(script_path) else getwd()
root_dir    <- find_project_root(start_dir)

# -----------------------------
# 0b) renv (best-effort)
# -----------------------------
if (file.exists(file.path(root_dir, "renv.lock"))) {
  if (!requireNamespace("renv", quietly = TRUE)) {
    install.packages("renv", repos = "https://cloud.r-project.org")
  }
  tryCatch({
    renv::load(project = root_dir)
    renv::restore(project = root_dir, prompt = FALSE)
  }, error = function(e) {
    message("WARNING: renv::restore() failed. You may need to run it manually.\nError: ", e$message)
  })
}

# -----------------------------
# 0c) Shared helpers
# -----------------------------
source(file.path(root_dir, "R", "utils", "pm_common.R"))
pm_load_packages()

# -----------------------------
# 0d) Output dirs + logging
# -----------------------------
out_dir <- file.path(root_dir, "statistics", "test_statistics", "econometrics")
fs::dir_create(out_dir)
fs::dir_create(file.path(out_dir, "logs"))

run_ts   <- format(Sys.time(), "%Y%m%dT%H%M%S")
log_path <- file.path(out_dir, "logs", paste0("HE_econ_run_", run_ts, ".log.txt"))

sink(log_path, split = TRUE)
on.exit(sink(), add = TRUE)

cat(glue::glue("Heckman/EventStudy run started: {Sys.time()}\n"))
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
# 1) Inputs
# =============================================================================
data_dir <- file.path(root_dir, "data")
paths <- list(
  markets      = file.path(data_dir, "markets", "markets.csv"),
  poly_prices  = file.path(data_dir, "poly_prices", "poly_prices_long.csv"),
  stock_prices = file.path(data_dir, "stock_prices", "stock_prices_daily.csv"),
  corporate    = file.path(data_dir, "corporate_info", "corporate_info.csv"),
  heck_comp    = file.path(data_dir, "heckman_selection_model", "heckman_universe_companies.csv"),
  heck_events  = file.path(data_dir, "heckman_selection_model", "heckman_universe_events.csv")
)

cat("Reading inputs...\n")
markets     <- janitor::clean_names(read_csv_required(paths$markets))
poly_prices <- janitor::clean_names(read_csv_required(paths$poly_prices))
stock_prices <- janitor::clean_names(read_csv_required(paths$stock_prices))
corporate   <- janitor::clean_names(read_csv_required(paths$corporate))
heck_comp   <- janitor::clean_names(read_csv_optional(paths$heck_comp))
heck_events <- janitor::clean_names(read_csv_optional(paths$heck_events))

# =============================================================================
# 2) Market sample (same filters as Brier script)
# =============================================================================
cat("\n[2] Preparing market sample...\n")

markets <- markets %>%
  dplyr::mutate(
    id = as.character(id),
    question = if ("question" %in% names(.)) as.character(question) else NA_character_,
    val_ric = if ("val_ric" %in% names(.)) normalize_ric(val_ric) else NA_character_,
    val_anchor_date = if ("val_anchor_date" %in% names(.)) parse_date_utc(val_anchor_date) else as.Date(NA),
    uma_end_date_utc = if ("uma_end_date" %in% names(.)) parse_ts_utc(uma_end_date) else as.POSIXct(NA),
    closed_time_utc  = if ("closed_time" %in% names(.)) parse_ts_utc(closed_time) else as.POSIXct(NA),
    updated_at_utc   = if ("updated_at" %in% names(.)) parse_ts_utc(updated_at) else as.POSIXct(NA),
    start_date_utc   = if ("start_date" %in% names(.)) parse_ts_utc(start_date) else as.POSIXct(NA),
    resolved_outcome_std = dplyr::case_when(
      "resolved_outcome" %in% names(.) & stringr::str_to_upper(resolved_outcome) %in% c("YES", "Y") ~ "YES",
      "resolved_outcome" %in% names(.) & stringr::str_to_upper(resolved_outcome) %in% c("NO", "N")  ~ "NO",
      TRUE ~ NA_character_
    ),
    volume_num    = if ("volume_num" %in% names(.)) safe_numeric(volume_num) else NA_real_,
    liquidity_num = if ("liquidity_num" %in% names(.)) safe_numeric(liquidity_num) else NA_real_,
    log_poly_volume = safe_log(volume_num),
    log_liquidity   = safe_log(liquidity_num),
    resolution_ts_utc = dplyr::coalesce(uma_end_date_utc, closed_time_utc, updated_at_utc),
    active_trading_hours = as.numeric(difftime(resolution_ts_utc, start_date_utc, units = "hours")),
    active_trading_hours = dplyr::if_else(is.finite(active_trading_hours), abs(active_trading_hours), NA_real_),
    val_yes_semantics = if ("val_yes_semantics" %in% names(.)) as.character(val_yes_semantics) else NA_character_
  )

cutoff_ts <- lubridate::with_tz(Sys.time(), tzone = "UTC") - lubridate::days(1)

markets_sample <- markets %>%
  dplyr::filter(resolved_outcome_std %in% c("YES", "NO")) %>%
  { if (all(c("val_ric", "val_anchor_date") %in% names(.))) dplyr::filter(., !is.na(val_ric), !is.na(val_anchor_date)) else . } %>%
  { if ("val_status" %in% names(.)) dplyr::filter(., !is.na(val_status), stringr::str_detect(val_status, "^MATCHED")) else . } %>%
  dplyr::filter(!is.na(resolution_ts_utc), resolution_ts_utc <= cutoff_ts) %>%
  dplyr::rename(market_id = id)

if (nrow(markets_sample) == 0) stop("No markets remain after sample filters.", call. = FALSE)

# =============================================================================
# 3) Non-stale snapshots (latest per cell)
# =============================================================================
cat("[3] Preparing non-stale Polymarket snapshots...\n")

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

prices_latest <- poly_prices %>%
  dplyr::filter(
    !is.na(market_id), !is.na(snapshot_label),
    !is.na(price_yes), !is.na(price_no),
    !is.na(src_yes_ts), !is.na(src_no_ts),
    is.finite(complement_error),
    complement_error <= complement_tolerance
  ) %>%
  dplyr::arrange(dplyr::desc(generated_utc), dplyr::desc(run_id)) %>%
  dplyr::group_by(market_id, snapshot_label) %>%
  dplyr::slice(1) %>%
  dplyr::ungroup()

if (nrow(prices_latest) == 0) stop("No valid non-stale snapshots remain.", call. = FALSE)

snapshot_levels <- prices_latest %>%
  dplyr::distinct(snapshot_label, snapshot_offset_seconds) %>%
  dplyr::arrange(dplyr::desc(snapshot_offset_seconds), snapshot_label) %>%
  dplyr::pull(snapshot_label) %>%
  unique()

# =============================================================================
# 4) Join + define BEAT + compute p_mkt + loss (needed for Heckman outcome)
# =============================================================================
cat("[4] Joining + computing p_mkt, y, and Brier loss...\n")

prices_sample <- prices_latest %>%
  dplyr::mutate(snapshot_label = factor(snapshot_label, levels = snapshot_levels)) %>%
  dplyr::inner_join(markets_sample, by = "market_id") %>%
  dplyr::mutate(
    y_yes = dplyr::if_else(resolved_outcome_std == "YES", 1, 0),
    yes_is_beat = purrr::pmap_lgl(list(val_yes_semantics, question), ~ yes_means_beat(..1, ..2)),
    y = dplyr::if_else(yes_is_beat, y_yes, 1 - y_yes),
    p_mkt = dplyr::if_else(yes_is_beat, price_yes, price_no),
    p_mkt = dplyr::if_else(p_mkt < 0, 0, dplyr::if_else(p_mkt > 1, 1, p_mkt)),
    loss_mkt = (p_mkt - y)^2
  )

# =============================================================================
# 5) Heckman two-step selection (requires universe files + 1d snapshot)
# =============================================================================
cat("\n[5] Heckman selection model...\n")

run_heckman <- (nrow(heck_comp) > 0 && nrow(heck_events) > 0 && any(as.character(prices_sample$snapshot_label) == "1d"))
if (!run_heckman) {
  cat("NOTE: Skipping Heckman (missing universe inputs OR missing 1d snapshot).\n")
} else {

  heck_events2 <- heck_events %>%
    dplyr::mutate(
      ric = if ("ric" %in% names(.)) normalize_ric(ric) else NA_character_,
      event_date = if ("event_date" %in% names(.)) parse_date_utc(event_date) else as.Date(NA)
    ) %>%
    dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
    dplyr::distinct(ric, event_date, .keep_all = TRUE)

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

  sample_events <- markets_sample %>%
    dplyr::transmute(ric = normalize_ric(val_ric), event_date = val_anchor_date) %>%
    dplyr::filter(!is.na(ric), !is.na(event_date)) %>%
    dplyr::distinct()

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
    dplyr::group_by(ric, event_date) %>%
    dplyr::summarise(
      loss_1d = mean(loss_1d, na.rm = TRUE),
      log_poly_volume = mean(log_poly_volume, na.rm = TRUE),
      log_liquidity   = mean(log_liquidity, na.rm = TRUE),
      active_trading_hours = mean(active_trading_hours, na.rm = TRUE),
      .groups = "drop"
    )

  sel_df <- heck_events2 %>%
    dplyr::left_join(heck_comp2, by = "ric") %>%
    dplyr::mutate(in_sample = as.integer(paste(ric, event_date) %in% paste(sample_events$ric, sample_events$event_date))) %>%
    dplyr::left_join(df_1d_event, by = c("ric", "event_date")) %>%
    dplyr::mutate(
      selected = as.integer(in_sample == 1 & is.finite(loss_1d)),
      gics_sector = as.factor(dplyr::coalesce(as.character(gics_sector), "Unknown"))
    )

  cat(glue::glue("Universe events: {nrow(sel_df)} | In-sample: {sum(sel_df$in_sample, na.rm=TRUE)} | Selected: {sum(sel_df$selected, na.rm=TRUE)}\n\n"))

  sel_model_df <- sel_df %>% tidyr::drop_na(selected, log_mcap, analysts_covering_latest, log_turnover, gics_sector)

  if (dplyr::n_distinct(sel_model_df$selected) < 2) {
    cat("NOTE: Selection variable has no variation; skipping Heckman.\n")
  } else {

    sel_terms <- c("log_mcap", "analysts_covering_latest", "log_turnover", "gics_sector")
    if (!factor_has_2plus(sel_model_df$gics_sector)) {
      sel_terms <- setdiff(sel_terms, "gics_sector")
      cat("NOTE: Dropping gics_sector in step 1 (only one level).\n")
    }
    if ("volatility_6m" %in% names(sel_model_df) && sum(is.finite(sel_model_df$volatility_6m)) >= 30) {
      sel_model_df <- sel_model_df %>% tidyr::drop_na(volatility_6m)
      sel_terms <- c(sel_terms, "volatility_6m")
    }

    sel_formula <- stats::as.formula(paste("selected ~", paste(sel_terms, collapse = " + ")))
    m_sel <- stats::glm(sel_formula, data = sel_model_df, family = binomial(link = "probit"))

    vc_sel <- sandwich::vcovHC(m_sel, type = "HC1")
    ct_sel <- lmtest::coeftest(m_sel, vcov. = vc_sel)
    sel_coef <- tidy_coeftest(ct_sel, "heckman_step1_probit_HC1")

    out_paths <- write_table_triple(sel_coef, "HE_01_heckman_step1_probit", out_dir)
    record_output("table", out_paths$csv, "Heckman step 1 probit (HC1) coefficients with CI + p-values.")

    # IMR
    eta <- stats::predict(m_sel, type = "link")
    Phi <- stats::pnorm(eta)
    phi <- stats::dnorm(eta)
    Phi_clip <- pmin(pmax(Phi, 1e-8), 1 - 1e-8)
    imr <- phi / Phi_clip

    sel_model_df <- sel_model_df %>% dplyr::mutate(eta = eta, imr = imr)

    sel_txt <- file.path(out_dir, "tables", "HE_01b_step1_summary.txt")
    sink(sel_txt); print(summary(m_sel)); sink()
    record_output("doc", sel_txt, "Heckman step 1 glm summary (TXT).")

    # Step 2 outcome regression on selected
    outcome_df <- sel_model_df %>%
      dplyr::filter(selected == 1, is.finite(loss_1d)) %>%
      tidyr::drop_na(log_mcap, analysts_covering_latest, log_poly_volume, log_liquidity, imr) %>%
      dplyr::mutate(gics_sector = as.factor(gics_sector))

    out_terms <- c("log_mcap", "analysts_covering_latest", "log_poly_volume", "log_liquidity",
                   "active_trading_hours", "gics_sector", "imr")
    if (!factor_has_2plus(outcome_df$gics_sector)) {
      out_terms <- setdiff(out_terms, "gics_sector")
      cat("NOTE: Dropping gics_sector in step 2 (only one level).\n")
    }
    if ("volatility_6m" %in% names(outcome_df) && sum(is.finite(outcome_df$volatility_6m)) >= 20) {
      outcome_df <- outcome_df %>% tidyr::drop_na(volatility_6m)
      out_terms <- c(out_terms, "volatility_6m")
    }

    out_formula <- stats::as.formula(paste("loss_1d ~", paste(out_terms, collapse = " + ")))
    m_out <- stats::lm(out_formula, data = outcome_df)

    vc2 <- sandwich::vcovHC(m_out, type = "HC1")
    ct2 <- lmtest::coeftest(m_out, vcov. = vc2)
    out_coef <- tidy_coeftest(ct2, "heckman_step2_outcome_HC1")

    out_paths <- write_table_triple(out_coef, "HE_02_heckman_step2_outcome_with_imr", out_dir)
    record_output("table", out_paths$csv, "Heckman step 2 OLS (HC1) with IMR: coefficients + CI + p-values.")

    out_txt <- file.path(out_dir, "tables", "HE_02b_step2_summary.txt")
    sink(out_txt); print(summary(m_out)); sink()
    record_output("doc", out_txt, "Heckman step 2 lm summary (TXT).")

    cat("\n--- TABLE: Heckman step 2 coefficients (HC1) ---\n")
    print(out_coef)
  }
}

# =============================================================================
# 6) Event study / trading interpretation (bins = 0.2) with significance
# =============================================================================
cat("\n[6] Event study: abnormal returns by Polymarket probability bins...\n")

run_event_study <- any(as.character(prices_sample$snapshot_label) == "1d") &&
  all(c("market_id", "offset_td", "close", "spx_close") %in% names(stock_prices))

if (!run_event_study) {
  cat("NOTE: Skipping event study (needs 1d snapshot + stock_prices_daily with market_id/offset_td/close/spx_close).\n")
} else {

  p_1d <- prices_sample %>%
    dplyr::filter(as.character(snapshot_label) == "1d") %>%
    dplyr::group_by(market_id) %>% dplyr::slice(1) %>% dplyr::ungroup() %>%
    dplyr::select(market_id, val_ric, val_anchor_date, p_mkt, y)

  stock_wide <- stock_prices %>%
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
    dplyr::group_by(market_id, offset_lab) %>%
    dplyr::summarise(
      close = mean(close, na.rm = TRUE),
      spx_close = mean(spx_close, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(
      names_from = offset_lab,
      values_from = c(close, spx_close),
      names_glue = "{.value}_{offset_lab}"
    )

  event_returns <- stock_wide %>%
    dplyr::mutate(
      ret_stock_m1_p1 = dplyr::if_else(is.finite(close_m1) & is.finite(close_p1) & close_m1 > 0 & close_p1 > 0, log(close_p1 / close_m1), NA_real_),
      ret_spx_m1_p1   = dplyr::if_else(is.finite(spx_close_m1) & is.finite(spx_close_p1) & spx_close_m1 > 0 & spx_close_p1 > 0, log(spx_close_p1 / spx_close_m1), NA_real_),
      abret_m1_p1 = ret_stock_m1_p1 - ret_spx_m1_p1
    ) %>%
    dplyr::select(market_id, abret_m1_p1)

  event_study_df <- p_1d %>%
    dplyr::left_join(event_returns, by = "market_id") %>%
    dplyr::filter(is.finite(p_mkt), p_mkt >= 0, p_mkt <= 1) %>%
    dplyr::mutate(p_bin = prob_bin_20pct(p_mkt))

  # Bin table with mean + 95% CI + p-values (H0: mean = 0)
  event_study_by_bin <- event_study_df %>%
    dplyr::group_by(p_bin) %>%
    dplyr::summarise(
      N = dplyr::n(),
      N_ret = sum(is.finite(abret_m1_p1)),
      n_markets = dplyr::n_distinct(market_id),
      mean_p = mean(p_mkt, na.rm = TRUE),
      realized_beat_rate = mean(y, na.rm = TRUE),

      mean_abret = mean(abret_m1_p1, na.rm = TRUE),
      sd_abret   = sd(abret_m1_p1, na.rm = TRUE),

      se_abret = dplyr::if_else(N_ret >= 2, sd_abret / sqrt(N_ret), NA_real_),
      t_abret  = dplyr::if_else(is.finite(se_abret) & se_abret > 0, mean_abret / se_abret, NA_real_),
      p_value  = dplyr::if_else(is.finite(t_abret), 2 * stats::pt(abs(t_abret), df = pmax(N_ret - 1, 1), lower.tail = FALSE), NA_real_),

      ci_low_95  = dplyr::if_else(
        N_ret >= 2,
        mean_abret - stats::qt(0.975, df = N_ret - 1) * se_abret,
        NA_real_
      ),
      ci_high_95 = dplyr::if_else(
        N_ret >= 2,
        mean_abret + stats::qt(0.975, df = N_ret - 1) * se_abret,
        NA_real_
      ),

      .groups = "drop"
    )

  out_paths <- write_table_triple(event_study_by_bin, "HE_03_event_study_abret_by_prob_bin_1d", out_dir)
  record_output("table", out_paths$csv, "Event study bin table: mean abnormal returns with CI + p-values.")

  cat("\n--- TABLE: Event study by probability bin (mean abnormal return, CI, p-value) ---\n")
  print(event_study_by_bin)

  # Plot (mean abnormal return by bin)
  p_ev <- ggplot2::ggplot(event_study_by_bin, ggplot2::aes(x = p_bin, y = mean_abret)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = COL_GREY_1) +
    ggplot2::geom_point(color = COL_RED, size = 2.5) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = ci_low_95, ymax = ci_high_95), width = 0.15, color = COL_GREY_2) +
    ggplot2::geom_line(ggplot2::aes(group = 1), color = COL_GREY_2) +
    ggplot2::labs(
      title = "Event study: mean market-adjusted return (t=-1 close to t=+1 close)\nby Polymarket P(BEAT) bin (1d snapshot)",
      x = "Polymarket implied probability bin",
      y = "Mean abnormal log return (stock - S&P 500) with 95% CI"
    ) +
    theme_corporate()

  plot_path <- save_plot_png(p_ev, "HE_04_event_study_plot_abret_by_bin_CI", out_dir, width = 10, height = 5)
  record_output("plot", plot_path, "Event study plot: mean abnormal return by bin with 95% CI.")

  # Continuous regression (abret ~ p_mkt) with firm-clustered SE where feasible
  ev_reg_df <- event_study_df %>%
    tidyr::drop_na(abret_m1_p1, p_mkt) %>%
    dplyr::mutate(ric = normalize_ric(val_ric))

  if (nrow(ev_reg_df) >= 30) {
    m_ev <- stats::lm(abret_m1_p1 ~ p_mkt, data = ev_reg_df)
    vc_ev <- vcov_cluster_or_hc(m_ev, cluster = ev_reg_df$ric, type = "HC1")
    ct_ev <- lmtest::coeftest(m_ev, vcov. = vc_ev)
    ev_coef <- tidy_coeftest(ct_ev, "event_study_abret_on_prob_1d_cluster_or_HC1")

    out_paths <- write_table_triple(ev_coef, "HE_05_event_study_reg_abret_on_prob_1d", out_dir)
    record_output("table", out_paths$csv, "Event study regression: abret ~ p_mkt with CI + p-values.")

    cat("\n--- TABLE: Event study regression (abret ~ p_mkt) ---\n")
    print(ev_coef)
  } else {
    cat("NOTE: Too few observations for continuous event-study regression.\n")
  }
}

# =============================================================================
# 7) Manifest + README + sessionInfo
# =============================================================================
cat("\n[7] Writing manifest + README + session info...\n")

manifest <- manifest %>% dplyr::arrange(match(type, c("table", "plot", "doc")), file)
out_paths <- write_table_triple(manifest, "HE_00_output_manifest", out_dir)
record_output("table", out_paths$csv, "Manifest of outputs (Heckman/EventStudy script).")

readme_path <- file.path(out_dir, "logs", "README.md")
readme_lines <- c(
  "# Heckman + Event study outputs",
  "",
  glue::glue("- Run timestamp: **{run_ts}**"),
  glue::glue("- Generated at: **{Sys.time()}**"),
  "- Script: `R/Heckman_EventStudy.R`",
  "",
  "## Key features",
  "- Same non-stale snapshot filtering as the Brier script (latest per market x snapshot).",
  "- Heckman two-step: probit selection (HC1) + outcome regression with IMR (HC1).",
  "- Event study: probability bins (width 0.2) and abnormal return tests with 95% CI + p-values.",
  "- Continuous regression: abret ~ p_mkt with cluster-robust SE where feasible.",
  "",
  "## Outputs",
  "- Tables: `tables/` (CSV + JSONL + JSON)",
  "- Plots:  `plots/` (PNG)",
  "- See `HE_00_output_manifest.csv` for full list."
)
writeLines(readme_lines, readme_path)
record_output("doc", readme_path, "README for Heckman/EventStudy script.")

sess_path <- file.path(out_dir, "logs", paste0("sessionInfo_", run_ts, ".txt"))
sink(sess_path); print(sessionInfo()); sink()
record_output("doc", sess_path, "sessionInfo() snapshot (Heckman/EventStudy script).")

cat("\n==================== ECON RUN COMPLETE ====================\n")
cat(glue::glue("Log: {log_path}\n"))
cat(glue::glue("Out: {out_dir}\n"))
cat("===========================================================\n\n")
