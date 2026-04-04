#!/usr/bin/env Rscript
# =============================================================================
# File:    R/statistics/event_study/run_polymarket_price_event_study.R
# Purpose: Conduct an earnings-announcement event study where the trading signal
#          is the Polymarket YES price observed no later than the purchase close.
#
# Main research question:
#   "What would have happened if we had used the Polymarket price as a signal
#    for which shares to purchase right before firms released earnings?"
#
# Methodological summary:
#   1. Load project data via R/utils/load_data.R.
#   2. Build one investable Polymarket signal per event using the latest
#      non-stale snapshot available by the relevant purchase close.
#   3. Bin the signal into five equal-width bins on [0, 1].
#   4. Estimate a standard market model for each event using an estimation
#      window of [-250, -30] trading days relative to the event trading day.
#   5. Compute abnormal returns and cumulative abnormal returns over the event
#      window [-5, +5] using the S&P 500 as benchmark.
#   6. Produce:
#        - event-level and bin-level output data (CSV + JSONL)
#        - CAAR plots with 95% confidence intervals
#        - regression-style HTML tables similar in spirit to the supplied
#          factor-analysis table
#
# Important timing convention:
#   The raw event timestamp is earnings_release_datetime.
#   Because the available prices are daily closes, the event is mapped to the
#   first close-to-close return that can reflect the earnings release:
#     - Before open release  -> t = 0 is the same trading day.
#     - After close release  -> t = 0 is the next trading day.
#     - During market hours  -> t = 0 is the same trading day, but the entry is
#                               still taken at the last close before release.
#   The purchase date is therefore the last trading day strictly before the
#   release for before-open/intraday releases, and the release-date trading
#   session itself for after-close releases.
#
# Outputs:
#   All output is written under:
#     statistics/event_study/
#
# Usage from project root:
#   source(file.path("R", "statistics", "event_study",
#                    "run_polymarket_price_event_study.R"))
#   results <- run_polymarket_event_study()
#
# Usage from another script:
#   source(file.path(ROOT, "R", "statistics", "event_study",
#                    "run_polymarket_price_event_study.R"))
#   results <- run_polymarket_event_study(root = ROOT,
#                                         event_window = -5:5,
#                                         estimation_window = c(-250, -30))
# =============================================================================

# -----------------------------------------------------------------------------
# Package checks
# -----------------------------------------------------------------------------
required_packages <- c(
  "readr",
  "dplyr",
  "ggplot2",
  "lubridate",
  "forcats",
  "tibble",
  "jsonlite",
  "sandwich",
  "lmtest",
  "gt",
  "scales"
)

check_required_packages <- function(pkgs = required_packages) {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  
  if (length(missing_pkgs) > 0) {
    stop(
      paste0(
        "This script requires the following R packages, which are not installed:\n",
        paste0(" - ", missing_pkgs, collapse = "\n"),
        "\n\nInstall them, for example, with:\n",
        "install.packages(c(",
        paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
        "))"
      ),
      call. = FALSE
    )
  }
}

# -----------------------------------------------------------------------------
# Reproducible paths
# -----------------------------------------------------------------------------
find_project_root <- function(root = NULL,
                              marker = file.path("R", "utils", "load_data.R"),
                              max_up = 10L) {
  if (!is.null(root)) {
    candidate <- normalizePath(root, winslash = "/", mustWork = FALSE)
    if (!file.exists(file.path(candidate, marker))) {
      stop(
        "The supplied root does not appear to be the project root. Missing: ",
        file.path(candidate, marker),
        call. = FALSE
      )
    }
    return(candidate)
  }
  
  current <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  candidates <- current
  
  for (i in seq_len(max_up)) {
    parent <- dirname(candidates[length(candidates)])
    if (identical(parent, candidates[length(candidates)])) {
      break
    }
    candidates <- c(candidates, parent)
  }
  
  hit <- candidates[vapply(
    candidates,
    function(x) file.exists(file.path(x, marker)),
    logical(1)
  )]
  
  if (length(hit) == 0L) {
    stop(
      paste0(
        "Could not locate the project root automatically.\n",
        "Start from the project root, a project subdirectory, or pass root = <path>.\n",
        "Expected to find: ", marker
      ),
      call. = FALSE
    )
  }
  
  normalizePath(hit[1], winslash = "/", mustWork = TRUE)
}

# -----------------------------------------------------------------------------
# Helpers for writing output
# -----------------------------------------------------------------------------
sanitize_for_json <- function(df) {
  out <- df
  
  for (nm in names(out)) {
    x <- out[[nm]]
    
    if (inherits(x, c("POSIXct", "POSIXt"))) {
      out[[nm]] <- format(x, "%Y-%m-%d %H:%M:%S", tz = "UTC")
    } else if (inherits(x, "Date")) {
      out[[nm]] <- as.character(x)
    } else if (is.factor(x)) {
      out[[nm]] <- as.character(x)
    } else if (is.numeric(x)) {
      x[is.infinite(x)] <- NA_real_
      out[[nm]] <- x
    }
  }
  
  out
}

write_jsonl <- function(df, path) {
  con <- file(path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  jsonlite::stream_out(sanitize_for_json(df), con = con, verbose = FALSE)
}

write_csv_jsonl <- function(df, stem) {
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(df, paste0(stem, ".csv"))
  write_jsonl(df, paste0(stem, ".jsonl"))
}

# -----------------------------------------------------------------------------
# Formatting helpers for regression-style tables
# -----------------------------------------------------------------------------
add_significance_stars <- function(estimate, p_value) {
  stars <- dplyr::case_when(
    is.na(p_value) ~ "",
    p_value < 0.01 ~ "***",
    p_value < 0.05 ~ "**",
    p_value < 0.10 ~ "*",
    TRUE ~ ""
  )
  
  paste0(sprintf("%.4f", estimate), stars)
}

format_parenthesized_se <- function(std_error) {
  ifelse(is.na(std_error), "", paste0("(", sprintf("%.4f", std_error), ")"))
}

make_gt_table <- function(df, title, subtitle) {
  gt::gt(df, rowname_col = "term") |>
    gt::tab_header(
      title = title,
      subtitle = subtitle
    ) |>
    gt::fmt_markdown(columns = gt::everything()) |>
    gt::cols_align(
      align = "center",
      columns = setdiff(names(df), "term")
    ) |>
    gt::tab_style(
      style = gt::cell_text(weight = "bold"),
      locations = gt::cells_title(groups = "title")
    ) |>
    gt::tab_options(
      table.font.size = gt::px(12),
      data_row.padding = gt::px(5),
      table.border.top.color = "#A8A8A8",
      table.border.bottom.color = "#A8A8A8",
      heading.align = "center"
    )
}

# -----------------------------------------------------------------------------
# Date-time parsing and event-time logic
# -----------------------------------------------------------------------------
parse_utc_datetime <- function(x) {
  x <- as.character(x)
  
  parsed <- suppressWarnings(lubridate::ymd_hms(x, tz = "UTC", quiet = TRUE))
  
  need_hm <- is.na(parsed) & !is.na(x)
  if (any(need_hm)) {
    parsed[need_hm] <- suppressWarnings(
      lubridate::ymd_hm(x[need_hm], tz = "UTC", quiet = TRUE)
    )
  }
  
  parsed
}

classify_release_timing <- function(release_dt_ny) {
  stopifnot(inherits(release_dt_ny, c("POSIXct", "POSIXt")))
  
  seconds <- lubridate::hour(release_dt_ny) * 3600 +
    lubridate::minute(release_dt_ny) * 60 +
    lubridate::second(release_dt_ny)
  
  dplyr::case_when(
    is.na(seconds) ~ NA_character_,
    seconds < (9 * 3600 + 30 * 60) ~ "before_open",
    seconds >= (16 * 3600) ~ "after_close",
    TRUE ~ "during_market"
  )
}

select_purchase_date <- function(trading_dates, release_date, release_timing) {
  trading_dates <- sort(unique(as.Date(trading_dates)))
  trading_dates <- trading_dates[!is.na(trading_dates)]
  
  if (length(trading_dates) == 0L || is.na(release_date) || is.na(release_timing)) {
    return(as.Date(NA))
  }
  
  if (release_timing == "after_close") {
    eligible <- trading_dates[trading_dates <= release_date]
  } else {
    eligible <- trading_dates[trading_dates < release_date]
  }
  
  if (length(eligible) == 0L) {
    return(as.Date(NA))
  }
  
  max(eligible)
}

select_event_trading_date <- function(trading_dates, purchase_date) {
  trading_dates <- sort(unique(as.Date(trading_dates)))
  trading_dates <- trading_dates[!is.na(trading_dates)]
  
  if (length(trading_dates) == 0L || is.na(purchase_date)) {
    return(as.Date(NA))
  }
  
  eligible <- trading_dates[trading_dates > purchase_date]
  if (length(eligible) == 0L) {
    return(as.Date(NA))
  }
  
  min(eligible)
}

# -----------------------------------------------------------------------------
# Event and signal preparation
# -----------------------------------------------------------------------------
prepare_event_master <- function(dataset_long,
                                 stock_prices,
                                 market_tz = "America/New_York") {
  dataset_long_prepped <- dataset_long |>
    dplyr::mutate(
      market_id = as.character(.data$id),
      release_dt_utc = parse_utc_datetime(.data$earnings_release_datetime)
    ) |>
    dplyr::mutate(
      release_dt_ny = lubridate::with_tz(.data$release_dt_utc, tzone = market_tz),
      release_date_ny = as.Date(.data$release_dt_ny),
      release_timing = classify_release_timing(.data$release_dt_ny)
    )
  
  prices_prepped <- stock_prices |>
    dplyr::mutate(
      market_id = as.character(.data$market_id),
      date = as.Date(.data$date)
    )
  
  price_date_map <- prices_prepped |>
    dplyr::distinct(.data$market_id, .data$date) |>
    dplyr::group_by(.data$market_id) |>
    dplyr::summarise(trading_dates = list(sort(unique(.data$date))), .groups = "drop")
  
  events <- dataset_long_prepped |>
    dplyr::distinct(
      .data$market_id,
      .data$ticker,
      .data$slug,
      .data$ric,
      .data$release_dt_utc,
      .data$release_dt_ny,
      .data$release_date_ny,
      .data$release_timing
    ) |>
    dplyr::left_join(price_date_map, by = "market_id") |>
    dplyr::rowwise() |>
    dplyr::mutate(
      purchase_date = select_purchase_date(
        trading_dates = .data$trading_dates,
        release_date = .data$release_date_ny,
        release_timing = .data$release_timing
      )
    ) |>
    dplyr::mutate(
      event_trading_date = select_event_trading_date(
        trading_dates = .data$trading_dates,
        purchase_date = .data$purchase_date
      )
    ) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      signal_cutoff_ny = as.POSIXct(
        ifelse(
          is.na(.data$purchase_date),
          NA_character_,
          paste0(format(.data$purchase_date, "%Y-%m-%d"), " 16:00:00")
        ),
        tz = market_tz
      ),
      signal_cutoff_utc = lubridate::with_tz(.data$signal_cutoff_ny, tzone = "UTC")
    ) |>
    dplyr::select(-trading_dates)
  
  list(
    dataset_long_prepped = dataset_long_prepped,
    prices_prepped = prices_prepped,
    events = events
  )
}

select_polymarket_signal <- function(dataset_long_prepped,
                                     events,
                                     allow_stale_signal = FALSE,
                                     market_tz = "America/New_York",
                                     signal_var = "p_polymarket_yes") {
  if (!signal_var %in% names(dataset_long_prepped)) {
    stop("Column not found in dataset_long: ", signal_var, call. = FALSE)
  }
  
  signal_rows <- dataset_long_prepped |>
    dplyr::mutate(
      snapshot_dt_utc_parsed = parse_utc_datetime(.data$snapshot_dt_utc),
      asof_date = as.Date(.data$asof_date),
      signal_value = .data[[signal_var]]
    ) |>
    dplyr::select(
      market_id,
      ticker,
      slug,
      ric,
      release_dt_utc,
      release_dt_ny,
      release_date_ny,
      release_timing,
      asof_date,
      snapshot_dt_utc_parsed,
      signal_value,
      dplyr::everything()
    ) |>
    dplyr::left_join(
      events |>
        dplyr::select(
          market_id,
          purchase_date,
          event_trading_date,
          signal_cutoff_utc
        ),
      by = "market_id"
    ) |>
    dplyr::mutate(
      snapshot_date_ny = dplyr::case_when(
        !is.na(.data$snapshot_dt_utc_parsed) ~ as.Date(
          lubridate::with_tz(.data$snapshot_dt_utc_parsed, tzone = market_tz)
        ),
        !is.na(.data$asof_date) ~ .data$asof_date,
        TRUE ~ as.Date(NA)
      ),
      observed_before_cutoff = dplyr::case_when(
        !is.na(.data$snapshot_dt_utc_parsed) & !is.na(.data$signal_cutoff_utc) ~
          .data$snapshot_dt_utc_parsed <= .data$signal_cutoff_utc,
        is.na(.data$snapshot_dt_utc_parsed) & !is.na(.data$asof_date) & !is.na(.data$purchase_date) ~
          .data$asof_date <= .data$purchase_date,
        TRUE ~ FALSE
      )
    ) |>
    dplyr::mutate(
      same_day_signal = !is.na(.data$snapshot_date_ny) &
        !is.na(.data$purchase_date) &
        .data$snapshot_date_ny == .data$purchase_date
    ) |>
    dplyr::filter(
      !is.na(.data$signal_value),
      dplyr::between(.data$signal_value, 0, 1),
      .data$observed_before_cutoff
    )
  
  if (!allow_stale_signal) {
    signal_rows <- signal_rows |>
      dplyr::filter(.data$same_day_signal)
  }
  
  selected <- signal_rows |>
    dplyr::arrange(
      .data$market_id,
      dplyr::desc(.data$snapshot_dt_utc_parsed),
      dplyr::desc(.data$asof_date)
    ) |>
    dplyr::group_by(.data$market_id) |>
    dplyr::slice(1L) |>
    dplyr::ungroup()
  
  selected
}

assign_signal_bins <- function(df,
                               breaks = seq(0, 1, by = 0.2),
                               labels = c("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")) {
  stopifnot(length(breaks) - 1L == length(labels))
  
  out <- df
  out$signal_bin <- cut(
    out$signal_value,
    breaks = breaks,
    include.lowest = TRUE,
    right = FALSE,
    labels = labels
  )
  
  # Ensure exactly 1.0 goes into the final bin.
  out$signal_bin[!is.na(out$signal_value) & out$signal_value == 1] <- tail(labels, 1)
  out$signal_bin <- forcats::fct_relevel(out$signal_bin, labels)
  out$signal_bin_id <- as.integer(out$signal_bin)
  out
}

# -----------------------------------------------------------------------------
# Price panel and event-study estimation
# -----------------------------------------------------------------------------
prepare_price_panel <- function(prices_prepped, selected_events) {
  panel <- prices_prepped |>
    dplyr::inner_join(
      selected_events |>
        dplyr::select(
          market_id,
          release_dt_utc,
          release_dt_ny,
          release_date_ny,
          release_timing,
          purchase_date,
          event_trading_date,
          signal_value,
          signal_bin,
          signal_bin_id
        ),
      by = "market_id"
    ) |>
    dplyr::arrange(.data$market_id, .data$date)
  
  panel_list <- split(panel, panel$market_id)
  panel_out <- vector("list", length(panel_list))
  
  for (i in seq_along(panel_list)) {
    df_i <- panel_list[[i]]
    df_i <- df_i[order(df_i$date), , drop = FALSE]
    
    idx0 <- match(df_i$event_trading_date[1], df_i$date)
    
    df_i$trading_index <- seq_len(nrow(df_i))
    df_i$stock_ret <- c(NA_real_, df_i$close[-1] / df_i$close[-nrow(df_i)] - 1)
    df_i$spx_ret <- c(NA_real_, df_i$spx_close[-1] / df_i$spx_close[-nrow(df_i)] - 1)
    df_i$event_day <- if (is.na(idx0)) NA_integer_ else df_i$trading_index - idx0
    
    panel_out[[i]] <- df_i
  }
  
  dplyr::bind_rows(panel_out)
}

fit_market_model_one_event <- function(df,
                                       event_window = -5:5,
                                       estimation_window = c(-250, -30),
                                       min_estimation_obs = 120L) {
  df <- df[order(df$date), , drop = FALSE]
  event_id <- df$market_id[1]
  
  template_info <- tibble::tibble(
    market_id = event_id,
    signal_bin = as.character(df$signal_bin[1]),
    signal_bin_id = df$signal_bin_id[1],
    signal_value = df$signal_value[1],
    release_dt_ny = df$release_dt_ny[1],
    purchase_date = df$purchase_date[1],
    event_trading_date = df$event_trading_date[1],
    est_n = NA_integer_,
    alpha = NA_real_,
    beta = NA_real_,
    sigma = NA_real_,
    status = "unknown"
  )
  
  if (all(is.na(df$event_day))) {
    template_info$status <- "missing_event_day"
    return(list(info = template_info, ar = tibble::tibble()))
  }
  
  estimation_df <- df |>
    dplyr::filter(
      !is.na(.data$event_day),
      .data$event_day >= estimation_window[1],
      .data$event_day <= estimation_window[2],
      is.finite(.data$stock_ret),
      is.finite(.data$spx_ret)
    )
  
  if (nrow(estimation_df) < min_estimation_obs) {
    template_info$est_n <- nrow(estimation_df)
    template_info$status <- "too_few_estimation_obs"
    return(list(info = template_info, ar = tibble::tibble()))
  }
  
  event_df <- df |>
    dplyr::filter(
      !is.na(.data$event_day),
      .data$event_day %in% event_window,
      is.finite(.data$stock_ret),
      is.finite(.data$spx_ret)
    ) |>
    dplyr::arrange(.data$event_day)
  
  if (nrow(event_df) != length(event_window) || !identical(event_df$event_day, event_window)) {
    template_info$est_n <- nrow(estimation_df)
    template_info$status <- "incomplete_event_window"
    return(list(info = template_info, ar = tibble::tibble()))
  }
  
  fit <- stats::lm(stock_ret ~ spx_ret, data = estimation_df)
  expected <- as.numeric(stats::predict(fit, newdata = event_df))
  
  event_df <- event_df |>
    dplyr::mutate(expected_ret = expected) |>
    dplyr::mutate(
      abnormal_ret = .data$stock_ret - .data$expected_ret,
      car_from_m5_raw = cumsum(.data$abnormal_ret)
    )
  
  baseline_m5 <- event_df$car_from_m5_raw[event_df$event_day == min(event_window)][1]
  
  event_df <- event_df |>
    dplyr::mutate(
      car_from_m5 = .data$car_from_m5_raw - baseline_m5
    )
  
  # CAR from t = 0 is zero before the position starts earning post-release returns.
  event_df$car_from_0 <- 0
  post_event_days <- event_df$event_day >= 0
  event_df$car_from_0[post_event_days] <- cumsum(event_df$abnormal_ret[post_event_days])
  
  model_sum <- summary(fit)
  template_info$est_n <- nrow(estimation_df)
  template_info$alpha <- unname(stats::coef(fit)[1])
  template_info$beta <- unname(stats::coef(fit)[2])
  template_info$sigma <- model_sum$sigma
  template_info$status <- "ok"
  
  list(info = template_info, ar = event_df)
}

estimate_all_events <- function(price_panel,
                                event_window = -5:5,
                                estimation_window = c(-250, -30),
                                min_estimation_obs = 120L) {
  by_event <- split(price_panel, price_panel$market_id)
  
  info_list <- vector("list", length(by_event))
  ar_list <- vector("list", length(by_event))
  
  for (i in seq_along(by_event)) {
    result_i <- fit_market_model_one_event(
      df = by_event[[i]],
      event_window = event_window,
      estimation_window = estimation_window,
      min_estimation_obs = min_estimation_obs
    )
    
    info_list[[i]] <- result_i$info
    ar_list[[i]] <- result_i$ar
  }
  
  list(
    model_info = dplyr::bind_rows(info_list),
    abnormal_returns = dplyr::bind_rows(ar_list)
  )
}

# -----------------------------------------------------------------------------
# Event-window summaries
# -----------------------------------------------------------------------------
compute_caar_by_bin <- function(abnormal_returns) {
  if (nrow(abnormal_returns) == 0L) {
    return(tibble::tibble())
  }
  
  event_level_paths <- abnormal_returns |>
    dplyr::select(
      market_id,
      signal_bin,
      signal_bin_id,
      signal_value,
      event_day,
      abnormal_ret,
      car_from_m5,
      car_from_0
    )
  
  event_level_paths |>
    dplyr::group_by(.data$signal_bin, .data$signal_bin_id, .data$event_day) |>
    dplyr::summarise(
      n = dplyr::n(),
      n_from_0 = sum(!is.na(.data$car_from_0)),
      mean_abnormal_ret = mean(.data$abnormal_ret, na.rm = TRUE),
      sd_abnormal_ret = stats::sd(.data$abnormal_ret, na.rm = TRUE),
      mean_caar_from_m5 = mean(.data$car_from_m5, na.rm = TRUE),
      sd_caar_from_m5 = stats::sd(.data$car_from_m5, na.rm = TRUE),
      mean_caar_from_0 = mean(.data$car_from_0, na.rm = TRUE),
      sd_caar_from_0 = stats::sd(.data$car_from_0, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      se_caar_from_m5 = .data$sd_caar_from_m5 / sqrt(.data$n),
      ci_low = .data$mean_caar_from_m5 - 1.96 * .data$se_caar_from_m5,
      ci_high = .data$mean_caar_from_m5 + 1.96 * .data$se_caar_from_m5,
      se_caar_from_0 = .data$sd_caar_from_0 / sqrt(.data$n_from_0),
      ci_low_from_0 = .data$mean_caar_from_0 - 1.96 * .data$se_caar_from_0,
      ci_high_from_0 = .data$mean_caar_from_0 + 1.96 * .data$se_caar_from_0
    ) |>
    dplyr::arrange(.data$signal_bin_id, .data$event_day)
}

window_specification <- function() {
  tibble::tribble(
    ~window_name,  ~start_day, ~end_day,
    "AR[0]",              0L,       0L,
    "CAR[0,1]",           0L,       1L,
    "CAR[0,3]",           0L,       3L,
    "CAR[0,5]",           0L,       5L,
    "CAR[-5,5]",         -5L,       5L
  )
}

compute_event_level_window_cars <- function(abnormal_returns,
                                            windows = window_specification()) {
  if (nrow(abnormal_returns) == 0L) {
    return(tibble::tibble())
  }
  
  out <- vector("list", nrow(windows))
  
  for (i in seq_len(nrow(windows))) {
    w <- windows[i, ]
    needed_days <- seq.int(w$start_day, w$end_day)
    
    out[[i]] <- abnormal_returns |>
      dplyr::filter(.data$event_day %in% needed_days) |>
      dplyr::group_by(
        .data$market_id,
        .data$ticker,
        .data$slug,
        .data$ric,
        .data$signal_bin,
        .data$signal_bin_id,
        .data$signal_value,
        .data$release_dt_ny,
        .data$purchase_date,
        .data$event_trading_date
      ) |>
      dplyr::summarise(
        n_days_observed = dplyr::n(),
        abnormal_return_raw = sum(.data$abnormal_ret, na.rm = FALSE),
        .groups = "drop"
      ) |>
      dplyr::mutate(
        abnormal_return = dplyr::if_else(
          .data$n_days_observed == length(needed_days),
          .data$abnormal_return_raw,
          NA_real_
        ),
        window_name = w$window_name,
        start_day = w$start_day,
        end_day = w$end_day
      ) |>
      dplyr::select(-abnormal_return_raw)
  }
  
  dplyr::bind_rows(out) |>
    dplyr::arrange(.data$signal_bin_id, .data$market_id, .data$start_day, .data$end_day)
}

# -----------------------------------------------------------------------------
# Regression helpers
# -----------------------------------------------------------------------------
robust_coeftest_to_df <- function(model) {
  vcov_hc3 <- sandwich::vcovHC(model, type = "HC3")
  ct <- lmtest::coeftest(model, vcov. = vcov_hc3)
  tibble::tibble(
    term = rownames(ct),
    estimate = unname(ct[, 1]),
    std_error = unname(ct[, 2]),
    statistic = unname(ct[, 3]),
    p_value = unname(ct[, 4])
  )
}

run_bin_mean_tests <- function(event_level_window_cars) {
  bins <- levels(event_level_window_cars$signal_bin)
  windows <- unique(event_level_window_cars$window_name)
  out <- list()
  idx <- 1L
  
  for (w in windows) {
    df_w <- event_level_window_cars |>
      dplyr::filter(.data$window_name == w)
    
    for (b in bins) {
      df_b <- df_w |>
        dplyr::filter(.data$signal_bin == b, !is.na(.data$abnormal_return))
      
      if (nrow(df_b) == 0L) {
        out[[idx]] <- tibble::tibble(
          window_name = w,
          bin = b,
          estimate = NA_real_,
          std_error = NA_real_,
          statistic = NA_real_,
          p_value = NA_real_,
          n = 0L
        )
      } else if (nrow(df_b) == 1L) {
        out[[idx]] <- tibble::tibble(
          window_name = w,
          bin = b,
          estimate = df_b$abnormal_return[1],
          std_error = NA_real_,
          statistic = NA_real_,
          p_value = NA_real_,
          n = 1L
        )
      } else {
        fit <- stats::lm(abnormal_return ~ 1, data = df_b)
        ct <- robust_coeftest_to_df(fit)
        out[[idx]] <- tibble::tibble(
          window_name = w,
          bin = b,
          estimate = ct$estimate[1],
          std_error = ct$std_error[1],
          statistic = ct$statistic[1],
          p_value = ct$p_value[1],
          n = nrow(df_b)
        )
      }
      
      idx <- idx + 1L
    }
  }
  
  dplyr::bind_rows(out)
}

run_bin_difference_models <- function(event_level_window_cars,
                                      reference_bin = "0.0-0.2") {
  windows <- unique(event_level_window_cars$window_name)
  out <- vector("list", length(windows))
  
  for (i in seq_along(windows)) {
    w <- windows[i]
    df_w <- event_level_window_cars |>
      dplyr::filter(.data$window_name == w, !is.na(.data$abnormal_return)) |>
      dplyr::mutate(signal_bin = stats::relevel(.data$signal_bin, ref = reference_bin))
    
    if (nrow(df_w) == 0L) {
      out[[i]] <- tibble::tibble(
        window_name = w,
        term = character(),
        estimate = numeric(),
        std_error = numeric(),
        statistic = numeric(),
        p_value = numeric(),
        n = integer(),
        r_squared = numeric()
      )
      next
    }
    
    fit <- stats::lm(abnormal_return ~ signal_bin, data = df_w)
    tidy_fit <- robust_coeftest_to_df(fit) |>
      dplyr::mutate(
        window_name = w,
        n = stats::nobs(fit),
        r_squared = summary(fit)$r.squared
      )
    
    out[[i]] <- tidy_fit
  }
  
  dplyr::bind_rows(out)
}

build_mean_test_table <- function(mean_test_results,
                                  window_order = c("AR[0]", "CAR[0,1]", "CAR[0,3]", "CAR[0,5]", "CAR[-5,5]")) {
  bins <- unique(as.character(mean_test_results$bin))
  rows <- c(rbind(bins, rep("", length(bins))))
  
  table_df <- tibble::tibble(term = rows)
  
  for (w in window_order) {
    df_w <- mean_test_results |>
      dplyr::filter(.data$window_name == w) |>
      dplyr::mutate(bin = as.character(.data$bin))
    
    col_values <- character(length(rows))
    ptr <- 1L
    
    for (b in bins) {
      row_b <- df_w[df_w$bin == b, , drop = FALSE]
      if (nrow(row_b) == 0L) {
        col_values[ptr] <- ""
        col_values[ptr + 1L] <- ""
      } else {
        col_values[ptr] <- add_significance_stars(row_b$estimate[1], row_b$p_value[1])
        col_values[ptr + 1L] <- format_parenthesized_se(row_b$std_error[1])
      }
      ptr <- ptr + 2L
    }
    
    table_df[[w]] <- col_values
  }
  
  table_df
}

build_difference_table <- function(diff_results,
                                   window_order = c("AR[0]", "CAR[0,1]", "CAR[0,3]", "CAR[0,5]", "CAR[-5,5]")) {
  term_map <- c(
    "(Intercept)" = "Constant (0.0-0.2 bin)",
    "signal_bin0.2-0.4" = "0.2-0.4 minus 0.0-0.2",
    "signal_bin0.4-0.6" = "0.4-0.6 minus 0.0-0.2",
    "signal_bin0.6-0.8" = "0.6-0.8 minus 0.0-0.2",
    "signal_bin0.8-1.0" = "0.8-1.0 minus 0.0-0.2"
  )
  
  row_terms <- unname(c(rbind(term_map, rep("", length(term_map)))))
  row_terms <- c(row_terms, "N", "R-squared")
  table_df <- tibble::tibble(term = row_terms)
  
  for (w in window_order) {
    df_w <- diff_results |>
      dplyr::filter(.data$window_name == w)
    
    term_sequence <- names(term_map)
    col_values <- character(length(row_terms))
    ptr <- 1L
    
    for (tm in term_sequence) {
      row_tm <- df_w[df_w$term == tm, , drop = FALSE]
      if (nrow(row_tm) == 0L) {
        col_values[ptr] <- ""
        col_values[ptr + 1L] <- ""
      } else {
        col_values[ptr] <- add_significance_stars(row_tm$estimate[1], row_tm$p_value[1])
        col_values[ptr + 1L] <- format_parenthesized_se(row_tm$std_error[1])
      }
      ptr <- ptr + 2L
    }
    
    if (nrow(df_w) > 0L) {
      col_values[length(row_terms) - 1L] <- as.character(df_w$n[1])
      col_values[length(row_terms)] <- sprintf("%.4f", df_w$r_squared[1])
    } else {
      col_values[length(row_terms) - 1L] <- ""
      col_values[length(row_terms)] <- ""
    }
    
    table_df[[w]] <- col_values
  }
  
  table_df
}

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plot_caar_individual_bins <- function(
    caar_by_bin,
    output_dir,
    colors = c("#808080", "#A9A9A9", "#E3170A", "#00008B", "#0000FF")
) {
  if (nrow(caar_by_bin) == 0L) {
    return(invisible(NULL))
  }
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  bin_levels <- levels(caar_by_bin$signal_bin)
  names(colors) <- bin_levels
  
  plots <- vector("list", length(bin_levels))
  names(plots) <- bin_levels
  
  for (b in bin_levels) {
    df_b <- caar_by_bin |>
      dplyr::filter(.data$signal_bin == b) |>
      dplyr::arrange(.data$event_day)
    
    if (nrow(df_b) == 0L) {
      next
    }
    
    y_min_raw <- min(df_b$ci_low, df_b$mean_caar_from_m5, 0, na.rm = TRUE)
    y_max_raw <- max(df_b$ci_high, df_b$mean_caar_from_m5, 0, na.rm = TRUE)
    
    y_span <- y_max_raw - y_min_raw
    if (!is.finite(y_span) || y_span <= 0) {
      y_span <- 0.01
    }
    
    y_pad <- max(0.0025, 0.10 * y_span)
    y_limits <- c(y_min_raw - y_pad, y_max_raw + y_pad)
    
    p <- ggplot2::ggplot(
      df_b,
      ggplot2::aes(
        x = .data$event_day,
        y = .data$mean_caar_from_m5
      )
    ) +
      ggplot2::annotate(
        "rect",
        xmin = -5,
        xmax = 5,
        ymin = -Inf,
        ymax = Inf,
        alpha = 0.03,
        fill = "#808080"
      ) +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$ci_low, ymax = .data$ci_high),
        fill = colors[[b]],
        alpha = 0.18,
        color = NA
      ) +
      ggplot2::geom_hline(yintercept = 0, linetype = "solid", linewidth = 0.3) +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
      ggplot2::geom_line(color = colors[[b]], linewidth = 0.9) +
      ggplot2::geom_point(color = colors[[b]], size = 1.5) +
      ggplot2::scale_x_continuous(breaks = -5:5) +
      ggplot2::scale_y_continuous(
        limits = y_limits,
        labels = scales::percent_format(accuracy = 0.1)
      ) +
      ggplot2::labs(
        title = paste0("Cumulative abnormal returns: Polymarket bin ", b),
        subtitle = paste(
          "Market-model abnormal returns relative to the S&P 500.",
          "t = 0 is the first daily return that can reflect the earnings release."
        ),
        x = "Event day",
        y = "CAAR from t = -5",
        caption = paste(
          "Shaded area shows the event window [-5, +5].",
          "Dashed vertical line marks t = 0.",
          "Ribbon is the 95% confidence interval."
        )
      ) +
      ggplot2::theme_minimal(base_size = 11) +
      ggplot2::theme(
        panel.grid.minor = ggplot2::element_blank()
      )
    
    safe_bin <- gsub("[^0-9\\.\\-]", "_", b)
    
    ggplot2::ggsave(
      filename = file.path(output_dir, paste0("caar_bin_", safe_bin, ".png")),
      plot = p,
      width = 8,
      height = 5,
      dpi = 300
    )
    
    ggplot2::ggsave(
      filename = file.path(output_dir, paste0("caar_bin_", safe_bin, ".pdf")),
      plot = p,
      width = 8,
      height = 5
    )
    
    plots[[b]] <- p
  }
  
  invisible(plots)
}

plot_caar_faceted <- function(caar_by_bin,
                              output_png,
                              output_pdf,
                              colors = c("#808080", "#A9A9A9", "#E3170A", "#00008B", "#0000FF")) {
  if (nrow(caar_by_bin) == 0L) {
    return(invisible(NULL))
  }
  
  bin_levels <- levels(caar_by_bin$signal_bin)
  names(colors) <- bin_levels
  
  p <- ggplot2::ggplot(
    caar_by_bin,
    ggplot2::aes(
      x = .data$event_day,
      y = .data$mean_caar_from_m5,
      group = .data$signal_bin,
      color = .data$signal_bin,
      fill = .data$signal_bin
    )
  ) +
    ggplot2::annotate(
      "rect",
      xmin = -5,
      xmax = 5,
      ymin = -Inf,
      ymax = Inf,
      alpha = 0.03,
      fill = "#808080"
    ) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = .data$ci_low, ymax = .data$ci_high),
      alpha = 0.18,
      color = NA
    ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "solid", linewidth = 0.3) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::geom_point(size = 1.2) +
    ggplot2::facet_wrap(~ signal_bin, ncol = 2, scales = "fixed") +
    ggplot2::scale_color_manual(values = colors, drop = FALSE) +
    ggplot2::scale_fill_manual(values = colors, drop = FALSE) +
    ggplot2::scale_x_continuous(breaks = -5:5) +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    ggplot2::labs(
      title = "Cumulative abnormal returns by Polymarket price bin",
      subtitle = paste(
        "Market-model abnormal returns relative to the S&P 500.",
        "t = 0 is the first daily return that can reflect the earnings release.",
        sep = " "
      ),
      x = "Event day",
      y = "CAAR from t = -5",
      color = "Polymarket bin",
      fill = "Polymarket bin",
      caption = "Shaded area shows the event window [-5, +5]. Dashed vertical line marks t = 0. Ribbons are 95% confidence intervals."
    ) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      legend.position = "none",
      panel.grid.minor = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold")
    )
  
  ggplot2::ggsave(output_png, plot = p, width = 10, height = 8, dpi = 300)
  ggplot2::ggsave(output_pdf, plot = p, width = 10, height = 8)
  invisible(p)
}

plot_caar_combined <- function(caar_by_bin,
                               output_png,
                               output_pdf,
                               colors = c("#808080", "#A9A9A9", "#E3170A", "#00008B", "#0000FF")) {
  if (nrow(caar_by_bin) == 0L) {
    return(invisible(NULL))
  }
  
  bin_levels <- levels(caar_by_bin$signal_bin)
  names(colors) <- bin_levels
  
  p <- ggplot2::ggplot(
    caar_by_bin,
    ggplot2::aes(
      x = .data$event_day,
      y = .data$mean_caar_from_m5,
      color = .data$signal_bin,
      fill = .data$signal_bin
    )
  ) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = .data$ci_low, ymax = .data$ci_high),
      alpha = 0.10,
      color = NA
    ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "solid", linewidth = 0.3) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    ggplot2::geom_line(linewidth = 0.9) +
    ggplot2::geom_point(size = 1.3) +
    ggplot2::scale_color_manual(values = colors, drop = FALSE) +
    ggplot2::scale_fill_manual(values = colors, drop = FALSE) +
    ggplot2::scale_x_continuous(breaks = -5:5) +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    ggplot2::labs(
      title = "CAAR by Polymarket price bin",
      subtitle = "Combined view across all five equal-width Polymarket bins.",
      x = "Event day",
      y = "CAAR from t = -5",
      color = "Polymarket bin",
      fill = "Polymarket bin",
      caption = "Dashed vertical line marks t = 0. Ribbons are 95% confidence intervals."
    ) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom"
    )
  
  ggplot2::ggsave(output_png, plot = p, width = 10, height = 6, dpi = 300)
  ggplot2::ggsave(output_pdf, plot = p, width = 10, height = 6)
  invisible(p)
}

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
run_polymarket_event_study <- function(root = NULL,
                                       output_dir = NULL,
                                       event_window = -5:5,
                                       estimation_window = c(-250, -30),
                                       min_estimation_obs = 120L,
                                       allow_stale_signal = FALSE,
                                       signal_var = "p_polymarket_yes",
                                       market_tz = "America/New_York",
                                       color_palette = c("#808080", "#A9A9A9", "#E3170A", "#00008B", "#0000FF")) {
  check_required_packages()
  
  ROOT <- find_project_root(root)
  source(file.path(ROOT, "R", "utils", "load_data.R"))
  
  if (is.null(output_dir)) {
    output_dir <- file.path(ROOT, "statistics", "event_study")
  }
  
  data_dir <- file.path(output_dir, "data")
  table_dir <- file.path(output_dir, "tables")
  figure_dir <- file.path(output_dir, "figures")
  
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  
  D <- load_project_data(ROOT)
  prep <- prepare_event_master(
    dataset_long = D$dataset_long,
    stock_prices = D$stock_prices,
    market_tz = market_tz
  )
  
  selected_signals <- select_polymarket_signal(
    dataset_long_prepped = prep$dataset_long_prepped,
    events = prep$events,
    allow_stale_signal = allow_stale_signal,
    market_tz = market_tz,
    signal_var = signal_var
  ) |>
    assign_signal_bins()
  
  price_panel <- prepare_price_panel(
    prices_prepped = prep$prices_prepped,
    selected_events = selected_signals
  )
  
  estimated <- estimate_all_events(
    price_panel = price_panel,
    event_window = event_window,
    estimation_window = estimation_window,
    min_estimation_obs = min_estimation_obs
  )
  
  final_events <- selected_signals |>
    dplyr::inner_join(
      estimated$model_info |>
        dplyr::filter(.data$status == "ok") |>
        dplyr::select(
          market_id,
          est_n,
          alpha,
          beta,
          sigma,
          status
        ),
      by = "market_id"
    )
  
  abnormal_returns <- estimated$abnormal_returns |>
    dplyr::semi_join(final_events |> dplyr::select(market_id), by = "market_id")
  
  caar_by_bin <- compute_caar_by_bin(abnormal_returns)
  event_level_window_cars <- compute_event_level_window_cars(abnormal_returns)
  mean_test_results <- run_bin_mean_tests(event_level_window_cars)
  diff_results <- run_bin_difference_models(event_level_window_cars)
  
  # ---------------------------------------------------------------------------
  # Diagnostics and sample-flow output
  # ---------------------------------------------------------------------------
  sample_flow <- tibble::tibble(
    step = c(
      "Unique events in dataset_long",
      "Events with matching daily price history",
      "Events with a derived purchase date and event trading date",
      if (allow_stale_signal) {
        "Events with a Polymarket signal observed by the purchase cutoff"
      } else {
        "Events with a non-stale same-day Polymarket signal by the purchase cutoff"
      },
      "Events with sufficient estimation data and a complete [-5, +5] event window"
    ),
    n = c(
      dplyr::n_distinct(prep$dataset_long_prepped$market_id),
      dplyr::n_distinct(intersect(prep$dataset_long_prepped$market_id, prep$prices_prepped$market_id)),
      sum(!is.na(prep$events$purchase_date) & !is.na(prep$events$event_trading_date)),
      dplyr::n_distinct(selected_signals$market_id),
      dplyr::n_distinct(final_events$market_id)
    )
  )
  
  # ---------------------------------------------------------------------------
  # Write tabular outputs (CSV + JSONL)
  # ---------------------------------------------------------------------------
  write_csv_jsonl(sample_flow, file.path(data_dir, "sample_flow"))
  write_csv_jsonl(prep$events, file.path(data_dir, "event_master"))
  write_csv_jsonl(selected_signals, file.path(data_dir, "selected_signals"))
  write_csv_jsonl(estimated$model_info, file.path(data_dir, "event_model_diagnostics"))
  write_csv_jsonl(abnormal_returns, file.path(data_dir, "abnormal_returns_event_window"))
  write_csv_jsonl(caar_by_bin, file.path(data_dir, "caar_by_bin"))
  write_csv_jsonl(event_level_window_cars, file.path(data_dir, "event_level_window_cars"))
  write_csv_jsonl(mean_test_results, file.path(data_dir, "bin_mean_tests"))
  write_csv_jsonl(diff_results, file.path(data_dir, "bin_difference_models"))
  
  # ---------------------------------------------------------------------------
  # Regression-style HTML tables
  # ---------------------------------------------------------------------------
  mean_table_df <- build_mean_test_table(mean_test_results)
  mean_gt <- make_gt_table(
    mean_table_df,
    title = "Abnormal returns by Polymarket price bin",
    subtitle = paste(
      "Cell entries are mean abnormal returns estimated with an event-study market model.",
      "Rows are the five equal-width Polymarket bins. Standard errors are shown in parentheses.",
      "Signals are selected at the last eligible pre-release close, using non-stale data by default."
    )
  )
  
  diff_table_df <- build_difference_table(diff_results)
  diff_gt <- make_gt_table(
    diff_table_df,
    title = "Cross-bin differences in abnormal returns: OLS",
    subtitle = paste(
      "Cross-sectional OLS with one row per event. Dependent variable is the event-level abnormal return",
      "for the stated horizon. The intercept is the 0.0-0.2 Polymarket bin and other coefficients are",
      "differences relative to that low-signal benchmark. HC3 robust standard errors are in parentheses."
    )
  )
  
  gt::gtsave(mean_gt, file.path(table_dir, "event_study_bin_mean_table.html"))
  gt::gtsave(diff_gt, file.path(table_dir, "event_study_bin_difference_table.html"))
  
  # Also save the display-ready table data in CSV/JSONL for reproducibility.
  write_csv_jsonl(mean_table_df, file.path(data_dir, "event_study_bin_mean_table_display"))
  write_csv_jsonl(diff_table_df, file.path(data_dir, "event_study_bin_difference_table_display"))
  
  # ---------------------------------------------------------------------------
  # Figures
  # ---------------------------------------------------------------------------
  plot_caar_faceted(
    caar_by_bin = caar_by_bin,
    output_png = file.path(figure_dir, "caar_by_bin_faceted.png"),
    output_pdf = file.path(figure_dir, "caar_by_bin_faceted.pdf"),
    colors = color_palette
  )
  
  plot_caar_combined(
    caar_by_bin = caar_by_bin,
    output_png = file.path(figure_dir, "caar_by_bin_combined.png"),
    output_pdf = file.path(figure_dir, "caar_by_bin_combined.pdf"),
    colors = color_palette
  )
  
  plot_caar_individual_bins(
    caar_by_bin = caar_by_bin,
    output_dir = file.path(figure_dir, "individual_bins"),
    colors = color_palette
  )
  
  # ---------------------------------------------------------------------------
  # Session info for reproducibility
  # ---------------------------------------------------------------------------
  utils::capture.output(sessionInfo(), file = file.path(output_dir, "session_info.txt"))
  
  invisible(list(
    root = ROOT,
    output_dir = output_dir,
    sample_flow = sample_flow,
    selected_signals = selected_signals,
    model_info = estimated$model_info,
    abnormal_returns = abnormal_returns,
    caar_by_bin = caar_by_bin,
    event_level_window_cars = event_level_window_cars,
    mean_test_results = mean_test_results,
    diff_results = diff_results
  ))
}

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if (sys.nframe() == 0L || interactive()) {
  ROOT <- "C:/Users/lasts/Desktop/Polymarket/Polymarket-Earnings-Study"
  results <- run_polymarket_event_study(root = ROOT)
  message(
    "Event study completed. Outputs saved under: ",
    file.path(ROOT, "statistics", "event_study")
  )
}