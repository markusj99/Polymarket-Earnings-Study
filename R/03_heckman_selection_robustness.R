#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/04_heckman_selection_robustness.R
# Purpose: Heckman sample-selection robustness analysis for Polymarket accuracy.
#
# Research question
# -----------------
# Are the cross-sectional determinants of Polymarket accuracy materially changed
# once we correct for the possibility that the observed Polymarket event sample
# is non-randomly selected from a broader universe of earnings events?
#
# Methodological choice
# ---------------------
# This script follows the classic two-step Heckman selection model described in
# the Chapter 1 handout:
#   1. Estimate a probit selection equation on the full universe of earnings
#      events.
#   2. Compute the inverse Mills ratio (IMR) from the first-stage probit.
#   3. Estimate the outcome equation on the selected sample, including the IMR.
#
# In implementation we use the standard R package 'sampleSelection' with the
# two-step estimator. This keeps the estimation transparent and close to the
# handout while using the package's purpose-built variance-covariance matrix for
# the tobit-2 / Heckman model.
#
# Data inputs (loaded via R/utils/load_data.R)
# --------------------------------------------
#   - data/complete_dataset_long.csv
#   - data/stock_prices/stock_prices_daily.csv     [loaded but not used here]
#   - data/heckman_selection_model/heckman_universe_events.csv
#   - data/heckman_selection_model/heckman_universe_companies.csv [optional crosswalk]
#
# Design
# ------
# Unit of observation: earnings event.
#
# Selection indicators
# --------------------
#   (1) selected_mean:
#       = 1 if a Polymarket event can be matched to the universe event and a
#         mean Polymarket Brier loss can be computed across the selected
#         horizons.
#   (2) selected_target:
#       = 1 if a Polymarket event can be matched to the universe event and a
#         Polymarket Brier loss is available at the target horizon.
#
# Outcome variables
# -----------------
#   (1) loss_mean   = mean Polymarket Brier loss across selected_horizons
#   (2) loss_target = Polymarket Brier loss at target_horizon
#
# Baseline ex-ante regressors (available for the full universe)
# -------------------------------------------------------------
#   - log_market_cap         : log market capitalization
#   - analysts               : number of analysts covering the stock
#   - log_turnover           : log 6m average daily turnover
#   - log_volatility_6m      : log 6m stock volatility
#
# Exclusion restrictions used only in the selection equation
# ----------------------------------------------------------
#   - release_timing_group   : before open / during market / after close / unknown
#   - exchange_country_group : grouped exchange country
#   - gics_sector_group      : grouped GICS sector
#
# These exclusion restrictions should be justified in the thesis text. The idea
# is that they can affect whether Polymarket chooses to list or support an
# earnings event, while the conditional accuracy of an existing Polymarket market
# is already controlled for by the outcome-equation covariates above.
#
# Outputs
# -------
# All outputs are written under:
#   statistics/heckman_selection/
#
# Machine-readable outputs:
#   - heckman_analysis_panel.csv / .jsonl
#   - heckman_matching_summary.csv / .jsonl
#   - heckman_market_match_diagnostics.csv / .jsonl
#   - heckman_selection_coefficients.csv / .jsonl
#   - heckman_outcome_coefficients.csv / .jsonl
#   - heckman_model_fit.csv / .jsonl
#
# Pretty tables:
#   - tables/table_selection_equations.html
#   - tables/table_outcome_mean_loss.html
#   - tables/table_outcome_target_loss.html
#   - tables/table_model_fit.html
#
# Usage
# -----
# Interactive / RStudio:
#   source(file.path("R", "04_heckman_selection_robustness.R"))
#
# From another script:
#   options(polymarket.autorun = FALSE)
#   source(file.path(ROOT, "R", "04_heckman_selection_robustness.R"))
#   results <- run_heckman_selection_robustness(
#     root = ROOT,
#     selected_horizons = c("4d", "3d", "2d", "1d", "12h", "6h"),
#     target_horizon = "6h"
#   )
# =============================================================================

options(stringsAsFactors = FALSE, scipen = 999)

# -----------------------------------------------------------------------------
# Package checks
# -----------------------------------------------------------------------------
required_packages <- c(
  "readr",
  "dplyr",
  "tidyr",
  "purrr",
  "stringr",
  "tibble",
  "jsonlite",
  "gt",
  "lmtest",
  "sandwich",
  "sampleSelection"
)

check_required_packages <- function(pkgs = required_packages) {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]

  if (length(missing_pkgs) > 0L) {
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
get_this_script_path <- function() {
  # Works for Rscript
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, cmd_args)
  if (length(match) > 0L) {
    return(normalizePath(sub(file_arg, "", cmd_args[match[1]]), winslash = "/", mustWork = FALSE))
  }
  
  # Works when sourced in many cases
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, winslash = "/", mustWork = FALSE))
  }
  
  # Fallback
  NULL
}

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
  
  start_points <- character()
  
  script_path <- get_this_script_path()
  if (!is.null(script_path) && nzchar(script_path)) {
    start_points <- c(start_points, dirname(script_path))
  }
  
  start_points <- c(start_points, normalizePath(getwd(), winslash = "/", mustWork = TRUE))
  start_points <- unique(start_points)
  
  for (start in start_points) {
    candidates <- start
    for (i in seq_len(max_up)) {
      parent <- dirname(candidates[length(candidates)])
      if (identical(parent, candidates[length(candidates)])) break
      candidates <- c(candidates, parent)
    }
    
    hit <- candidates[vapply(
      candidates,
      function(x) file.exists(file.path(x, marker)),
      logical(1)
    )]
    
    if (length(hit) > 0L) {
      return(normalizePath(hit[1], winslash = "/", mustWork = TRUE))
    }
  }
  
  stop(
    paste0(
      "Could not locate the project root automatically.\n",
      "Start from the project root, a project subdirectory, or pass root = <path>.\n",
      "Expected to find: ", marker
    ),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
write_jsonl <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  con <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)

  if (nrow(df) == 0L) {
    return(invisible(NULL))
  }

  for (i in seq_len(nrow(df))) {
    row_list <- lapply(df[i, , drop = FALSE], function(col) {
      value <- col[[1]]
      if (inherits(value, "Date")) {
        return(as.character(value))
      }
      if (inherits(value, c("POSIXct", "POSIXt"))) {
        return(format(value, "%Y-%m-%d %H:%M:%S", tz = "UTC"))
      }
      value
    })

    line <- jsonlite::toJSON(
      row_list,
      auto_unbox = TRUE,
      na = "null",
      null = "null",
      digits = NA
    )
    writeLines(line, con = con)
  }

  invisible(NULL)
}

write_csv_jsonl <- function(df, stem) {
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(df, paste0(stem, ".csv"))
  write_jsonl(df, paste0(stem, ".jsonl"))
}

first_non_missing <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0L) return(NA)
  x[1]
}

safe_num <- function(x) {
  suppressWarnings(as.numeric(x))
}

safe_log <- function(x) {
  x <- safe_num(x)
  ifelse(is.finite(x) & x > 0, log(x), NA_real_)
}

safe_log1p <- function(x) {
  x <- safe_num(x)
  ifelse(is.finite(x) & x >= 0, log1p(x), NA_real_)
}

parse_event_date <- function(x) {
  x_chr <- as.character(x)
  x_chr <- substr(x_chr, 1L, 10L)
  suppressWarnings(as.Date(x_chr))
}

parse_datetime_utc <- function(x, tz = "UTC") {
  x_chr <- stringr::str_trim(as.character(x))
  x_chr[x_chr %in% c("", "NA", "NaN", "NULL", "null")] <- NA_character_

  out <- rep(as.POSIXct(NA_real_, origin = "1970-01-01", tz = tz), length(x_chr))

  formats <- c(
    "%Y-%m-%dT%H:%M:%OSZ",
    "%Y-%m-%dT%H:%M:%OS",
    "%Y-%m-%d %H:%M:%OS",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S"
  )

  remaining <- is.na(out) & !is.na(x_chr)
  for (fmt in formats) {
    if (!any(remaining)) break
    parsed <- suppressWarnings(as.POSIXct(x_chr[remaining], format = fmt, tz = tz))
    out[which(remaining)] <- parsed
    remaining <- is.na(out) & !is.na(x_chr)
  }

  out
}

combine_date_time_utc <- function(date_x, time_x, tz = "UTC") {
  date_chr <- as.character(parse_event_date(date_x))
  time_chr <- stringr::str_trim(as.character(time_x))
  time_chr[time_chr %in% c("", "NA", "NaN", "NULL", "null")] <- NA_character_

  has_both <- !is.na(date_chr) & !is.na(time_chr)
  out <- rep(as.POSIXct(NA_real_, origin = "1970-01-01", tz = tz), length(date_chr))

  if (any(has_both)) {
    out[has_both] <- parse_datetime_utc(
      paste(date_chr[has_both], time_chr[has_both]),
      tz = tz
    )
  }

  out
}

normalize_ric <- function(x) {
  x_chr <- stringr::str_trim(stringr::str_to_upper(as.character(x)))
  dplyr::na_if(x_chr, "")
}

normalize_ticker <- function(x) {
  x_chr <- stringr::str_trim(stringr::str_to_upper(as.character(x)))
  dplyr::na_if(x_chr, "")
}

extract_quarter_from_text <- function(x) {
  x_chr <- stringr::str_to_upper(as.character(x))
  x_chr[is.na(x_chr)] <- ""

  out <- rep(NA_character_, length(x_chr))
  m1 <- stringr::str_match(x_chr, "\\bQ([1-4])\\b")
  has_m1 <- !is.na(m1[, 2])
  out[has_m1] <- paste0("Q", m1[has_m1, 2])

  m2 <- stringr::str_match(x_chr, "\\bQUARTER\\s*([1-4])\\b")
  has_m2 <- is.na(out) & !is.na(m2[, 2])
  out[has_m2] <- paste0("Q", m2[has_m2, 2])

  out
}

classify_release_timing <- function(x) {
  x_chr <- stringr::str_trim(as.character(x))
  is_missing <- is.na(x_chr) | !nzchar(x_chr)

  hh <- suppressWarnings(as.integer(substr(x_chr, 1L, 2L)))
  mm <- suppressWarnings(as.integer(substr(x_chr, 4L, 5L)))
  ss <- suppressWarnings(as.integer(substr(x_chr, 7L, 8L)))
  total_seconds <- hh * 3600 + mm * 60 + ss

  out <- dplyr::case_when(
    is_missing ~ "unknown",
    !is.finite(total_seconds) ~ "unknown",
    total_seconds < (9 * 3600 + 30 * 60) ~ "before_open",
    total_seconds >= (16 * 3600) ~ "after_close",
    TRUE ~ "during_market"
  )

  factor(out, levels = c("before_open", "during_market", "after_close", "unknown"))
}

lump_small_levels <- function(x, min_n = 25L, other_label = "Other / sparse") {
  x_chr <- as.character(x)
  x_chr[is.na(x_chr) | !nzchar(stringr::str_trim(x_chr))] <- "Unknown"

  counts <- table(x_chr)
  small <- names(counts)[counts < min_n]
  x_chr[x_chr %in% small] <- other_label

  factor(x_chr)
}

p_to_stars <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.01 ~ "***",
    p < 0.05 ~ "**",
    p < 0.10 ~ "*",
    TRUE ~ ""
  )
}

fmt_num <- function(x, digits = 4) {
  ifelse(is.na(x), "", sprintf(paste0("%.", digits, "f"), x))
}

fmt_p <- function(x) {
  dplyr::case_when(
    is.na(x) ~ "",
    x < 0.001 ~ "<0.001",
    TRUE ~ sprintf("%.3f", x)
  )
}

fmt_ci <- function(lo, hi, digits = 4) {
  dplyr::if_else(
    is.na(lo) | is.na(hi),
    "",
    sprintf(paste0("[%.", digits, "f, %.", digits, "f]"), lo, hi)
  )
}

make_gt_table <- function(df, title, subtitle = NULL, note = NULL) {
  g <- gt::gt(df) %>%
    gt::tab_header(title = title, subtitle = subtitle) %>%
    gt::fmt_markdown(columns = gt::everything()) %>%
    gt::opt_row_striping() %>%
    gt::tab_options(
      table.font.size = gt::px(12),
      data_row.padding = gt::px(6),
      heading.align = "center"
    )

  if (!is.null(note)) {
    g <- g %>% gt::tab_source_note(note)
  }

  g
}

save_gt_html <- function(gt_obj, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  gt::gtsave(gt_obj, filename = path)
}

tidy_glm_hc3 <- function(model, model_label, component_label) {
  ct <- lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC3"))
  est <- as.numeric(ct[, 1])
  se <- as.numeric(ct[, 2])
  stat <- as.numeric(ct[, 3])
  pval <- as.numeric(ct[, 4])
  crit <- stats::qnorm(0.975)

  tibble::tibble(
    model = model_label,
    component = component_label,
    estimator = "Probit",
    term = rownames(ct),
    estimate = est,
    std_error = se,
    statistic = stat,
    p_value = pval,
    conf_low = est - crit * se,
    conf_high = est + crit * se,
    stars = p_to_stars(pval)
  )
}

tidy_lm_hc3 <- function(model, model_label, component_label, estimator_label) {
  ct <- lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC3"))
  est <- as.numeric(ct[, 1])
  se <- as.numeric(ct[, 2])
  stat <- as.numeric(ct[, 3])
  pval <- as.numeric(ct[, 4])
  crit <- stats::qt(0.975, df = model$df.residual)

  tibble::tibble(
    model = model_label,
    component = component_label,
    estimator = estimator_label,
    term = rownames(ct),
    estimate = est,
    std_error = se,
    statistic = stat,
    p_value = pval,
    conf_low = est - crit * se,
    conf_high = est + crit * se,
    stars = p_to_stars(pval)
  )
}

tidy_heckman_outcome <- function(heckman_fit, naive_fit, model_label, component_label) {
  est <- stats::coef(heckman_fit$lm)
  term_names_raw <- names(est)
  
  normalize_heckman_term <- function(x) {
    x <- sub("^XO", "", x)
    x <- sub("^imrData\\$IMR1$", "invMillsRatio", x)
    x
  }
  
  term_names_clean <- vapply(term_names_raw, normalize_heckman_term, character(1))
  
  vc <- heckman_fit$vcov
  if (is.null(dim(vc))) {
    stop("Heckman vcov is not a matrix.", call. = FALSE)
  }
  
  vc_names <- rownames(vc)
  if (is.null(vc_names)) {
    stop("Heckman vcov does not have row names.", call. = FALSE)
  }
  
  # Keep the last occurrence of duplicated vcov names, which correspond to the
  # outcome-equation block plus invMillsRatio, sigma, and rho in 2-step output.
  vc_name_positions <- seq_along(vc_names)
  vc_last_pos <- tapply(vc_name_positions, vc_names, max)
  vc_keep_pos <- as.integer(vc_last_pos)
  vc_outcome <- vc[vc_keep_pos, vc_keep_pos, drop = FALSE]
  vc_outcome_names <- rownames(vc_outcome)
  
  common_terms <- intersect(term_names_clean, vc_outcome_names)
  
  if (length(common_terms) == 0L) {
    stop(
      paste0(
        "No overlap between normalized outcome coefficient names and Heckman vcov names.\n",
        "Raw outcome terms: ", paste(term_names_raw, collapse = ", "), "\n",
        "Normalized outcome terms: ", paste(term_names_clean, collapse = ", "), "\n",
        "Outcome vcov names: ", paste(vc_outcome_names, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  
  missing_terms <- setdiff(term_names_clean, common_terms)
  if (length(missing_terms) > 0L) {
    message(
      "Dropping Heckman outcome terms without matching vcov entries: ",
      paste(missing_terms, collapse = ", ")
    )
  }
  
  est_common <- est[match(common_terms, term_names_clean)]
  names(est_common) <- common_terms
  
  vc_sub <- vc_outcome[common_terms, common_terms, drop = FALSE]
  se <- sqrt(diag(vc_sub))
  z_stat <- est_common / se
  pval <- 2 * stats::pnorm(abs(z_stat), lower.tail = FALSE)
  crit <- stats::qnorm(0.975)
  
  naive_terms <- names(stats::coef(naive_fit))
  imr_candidates <- setdiff(common_terms, naive_terms)
  
  tibble::tibble(
    model = model_label,
    component = component_label,
    estimator = "Heckman 2-step",
    term = common_terms,
    estimate = as.numeric(est_common),
    std_error = as.numeric(se),
    statistic = as.numeric(z_stat),
    p_value = as.numeric(pval),
    conf_low = as.numeric(est_common - crit * se),
    conf_high = as.numeric(est_common + crit * se),
    stars = p_to_stars(pval),
    is_imr = common_terms %in% imr_candidates
  )
}

clean_term_label <- function(term) {
  dplyr::case_when(
    term == "(Intercept)" ~ "Constant",
    term == "log_market_cap" ~ "log(Market cap)",
    term == "analysts" ~ "Analysts covering",
    term == "log_turnover" ~ "log(6m avg. daily turnover + 1)",
    term == "log_volatility_6m" ~ "log(6m stock volatility + 1)",
    grepl("release_timing_group", term, fixed = TRUE) ~
      paste0("Release timing: ", gsub("release_timing_group", "", term, fixed = TRUE)),
    grepl("exchange_country_group", term, fixed = TRUE) ~
      paste0("Exchange country: ", gsub("exchange_country_group", "", term, fixed = TRUE)),
    grepl("gics_sector_group", term, fixed = TRUE) ~
      paste0("Sector: ", gsub("gics_sector_group", "", term, fixed = TRUE)),
    grepl("invMillsRatio", term, ignore.case = TRUE) ~ "Lambda (inverse Mills ratio)",
    grepl("IMR", term, ignore.case = TRUE) ~ "Lambda (inverse Mills ratio)",
    TRUE ~ term
  )
}

selection_fit_stats <- function(model, selected, model_label, component_label) {
  model_df <- stats::model.frame(model)
  
  null_model <- stats::glm(
    formula = stats::update(stats::formula(model), . ~ 1),
    data = model_df,
    family = stats::binomial(link = "probit")
  )
  
  loglik_full <- as.numeric(stats::logLik(model))
  loglik_null <- as.numeric(stats::logLik(null_model))
  pseudo_r2 <- if (is.finite(loglik_full) && is.finite(loglik_null) && loglik_null != 0) {
    1 - (loglik_full / loglik_null)
  } else {
    NA_real_
  }
  
  phat <- stats::fitted(model)
  threshold <- mean(selected, na.rm = TRUE)
  
  pcp <- mean(
    (phat > threshold & selected == 1) |
      (phat < threshold & selected == 0),
    na.rm = TRUE
  )
  
  tibble::tibble(
    model = model_label,
    component = component_label,
    n_total = nrow(model_df),
    n_selected = sum(selected == 1, na.rm = TRUE),
    selection_rate = mean(selected == 1, na.rm = TRUE),
    logLik = loglik_full,
    logLik_null = loglik_null,
    pseudo_r2 = pseudo_r2,
    pcp = pcp
  )
}

build_polymarket_market_registry <- function(dataset_long,
                                            selected_horizons,
                                            target_horizon,
                                            require_complete_panel_for_mean = FALSE,
                                            polymarket_event_tz = "UTC") {
  volume_col <- if ("volume_num" %in% names(dataset_long)) {
    "volume_num"
  } else if ("volumeNum" %in% names(dataset_long)) {
    "volumeNum"
  } else {
    stop("Neither 'volume_num' nor 'volumeNum' exists in dataset_long.", call. = FALSE)
  }

  available_horizons <- sort(unique(as.character(dataset_long$horizon)))
  missing_horizons <- setdiff(selected_horizons, available_horizons)

  if (length(missing_horizons) > 0L) {
    stop(
      paste0(
        "These selected_horizons were not found in dataset_long: ",
        paste(missing_horizons, collapse = ", "),
        "\nAvailable horizons are: ",
        paste(available_horizons, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  d <- dataset_long %>%
    dplyr::mutate(
      market_id = as.character(.data$id),
      ric = normalize_ric(.data$ric),
      ticker = normalize_ticker(.data$ticker),
      slug = dplyr::na_if(stringr::str_trim(as.character(.data$slug)), ""),
      event_datetime_utc = parse_datetime_utc(.data$earnings_release_datetime, tz = polymarket_event_tz),
      event_date = as.Date(.data$event_datetime_utc),
      event_quarter = extract_quarter_from_text(.data$slug),
      horizon = as.character(.data$horizon),
      status = tolower(trimws(as.character(.data$status))),
      loss_polymarket = safe_num(.data$loss_polymarket),
      p_polymarket_yes = safe_num(.data$p_polymarket_yes),
      volume_num = safe_num(.data[[volume_col]]),
      market_cap_usd_asof = safe_num(.data$market_cap_usd_asof),
      analysts_covering_asof = safe_num(.data$analysts_covering_asof),
      turnover_6m_avg_daily_volume = safe_num(.data$turnover_6m_avg_daily_volume),
      volatility_6m = safe_num(.data$volatility_6m)
    ) %>%
    dplyr::filter(.data$horizon %in% selected_horizons)

  market_level <- d %>%
    dplyr::group_by(.data$market_id) %>%
    dplyr::summarise(
      ric = first_non_missing(.data$ric),
      ticker = first_non_missing(.data$ticker),
      slug = first_non_missing(.data$slug),
      event_datetime_utc = first_non_missing(.data$event_datetime_utc),
      event_date = as.Date(first_non_missing(.data$event_datetime_utc)),
      event_quarter = first_non_missing(.data$event_quarter),
      n_rows_selected_horizons = dplyr::n_distinct(.data$horizon),
      n_snapshots_ok = dplyr::n_distinct(
        .data$horizon[
          .data$status == "ok" &
            !is.na(.data$loss_polymarket) &
            is.finite(.data$loss_polymarket)
        ]
      ),
      any_ok_snapshot = as.integer(any(
        .data$status == "ok" &
          !is.na(.data$loss_polymarket) &
          is.finite(.data$loss_polymarket)
      )),
      loss_mean = {
        z <- .data$loss_polymarket[
          .data$status == "ok" &
            !is.na(.data$loss_polymarket) &
            is.finite(.data$loss_polymarket)
        ]
        if (length(z) == 0L) NA_real_ else mean(z, na.rm = TRUE)
      },
      loss_target = {
        z <- .data$loss_polymarket[
          .data$horizon == target_horizon &
            .data$status == "ok" &
            !is.na(.data$loss_polymarket) &
            is.finite(.data$loss_polymarket)
        ]
        if (length(z) == 0L) NA_real_ else z[1]
      },
      volume_num = first_non_missing(.data$volume_num),
      market_cap_usd_asof = first_non_missing(.data$market_cap_usd_asof),
      analysts_covering_asof = first_non_missing(.data$analysts_covering_asof),
      turnover_6m_avg_daily_volume = first_non_missing(.data$turnover_6m_avg_daily_volume),
      volatility_6m = first_non_missing(.data$volatility_6m),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      loss_mean = dplyr::if_else(
        require_complete_panel_for_mean & .data$n_snapshots_ok < length(selected_horizons),
        NA_real_,
        .data$loss_mean
      ),
      selected_mean = as.integer(!is.na(.data$loss_mean)),
      selected_target = as.integer(!is.na(.data$loss_target)),
      log_market_cap = safe_log(.data$market_cap_usd_asof),
      analysts = .data$analysts_covering_asof,
      log_turnover = safe_log1p(.data$turnover_6m_avg_daily_volume),
      log_volatility_6m = safe_log1p(.data$volatility_6m),
      log_volume = safe_log1p(.data$volume_num),
      approx_event_key = paste(.data$ric, .data$event_date, sep = "__")
    )

  market_level
}

build_universe_events <- function(heckman_universe_events,
                                  min_level_n = 25L,
                                  universe_event_tz = "UTC") {
  u <- heckman_universe_events %>%
    dplyr::mutate(
      ric = normalize_ric(.data$ric),
      ticker = normalize_ticker(.data$ticker),
      event_date = parse_event_date(.data$event_date),
      event_time = as.character(.data$event_time),
      event_datetime_utc = combine_date_time_utc(.data$event_date, .data$event_time, tz = universe_event_tz),
      event_quarter = extract_quarter_from_text(.data$event_title),
      market_cap_usd_asof_evt = safe_num(.data$market_cap_usd_asof_evt),
      analysts_covering_asof_evt = safe_num(.data$analysts_covering_asof_evt),
      turnover_lookback_avg_daily_volume_asof_evt = safe_num(.data$turnover_lookback_avg_daily_volume_asof_evt),
      volatility_lookback_asof_evt = safe_num(.data$volatility_lookback_asof_evt)
    ) %>%
    dplyr::filter(!is.na(.data$ric), !is.na(.data$event_date)) %>%
    dplyr::mutate(
      event_key = paste(.data$ric, .data$event_date, sep = "__"),
      log_market_cap = safe_log(.data$market_cap_usd_asof_evt),
      analysts = .data$analysts_covering_asof_evt,
      log_turnover = safe_log1p(.data$turnover_lookback_avg_daily_volume_asof_evt),
      log_volatility_6m = safe_log1p(.data$volatility_lookback_asof_evt),
      release_timing_group = classify_release_timing(.data$event_time),
      exchange_country_group = lump_small_levels(.data$exchange_country, min_n = min_level_n),
      gics_sector_group = lump_small_levels(.data$gics_sector, min_n = min_level_n)
    ) %>%
    dplyr::arrange(.data$event_key)

  u %>%
    dplyr::group_by(.data$event_key) %>%
    dplyr::mutate(n_universe_rows_same_event = dplyr::n()) %>%
    dplyr::slice(1L) %>%
    dplyr::ungroup()
}

build_ticker_crosswalk <- function(universe_events, universe_companies = NULL) {
  candidates <- universe_events %>%
    dplyr::transmute(
      ticker = normalize_ticker(.data$ticker),
      ric = normalize_ric(.data$ric)
    )

  if (!is.null(universe_companies) && nrow(universe_companies) > 0L) {
    companies_xwalk <- universe_companies %>%
      dplyr::transmute(
        ticker = normalize_ticker(.data$ticker),
        ric = normalize_ric(.data$ric)
      )
    candidates <- dplyr::bind_rows(candidates, companies_xwalk)
  }

  candidates %>%
    dplyr::filter(!is.na(.data$ticker), !is.na(.data$ric)) %>%
    dplyr::distinct() %>%
    dplyr::add_count(.data$ticker, name = "n_ric_per_ticker") %>%
    dplyr::filter(.data$n_ric_per_ticker == 1L) %>%
    dplyr::select(.data$ticker, .data$ric)
}

match_polymarket_to_universe <- function(markets,
                                         universe_events,
                                         universe_companies = NULL,
                                         max_time_diff_hours = 18,
                                         use_unique_ticker_crosswalk = TRUE) {
  if (nrow(markets) == 0L) {
    stop("No Polymarket market rows were supplied to the matching function.", call. = FALSE)
  }
  
  # ---------------------------------------------------------------------------
  # Prepare normalized copies
  # ---------------------------------------------------------------------------
  markets_aug <- markets %>%
    dplyr::mutate(
      market_row_id = dplyr::row_number(),
      ticker_norm = normalize_ticker(.data$ticker)
    )
  
  universe_aug <- universe_events %>%
    dplyr::mutate(
      ticker_norm = normalize_ticker(.data$ticker)
    )
  
  # ---------------------------------------------------------------------------
  # Candidate set 1: exact RIC-based linkage
  # ---------------------------------------------------------------------------
  ric_candidates <- markets_aug %>%
    dplyr::filter(!is.na(.data$ric)) %>%
    dplyr::inner_join(
      universe_aug,
      by = "ric",
      suffix = c("_market", "_universe")
    ) %>%
    dplyr::mutate(
      identifier_method = "ric_exact",
      matched_universe_ric = .data$ric
    )
  
  # ---------------------------------------------------------------------------
  # Candidate set 2: ticker -> unique RIC crosswalk fallback
  #
  # Important change relative to your current script:
  # We do NOT restrict this to markets with zero RIC candidates.
  # Instead, we generate these candidates for all markets with a usable ticker,
  # and let the ranking stage prefer RIC matches first.
  # ---------------------------------------------------------------------------
  ticker_candidates <- tibble::tibble()
  
  if (isTRUE(use_unique_ticker_crosswalk)) {
    ticker_xwalk <- build_ticker_crosswalk(
      universe_events = universe_aug,
      universe_companies = universe_companies
    ) %>%
      dplyr::rename(ric_candidate = .data$ric)
    
    universe_for_ticker <- universe_aug %>%
      dplyr::rename(ric_candidate = .data$ric)
    
    ticker_candidates <- markets_aug %>%
      dplyr::filter(!is.na(.data$ticker_norm)) %>%
      dplyr::inner_join(
        ticker_xwalk,
        by = c("ticker_norm" = "ticker")
      ) %>%
      dplyr::mutate(
        market_ric_present = !is.na(.data$ric),
        market_ric_matches_ticker_ric = !is.na(.data$ric) & .data$ric == .data$ric_candidate
      ) %>%
      dplyr::inner_join(
        universe_for_ticker,
        by = "ric_candidate",
        suffix = c("_market", "_universe")
      ) %>%
      dplyr::mutate(
        identifier_method = "ticker_to_unique_ric",
        matched_universe_ric = .data$ric_candidate
      )
  }
  
  # ---------------------------------------------------------------------------
  # Combine all candidate sets
  # ---------------------------------------------------------------------------
  all_candidates <- dplyr::bind_rows(
    ric_candidates,
    ticker_candidates
  ) %>%
    dplyr::distinct(
      .data$market_row_id,
      .data$event_key,
      .data$identifier_method,
      .keep_all = TRUE
    )
  
  # ---------------------------------------------------------------------------
  # Score and filter candidates
  # ---------------------------------------------------------------------------
  if (nrow(all_candidates) > 0L) {
    all_candidates <- all_candidates %>%
      dplyr::mutate(
        date_diff_days = abs(as.numeric(difftime(
          .data$event_date_market,
          .data$event_date_universe,
          units = "days"
        ))),
        
        time_diff_hours = dplyr::if_else(
          !is.na(.data$event_datetime_utc_market) & !is.na(.data$event_datetime_utc_universe),
          abs(as.numeric(difftime(
            .data$event_datetime_utc_market,
            .data$event_datetime_utc_universe,
            units = "hours"
          ))),
          NA_real_
        ),
        
        quarter_consistent = dplyr::case_when(
          is.na(.data$event_quarter_market) | is.na(.data$event_quarter_universe) ~ TRUE,
          .data$event_quarter_market == .data$event_quarter_universe ~ TRUE,
          TRUE ~ FALSE
        ),
        
        tier_exact_date = .data$date_diff_days == 0,
        tier_time_window = !is.na(.data$time_diff_hours) &
          .data$time_diff_hours <= max_time_diff_hours &
          .data$quarter_consistent,
        tier_adjacent_date = .data$date_diff_days <= 1 & .data$quarter_consistent,
        
        # Conservative acceptance rule:
        # - RIC matches can pass exact date, time-window, or adjacent-date rules
        # - ticker fallback must also satisfy quarter consistency
        accepted_match = dplyr::case_when(
          .data$identifier_method == "ric_exact" ~
            .data$tier_exact_date | .data$tier_time_window | .data$tier_adjacent_date,
          
          .data$identifier_method == "ticker_to_unique_ric" ~
            (.data$tier_exact_date | .data$tier_time_window | .data$tier_adjacent_date) &
            .data$quarter_consistent,
          
          TRUE ~ FALSE
        ),
        
        match_tier = dplyr::case_when(
          .data$tier_exact_date ~ 1L,
          .data$tier_time_window ~ 2L,
          .data$tier_adjacent_date ~ 3L,
          TRUE ~ 99L
        ),
        
        match_rule = dplyr::case_when(
          .data$tier_exact_date & .data$identifier_method == "ric_exact" ~
            "exact_ric_plus_event_date",
          
          .data$tier_exact_date & .data$identifier_method == "ticker_to_unique_ric" ~
            "exact_ticker_crosswalk_plus_event_date",
          
          .data$tier_time_window & .data$identifier_method == "ric_exact" ~
            paste0("ric_nearest_event_datetime_within_", max_time_diff_hours, "h"),
          
          .data$tier_time_window & .data$identifier_method == "ticker_to_unique_ric" ~
            paste0("ticker_crosswalk_nearest_event_datetime_within_", max_time_diff_hours, "h"),
          
          .data$tier_adjacent_date & .data$identifier_method == "ric_exact" ~
            "ric_adjacent_date_with_quarter_check",
          
          .data$tier_adjacent_date & .data$identifier_method == "ticker_to_unique_ric" ~
            "ticker_crosswalk_adjacent_date_with_quarter_check",
          
          TRUE ~ "rejected"
        ),
        
        time_diff_hours_for_rank = dplyr::coalesce(.data$time_diff_hours, Inf),
        
        identifier_rank = dplyr::case_when(
          .data$identifier_method == "ric_exact" ~ 1L,
          .data$identifier_method == "ticker_to_unique_ric" &
            isTRUE(use_unique_ticker_crosswalk) ~ 2L,
          TRUE ~ 99L
        ),
        
        ticker_rank = dplyr::case_when(
          .data$identifier_method != "ticker_to_unique_ric" ~ 0L,
          .data$market_ric_matches_ticker_ric ~ 1L,
          .data$market_ric_present & !.data$market_ric_matches_ticker_ric ~ 2L,
          TRUE ~ 3L
        )
      )
  }
  
  # ---------------------------------------------------------------------------
  # Candidate diagnostics per market
  # ---------------------------------------------------------------------------
  candidate_counts <- if (nrow(all_candidates) > 0L) {
    all_candidates %>%
      dplyr::group_by(.data$market_row_id) %>%
      dplyr::summarise(
        n_candidates_total = dplyr::n(),
        n_candidates_accepted = sum(.data$accepted_match, na.rm = TRUE),
        closest_candidate_time_diff_hours = {
          z <- .data$time_diff_hours[!is.na(.data$time_diff_hours)]
          if (length(z) == 0L) NA_real_ else min(z)
        },
        .groups = "drop"
      )
  } else {
    markets_aug %>%
      dplyr::transmute(
        market_row_id = .data$market_row_id,
        n_candidates_total = 0L,
        n_candidates_accepted = 0L,
        closest_candidate_time_diff_hours = NA_real_
      )
  }
  
  # ---------------------------------------------------------------------------
  # Choose best accepted match
  # Preference order:
  #   1. lower match tier (exact date first)
  #   2. identifier method (RIC before ticker crosswalk)
  #   3. for ticker fallback, prefer cases where market RIC agrees with the
  #      ticker-implied RIC
  #   4. smaller time difference
  #   5. stronger market data availability
  # ---------------------------------------------------------------------------
  best_match <- if (nrow(all_candidates) > 0L) {
    all_candidates %>%
      dplyr::filter(.data$accepted_match) %>%
      dplyr::arrange(
        .data$market_row_id,
        .data$match_tier,
        .data$identifier_rank,
        .data$ticker_rank,
        .data$time_diff_hours_for_rank,
        dplyr::desc(.data$selected_mean),
        dplyr::desc(.data$selected_target),
        dplyr::desc(.data$n_snapshots_ok),
        dplyr::desc(.data$volume_num),
        .data$market_id
      ) %>%
      dplyr::group_by(.data$market_row_id) %>%
      dplyr::slice(1L) %>%
      dplyr::ungroup() %>%
      dplyr::transmute(
        market_row_id = .data$market_row_id,
        matched_event_key = .data$event_key,
        matched_universe_ric = .data$matched_universe_ric,
        matched_universe_ticker = .data$ticker_universe,
        matched_event_date = .data$event_date_universe,
        matched_event_datetime_utc = .data$event_datetime_utc_universe,
        match_tier = .data$match_tier,
        match_rule = .data$match_rule,
        identifier_method = .data$identifier_method,
        match_time_diff_hours = .data$time_diff_hours,
        quarter_consistent = .data$quarter_consistent
      )
  } else {
    tibble::tibble(
      market_row_id = integer(),
      matched_event_key = character(),
      matched_universe_ric = character(),
      matched_universe_ticker = character(),
      matched_event_date = as.Date(character()),
      matched_event_datetime_utc = as.POSIXct(character()),
      match_tier = integer(),
      match_rule = character(),
      identifier_method = character(),
      match_time_diff_hours = numeric(),
      quarter_consistent = logical()
    )
  }
  
  # ---------------------------------------------------------------------------
  # Final market-level diagnostics
  # ---------------------------------------------------------------------------
  markets_matched <- markets_aug %>%
    dplyr::left_join(candidate_counts, by = "market_row_id") %>%
    dplyr::left_join(best_match, by = "market_row_id") %>%
    dplyr::mutate(
      match_status = dplyr::if_else(!is.na(.data$matched_event_key), "matched", "unmatched"),
      unmatched_reason = dplyr::case_when(
        .data$match_status == "matched" ~ NA_character_,
        is.na(.data$ric) & is.na(.data$ticker_norm) ~ "missing_ric_and_ticker",
        dplyr::coalesce(.data$n_candidates_total, 0L) == 0L ~ "no_candidate_company_in_universe",
        dplyr::coalesce(.data$n_candidates_total, 0L) > 0L &
          dplyr::coalesce(.data$n_candidates_accepted, 0L) == 0L ~
          "candidate_found_but_failed_time_or_quarter_rules",
        TRUE ~ "unclassified"
      )
    ) %>%
    dplyr::select(-.data$ticker_norm)
  
  list(
    markets = markets_matched,
    candidate_matches = all_candidates
  )
}

rhs_terms_from_formula <- function(formula_obj) {
  attr(stats::terms(formula_obj), "term.labels")
}

variable_is_usable <- function(x) {
  if (is.factor(x) || is.character(x)) {
    x2 <- x[!is.na(x)]
    return(length(unique(x2)) >= 2L)
  }
  
  x2 <- x[is.finite(x)]
  return(length(unique(x2)) >= 2L)
}

keep_varying_terms <- function(data, rhs_terms) {
  keep <- vapply(
    rhs_terms,
    function(term) {
      # Only intended for plain variable names, which is what your formulas use
      if (!term %in% names(data)) return(FALSE)
      variable_is_usable(data[[term]])
    },
    logical(1)
  )
  
  rhs_terms[keep]
}

fit_heckman_spec <- function(panel,
                             selected_var,
                             outcome_var,
                             selection_rhs,
                             outcome_rhs,
                             model_label) {
  selection_formula_initial <- stats::as.formula(
    paste(selected_var, "~", selection_rhs)
  )
  outcome_formula_initial <- stats::as.formula(
    paste(outcome_var, "~", outcome_rhs)
  )
  
  selection_terms_raw <- rhs_terms_from_formula(selection_formula_initial)
  outcome_terms_raw <- rhs_terms_from_formula(outcome_formula_initial)
  
  rhs_needed_vars <- unique(c(selection_terms_raw, outcome_terms_raw))
  
  stage_df <- panel %>%
    dplyr::filter(stats::complete.cases(dplyr::across(dplyr::all_of(rhs_needed_vars)))) %>%
    dplyr::filter(!is.na(.data[[selected_var]])) %>%
    dplyr::filter(.data[[selected_var]] == 0L | !is.na(.data[[outcome_var]])) %>%
    droplevels()
  
  message("Post-filter selection counts for ", model_label, ":")
  print(table(stage_df[[selected_var]], useNA = "ifany"))
  
  if (nrow(stage_df) == 0L) {
    stop("No usable observations remain for model: ", model_label, call. = FALSE)
  }
  
  if (length(unique(stage_df[[selected_var]])) < 2L) {
    stop(
      "Selection indicator has fewer than two classes for model: ",
      model_label,
      call. = FALSE
    )
  }
  
  selection_terms_raw <- rhs_terms_from_formula(selection_formula_initial)
  outcome_terms_raw <- rhs_terms_from_formula(outcome_formula_initial)
  
  selection_terms <- keep_varying_terms(stage_df, selection_terms_raw)
  outcome_terms <- keep_varying_terms(stage_df, outcome_terms_raw)
  
  if (length(outcome_terms) == 0L) {
    stop(
      "No varying outcome regressors remain after filtering for model: ",
      model_label,
      call. = FALSE
    )
  }
  
  if (length(selection_terms) == 0L) {
    stop(
      "No varying selection regressors remain after filtering for model: ",
      model_label,
      call. = FALSE
    )
  }
  
  selection_dropped <- setdiff(selection_terms_raw, selection_terms)
  outcome_dropped <- setdiff(outcome_terms_raw, outcome_terms)
  
  if (length(selection_dropped) > 0L) {
    message(
      "Dropping non-varying selection terms in ", model_label, ": ",
      paste(selection_dropped, collapse = ", ")
    )
  }
  
  if (length(outcome_dropped) > 0L) {
    message(
      "Dropping non-varying outcome terms in ", model_label, ": ",
      paste(outcome_dropped, collapse = ", ")
    )
  }
  
  selection_formula <- stats::as.formula(
    paste(selected_var, "~", paste(selection_terms, collapse = " + "))
  )
  
  outcome_formula <- stats::as.formula(
    paste(outcome_var, "~", paste(outcome_terms, collapse = " + "))
  )
  
  stage_df <- droplevels(stage_df)
  
  probit_fit <- stats::glm(
    formula = selection_formula,
    data = stage_df,
    family = stats::binomial(link = "probit")
  )
  
  selected_df <- stage_df %>%
    dplyr::filter(.data[[selected_var]] == 1L, !is.na(.data[[outcome_var]])) %>%
    droplevels()
  
  if (nrow(selected_df) == 0L) {
    stop("No selected observations remain for outcome model: ", model_label, call. = FALSE)
  }
  
  # Re-check outcome regressors on selected sample too
  outcome_terms_selected <- keep_varying_terms(selected_df, outcome_terms)
  
  if (length(outcome_terms_selected) == 0L) {
    stop(
      "No varying outcome regressors remain in selected sample for model: ",
      model_label,
      call. = FALSE
    )
  }
  
  if (!identical(outcome_terms_selected, outcome_terms)) {
    dropped_selected <- setdiff(outcome_terms, outcome_terms_selected)
    message(
      "Dropping selected-sample non-varying outcome terms in ", model_label, ": ",
      paste(dropped_selected, collapse = ", ")
    )
    
    outcome_formula <- stats::as.formula(
      paste(outcome_var, "~", paste(outcome_terms_selected, collapse = " + "))
    )
  }
  
  naive_fit <- stats::lm(outcome_formula, data = selected_df)
  
  heckman_fit <- sampleSelection::selection(
    selection = selection_formula,
    outcome = outcome_formula,
    data = stage_df,
    method = "2step"
  )
  
  selection_tidy <- tidy_glm_hc3(
    model = probit_fit,
    model_label = model_label,
    component_label = "Selection equation"
  )
  
  naive_tidy <- tidy_lm_hc3(
    model = naive_fit,
    model_label = model_label,
    component_label = "Outcome equation",
    estimator_label = "Naive OLS"
  )
  
  heckman_tidy <- tidy_heckman_outcome(
    heckman_fit = heckman_fit,
    naive_fit = naive_fit,
    model_label = model_label,
    component_label = "Outcome equation"
  )
  
  fit_stats <- selection_fit_stats(
    model = probit_fit,
    selected = stage_df[[selected_var]],
    model_label = model_label,
    component_label = "Selection equation"
  ) %>%
    dplyr::mutate(
      outcome_n = nrow(selected_df),
      naive_r_squared = summary(naive_fit)$r.squared,
      heckman_rho = as.numeric(heckman_fit$rho),
      heckman_sigma = as.numeric(heckman_fit$sigma),
      imr_term = {
        imr_row <- heckman_tidy %>% dplyr::filter(.data$is_imr)
        if (nrow(imr_row) == 0L) NA_character_ else imr_row$term[1]
      },
      imr_estimate = {
        imr_row <- heckman_tidy %>% dplyr::filter(.data$is_imr)
        if (nrow(imr_row) == 0L) NA_real_ else imr_row$estimate[1]
      },
      imr_p_value = {
        imr_row <- heckman_tidy %>% dplyr::filter(.data$is_imr)
        if (nrow(imr_row) == 0L) NA_real_ else imr_row$p_value[1]
      }
    )
  
  list(
    selection_formula = selection_formula,
    outcome_formula = outcome_formula,
    stage_df = stage_df,
    selected_df = selected_df,
    probit_fit = probit_fit,
    naive_fit = naive_fit,
    heckman_fit = heckman_fit,
    selection_tidy = selection_tidy,
    naive_tidy = naive_tidy,
    heckman_tidy = heckman_tidy,
    fit_stats = fit_stats,
    selection_terms_dropped = selection_dropped,
    outcome_terms_dropped = outcome_dropped
  )
}

`%+%` <- function(a, b) paste0(a, b)

build_outcome_pretty_table <- function(df, title, subtitle) {
  df_pretty <- df %>%
    dplyr::mutate(
      term_clean = clean_term_label(.data$term),
      display = paste0(
        sprintf("%.4f", .data$estimate),
        .data$stars,
        "<br>(",
        sprintf("%.4f", .data$std_error),
        ")"
      )
    ) %>%
    dplyr::select(.data$term_clean, .data$estimator, .data$display) %>%
    tidyr::pivot_wider(
      names_from = .data$estimator,
      values_from = .data$display
    ) %>%
    dplyr::rename(term = .data$term_clean)

  make_gt_table(
    df_pretty,
    title = title,
    subtitle = subtitle,
    note = gt::md(
      "Cells show coefficient estimates with standard errors in parentheses. "
      %+% "For the Heckman model, the two-step variance-covariance matrix from "
      %+% "`sampleSelection` is used. "
      %+% "Stars: * p < 0.10, ** p < 0.05, *** p < 0.01."
    )
  )
}

run_heckman_selection_robustness <- function(
    root = NULL,
    selected_horizons = c("1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h"),
    target_horizon = "6h",
    require_complete_panel_for_mean = FALSE,
    min_level_n = 25L,
    output_dir = NULL,
    polymarket_event_tz = "UTC",
    universe_event_tz = "UTC",
    max_match_time_diff_hours = 18,
    use_unique_ticker_crosswalk = TRUE
) {
  check_required_packages()

  suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
    library(tidyr)
    library(purrr)
    library(stringr)
    library(tibble)
    library(jsonlite)
    library(gt)
    library(lmtest)
    library(sandwich)
    library(sampleSelection)
  })

  ROOT <- find_project_root(root = root)

  if (is.null(output_dir)) {
    output_dir <- file.path(ROOT, "statistics", "heckman_selection")
  }

  tables_dir <- file.path(output_dir, "tables")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

  source(file.path(ROOT, "R", "utils", "load_data.R"))
  D <- load_project_data(ROOT)

  dataset_long <- D$dataset_long
  universe_raw <- D$heckman_universe_events

  universe_companies_path <- file.path(
    ROOT,
    "data",
    "heckman_selection_model",
    "heckman_universe_companies.csv"
  )

  universe_companies <- if (file.exists(universe_companies_path)) {
    readr::read_csv(universe_companies_path, show_col_types = FALSE)
  } else {
    NULL
  }

  message("Building Polymarket market registry ...")
  market_registry <- build_polymarket_market_registry(
    dataset_long = dataset_long,
    selected_horizons = selected_horizons,
    target_horizon = target_horizon,
    require_complete_panel_for_mean = require_complete_panel_for_mean,
    polymarket_event_tz = polymarket_event_tz
  )

  message("Building full-universe event panel ...")
  universe_events <- build_universe_events(
    heckman_universe_events = universe_raw,
    min_level_n = min_level_n,
    universe_event_tz = universe_event_tz
  )

  message("Matching Polymarket markets to universe events ...")
  match_results <- match_polymarket_to_universe(
    markets = market_registry,
    universe_events = universe_events,
    universe_companies = universe_companies,
    max_time_diff_hours = max_match_time_diff_hours,
    use_unique_ticker_crosswalk = use_unique_ticker_crosswalk
  )

  market_matches <- match_results$markets
  
  message("Identifier-method diagnostics:")
  print(table(market_matches$identifier_method, useNA = "ifany"))

  market_outcomes <- market_matches %>%
    dplyr::filter(!is.na(.data$matched_event_key)) %>%
    dplyr::arrange(
      .data$match_tier,
      dplyr::coalesce(.data$match_time_diff_hours, Inf),
      dplyr::desc(.data$selected_mean),
      dplyr::desc(.data$selected_target),
      dplyr::desc(.data$n_snapshots_ok),
      dplyr::desc(.data$volume_num),
      .data$market_id
    ) %>%
    dplyr::group_by(.data$matched_event_key) %>%
    dplyr::mutate(n_markets_same_event = dplyr::n()) %>%
    dplyr::slice(1L) %>%
    dplyr::ungroup()

  analysis_panel <- universe_events %>%
    dplyr::left_join(
      market_outcomes %>%
        dplyr::select(
          matched_event_key = .data$matched_event_key,
          market_id = .data$market_id,
          ticker_pm = .data$ticker,
          slug = .data$slug,
          n_snapshots_ok = .data$n_snapshots_ok,
          loss_mean = .data$loss_mean,
          loss_target = .data$loss_target,
          selected_mean = .data$selected_mean,
          selected_target = .data$selected_target,
          match_tier = .data$match_tier,
          match_rule = .data$match_rule,
          identifier_method = .data$identifier_method,
          match_time_diff_hours = .data$match_time_diff_hours,
          n_markets_same_event = .data$n_markets_same_event
        ),
      by = c("event_key" = "matched_event_key")
    ) %>%
    dplyr::mutate(
      selected_mean = dplyr::coalesce(.data$selected_mean, 0L),
      selected_target = dplyr::coalesce(.data$selected_target, 0L)
    )

  matching_summary <- tibble::tibble(
    n_universe_events = nrow(universe_events),
    n_polymarket_markets_after_collapse = nrow(market_registry),
    n_matched_universe_events = sum(!is.na(analysis_panel$market_id)),
    n_unmatched_universe_events = sum(is.na(analysis_panel$market_id)),
    n_matched_polymarket_markets = sum(!is.na(market_matches$matched_event_key)),
    n_unmatched_polymarket_markets = sum(is.na(market_matches$matched_event_key)),
    n_exact_date_matches = sum(market_matches$match_tier == 1L, na.rm = TRUE),
    n_time_window_matches = sum(market_matches$match_tier == 2L, na.rm = TRUE),
    n_adjacent_date_matches = sum(market_matches$match_tier == 3L, na.rm = TRUE),
    n_matches_via_ric = sum(market_matches$identifier_method == "ric_exact", na.rm = TRUE),
    n_matches_via_ticker_crosswalk = sum(market_matches$identifier_method == "ticker_to_unique_ric", na.rm = TRUE),
    n_selected_mean = sum(analysis_panel$selected_mean == 1L, na.rm = TRUE),
    n_selected_target = sum(analysis_panel$selected_target == 1L, na.rm = TRUE),
    require_complete_panel_for_mean = require_complete_panel_for_mean,
    target_horizon = target_horizon,
    polymarket_event_tz = polymarket_event_tz,
    universe_event_tz = universe_event_tz,
    max_match_time_diff_hours = max_match_time_diff_hours,
    use_unique_ticker_crosswalk = use_unique_ticker_crosswalk,
    selected_horizons = paste(selected_horizons, collapse = ", ")
  )

  write_csv_jsonl(analysis_panel, file.path(output_dir, "heckman_analysis_panel"))
  write_csv_jsonl(matching_summary, file.path(output_dir, "heckman_matching_summary"))
  write_csv_jsonl(market_matches, file.path(output_dir, "heckman_market_match_diagnostics"))

  outcome_rhs <- paste(
    c("log_market_cap", "analysts", "log_turnover", "log_volatility_6m"),
    collapse = " + "
  )

  selection_rhs <- paste(
    c(
      "log_market_cap",
      "analysts",
      "log_turnover",
      "log_volatility_6m",
      "release_timing_group",
      "gics_sector_group"
    ),
    collapse = " + "
  )
  
  message("Selection-rate diagnostics:")
  print(table(analysis_panel$selected_mean, useNA = "ifany"))
  print(table(analysis_panel$selected_target, useNA = "ifany"))
  
  message("Factor diagnostics before estimation:")
  print(table(analysis_panel$release_timing_group, useNA = "ifany"))
  print(table(analysis_panel$exchange_country_group, useNA = "ifany"))
  print(table(analysis_panel$gics_sector_group, useNA = "ifany"))

  message("Estimating mean-loss Heckman model ...")
  mean_results <- fit_heckman_spec(
    panel = analysis_panel,
    selected_var = "selected_mean",
    outcome_var = "loss_mean",
    selection_rhs = selection_rhs,
    outcome_rhs = outcome_rhs,
    model_label = "Mean Brier loss"
  )

  message("Estimating target-horizon Heckman model ...")
  target_results <- fit_heckman_spec(
    panel = analysis_panel,
    selected_var = "selected_target",
    outcome_var = "loss_target",
    selection_rhs = selection_rhs,
    outcome_rhs = outcome_rhs,
    model_label = paste0("Brier loss at ", target_horizon)
  )

  selection_coefficients <- dplyr::bind_rows(
    mean_results$selection_tidy,
    target_results$selection_tidy
  ) %>%
    dplyr::mutate(term_clean = clean_term_label(.data$term)) %>%
    dplyr::select(
      .data$model,
      .data$component,
      .data$estimator,
      .data$term,
      .data$term_clean,
      .data$estimate,
      .data$std_error,
      .data$statistic,
      .data$p_value,
      .data$conf_low,
      .data$conf_high,
      .data$stars
    )

  outcome_coefficients <- dplyr::bind_rows(
    mean_results$naive_tidy,
    mean_results$heckman_tidy %>% dplyr::select(-.data$is_imr),
    target_results$naive_tidy,
    target_results$heckman_tidy %>% dplyr::select(-.data$is_imr)
  ) %>%
    dplyr::mutate(term_clean = clean_term_label(.data$term)) %>%
    dplyr::select(
      .data$model,
      .data$component,
      .data$estimator,
      .data$term,
      .data$term_clean,
      .data$estimate,
      .data$std_error,
      .data$statistic,
      .data$p_value,
      .data$conf_low,
      .data$conf_high,
      .data$stars
    )

  model_fit <- dplyr::bind_rows(
    mean_results$fit_stats,
    target_results$fit_stats
  )

  write_csv_jsonl(selection_coefficients, file.path(output_dir, "heckman_selection_coefficients"))
  write_csv_jsonl(outcome_coefficients, file.path(output_dir, "heckman_outcome_coefficients"))
  write_csv_jsonl(model_fit, file.path(output_dir, "heckman_model_fit"))

  selection_pretty <- selection_coefficients %>%
    dplyr::mutate(
      `Estimate` = paste0(fmt_num(.data$estimate), .data$stars),
      `Std. error` = fmt_num(.data$std_error),
      `95% CI` = fmt_ci(.data$conf_low, .data$conf_high),
      `p-value` = fmt_p(.data$p_value)
    ) %>%
    dplyr::select(
      Model = .data$model,
      Term = .data$term_clean,
      `Estimate`,
      `Std. error`,
      `95% CI`,
      `p-value`
    )

  selection_gt <- make_gt_table(
    selection_pretty,
    title = "Heckman selection equations (probit)",
    subtitle = "First-stage probit estimates for sample inclusion in the observed Polymarket event sample",
    note = gt::md(
      "HC3 robust standard errors are reported for the standalone probit display model. "
      %+% "These tables are descriptive complements to the two-step Heckman estimation."
    )
  )

  save_gt_html(selection_gt, file.path(tables_dir, "table_selection_equations.html"))

  mean_outcome_gt <- build_outcome_pretty_table(
    df = outcome_coefficients %>% dplyr::filter(.data$model == "Mean Brier loss"),
    title = "Outcome equation: mean Polymarket Brier loss",
    subtitle = paste0(
      "Comparison of naive selected-sample OLS and Heckman two-step correction across horizons: ",
      paste(selected_horizons, collapse = ", ")
    )
  )

  target_outcome_gt <- build_outcome_pretty_table(
    df = outcome_coefficients %>% dplyr::filter(.data$model == paste0("Brier loss at ", target_horizon)),
    title = paste0("Outcome equation: Polymarket Brier loss at ", target_horizon),
    subtitle = "Comparison of naive selected-sample OLS and Heckman two-step correction"
  )

  save_gt_html(mean_outcome_gt, file.path(tables_dir, "table_outcome_mean_loss.html"))
  save_gt_html(target_outcome_gt, file.path(tables_dir, "table_outcome_target_loss.html"))

  model_fit_pretty <- model_fit %>%
    dplyr::mutate(
      `Selection rate` = sprintf("%.1f%%", 100 * .data$selection_rate),
      `Pseudo R²` = fmt_num(.data$pseudo_r2),
      `PCP` = sprintf("%.1f%%", 100 * .data$pcp),
      `Naive OLS R²` = fmt_num(.data$naive_r_squared),
      `rho` = fmt_num(.data$heckman_rho),
      `sigma` = fmt_num(.data$heckman_sigma),
      `IMR estimate` = fmt_num(.data$imr_estimate),
      `IMR p-value` = fmt_p(.data$imr_p_value)
    ) %>%
    dplyr::select(
      Model = .data$model,
      `N total` = .data$n_total,
      `N selected` = .data$n_selected,
      `N outcome` = .data$outcome_n,
      `Selection rate`,
      `Pseudo R²`,
      `PCP`,
      `Naive OLS R²`,
      `rho`,
      `sigma`,
      `IMR estimate`,
      `IMR p-value`
    )

  model_fit_gt <- make_gt_table(
    model_fit_pretty,
    title = "Heckman model fit and selection diagnostics",
    subtitle = "Pseudo R² and PCP refer to the first-stage probit selection equation",
    note = gt::md(
      "A statistically meaningful IMR coefficient suggests that selected-sample OLS may be affected by sample-selection bias. "
      %+% "`rho` and `sigma` come from the two-step Heckman model estimated by `sampleSelection`."
    )
  )

  save_gt_html(model_fit_gt, file.path(tables_dir, "table_model_fit.html"))

  message("Heckman robustness outputs written to: ", output_dir)

  invisible(list(
    root = ROOT,
    output_dir = output_dir,
    matching_summary = matching_summary,
    analysis_panel = analysis_panel,
    market_matches = market_matches,
    selection_coefficients = selection_coefficients,
    outcome_coefficients = outcome_coefficients,
    model_fit = model_fit,
    mean_results = mean_results,
    target_results = target_results
  ))
}

if (isTRUE(getOption("polymarket.autorun", interactive()))) {
  heckman_selection_results <- run_heckman_selection_robustness()
}
