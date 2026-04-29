#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/factor_analysis.R
# Purpose: Cross-sectional OLS factor analysis for Polymarket accuracy.
#
# Design
# ------
# One observation per market.
#
# Dependent variables:
#   (1) Mean Polymarket Brier loss across the LAST 4 DAYS of snapshots:
#       4d, 3d, 2d, 1d, 12h, 6h
#   (2) Polymarket Brier loss at 6h before resolution
#
# Model split:
#   - Ex-ante model: excludes val_surprise and volumeNum
#   - Full model: includes val_surprise and volumeNum
#
# Interpretation:
#   Lower Brier loss = higher accuracy
#   Therefore, NEGATIVE coefficients imply HIGHER accuracy.
#
# Outputs
# -------
# Saves the following (relative to project root):
#   statistics/factor_analysis/factor_analysis_regression_table.html
#   statistics/factor_analysis/factor_analysis_regression_coefficients.csv
#   statistics/factor_analysis/factor_analysis_regression_coefficients.jsonl
#   statistics/factor_analysis/factor_analysis_model_fit.csv
#   statistics/factor_analysis/factor_analysis_model_fit.jsonl
#   statistics/factor_analysis/factor_analysis_market_level.csv
#   statistics/factor_analysis/factor_analysis_market_level.jsonl
#   statistics/factor_analysis/factor_analysis_plot_data.csv
#   statistics/factor_analysis/factor_analysis_plot_data.jsonl
#   statistics/factor_analysis/factor_analysis_coefficients_plot.png
#
# Notes
# -----
# 1) The mean-loss model now requires complete data only for the LAST 4 DAYS
#    window: 4d, 3d, 2d, 1d, 12h, 6h.
#
# 2) If val_surprise is realized surprise, it is ex-post and should not be
#    interpreted as an ex-ante tradable signal.
#
# 3) If volumeNum is final realized volume rather than volume known before
#    resolution, it is also ex-post. That is why it is excluded from the
#    ex-ante specification and included only in the full specification.
# =============================================================================

options(stringsAsFactors = FALSE, scipen = 999)

# ------------------------------ #
# 1. Package handling
# ------------------------------ #
required_packages <- c(
  "dplyr",
  "broom",
  "lmtest",
  "sandwich",
  "gt",
  "modelsummary",
  "jsonlite",
  "ggplot2"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(dplyr)
  library(broom)
  library(lmtest)
  library(sandwich)
  library(gt)
  library(modelsummary)
  library(jsonlite)
  library(ggplot2)
})

# ------------------------------ #
# 2. Helper functions
# ------------------------------ #

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
  stop(
    "Could not find project root (renv.lock or .Rproj). ",
    "Run from inside the project.",
    call. = FALSE
  )
}

get_start_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    p <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(p, winslash = "/", mustWork = FALSE)))
  }
  
  ofile <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) "")
  if (is.character(ofile) && nzchar(ofile)) {
    return(dirname(normalizePath(ofile, winslash = "/", mustWork = FALSE)))
  }
  
  if (interactive() &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nzchar(p)) {
      return(dirname(normalizePath(p, winslash = "/", mustWork = FALSE)))
    }
  }
  
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

to_logical_safe <- function(x) {
  x_chr <- trimws(tolower(as.character(x)))
  x_chr %in% c("true", "t", "1", "yes", "y")
}

parse_datetime_utc <- function(x, colname = deparse(substitute(x))) {
  x <- trimws(as.character(x))
  x[x %in% c("", "NA", "N/A", "NULL", "null")] <- NA_character_
  
  # Normalize timezone offsets like +00:00 -> +0000
  x <- sub("([+-][0-9]{2}):([0-9]{2})$", "\\1\\2", x)
  
  fmts <- c(
    "%Y-%m-%dT%H:%M:%OSZ",
    "%Y-%m-%dT%H:%M:%OS%z",
    "%Y-%m-%d %H:%M:%OS%z",
    "%Y-%m-%d %H:%M:%OS",
    "%Y-%m-%d"
  )
  
  out <- rep(as.POSIXct(NA, tz = "UTC"), length(x))
  
  for (fmt in fmts) {
    idx <- !is.na(x) & is.na(out)
    if (any(idx)) {
      parsed <- as.POSIXct(x[idx], format = fmt, tz = "UTC")
      out[idx] <- parsed
    }
  }
  
  bad <- unique(x[!is.na(x) & is.na(out)])
  if (length(bad) > 0) {
    stop(
      "Unparseable datetime values in ", colname, ": ",
      paste(utils::head(bad, 10), collapse = " | "),
      call. = FALSE
    )
  }
  
  out
}
first_non_missing <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  x[1]
}

safe_log <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.finite(x) & x > 0, log(x), NA_real_)
}

safe_log1p <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.finite(x) & x >= 0, log1p(x), NA_real_)
}

winsorize_vec <- function(x, probs = c(0.05, 0.95)) {
  x <- suppressWarnings(as.numeric(x))
  qs <- stats::quantile(x, probs = probs, na.rm = TRUE, names = FALSE, type = 7)
  x <- pmax(x, qs[1])
  x <- pmin(x, qs[2])
  x
}

bootstrap_ci_model <- function(
    formula,
    data,
    fit_type = c("lm", "glm"),
    link = NULL,
    R = 25000,
    seed = 123,
    conf_level = 0.95,
    show_progress = TRUE,
    progress_label = ""
) {
  fit_type <- match.arg(fit_type)
  set.seed(seed)

  original_fit <- if (fit_type == "lm") {
    lm(formula, data = data)
  } else {
    glm(formula, data = data, family = quasibinomial(link = link))
  }

  term_names <- names(coef(original_fit))
  boot_coefs <- matrix(
    NA_real_,
    nrow = R,
    ncol = length(term_names),
    dimnames = list(NULL, term_names)
  )

  start_time <- Sys.time()

  if (isTRUE(show_progress)) {
    cat("\nStarting bootstrap:", progress_label, "\n")
    pb <- utils::txtProgressBar(min = 0, max = R, style = 3)
    on.exit(close(pb), add = TRUE)
  }

  for (b in seq_len(R)) {
    idx <- sample.int(nrow(data), size = nrow(data), replace = TRUE)
    d_b <- data[idx, , drop = FALSE]

    fit_b <- tryCatch(
      suppressWarnings(
        if (fit_type == "lm") {
          lm(formula, data = d_b)
        } else {
          glm(formula, data = d_b, family = quasibinomial(link = link))
        }
      ),
      error = function(e) NULL
    )

    if (!is.null(fit_b)) {
      cf <- coef(fit_b)
      boot_coefs[b, match(names(cf), term_names)] <- unname(cf)
    }

    if (isTRUE(show_progress)) {
      utils::setTxtProgressBar(pb, b)

      if (b %% 250 == 0 || b == R) {
        elapsed_sec <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        sec_per_iter <- elapsed_sec / b
        eta_sec <- sec_per_iter * (R - b)

        cat(
          sprintf(
            "\n[%s] %d/%d | elapsed: %.1f min | ETA: %.1f min",
            progress_label,
            b, R,
            elapsed_sec / 60,
            eta_sec / 60
          )
        )
        flush.console()
      }
    }
  }

  alpha <- 1 - conf_level

  data.frame(
    term = term_names,
    conf.low_boot = apply(
      boot_coefs, 2,
      function(x) stats::quantile(
        x, probs = alpha / 2, na.rm = TRUE, names = FALSE, type = 7
      )
    ),
    conf.high_boot = apply(
      boot_coefs, 2,
      function(x) stats::quantile(
        x, probs = 1 - alpha / 2, na.rm = TRUE, names = FALSE, type = 7
      )
    ),
    n_boot = R,
    n_boot_success = colSums(is.finite(boot_coefs)),
    stringsAsFactors = FALSE
  )
}

p_to_stars <- function(p) {
  ifelse(
    is.na(p), "",
    ifelse(p < 0.01, "***",
           ifelse(p < 0.05, "**",
                  ifelse(p < 0.10, "*", "")))
  )
}

write_jsonl <- function(df, path) {
  con <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  
  if (nrow(df) == 0) {
    return(invisible(NULL))
  }
  
  for (i in seq_len(nrow(df))) {
    row_list <- lapply(df[i, , drop = FALSE], function(col) col[[1]])
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

tidy_lm_hc3 <- function(model, model_name) {
  ct <- lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC3"))
  out <- broom::tidy(ct)
  
  crit <- qt(0.975, df = model$df.residual)
  
  out <- out %>%
    mutate(
      conf.low = estimate - crit * std.error,
      conf.high = estimate + crit * std.error,
      stars = p_to_stars(p.value),
      model = model_name
    ) %>%
    select(model, term, estimate, std.error, statistic, p.value, conf.low, conf.high, stars)
  
  out
}

run_lm_diagnostics <- function(model, model_name, output_dir) {
  safe_name <- gsub("[^A-Za-z0-9]+", "_", tolower(model_name))
  
  resid_vec <- residuals(model)
  fitted_vec <- fitted(model)
  std_resid <- rstandard(model)
  stud_resid <- rstudent(model)
  leverage <- hatvalues(model)
  cooks_d <- cooks.distance(model)
  
  bp <- tryCatch(lmtest::bptest(model), error = function(e) NULL)
  reset <- tryCatch(
    lmtest::resettest(model, power = 2:3, type = "fitted"),
    error = function(e) NULL
  )
  
  n <- length(resid_vec)
  p <- length(coef(model))
  
  diag_summary <- data.frame(
    model = model_name,
    n = n,
    p = p,
    mean_residual = mean(resid_vec, na.rm = TRUE),
    sd_residual = sd(resid_vec, na.rm = TRUE),
    bp_statistic = if (is.null(bp)) NA_real_ else unname(bp$statistic),
    bp_p_value = if (is.null(bp)) NA_real_ else bp$p.value,
    reset_statistic = if (is.null(reset)) NA_real_ else unname(reset$statistic),
    reset_p_value = if (is.null(reset)) NA_real_ else reset$p.value,
    max_abs_std_resid = max(abs(std_resid), na.rm = TRUE),
    n_abs_std_resid_gt_2 = sum(abs(std_resid) > 2, na.rm = TRUE),
    n_abs_std_resid_gt_3 = sum(abs(std_resid) > 3, na.rm = TRUE),
    max_leverage = max(leverage, na.rm = TRUE),
    leverage_cutoff = 2 * p / n,
    n_high_leverage = sum(leverage > (2 * p / n), na.rm = TRUE),
    max_cooks_d = max(cooks_d, na.rm = TRUE),
    cooks_d_cutoff = 4 / n,
    n_high_cooks_d = sum(cooks_d > (4 / n), na.rm = TRUE),
    corr_fitted_residual = suppressWarnings(cor(fitted_vec, resid_vec, use = "complete.obs"))
  )
  
  diag_obs <- broom::augment(model) %>%
    mutate(
      model = model_name,
      residual = resid_vec,
      fitted = fitted_vec,
      std_resid = std_resid,
      stud_resid = stud_resid,
      leverage = leverage,
      cooks_d = cooks_d
    )
  
  write.csv(
    diag_summary,
    file.path(output_dir, paste0("diagnostics_summary_", safe_name, ".csv")),
    row.names = FALSE
  )
  write_jsonl(
    diag_summary,
    file.path(output_dir, paste0("diagnostics_summary_", safe_name, ".jsonl"))
  )
  
  write.csv(
    diag_obs,
    file.path(output_dir, paste0("diagnostics_observations_", safe_name, ".csv")),
    row.names = FALSE
  )
  write_jsonl(
    diag_obs,
    file.path(output_dir, paste0("diagnostics_observations_", safe_name, ".jsonl"))
  )
  
  png(
    filename = file.path(output_dir, paste0("diagnostics_plot_", safe_name, ".png")),
    width = 1600,
    height = 1600,
    res = 200
  )
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)
  
  par(mfrow = c(2, 2))
  plot(model, which = 1:4)
  
  cat("\n================ DIAGNOSTICS: ", model_name, " ================\n", sep = "")
  print(diag_summary)
  
  invisible(list(summary = diag_summary, observations = diag_obs))
}

tidy_glm_hc3 <- function(model, model_name, model_class) {
  ct <- lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC3"))
  out <- broom::tidy(ct)
  
  crit <- qnorm(0.975)
  
  out <- out %>%
    mutate(
      conf.low = estimate - crit * std.error,
      conf.high = estimate + crit * std.error,
      stars = p_to_stars(p.value),
      model_class = model_class,
      model = model_name
    ) %>%
    select(model_class, model, term, estimate, std.error, statistic, p.value, conf.low, conf.high, stars)
  
  out
}

glance_fractional_model <- function(model, model_name, model_class) {
  data.frame(
    model_class = model_class,
    model = model_name,
    nobs = stats::nobs(model),
    null.deviance = model$null.deviance,
    deviance = model$deviance,
    df.residual = model$df.residual,
    dispersion = summary(model)$dispersion,
    pseudo_r2 = if (is.finite(model$null.deviance) && model$null.deviance > 0) {
      1 - model$deviance / model$null.deviance
    } else {
      NA_real_
    },
    AIC = suppressWarnings(tryCatch(AIC(model), error = function(e) NA_real_)),
    BIC = suppressWarnings(tryCatch(BIC(model), error = function(e) NA_real_)),
    stringsAsFactors = FALSE
  )
}

run_glm_diagnostics <- function(model, model_name, output_dir) {
  safe_name <- gsub("[^A-Za-z0-9]+", "_", tolower(model_name))
  
  fitted_vec <- fitted(model)
  dev_resid <- residuals(model, type = "deviance")
  pearson_resid <- residuals(model, type = "pearson")
  leverage <- hatvalues(model)
  cooks_d <- cooks.distance(model)
  
  std_deviance_resid <- dev_resid / sqrt(pmax(1 - leverage, 1e-8))
  std_pearson_resid <- pearson_resid / sqrt(pmax(1 - leverage, 1e-8))
  
  n <- length(dev_resid)
  p <- length(coef(model))
  
  diag_summary <- data.frame(
    model = model_name,
    n = n,
    p = p,
    mean_deviance_residual = mean(dev_resid, na.rm = TRUE),
    sd_deviance_residual = sd(dev_resid, na.rm = TRUE),
    mean_pearson_residual = mean(pearson_resid, na.rm = TRUE),
    sd_pearson_residual = sd(pearson_resid, na.rm = TRUE),
    overdispersion_phi = sum(pearson_resid^2, na.rm = TRUE) / model$df.residual,
    pseudo_r2 = if (is.finite(model$null.deviance) && model$null.deviance > 0) {
      1 - model$deviance / model$null.deviance
    } else {
      NA_real_
    },
    max_abs_std_deviance_resid = max(abs(std_deviance_resid), na.rm = TRUE),
    n_abs_std_dev_resid_gt_2 = sum(abs(std_deviance_resid) > 2, na.rm = TRUE),
    n_abs_std_dev_resid_gt_3 = sum(abs(std_deviance_resid) > 3, na.rm = TRUE),
    max_leverage = max(leverage, na.rm = TRUE),
    leverage_cutoff = 2 * p / n,
    n_high_leverage = sum(leverage > (2 * p / n), na.rm = TRUE),
    max_cooks_d = max(cooks_d, na.rm = TRUE),
    cooks_d_cutoff = 4 / n,
    n_high_cooks_d = sum(cooks_d > (4 / n), na.rm = TRUE),
    corr_fitted_dev_residual = suppressWarnings(cor(fitted_vec, dev_resid, use = "complete.obs"))
  )
  
  diag_obs <- data.frame(
    obs_number = seq_len(n),
    model = model_name,
    fitted = fitted_vec,
    deviance_residual = dev_resid,
    pearson_residual = pearson_resid,
    std_deviance_resid = std_deviance_resid,
    std_pearson_resid = std_pearson_resid,
    leverage = leverage,
    cooks_d = cooks_d,
    stringsAsFactors = FALSE
  )
  
  write.csv(
    diag_summary,
    file.path(output_dir, paste0("diagnostics_summary_", safe_name, ".csv")),
    row.names = FALSE
  )
  write_jsonl(
    diag_summary,
    file.path(output_dir, paste0("diagnostics_summary_", safe_name, ".jsonl"))
  )
  
  write.csv(
    diag_obs,
    file.path(output_dir, paste0("diagnostics_observations_", safe_name, ".csv")),
    row.names = FALSE
  )
  write_jsonl(
    diag_obs,
    file.path(output_dir, paste0("diagnostics_observations_", safe_name, ".jsonl"))
  )
  
  png(
    filename = file.path(output_dir, paste0("diagnostics_plot_", safe_name, ".png")),
    width = 1600,
    height = 1600,
    res = 200
  )
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)
  
  par(mfrow = c(2, 2))
  
  plot(
    fitted_vec, dev_resid,
    xlab = "Fitted values",
    ylab = "Deviance residuals",
    main = "Residuals vs Fitted",
    pch = 1
  )
  abline(h = 0, lty = 3, col = "#808080")
  lines(lowess(fitted_vec, dev_resid), col = "#E3170A", lwd = 2)
  
  qqnorm(std_deviance_resid, main = "Q-Q Deviance Residuals")
  qqline(std_deviance_resid, lty = 3, col = "#808080")
  
  plot(
    fitted_vec, sqrt(abs(std_pearson_resid)),
    xlab = "Fitted values",
    ylab = expression(sqrt("|Standardized Pearson residuals|")),
    main = "Scale-Location",
    pch = 1
  )
  lines(lowess(fitted_vec, sqrt(abs(std_pearson_resid))), col = "#E3170A", lwd = 2)
  
  plot(
    cooks_d,
    type = "h",
    xlab = "Obs. number",
    ylab = "Cook's distance",
    main = "Cook's distance"
  )
  
  cat("\n================ DIAGNOSTICS: ", model_name, " ================\n", sep = "")
  print(diag_summary)
  
  invisible(list(summary = diag_summary, observations = diag_obs))
}

# ------------------------------ #
# 3. Main analysis function
# ------------------------------ #
run_factor_analysis <- function(
    selected_horizons = c("1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h"),
    robustness_horizon = "6h",
    require_complete_panel_for_mean = FALSE,
    output_dir = NULL,
    winsorize = TRUE,
    winsor_probs = c(0.05, 0.95),
    bootstrap_runs = 25000,
    bootstrap_conf_level = 0.95,
    bootstrap_seed = 123
) {
  project_root <- find_project_root(get_start_dir())
  
  data_path  <- file.path(project_root, "data", "complete_dataset_long.csv")
  brier_path <- file.path(project_root, "data", "brier_scores", "brier_scores_market_horizon.csv")
  
  if (is.null(output_dir)) {
    output_dir <- file.path(project_root, "statistics", "factor_analysis")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  diagnostics_dir <- file.path(output_dir, "diagnostics")
  dir.create(diagnostics_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (!file.exists(data_path)) {
    stop("Data file not found: ", data_path, call. = FALSE)
  }
  if (!file.exists(brier_path)) {
    stop("Brier score file not found: ", brier_path, call. = FALSE)
  }
  
  cat("Project root: ", project_root, "\n", sep = "")
  cat("Reading data...\n")
  
  # ------------------------------ #
  # 4. Read input files
  # ------------------------------ #
  main_df  <- read.csv(data_path, strip.white = TRUE)
  brier_df <- read.csv(brier_path, strip.white = TRUE)
  
  # ------------------------------ #
  # 5. Standardize and construct variables
  # ------------------------------ #
  if (!"market_id" %in% names(main_df) && "id" %in% names(main_df)) {
    names(main_df)[names(main_df) == "id"] <- "market_id"
  }
  
  main_df$market_id <- as.character(main_df$market_id)
  brier_df$market_id <- as.character(brier_df$market_id)
  
  required_main_cols <- c(
    "market_id",
    "horizon",
    "acceptingOrdersTimestamp",
    "umaEndDate",
    "volumeNum",
    "val_eikon_eps_stddev_estimate",
    "val_surprise",
    "market_cap_usd_asof",
    "analysts_covering_asof",
    "turnover_6m_avg_daily_volume",
    "volatility_6m"
  )
  
  required_brier_cols <- c(
    "market_id",
    "horizon",
    "loss_polymarket"
  )
  
  missing_main <- setdiff(required_main_cols, names(main_df))
  missing_brier <- setdiff(required_brier_cols, names(brier_df))
  
  if (length(missing_main) > 0) {
    stop(
      "Missing required columns in complete_dataset_long.csv: ",
      paste(missing_main, collapse = ", "),
      call. = FALSE
    )
  }
  
  if (length(missing_brier) > 0) {
    stop(
      "Missing required columns in brier_scores_market_horizon.csv: ",
      paste(missing_brier, collapse = ", "),
      call. = FALSE
    )
  }
  
  main_df$acceptingOrdersTimestamp <- parse_datetime_utc(main_df$acceptingOrdersTimestamp)
  main_df$umaEndDate <- parse_datetime_utc(main_df$umaEndDate)
  
  main_df$open_time_days <- as.numeric(
    difftime(main_df$umaEndDate, main_df$acceptingOrdersTimestamp, units = "days")
  )
  
  main_df$volume_per_analyst <- ifelse(
    is.finite(suppressWarnings(as.numeric(main_df$volumeNum))) &
      is.finite(suppressWarnings(as.numeric(main_df$analysts_covering_asof))) &
      suppressWarnings(as.numeric(main_df$analysts_covering_asof)) > 0,
    suppressWarnings(as.numeric(main_df$volumeNum)) /
      suppressWarnings(as.numeric(main_df$analysts_covering_asof)),
    NA_real_
  )
  
  xvars_raw <- c(
    "volumeNum",
    "volume_per_analyst",
    "val_eikon_eps_stddev_estimate",
    "val_surprise",
    "market_cap_usd_asof",
    "analysts_covering_asof",
    "turnover_6m_avg_daily_volume",
    "volatility_6m",
    "open_time_days"
  )
  
  main_keep <- main_df[, c("market_id", "horizon", xvars_raw)]
  
  brier_keep_cols <- c("market_id", "horizon", "loss_polymarket")
  optional_brier_cols <- c("usable_polymarket", "status")
  optional_brier_cols <- optional_brier_cols[optional_brier_cols %in% names(brier_df)]
  
  brier_keep <- brier_df[, c(brier_keep_cols, optional_brier_cols)]
  
  df <- merge(
    x = brier_keep,
    y = main_keep,
    by = c("market_id", "horizon"),
    all.x = FALSE,
    all.y = FALSE
  )
  
  if ("usable_polymarket" %in% names(df)) {
    df <- df[to_logical_safe(df$usable_polymarket), ]
  }
  
  if ("status" %in% names(df)) {
    df <- df[tolower(trimws(as.character(df$status))) == "ok", ]
  }
  
  numeric_cols <- c("loss_polymarket", xvars_raw)
  for (v in numeric_cols) {
    df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
  }
  
  df$market_id <- as.character(df$market_id)
  
  df$horizon <- factor(df$horizon, levels = selected_horizons, ordered = TRUE)
  df <- df[!is.na(df$horizon), ]

  if (nrow(df) == 0) {
    stop("No usable observations remain after merging and cleaning.", call. = FALSE)
  }
  
  df <- df[order(df$market_id, df$horizon), ]
  
  cat("\n================ SAMPLE SUMMARY (LONG DATA) ================\n")
  cat("Observations: ", nrow(df), "\n", sep = "")
  cat("Markets: ", length(unique(df$market_id)), "\n", sep = "")
  cat("Horizons used for mean model: ", paste(selected_horizons, collapse = ", "), "\n", sep = "")
  cat("Dependent variable in regressions: Polymarket Brier loss\n")
  cat("Interpretation: lower Brier loss = higher Polymarket accuracy\n")
  
  market_df <- df %>%
    group_by(market_id) %>%
    summarise(
      n_snapshots = n_distinct(horizon),
      loss_mean = mean(loss_polymarket, na.rm = TRUE),
      loss_target = {
        z <- loss_polymarket[as.character(horizon) == robustness_horizon]
        if (length(z) == 0) NA_real_ else z[1]
      },
      volumeNum = first_non_missing(volumeNum),
      volume_per_analyst = first_non_missing(volume_per_analyst),
      val_eikon_eps_stddev_estimate = first_non_missing(val_eikon_eps_stddev_estimate),
      val_surprise = first_non_missing(val_surprise),
      market_cap_usd_asof = first_non_missing(market_cap_usd_asof),
      analysts_covering_asof = first_non_missing(analysts_covering_asof),
      turnover_6m_avg_daily_volume = first_non_missing(turnover_6m_avg_daily_volume),
      volatility_6m = first_non_missing(volatility_6m),
      open_time_days = first_non_missing(open_time_days),
      .groups = "drop"
    ) %>%
    mutate(
      log_volume = safe_log1p(volumeNum),
      log_volume_per_analyst = safe_log1p(volume_per_analyst),
      eps_stddev = val_eikon_eps_stddev_estimate,
      surprise = val_surprise,
      log_market_cap = safe_log(market_cap_usd_asof),
      analysts = analysts_covering_asof,
      log_turnover = safe_log1p(turnover_6m_avg_daily_volume),
      log_volatility_6m = safe_log1p(volatility_6m),
      open_time_days = open_time_days
    )

  if (isTRUE(winsorize)) {
  winsor_vars <- c(
    "log_volume",
    "log_volume_per_analyst",
    "eps_stddev",
    "surprise",
    "log_market_cap",
    "analysts",
    "log_turnover",
    "log_volatility_6m",
    "open_time_days"
  )
    
    market_df[winsor_vars] <- lapply(
      market_df[winsor_vars],
      winsorize_vec,
      probs = winsor_probs
    )
  }
  
  write.csv(
    market_df,
    file.path(output_dir, "factor_analysis_market_level.csv"),
    row.names = FALSE
  )
  write_jsonl(
    market_df,
    file.path(output_dir, "factor_analysis_market_level.jsonl")
  )
  
  # ------------------------------ #
  # 7. Define model variable sets
  # ------------------------------ #
  ex_ante_vars <- c(
    "eps_stddev",
    "log_market_cap",
    "analysts",
    "log_turnover",
    "log_volatility_6m",
    "open_time_days"
  )
  
  full_vars <- c(
    "log_volume",
    "log_volume_per_analyst",
    "eps_stddev",
    "surprise",
    "log_market_cap",
    "analysts",
    "log_turnover",
    "log_volatility_6m",
    "open_time_days"
  )
  
  coef_map <- c(
    "(Intercept)"            = "Constant",
    "log_volume"             = "log(Polymarket volume + 1)",
    "log_volume_per_analyst" = "log(Polymarket volume / analysts + 1)",
    "eps_stddev"             = "Std. dev. of analyst estimates",
    "surprise"               = "Earnings surprise",
    "log_market_cap"         = "log(Market cap)",
    "analysts"               = "Analysts covering",
    "log_turnover"           = "log(6m avg. daily turnover + 1)",
    "log_volatility_6m"      = "log(6m stock volatility)",
    "open_time_days"         = "Market open-to-resolution (days)"
  )
  
  # ------------------------------ #
  # 8. Build estimation samples
  # ------------------------------ #
  mean_df <- market_df
  
  if (isTRUE(require_complete_panel_for_mean)) {
    mean_df <- mean_df %>%
      filter(n_snapshots == length(selected_horizons))
  }
  
  mean_ex_ante_df <- mean_df[complete.cases(mean_df[, c("loss_mean", ex_ante_vars)]), ]
  mean_full_df    <- mean_df[complete.cases(mean_df[, c("loss_mean", full_vars)]), ]
  
  target_ex_ante_df <- market_df[complete.cases(market_df[, c("loss_target", ex_ante_vars)]), ]
  target_full_df    <- market_df[complete.cases(market_df[, c("loss_target", full_vars)]), ]
  
  if (nrow(mean_ex_ante_df) == 0 || nrow(mean_full_df) == 0 ||
      nrow(target_ex_ante_df) == 0 || nrow(target_full_df) == 0) {
    stop(
      "At least one model has no usable observations after filtering and complete-case selection.",
      call. = FALSE
    )
  }
  
  cat("\n================ SAMPLE SUMMARY (MARKET LEVEL) ================\n")
  cat("Mean loss, ex-ante model markets: ", nrow(mean_ex_ante_df), "\n", sep = "")
  cat("Mean loss, full model markets: ", nrow(mean_full_df), "\n", sep = "")
  cat("Loss at ", robustness_horizon, ", ex-ante model markets: ", nrow(target_ex_ante_df), "\n", sep = "")
  cat("Loss at ", robustness_horizon, ", full model markets: ", nrow(target_full_df), "\n", sep = "")
  cat("Negative coefficients imply higher accuracy because the dependent variable is Brier loss.\n")
  
  # ------------------------------ #
  # 9. Estimate OLS + fractional response models
  # ------------------------------ #
  formula_mean_ex_ante   <- reformulate(ex_ante_vars, response = "loss_mean")
  formula_mean_full      <- reformulate(full_vars, response = "loss_mean")
  formula_target_ex_ante <- reformulate(ex_ante_vars, response = "loss_target")
  formula_target_full    <- reformulate(full_vars, response = "loss_target")
  
  check_unit_interval <- function(x, x_name) {
    bad <- is.finite(x) & (x < 0 | x > 1)
    if (any(bad)) {
      stop(x_name, " contains values outside [0, 1].", call. = FALSE)
    }
  }
  
  check_unit_interval(mean_ex_ante_df$loss_mean, "mean_ex_ante_df$loss_mean")
  check_unit_interval(mean_full_df$loss_mean, "mean_full_df$loss_mean")
  check_unit_interval(target_ex_ante_df$loss_target, "target_ex_ante_df$loss_target")
  check_unit_interval(target_full_df$loss_target, "target_full_df$loss_target")
  
  # OLS
  model_mean_ex_ante   <- lm(formula_mean_ex_ante, data = mean_ex_ante_df)
  model_mean_full      <- lm(formula_mean_full, data = mean_full_df)
  model_target_ex_ante <- lm(formula_target_ex_ante, data = target_ex_ante_df)
  model_target_full    <- lm(formula_target_full, data = target_full_df)
  
  # Fractional logit
  model_mean_ex_ante_logit <- glm(
    formula_mean_ex_ante,
    data = mean_ex_ante_df,
    family = quasibinomial(link = "logit")
  )
  model_mean_full_logit <- glm(
    formula_mean_full,
    data = mean_full_df,
    family = quasibinomial(link = "logit")
  )
  model_target_ex_ante_logit <- glm(
    formula_target_ex_ante,
    data = target_ex_ante_df,
    family = quasibinomial(link = "logit")
  )
  model_target_full_logit <- glm(
    formula_target_full,
    data = target_full_df,
    family = quasibinomial(link = "logit")
  )
  
  # Fractional probit
  model_mean_ex_ante_probit <- glm(
    formula_mean_ex_ante,
    data = mean_ex_ante_df,
    family = quasibinomial(link = "probit")
  )
  model_mean_full_probit <- glm(
    formula_mean_full,
    data = mean_full_df,
    family = quasibinomial(link = "probit")
  )
  model_target_ex_ante_probit <- glm(
    formula_target_ex_ante,
    data = target_ex_ante_df,
    family = quasibinomial(link = "probit")
  )
  model_target_full_probit <- glm(
    formula_target_full,
    data = target_full_df,
    family = quasibinomial(link = "probit")
  )
  
  # Robust covariance matrices
  vcov_mean_ex_ante   <- sandwich::vcovHC(model_mean_ex_ante, type = "HC3")
  vcov_mean_full      <- sandwich::vcovHC(model_mean_full, type = "HC3")
  vcov_target_ex_ante <- sandwich::vcovHC(model_target_ex_ante, type = "HC3")
  vcov_target_full    <- sandwich::vcovHC(model_target_full, type = "HC3")
  
  vcov_mean_ex_ante_logit   <- sandwich::vcovHC(model_mean_ex_ante_logit, type = "HC3")
  vcov_mean_full_logit      <- sandwich::vcovHC(model_mean_full_logit, type = "HC3")
  vcov_target_ex_ante_logit <- sandwich::vcovHC(model_target_ex_ante_logit, type = "HC3")
  vcov_target_full_logit    <- sandwich::vcovHC(model_target_full_logit, type = "HC3")
  
  vcov_mean_ex_ante_probit   <- sandwich::vcovHC(model_mean_ex_ante_probit, type = "HC3")
  vcov_mean_full_probit      <- sandwich::vcovHC(model_mean_full_probit, type = "HC3")
  vcov_target_ex_ante_probit <- sandwich::vcovHC(model_target_ex_ante_probit, type = "HC3")
  vcov_target_full_probit    <- sandwich::vcovHC(model_target_full_probit, type = "HC3")

  cat("\nRunning bootstrap confidence intervals...\n")

  bootstrap_results <- bind_rows(
    bootstrap_ci_model(
      formula_mean_ex_ante, mean_ex_ante_df,
      fit_type = "lm",
      R = bootstrap_runs,
      seed = bootstrap_seed + 1,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = "OLS Mean loss | Ex-ante"
    ) %>% mutate(
      model_class = "OLS",
      model = "Mean loss | Ex-ante",
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_mean_full, mean_full_df,
      fit_type = "lm",
      R = bootstrap_runs,
      seed = bootstrap_seed + 2,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = "OLS Mean loss | Full"
    ) %>% mutate(
      model_class = "OLS",
      model = "Mean loss | Full",
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_target_ex_ante, target_ex_ante_df,
      fit_type = "lm",
      R = bootstrap_runs,
      seed = bootstrap_seed + 3,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = paste0("OLS Loss at ", robustness_horizon, " | Ex-ante")
    ) %>% mutate(
      model_class = "OLS",
      model = paste0("Loss at ", robustness_horizon, " | Ex-ante"),
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_target_full, target_full_df,
      fit_type = "lm",
      R = bootstrap_runs,
      seed = bootstrap_seed + 4,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = paste0("OLS Loss at ", robustness_horizon, " | Full")
    ) %>% mutate(
      model_class = "OLS",
      model = paste0("Loss at ", robustness_horizon, " | Full"),
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_mean_ex_ante, mean_ex_ante_df,
      fit_type = "glm",
      link = "logit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 5,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = "Logit Mean loss | Ex-ante"
    ) %>% mutate(
      model_class = "Fractional logit",
      model = "Mean loss | Ex-ante",
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_mean_full, mean_full_df,
      fit_type = "glm",
      link = "logit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 6,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = "Logit Mean loss | Full"
    ) %>% mutate(
      model_class = "Fractional logit",
      model = "Mean loss | Full",
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_target_ex_ante, target_ex_ante_df,
      fit_type = "glm",
      link = "logit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 7,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = paste0("Logit Loss at ", robustness_horizon, " | Ex-ante")
    ) %>% mutate(
      model_class = "Fractional logit",
      model = paste0("Loss at ", robustness_horizon, " | Ex-ante"),
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_target_full, target_full_df,
      fit_type = "glm",
      link = "logit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 8,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = paste0("Logit Loss at ", robustness_horizon, " | Full")
    ) %>% mutate(
      model_class = "Fractional logit",
      model = paste0("Loss at ", robustness_horizon, " | Full"),
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_mean_ex_ante, mean_ex_ante_df,
      fit_type = "glm",
      link = "probit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 9,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = "Probit Mean loss | Ex-ante"
    ) %>% mutate(
      model_class = "Fractional probit",
      model = "Mean loss | Ex-ante",
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_mean_full, mean_full_df,
      fit_type = "glm",
      link = "probit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 10,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = "Probit Mean loss | Full"
    ) %>% mutate(
      model_class = "Fractional probit",
      model = "Mean loss | Full",
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_target_ex_ante, target_ex_ante_df,
      fit_type = "glm",
      link = "probit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 11,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = paste0("Probit Loss at ", robustness_horizon, " | Ex-ante")
    ) %>% mutate(
      model_class = "Fractional probit",
      model = paste0("Loss at ", robustness_horizon, " | Ex-ante"),
      .before = 1
    ),
    
    bootstrap_ci_model(
      formula_target_full, target_full_df,
      fit_type = "glm",
      link = "probit",
      R = bootstrap_runs,
      seed = bootstrap_seed + 12,
      conf_level = bootstrap_conf_level,
      show_progress = TRUE,
      progress_label = paste0("Probit Loss at ", robustness_horizon, " | Full")
    ) %>% mutate(
      model_class = "Fractional probit",
      model = paste0("Loss at ", robustness_horizon, " | Full"),
      .before = 1
    )
  )
  
  # ------------------------------ #
  # 10. Console regression tables
  # ------------------------------ #
  ols_gof_map <- data.frame(
    raw   = c("nobs", "r.squared", "adj.r.squared", "sigma", "statistic", "p.value"),
    clean = c("N", "R²", "Adj. R²", "Residual SD", "F statistic", "Model p-value"),
    fmt   = c(0, 3, 3, 3, 2, 3),
    stringsAsFactors = FALSE
  )
  
  glm_gof_map <- data.frame(
    raw   = c("nobs", "deviance", "df.residual", "AIC"),
    clean = c("N", "Residual deviance", "Residual df", "AIC"),
    fmt   = c(0, 3, 0, 3),
    stringsAsFactors = FALSE
  )
  
  ols_models_list <- list(
    model_mean_ex_ante,
    model_mean_full,
    model_target_ex_ante,
    model_target_full
  )
  names(ols_models_list) <- c(
    "Mean loss\nEx-ante",
    "Mean loss\nFull",
    paste0("Loss at ", robustness_horizon, "\nEx-ante"),
    paste0("Loss at ", robustness_horizon, "\nFull")
  )
  
  ols_vcov_list <- list(
    vcov_mean_ex_ante,
    vcov_mean_full,
    vcov_target_ex_ante,
    vcov_target_full
  )
  
  logit_models_list <- list(
    model_mean_ex_ante_logit,
    model_mean_full_logit,
    model_target_ex_ante_logit,
    model_target_full_logit
  )
  names(logit_models_list) <- names(ols_models_list)
  
  logit_vcov_list <- list(
    vcov_mean_ex_ante_logit,
    vcov_mean_full_logit,
    vcov_target_ex_ante_logit,
    vcov_target_full_logit
  )
  
  probit_models_list <- list(
    model_mean_ex_ante_probit,
    model_mean_full_probit,
    model_target_ex_ante_probit,
    model_target_full_probit
  )
  names(probit_models_list) <- names(ols_models_list)
  
  probit_vcov_list <- list(
    vcov_mean_ex_ante_probit,
    vcov_mean_full_probit,
    vcov_target_ex_ante_probit,
    vcov_target_full_probit
  )
  
  cat("\n================ OLS REGRESSION TABLE ================\n")
  print(
    modelsummary::msummary(
      models = ols_models_list,
      vcov = ols_vcov_list,
      estimate = "{estimate}{stars}",
      statistic = "({std.error})",
      coef_map = coef_map,
      coef_omit = NULL,
      gof_map = ols_gof_map,
      stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
      fmt = 4,
      output = "markdown"
    )
  )
  
  cat("\n================ FRACTIONAL LOGIT TABLE ================\n")
  print(
    modelsummary::msummary(
      models = logit_models_list,
      vcov = logit_vcov_list,
      estimate = "{estimate}{stars}",
      statistic = "({std.error})",
      coef_map = coef_map,
      coef_omit = NULL,
      gof_map = glm_gof_map,
      stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
      fmt = 4,
      output = "markdown"
    )
  )
  
  cat("\n================ FRACTIONAL PROBIT TABLE ================\n")
  print(
    modelsummary::msummary(
      models = probit_models_list,
      vcov = probit_vcov_list,
      estimate = "{estimate}{stars}",
      statistic = "({std.error})",
      coef_map = coef_map,
      coef_omit = NULL,
      gof_map = glm_gof_map,
      stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
      fmt = 4,
      output = "markdown"
    )
  )
  
  # ------------------------------ #
  # 11. HTML regression tables
  # ------------------------------ #
  gt_tbl_ols <- modelsummary::msummary(
    models = ols_models_list,
    vcov = ols_vcov_list,
    estimate = "{estimate}{stars}",
    statistic = "({std.error})",
    coef_map = coef_map,
    coef_omit = NULL,
    gof_map = ols_gof_map,
    stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
    fmt = 4,
    output = "gt"
  ) %>%
    gt::tab_header(
      title = gt::md("Factors associated with Polymarket accuracy: OLS"),
      subtitle = gt::md(
        paste0(
          "Cross-sectional OLS with one row per market. ",
          "Dependent variable is Polymarket Brier loss, so lower values indicate higher accuracy. ",
          "HC3 robust standard errors in parentheses."
        )
      )
    ) %>%
    gt::tab_source_note(
      source_note = gt::md(
        "Significance: * p < 0.10, ** p < 0.05, *** p < 0.01. Negative coefficients imply higher accuracy because the dependent variable is Brier loss."
      )
    ) %>%
    gt::opt_row_striping() %>%
    gt::tab_options(
      table.font.size = 12,
      data_row.padding = gt::px(6),
      heading.align = "center"
    )
  
  gt_tbl_logit <- modelsummary::msummary(
    models = logit_models_list,
    vcov = logit_vcov_list,
    estimate = "{estimate}{stars}",
    statistic = "({std.error})",
    coef_map = coef_map,
    coef_omit = NULL,
    gof_map = glm_gof_map,
    stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
    fmt = 4,
    output = "gt"
  ) %>%
    gt::tab_header(
      title = gt::md("Factors associated with Polymarket accuracy: Fractional logit"),
      subtitle = gt::md(
        "Fractional response model with logit link and HC3 robust standard errors."
      )
    ) %>%
    gt::tab_source_note(
      source_note = gt::md(
        "Because the dependent variable is a bounded fractional outcome in [0,1], coefficients are on the logit link scale and are not directly comparable in magnitude to OLS coefficients."
      )
    ) %>%
    gt::opt_row_striping() %>%
    gt::tab_options(
      table.font.size = 12,
      data_row.padding = gt::px(6),
      heading.align = "center"
    )
  
  gt_tbl_probit <- modelsummary::msummary(
    models = probit_models_list,
    vcov = probit_vcov_list,
    estimate = "{estimate}{stars}",
    statistic = "({std.error})",
    coef_map = coef_map,
    coef_omit = NULL,
    gof_map = glm_gof_map,
    stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
    fmt = 4,
    output = "gt"
  ) %>%
    gt::tab_header(
      title = gt::md("Factors associated with Polymarket accuracy: Fractional probit"),
      subtitle = gt::md(
        "Fractional response model with probit link and HC3 robust standard errors."
      )
    ) %>%
    gt::tab_source_note(
      source_note = gt::md(
        "Because the dependent variable is a bounded fractional outcome in [0,1], coefficients are on the probit link scale and are not directly comparable in magnitude to OLS coefficients."
      )
    ) %>%
    gt::opt_row_striping() %>%
    gt::tab_options(
      table.font.size = 12,
      data_row.padding = gt::px(6),
      heading.align = "center"
    )
  
  gt::gtsave(
    data = gt_tbl_ols,
    filename = file.path(output_dir, "factor_analysis_regression_table.html")
  )
  gt::gtsave(
    data = gt_tbl_ols,
    filename = file.path(output_dir, "factor_analysis_regression_table_ols.html")
  )
  gt::gtsave(
    data = gt_tbl_logit,
    filename = file.path(output_dir, "factor_analysis_regression_table_logit.html")
  )
  gt::gtsave(
    data = gt_tbl_probit,
    filename = file.path(output_dir, "factor_analysis_regression_table_probit.html")
  )
  
  # ------------------------------ #
  # 12. Save coefficient and fit outputs
  # ------------------------------ #
  coef_results <- bind_rows(
    tidy_lm_hc3(model_mean_ex_ante, "Mean loss | Ex-ante") %>% mutate(model_class = "OLS", .before = 1),
    tidy_lm_hc3(model_mean_full, "Mean loss | Full") %>% mutate(model_class = "OLS", .before = 1),
    tidy_lm_hc3(model_target_ex_ante, paste0("Loss at ", robustness_horizon, " | Ex-ante")) %>% mutate(model_class = "OLS", .before = 1),
    tidy_lm_hc3(model_target_full, paste0("Loss at ", robustness_horizon, " | Full")) %>% mutate(model_class = "OLS", .before = 1),
    
    tidy_glm_hc3(model_mean_ex_ante_logit, "Mean loss | Ex-ante", "Fractional logit"),
    tidy_glm_hc3(model_mean_full_logit, "Mean loss | Full", "Fractional logit"),
    tidy_glm_hc3(model_target_ex_ante_logit, paste0("Loss at ", robustness_horizon, " | Ex-ante"), "Fractional logit"),
    tidy_glm_hc3(model_target_full_logit, paste0("Loss at ", robustness_horizon, " | Full"), "Fractional logit"),
    
    tidy_glm_hc3(model_mean_ex_ante_probit, "Mean loss | Ex-ante", "Fractional probit"),
    tidy_glm_hc3(model_mean_full_probit, "Mean loss | Full", "Fractional probit"),
    tidy_glm_hc3(model_target_ex_ante_probit, paste0("Loss at ", robustness_horizon, " | Ex-ante"), "Fractional probit"),
    tidy_glm_hc3(model_target_full_probit, paste0("Loss at ", robustness_horizon, " | Full"), "Fractional probit")
  ) %>%
    mutate(
      term_label = dplyr::recode(term, !!!coef_map),
      outcome_group = case_when(
        grepl("^Mean loss", model) ~ "Mean loss across selected horizons",
        grepl(paste0("^Loss at ", robustness_horizon), model) ~ paste0("Loss at ", robustness_horizon),
        TRUE ~ "Other"
      ),
      spec_group = case_when(
        grepl("Ex-ante$", model) ~ "Ex-ante",
        grepl("Full$", model) ~ "Full",
        TRUE ~ "Other"
      )
    ) %>%
    left_join(
      bootstrap_results,
      by = c("model_class", "model", "term")
    ) %>%
    mutate(
      conf.low_hc3 = conf.low,
      conf.high_hc3 = conf.high,
      conf.low = conf.low_boot,
      conf.high = conf.high_boot
    )
  
  fit_results <- bind_rows(
    broom::glance(model_mean_ex_ante) %>% mutate(model_class = "OLS", model = "Mean loss | Ex-ante"),
    broom::glance(model_mean_full) %>% mutate(model_class = "OLS", model = "Mean loss | Full"),
    broom::glance(model_target_ex_ante) %>% mutate(model_class = "OLS", model = paste0("Loss at ", robustness_horizon, " | Ex-ante")),
    broom::glance(model_target_full) %>% mutate(model_class = "OLS", model = paste0("Loss at ", robustness_horizon, " | Full")),
    
    glance_fractional_model(model_mean_ex_ante_logit, "Mean loss | Ex-ante", "Fractional logit"),
    glance_fractional_model(model_mean_full_logit, "Mean loss | Full", "Fractional logit"),
    glance_fractional_model(model_target_ex_ante_logit, paste0("Loss at ", robustness_horizon, " | Ex-ante"), "Fractional logit"),
    glance_fractional_model(model_target_full_logit, paste0("Loss at ", robustness_horizon, " | Full"), "Fractional logit"),
    
    glance_fractional_model(model_mean_ex_ante_probit, "Mean loss | Ex-ante", "Fractional probit"),
    glance_fractional_model(model_mean_full_probit, "Mean loss | Full", "Fractional probit"),
    glance_fractional_model(model_target_ex_ante_probit, paste0("Loss at ", robustness_horizon, " | Ex-ante"), "Fractional probit"),
    glance_fractional_model(model_target_full_probit, paste0("Loss at ", robustness_horizon, " | Full"), "Fractional probit")
  ) %>%
    select(
      model_class, model,
      any_of(c(
        "r.squared", "adj.r.squared", "sigma", "statistic", "p.value",
        "null.deviance", "deviance", "dispersion", "pseudo_r2",
        "df.residual", "nobs", "AIC", "BIC"
      ))
    )
  
  write.csv(
    coef_results,
    file.path(output_dir, "factor_analysis_regression_coefficients.csv"),
    row.names = FALSE
  )
  write_jsonl(
    coef_results,
    file.path(output_dir, "factor_analysis_regression_coefficients.jsonl")
  )
  
  write.csv(
    fit_results,
    file.path(output_dir, "factor_analysis_model_fit.csv"),
    row.names = FALSE
  )
  write_jsonl(
    fit_results,
    file.path(output_dir, "factor_analysis_model_fit.jsonl")
  )

    write.csv(
    bootstrap_results,
    file.path(output_dir, "factor_analysis_bootstrap_coefficients.csv"),
    row.names = FALSE
  )
  write_jsonl(
    bootstrap_results,
    file.path(output_dir, "factor_analysis_bootstrap_coefficients.jsonl")
  )
  
  # ------------------------------ #
  # 12B. Diagnostics for OLS + GLM
  # ------------------------------ #
  diagnostics_list <- list(
    run_lm_diagnostics(model_mean_ex_ante, "OLS Mean loss | Ex-ante", diagnostics_dir),
    run_lm_diagnostics(model_mean_full, "OLS Mean loss | Full", diagnostics_dir),
    run_lm_diagnostics(model_target_ex_ante, paste0("OLS Loss at ", robustness_horizon, " | Ex-ante"), diagnostics_dir),
    run_lm_diagnostics(model_target_full, paste0("OLS Loss at ", robustness_horizon, " | Full"), diagnostics_dir),
    
    run_glm_diagnostics(model_mean_ex_ante_logit, "Logit Mean loss | Ex-ante", diagnostics_dir),
    run_glm_diagnostics(model_mean_full_logit, "Logit Mean loss | Full", diagnostics_dir),
    run_glm_diagnostics(model_target_ex_ante_logit, paste0("Logit Loss at ", robustness_horizon, " | Ex-ante"), diagnostics_dir),
    run_glm_diagnostics(model_target_full_logit, paste0("Logit Loss at ", robustness_horizon, " | Full"), diagnostics_dir),
    
    run_glm_diagnostics(model_mean_ex_ante_probit, "Probit Mean loss | Ex-ante", diagnostics_dir),
    run_glm_diagnostics(model_mean_full_probit, "Probit Mean loss | Full", diagnostics_dir),
    run_glm_diagnostics(model_target_ex_ante_probit, paste0("Probit Loss at ", robustness_horizon, " | Ex-ante"), diagnostics_dir),
    run_glm_diagnostics(model_target_full_probit, paste0("Probit Loss at ", robustness_horizon, " | Full"), diagnostics_dir)
  )
  
  # ------------------------------ #
  # 13. Plot data and coefficient plot
  # ------------------------------ #
  plot_data <- coef_results %>%
    filter(term != "(Intercept)") %>%
    mutate(
      term_label = factor(
        term_label,
        levels = rev(c(
          "log(Polymarket volume + 1)",
          "log(Polymarket volume / analysts + 1)",
          "Std. dev. of analyst estimates",
          "Earnings surprise",
          "log(Market cap)",
          "Analysts covering",
          "log(6m avg. daily turnover + 1)",
          "log(6m stock volatility)",
          "Market open-to-resolution (days)"
        ))
      ),
      model_plot = factor(
        model,
        levels = c(
          "Mean loss | Ex-ante",
          "Mean loss | Full",
          paste0("Loss at ", robustness_horizon, " | Ex-ante"),
          paste0("Loss at ", robustness_horizon, " | Full")
        )
      ),
      model_class = factor(
        model_class,
        levels = c("OLS", "Fractional logit", "Fractional probit")
      )
    )
  
  write.csv(
    plot_data,
    file.path(output_dir, "factor_analysis_plot_data.csv"),
    row.names = FALSE
  )
  write_jsonl(
    plot_data,
    file.path(output_dir, "factor_analysis_plot_data.jsonl")
  )
  
  color_values <- c("#808080", "#A9A9A9", "#00008B", "#E3170A")
  names(color_values) <- c(
    "Mean loss | Ex-ante",
    "Mean loss | Full",
    paste0("Loss at ", robustness_horizon, " | Ex-ante"),
    paste0("Loss at ", robustness_horizon, " | Full")
  )
  
  coeff_plot <- ggplot(
    plot_data,
    aes(x = estimate, y = term_label, color = model_plot)
  ) +
    geom_vline(xintercept = 0, linewidth = 0.5, linetype = "dashed", color = "#0000FF") +
    geom_errorbarh(
      aes(xmin = conf.low, xmax = conf.high),
      position = position_dodge(width = 0.7),
      height = 0.20,
      linewidth = 0.5
    ) +
    geom_point(
      position = position_dodge(width = 0.7),
      size = 2.4
    ) +
    scale_color_manual(values = color_values) +
    facet_wrap(~ model_class, ncol = 1, scales = "free_x") +
    labs(
      title = "Estimated factor effects on Polymarket accuracy",
      subtitle = paste0(
        "OLS coefficients are on the Brier-loss scale. ",
        "Fractional logit and fractional probit coefficients are on link-function scales, ",
        "so magnitudes should not be compared directly across model classes."
      ),
      x = "Coefficient estimate",
      y = NULL,
      color = "Model"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold")
    )
  
  ggsave(
    filename = file.path(output_dir, "factor_analysis_coefficients_plot.png"),
    plot = coeff_plot,
    width = 12,
    height = 14,
    dpi = 300
  )

  cat("Bootstrap percentile confidence intervals based on ", bootstrap_runs, " resamples were added.\n", sep = "")
  cat("\nSaved outputs to:\n")
  cat(normalizePath(output_dir, winslash = "/", mustWork = FALSE), "\n", sep = "")
  cat("\nDone.\n")
  
  invisible(list(
    ols_models = list(
      model_mean_ex_ante = model_mean_ex_ante,
      model_mean_full = model_mean_full,
      model_target_ex_ante = model_target_ex_ante,
      model_target_full = model_target_full
    ),
    logit_models = list(
      model_mean_ex_ante_logit = model_mean_ex_ante_logit,
      model_mean_full_logit = model_mean_full_logit,
      model_target_ex_ante_logit = model_target_ex_ante_logit,
      model_target_full_logit = model_target_full_logit
    ),
    probit_models = list(
      model_mean_ex_ante_probit = model_mean_ex_ante_probit,
      model_mean_full_probit = model_mean_full_probit,
      model_target_ex_ante_probit = model_target_ex_ante_probit,
      model_target_full_probit = model_target_full_probit
    ),
    market_data = market_df,
    coefficients = coef_results,
    fit = fit_results,
    plot_data = plot_data,
    coefficient_plot = coeff_plot,
    output_dir = output_dir
  ))
}

# ------------------------------ #
# 14. Run
# ------------------------------ #
results <- run_factor_analysis()