# =============================================================================
# File:    Polymarket-Earnings-Study/R/brier_regression_test_statistics.R
# Purpose: Run one OLS regression per snapshot (horizon) to test which factors
#          affect the Polymarket Brier score (loss_polymarket).
# Output:  Stata-like tables printed to console + saved under:
#          Polymarket-Earnings-Study/statistics/test_statistics/
#
# Run:
#   - RStudio: Source this file
#   - Terminal: Rscript Polymarket-Earnings-Study/R/brier_regression_test_statistics.R
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
  stop("Could not find project root (renv.lock or .Rproj). Run from inside the project.", call. = FALSE)
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

# ---------- Load data (shared loader) ----------
source(file.path(ROOT, "R", "utils", "load_data.R"))
D <- load_project_data(ROOT)

dataset_long   <- D$dataset_long
stock_prices   <- D$stock_prices          # loaded for convenience / future use
heckman_events <- D$heckman_universe_events

# =============================================================================
# Model specification
# =============================================================================
dep_var <- "loss_polymarket"  # Polymarket Brier score = (p - y)^2

x_vars <- c(
  "volumeNum",
  "val_eikon_eps_stddev_estimate",
  "analysts_covering_asof",
  "turnover_6m_avg_daily_volume",
  "volatility_6m"
)

# Required columns
needed <- c("horizon", dep_var, x_vars)
missing_cols <- setdiff(needed, names(dataset_long))
if (length(missing_cols) > 0) {
  stop(
    "dataset_long is missing required columns: ",
    paste(missing_cols, collapse = ", "),
    call. = FALSE
  )
}

# =============================================================================
# Stata-like regression table printer (base R)
# =============================================================================

fmt_int <- function(x) format(as.integer(x), big.mark = ",", scientific = FALSE)

fmt_p4 <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.00005) return("0.0000")
  formatC(p, format = "f", digits = 4)
}

fmt_p3 <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.0005) return("0.000")
  formatC(p, format = "f", digits = 3)
}

fmt_2 <- function(x) {
  if (is.na(x)) return("")
  formatC(x, format = "f", digits = 2)
}

fmt_r2 <- function(x) {
  if (is.na(x)) return("")
  formatC(x, format = "f", digits = 4)
}

fmt_rootmse <- function(x) {
  if (is.na(x)) return("")
  s <- formatC(x, format = "f", digits = 5)
  sub("^0\\.", ".", s)
}

fmt_stata_num <- function(x, fixed_digits = 7, sci_digits = 2, sci_cut = 1e-4) {
  if (is.na(x)) return("")
  if (x == 0) {
    s <- formatC(0, format = "f", digits = fixed_digits)
  } else if (abs(x) < sci_cut) {
    s <- formatC(x, format = "e", digits = sci_digits)   # e.g., 2.48e-07
  } else {
    s <- formatC(x, format = "f", digits = fixed_digits) # e.g., 0.1814131
    # Stata-style leading dot for |x|<1
    if (abs(x) < 1) s <- sub("^(-?)0\\.", "\\1.", s)
  }
  s
}

stata_reg_table_lines <- function(model, depvar_label = NULL, cmd = NULL, extra_lines = NULL) {
  stopifnot(inherits(model, "lm"))
  
  sm <- summary(model)
  y  <- model.response(model.frame(model))
  e  <- residuals(model)
  yhat <- fitted(model)
  
  n <- length(y)
  k <- length(coef(model)) - 1
  df_m <- k
  df_r <- df.residual(model)
  df_t <- n - 1
  
  sse <- sum(e^2)
  ssr <- sum((yhat - mean(y))^2)
  sst <- sum((y - mean(y))^2)
  
  ms_m <- if (df_m > 0) ssr / df_m else NA_real_
  ms_r <- sse / df_r
  ms_t <- sst / df_t
  
  fstat <- if (df_m > 0) ms_m / ms_r else NA_real_
  pfval <- if (df_m > 0) stats::pf(fstat, df_m, df_r, lower.tail = FALSE) else NA_real_
  
  r2  <- sm$r.squared
  ar2 <- sm$adj.r.squared
  rmse <- sqrt(ms_r)
  
  # correlation between residuals and dependent variable
  corr_resid_y <- suppressWarnings(stats::cor(e, y, use = "complete.obs"))
  
  coef_mat <- sm$coefficients
  ci <- confint(model, level = 0.95)
  
  # variable names in Stata style
  vars <- rownames(coef_mat)
  vars_disp <- ifelse(vars == "(Intercept)", "_cons", vars)
  
  dep_disp <- if (!is.null(depvar_label) && nzchar(depvar_label)) depvar_label else all.vars(formula(model))[1]
  cmd_line <- if (!is.null(cmd) && nzchar(cmd)) paste0(" . ", cmd) else NULL
  
  lines <- character(0)
  if (!is.null(cmd_line)) lines <- c(lines, cmd_line, "")
  
  # ANOVA-style block
  lines <- c(
    lines,
    sprintf("      Source |%13s%9s%13s", "SS", "df", "MS"),
    "-------------+----------------------------------",
    sprintf("%12s | %12s %9d %12s", "Model",    fmt_stata_num(ssr, fixed_digits = 8), df_m, fmt_stata_num(ms_m, fixed_digits = 9)),
    sprintf("%12s | %12s %9d %12s", "Residual", fmt_stata_num(sse, fixed_digits = 8), df_r, fmt_stata_num(ms_r, fixed_digits = 9)),
    "-------------+----------------------------------",
    sprintf("%12s | %12s %9d %12s", "Total",    fmt_stata_num(sst, fixed_digits = 8), df_t, fmt_stata_num(ms_t, fixed_digits = 9)),
    ""
  )
  
  # Summary metrics
  lines <- c(
    lines,
    sprintf("  Number of obs = %s", fmt_int(n)),
    if (df_m > 0) sprintf("        F(%d, %d) = %s", df_m, df_r, fmt_2(fstat)) else "        F(., .) = .",
    sprintf("   Prob > F = %s", fmt_p4(pfval)),
    sprintf("   R-squared = %s", fmt_r2(r2)),
    sprintf("Adj R-squared = %s", fmt_r2(ar2)),
    sprintf("    Root MSE = %s", fmt_rootmse(rmse)),
    sprintf("Corr(resid, y) = %.4f", corr_resid_y)
  )
  
  if (!is.null(extra_lines) && length(extra_lines) > 0) {
    lines <- c(lines, extra_lines)
  }
  
  # Coefficient table
  lines <- c(
    lines,
    "",
    sprintf("      %s |%12s%12s%8s%8s%25s", dep_disp, "Coefficient", "Std. err.", "t", "P>|t|", "[95% conf. interval]"),
    "-------------+----------------------------------------------------------------"
  )
  
  for (i in seq_len(nrow(coef_mat))) {
    b  <- coef_mat[i, 1]
    se <- coef_mat[i, 2]
    t  <- coef_mat[i, 3]
    p  <- coef_mat[i, 4]
    
    lo <- ci[i, 1]
    hi <- ci[i, 2]
    
    lines <- c(
      lines,
      sprintf(
        "%12s | %12s %12s %8s %8s %12s %12s",
        vars_disp[i],
        fmt_stata_num(b),
        fmt_stata_num(se),
        fmt_2(t),
        fmt_p3(p),
        fmt_stata_num(lo),
        fmt_stata_num(hi)
      )
    )
  }
  
  c(lines, "")
}

write_stata_table <- function(lines, txt_path, html_path) {
  # Force print to console even when using source() in RStudio
  cat(paste(lines, collapse = "\n"), "\n")
  
  dir.create(dirname(txt_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, con = txt_path)
  
  html <- c("<html><head><meta charset='utf-8'></head><body><pre>",
            lines,
            "</pre></body></html>")
  writeLines(html, con = html_path)
}

safe_slug <- function(x) {
  x <- as.character(x)
  x <- gsub("[^A-Za-z0-9]+", "_", x)
  x <- gsub("_+", "_", x)
  gsub("^_|_$", "", x)
}

# =============================================================================
# Run one OLS per snapshot (horizon)
# =============================================================================

out_dir <- file.path(ROOT, "statistics", "test_statistics")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Choose ordering of horizons:
# Prefer horizon_seconds (numeric ordering) if available, else alphabetical.
if ("horizon_seconds" %in% names(dataset_long)) {
  tmp <- unique(dataset_long[, c("horizon", "horizon_seconds")])
  tmp <- tmp[!is.na(tmp$horizon), , drop = FALSE]
  if (any(!is.na(tmp$horizon_seconds))) {
    tmp <- tmp[order(tmp$horizon_seconds, tmp$horizon), , drop = FALSE]
    horizons <- as.character(tmp$horizon)
  } else {
    horizons <- sort(unique(as.character(dataset_long$horizon)))
  }
} else {
  horizons <- sort(unique(as.character(dataset_long$horizon)))
}

# Model formula
fml <- stats::as.formula(paste(dep_var, "~", paste(x_vars, collapse = " + ")))

# Run regressions
for (h in horizons) {
  df_h <- dataset_long[as.character(dataset_long$horizon) == h, , drop = FALSE]
  
  # Keep only relevant columns + complete cases
  df_h <- df_h[, needed, drop = FALSE]
  df_h <- df_h[stats::complete.cases(df_h), , drop = FALSE]
  
  # Drop non-finite y (rare but safe)
  df_h <- df_h[is.finite(df_h[[dep_var]]), , drop = FALSE]
  
  # Guardrail: require enough observations
  min_n <- length(x_vars) + 10
  if (nrow(df_h) < min_n) {
    msg <- paste0("Skipping horizon=", h, " because n=", nrow(df_h), " < ", min_n, " after filtering.")
    message(msg)
    writeLines(msg, con = file.path(out_dir, paste0("brier_regression__", safe_slug(h), "__SKIPPED.txt")))
    next
  }
  
  # Fit model
  mod <- tryCatch(stats::lm(fml, data = df_h), error = function(e) e)
  if (inherits(mod, "error")) {
    msg <- paste0("FAILED horizon=", h, " : ", mod$message)
    message(msg)
    writeLines(msg, con = file.path(out_dir, paste0("brier_regression__", safe_slug(h), "__FAILED.txt")))
    next
  }
  
  # Stata-like "command"
  cmd_label <- paste("regress", dep_var, paste(x_vars, collapse = " "))
  
  # Build output
  lines <- stata_reg_table_lines(
    model = mod,
    depvar_label = dep_var,
    cmd = cmd_label,
    extra_lines = c(paste0("Snapshot (horizon) = ", h))
  )
  
  txt_path  <- file.path(out_dir, paste0("brier_regression__", safe_slug(h), ".txt"))
  html_path <- file.path(out_dir, paste0("brier_regression__", safe_slug(h), ".html"))
  
  write_stata_table(lines, txt_path, html_path)
  
  message("Saved snapshot outputs for horizon=", h,
          "\n - ", txt_path,
          "\n - ", html_path)
}

message("Done. Output folder: ", out_dir)
