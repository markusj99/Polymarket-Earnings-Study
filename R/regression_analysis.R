# =============================================================================
# File:    Polymarket-Earnings-Study/R/brier_regression_test_statistics.R
# Purpose: Pooled OLS to test which factors affect Polymarket Brier loss,
#          controlling flexibly for probability level + horizon FE,
#          with cluster-robust SEs by market_id.
#
# Outputs: Tables + coefficient exports under:
#          Polymarket-Earnings-Study/statistics/regression/
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
dataset_long <- D$dataset_long

suppressPackageStartupMessages({
  library(dplyr)
  library(stats)
})

# ---- Optional packages ----
have_modelsummary <- requireNamespace("modelsummary", quietly = TRUE)
have_broom <- requireNamespace("broom", quietly = TRUE)
have_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
have_sandwich <- requireNamespace("sandwich", quietly = TRUE)
have_lmtest <- requireNamespace("lmtest", quietly = TRUE)

OUT_DIR <- file.path(ROOT, "statistics", "regression")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- Safety check ----
if (!exists("dataset_long")) stop("dataset_long not found. Load it before running this script.", call. = FALSE)
df <- dataset_long

# ---- Helpers ----
pick_col <- function(data, candidates) {
  for (nm in candidates) if (nm %in% names(data)) return(nm)
  return(NA_character_)
}

safe_log <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  out <- rep(NA_real_, length(x))
  ok <- is.finite(x) & (x > 0)
  out[ok] <- log(x[ok])
  out
}

parse_dt_utc <- function(x) {
  if (inherits(x, "POSIXct")) return(as.POSIXct(x, tz = "UTC"))
  if (inherits(x, "Date")) return(as.POSIXct(x, tz = "UTC"))
  
  if (is.numeric(x)) {
    v <- x
    v[is.na(v)] <- NA_real_
    is_ms <- is.finite(v) & v >= 1e12
    v[is_ms] <- v[is_ms] / 1000
    return(as.POSIXct(v, origin = "1970-01-01", tz = "UTC"))
  }
  
  s <- trimws(as.character(x))
  s[s == ""] <- NA_character_
  s <- ifelse(!is.na(s) & grepl("Z$", s), sub("Z$", "+00:00", s), s)
  suppressWarnings(as.POSIXct(s, tz = "UTC"))
}

write_jsonl <- function(path, df_rows) {
  if (!have_jsonlite) return(invisible(NULL))
  con <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  for (i in seq_len(nrow(df_rows))) {
    line <- jsonlite::toJSON(df_rows[i, , drop = FALSE], auto_unbox = TRUE, na = "null")
    writeLines(line, con)
  }
  invisible(NULL)
}

# Cluster-robust VCOV helper
cluster_vcov <- function(fit, cluster_vec) {
  if (!have_sandwich) return(stats::vcov(fit))
  
  cluster_vec <- as.factor(cluster_vec)
  n_cl <- length(levels(cluster_vec))
  if (n_cl < 2) return(stats::vcov(fit))
  
  # vcovCL handles clustering; HC1 is a common choice
  sandwich::vcovCL(fit, cluster = cluster_vec, type = "HC1")
}

# Tidy coefficients using robust vcov (if lmtest available)
tidy_with_vcov <- function(fit, V, model_name) {
  b <- coef(fit)
  se <- sqrt(diag(V))
  tval <- b / se
  
  if (have_lmtest) {
    ct <- lmtest::coeftest(fit, vcov. = V)
    out <- data.frame(
      model = model_name,
      term = rownames(ct),
      estimate = ct[, 1],
      std_error = ct[, 2],
      statistic = ct[, 3],
      p_value = ct[, 4],
      row.names = NULL
    )
    return(out)
  }
  
  pval <- 2 * (1 - pnorm(abs(tval)))
  data.frame(
    model = model_name,
    term = names(b),
    estimate = as.numeric(b),
    std_error = as.numeric(se),
    statistic = as.numeric(tval),
    p_value = as.numeric(pval),
    row.names = NULL
  )
}

# Add polynomial powers of p (raw polynomial terms p^2...p^degree)
add_p_powers <- function(data, p_col = "p_pm_yes", degree = 4) {
  if (!is.numeric(data[[p_col]])) data[[p_col]] <- suppressWarnings(as.numeric(data[[p_col]]))
  if (!is.finite(degree) || degree < 2) stop("degree must be >= 2", call. = FALSE)
  
  for (k in 2:degree) {
    nm <- paste0("p", k)
    data[[nm]] <- data[[p_col]]^k
  }
  data
}

# ---- Column mapping (flexible) ----
col_market_id <- pick_col(df, c("market_id", "id", "conditionId", "questionID"))
col_horizon   <- pick_col(df, c("horizon", "snapshot_label", "time_horizon"))
col_status    <- pick_col(df, c("status"))
col_p         <- pick_col(df, c("p_polymarket_yes", "p_yes", "prob_yes"))
col_brier     <- pick_col(df, c("loss_polymarket", "brier_loss", "brier", "loss"))

col_uma_end <- pick_col(df, c("umaEndDate", "uma_end_dt_utc", "uma_end_dt", "umaEndDate_utc"))
col_accept  <- pick_col(df, c("acceptingOrdersTimestamp", "accepting_orders_ts", "acceptingOrdersTimestamp_utc"))

needed <- c(col_market_id, col_horizon, col_status, col_p, col_brier)
if (any(is.na(needed))) {
  stop(
    "Missing required columns. Found:\n",
    "  market_id col: ", col_market_id, "\n",
    "  horizon col:   ", col_horizon, "\n",
    "  status col:    ", col_status, "\n",
    "  p col:         ", col_p, "\n",
    "  brier col:     ", col_brier, "\n",
    "Please ensure dataset_long has these fields.",
    call. = FALSE
  )
}

# ---- Clean + construct variables ----
# (Minimal robustness: if a raw column is missing, return NA vector instead of erroring)
get_or_na <- function(data, nm) {
  if (nm %in% names(data)) data[[nm]] else rep(NA, nrow(data))
}

df2 <- df %>%
  mutate(
    market_id = as.character(.data[[col_market_id]]),
    horizon   = as.character(.data[[col_horizon]]),
    
    # treat non-ok as missing (per your rule)
    .status_ok = (.data[[col_status]] == "ok"),
    brier_loss = ifelse(.status_ok, suppressWarnings(as.numeric(.data[[col_brier]])), NA_real_),
    p_pm_yes   = ifelse(.status_ok, suppressWarnings(as.numeric(.data[[col_p]])), NA_real_),
    
    # bins for DESCRIPTIVE accuracy only (NOT used in OLS)
    price_bin = cut(
      p_pm_yes,
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0000001),
      include.lowest = TRUE,
      right = FALSE,
      labels = c("[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]")
    ),
    
    # logs
    log_volumeNum = safe_log(get_or_na(cur_data(), "volumeNum")),
    log_mcap_usd  = safe_log(get_or_na(cur_data(), "market_cap_usd_asof")),
    log_surprise  = safe_log(get_or_na(cur_data(), "val_surprise"))
  )

# open_time (days), if time columns exist
if (!is.na(col_uma_end) && !is.na(col_accept)) {
  uma_dt <- parse_dt_utc(df2[[col_uma_end]])
  acc_dt_parsed <- parse_dt_utc(df2[[col_accept]])
  df2 <- df2 %>% mutate(open_time_days = as.numeric(difftime(uma_dt, acc_dt_parsed, units = "days")))
} else {
  df2 <- df2 %>% mutate(open_time_days = NA_real_)
}

# ---- Exclude horizons ----
excluded <- c("4w", "3w", "2w")
df2 <- df2 %>% filter(!(horizon %in% excluded))

# ---- Horizon order (for fixed effects reference / reporting) ----
HORIZON_ORDER <- c("1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h")
h_avail <- unique(df2$horizon)
h_levels <- c(HORIZON_ORDER[HORIZON_ORDER %in% h_avail], sort(setdiff(h_avail, HORIZON_ORDER)))
df2 <- df2 %>% mutate(horizon = factor(horizon, levels = h_levels))

# ---- Price-bin accuracy table (descriptive) ----
acc_tab <- df2 %>%
  filter(is.finite(brier_loss), !is.na(price_bin)) %>%
  group_by(horizon, price_bin) %>%
  summarise(
    n = dplyr::n(),
    mean_brier = mean(brier_loss, na.rm = TRUE),
    sd_brier = sd(brier_loss, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(horizon, price_bin)

if (requireNamespace("readr", quietly = TRUE)) {
  readr::write_csv(acc_tab, file.path(OUT_DIR, "accuracy_by_price_bin.csv"))
} else {
  write.csv(acc_tab, file.path(OUT_DIR, "accuracy_by_price_bin.csv"), row.names = FALSE)
}
if (have_jsonlite) write_jsonl(file.path(OUT_DIR, "accuracy_by_price_bin.jsonl"), acc_tab)

# =============================================================================
# MAIN TEST: pooled OLS with (i) polynomial control for p, (ii) horizon FE,
#            (iii) clustered SEs by market_id
# =============================================================================

# --- Polynomial degree for p ---
# Degree 2 is the theoretical minimum for squared loss; degree 4 gives a flexible shape
# similar to the prior ns(..., df=4) choice.
P_DEGREE <- 4

# Covariates (your requested set)
X_vars <- c(
  "log_volumeNum",
  "val_eikon_eps_stddev_estimate",
  "analysts_covering_asof",
  "turnover_6m_avg_daily_volume",
  "volatility_6m",
  "log_mcap_usd",
  "log_surprise",
  "open_time_days"
)

# Keep only complete cases for pooled regression
df_pool <- df2 %>%
  select(market_id, horizon, brier_loss, p_pm_yes, all_of(X_vars)) %>%
  filter(is.finite(brier_loss), is.finite(p_pm_yes)) %>%
  # optional sanity: p should be in [0,1]
  filter(p_pm_yes >= 0, p_pm_yes <= 1) %>%
  filter(complete.cases(.))

if (nrow(df_pool) < 100) {
  stop("Too few usable pooled observations after filtering. Check data availability.", call. = FALSE)
}

# Add polynomial terms of p (p2...p{P_DEGREE})
df_pool <- add_p_powers(df_pool, p_col = "p_pm_yes", degree = P_DEGREE)
p_terms <- c("p_pm_yes", paste0("p", 2:P_DEGREE))

# Model 1: pooled, horizon FE + poly(p) + X
fml_main <- as.formula(paste(
  "brier_loss ~", paste(p_terms, collapse = " + "), "+ horizon +",
  paste(X_vars, collapse = " + ")
))
fit_main <- lm(fml_main, data = df_pool)

# Model 2: pooled with horizon-varying effects for TWO key variables
# (keeps table readable; expands only open_time_days and log_surprise by horizon)
fml_int <- as.formula(paste(
  "brier_loss ~", paste(p_terms, collapse = " + "), "+ horizon +",
  paste(setdiff(X_vars, c("open_time_days", "log_surprise")), collapse = " + "),
  "+ horizon:(open_time_days + log_surprise)"
))
fit_int <- lm(fml_int, data = df_pool)

# Clustered vcov matrices
V_main <- cluster_vcov(fit_main, df_pool$market_id)
V_int  <- cluster_vcov(fit_int,  df_pool$market_id)

# ---- Print + save regression tables ----
if (have_modelsummary) {
  models <- setNames(
    list(fit_main, fit_int),
    c(
      paste0("Pooled FE + poly(p), degree=", P_DEGREE),
      "Pooled + horizon interactions"
    )
  )
  
  vcovs <- list(V_main, V_int)
  
  # Save tables (keep filenames stable)
  modelsummary::modelsummary(
    models,
    vcov = vcovs,
    stars = TRUE,
    output = file.path(OUT_DIR, "ols_brier_pooled.html"),
    title = paste0("Pooled OLS: Brier loss ~ polynomial(p), degree=", P_DEGREE,
                   " + horizon FE + X (cluster SE by market_id)")
  )
  modelsummary::modelsummary(
    models,
    vcov = vcovs,
    stars = TRUE,
    output = file.path(OUT_DIR, "ols_brier_pooled.tex"),
    title = paste0("Pooled OLS: Brier loss ~ polynomial(p), degree=", P_DEGREE,
                   " + horizon FE + X (cluster SE by market_id)")
  )
  modelsummary::modelsummary(
    models,
    vcov = vcovs,
    stars = TRUE,
    output = file.path(OUT_DIR, "ols_brier_pooled.md"),
    title = paste0("Pooled OLS: Brier loss ~ polynomial(p), degree=", P_DEGREE,
                   " + horizon FE + X (cluster SE by market_id)")
  )
  
  # Console print
  tab_console <- modelsummary::modelsummary(models, vcov = vcovs, stars = TRUE, output = "markdown")
  print(tab_console)
} else {
  message("Package 'modelsummary' not installed; skipping formatted regression tables.")
  message("Install with: install.packages('modelsummary')")
}

# ---- Export coefficients with clustered SEs + fit stats ----
coef_main <- tidy_with_vcov(fit_main, V_main, paste0("Pooled FE + poly(p), degree=", P_DEGREE))
coef_int  <- tidy_with_vcov(fit_int,  V_int,  "Pooled + horizon interactions")
coef_out <- bind_rows(coef_main, coef_int)

if (requireNamespace("readr", quietly = TRUE)) {
  readr::write_csv(coef_out, file.path(OUT_DIR, "coefficients_pooled_clustered.csv"))
} else {
  write.csv(coef_out, file.path(OUT_DIR, "coefficients_pooled_clustered.csv"), row.names = FALSE)
}
if (have_jsonlite) write_jsonl(file.path(OUT_DIR, "coefficients_pooled_clustered.jsonl"), coef_out)

if (have_broom) {
  fitstats <- bind_rows(
    broom::glance(fit_main) %>% mutate(model = paste0("Pooled FE + poly(p), degree=", P_DEGREE)),
    broom::glance(fit_int)  %>% mutate(model = "Pooled + horizon interactions")
  )
  if (requireNamespace("readr", quietly = TRUE)) {
    readr::write_csv(fitstats, file.path(OUT_DIR, "fitstats_pooled.csv"))
  } else {
    write.csv(fitstats, file.path(OUT_DIR, "fitstats_pooled.csv"), row.names = FALSE)
  }
  if (have_jsonlite) write_jsonl(file.path(OUT_DIR, "fitstats_pooled.jsonl"), fitstats)
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# --- Choose a realistic p-range (avoid extreme tails if you want) ---
p_rng <- quantile(df_pool$p_pm_yes, probs = c(0.01, 0.99), na.rm = TRUE)
p_seq <- seq(from = max(0, p_rng[1]), to = min(1, p_rng[2]), length.out = 200)

# --- Hold covariates fixed at medians ---
X_med <- df_pool %>%
  summarise(across(all_of(X_vars), ~ median(.x, na.rm = TRUE)))

# --- Prediction grid: p x horizon (others fixed) ---
pred_grid <- expand.grid(
  p_pm_yes = p_seq,
  horizon  = levels(df_pool$horizon)
) %>%
  as_tibble() %>%
  mutate(horizon = factor(horizon, levels = levels(df_pool$horizon))) %>%
  bind_cols(X_med[rep(1, nrow(.)), , drop = FALSE])

# Add polynomial terms for prediction to match the regression specification
pred_grid <- add_p_powers(pred_grid, p_col = "p_pm_yes", degree = P_DEGREE)

# --- Predict fitted values and pointwise SE (model-based) ---
pr <- predict(fit_main, newdata = pred_grid, se.fit = TRUE)

pred_grid <- pred_grid %>%
  mutate(
    brier_hat = as.numeric(pr$fit),
    se_hat    = as.numeric(pr$se.fit),
    ci_low    = brier_hat - 1.96 * se_hat,
    ci_high   = brier_hat + 1.96 * se_hat
  )

# --- Plot (facets by horizon) ---
g <- ggplot(pred_grid, aes(x = p_pm_yes, y = brier_hat)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), fill = "#A9A9A9", alpha = 0.5) +
  geom_line(color = "#0000FF", linewidth = 0.9) +
  facet_wrap(~ horizon, ncol = 3) +
  labs(
    x = "Polymarket implied probability p (YES)",
    y = "Predicted Brier loss",
    title = paste0("Predicted Brier loss vs implied probability (poly(p), degree=", P_DEGREE, "; covariates at medians)")
  ) +
  theme_minimal(base_size = 12)

print(g)

# --- Save ---
ggplot2::ggsave(
  filename = file.path(OUT_DIR, "predicted_brier_vs_p_facets.png"),
  plot = g, width = 10, height = 7.5, dpi = 300
)
ggplot2::ggsave(
  filename = file.path(OUT_DIR, "predicted_brier_vs_p_facets.pdf"),
  plot = g, width = 10, height = 7.5
)

message("\nDone. Wrote outputs to: ", OUT_DIR)
