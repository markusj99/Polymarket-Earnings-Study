#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/scripts/brier_score_factors.R
# Purpose: Explain which market/firm characteristics correlate with forecast
#          accuracy (Brier loss) across time horizons.
#
# Scientific choices implemented
# ------------------------------
# 1) Dependent variable: loss_polymarket = (p - y)^2 in [0, 1] (lower = better).
# 2) Main specification: OLS with horizon fixed effects + clustered SE by market id.
#    - This handles repeated observations per market across horizons.
# 3) Controls: sector FE + time FE (event-month) as robustness.
# 4) Robustness: run models separately by horizon.
# 5) Optional selection correction (Heckman-style control function):
#    - Probit selection on full event universe (heck_universe)
#    - Add inverse Mills ratio (IMR) to outcome regression on selected sample
#
# Inputs (relative to project root)
# --------------------------------
# - data/.../dataset_long.csv  (your long panel with loss_polymarket)
# - data/heckman_selection_model/heckman_universe_events.csv (or equivalent)
#
# Outputs (relative to project root)
# ---------------------------------
# - outputs/brier_factors/analysis_panel.{csv,jsonl}
# - outputs/brier_factors/analysis_final_snapshot.{csv,jsonl}
# - outputs/brier_factors/tables_main.tex
# - outputs/brier_factors/tables_by_horizon.tex
# - outputs/brier_factors/coefplot_volume_open_days.png
#
# Run (from project root)
# ----------------------
# Rscript R/scripts/brier_score_factors.R
#
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
  # (A) Rscript --file=...
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    p <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(p, winslash = "/", mustWork = FALSE)))
  }
  
  # (B) source("...") case
  ofile <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) "")
  if (is.character(ofile) && nzchar(ofile)) {
    return(dirname(normalizePath(ofile, winslash = "/", mustWork = FALSE)))
  }
  
  # (C) RStudio active document (optional)
  if (interactive() &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nzchar(p)) {
      return(dirname(normalizePath(p, winslash = "/", mustWork = FALSE)))
    }
  }
  
  # (D) Fallback
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

ROOT <- find_project_root(get_start_dir())
setwd(ROOT)

# ------------------------ Helper: robust timestamp parsing -------------------

parse_epoch_to_posix <- function(x, tz = "UTC") {
  # Accepts numeric epoch in seconds or milliseconds; returns POSIXct
  x_num <- suppressWarnings(as.numeric(x))
  out <- rep(as.POSIXct(NA), length(x_num))
  
  ok <- is.finite(x_num)
  if (!any(ok)) return(out)
  
  # Heuristic: if > 1e12 => milliseconds
  is_ms <- x_num[ok] > 1e12
  secs <- x_num[ok]
  secs[is_ms] <- secs[is_ms] / 1000
  
  out[ok] <- as.POSIXct(secs, origin = "1970-01-01", tz = tz)
  out
}

collapse_rare_levels <- function(x, min_n = 25, other = "Other") {
  # Make factor, collapse rare levels to "Other"
  x <- as.character(x)
  x[is.na(x) | trimws(x) == ""] <- "Unknown"
  tab <- table(x)
  rare <- names(tab)[tab < min_n]
  x[x %in% rare] <- other
  factor(x)
}

write_jsonl <- function(df, path) {
  con <- file(path, open = "wb")
  on.exit(close(con), add = TRUE)
  jsonlite::stream_out(df, con = con, verbose = FALSE)
}

# ------------------------------- I/O paths -----------------------------------

OUT_DIR <- file.path(ROOT, "statistics", "brier_factors")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# dt is already loaded above (via load_data.R fallback search).
# Sanity check:
if (is.null(dt) || nrow(dt) == 0) stop("dt is NULL/empty after loading dataset_long.")
message("Using ROOT: ", ROOT)
message("Rows in dataset_long: ", nrow(dt))

# -------------------- Heckman universe detection (optional) -------------------

heck_path <- NA_character_
has_heck <- FALSE

heck_candidates <- c(
  file.path(ROOT, "data", "heckman_selection_model", "heckman_universe_events.csv"),
  file.path(ROOT, "data", "heckman_selection_model", "heckman_universe_events.parquet"),
  file.path(ROOT, "data", "heckman_universe_events.csv")
)

heck_path <- heck_candidates[file.exists(heck_candidates)][1]

# Fallback: recursive search for heckman_universe_events.csv under /data
if (is.na(heck_path)) {
  data_root <- file.path(ROOT, "data")
  if (dir.exists(data_root)) {
    m <- list.files(
      data_root,
      pattern = "^heckman_universe_events\\.csv$",
      recursive = TRUE,
      full.names = TRUE,
      ignore.case = TRUE
    )
    if (length(m) > 0) {
      fi <- file.info(m)
      heck_path <- m[which.max(fi$mtime)]
    }
  }
}

has_heck <- !is.na(heck_path)

if (has_heck) {
  message("Using heck universe: ", heck_path)
} else {
  message("No heck_universe found (selection correction will be skipped).")
}



# ---------------------------- Feature engineering ----------------------------

# Parse timestamps/dates
dt[, umaEndDate := ymd_hms(umaEndDate, tz = "UTC", quiet = TRUE)]
dt[, acceptingOrders_dt_utc := parse_epoch_to_posix(acceptingOrdersTimestamp, tz = "UTC")]

# Market open length (in days)
dt[, market_open_days := as.numeric(difftime(umaEndDate, acceptingOrders_dt_utc, units = "days"))]
# Some feeds can have missing/odd timestamps; keep NAs rather than inventing
dt[!is.finite(market_open_days), market_open_days := NA_real_]

# Event-month fixed effects (time trends / regime changes)
dt[, event_month := format(as.Date(umaEndDate), "%Y-%m")]

# Core covariates (use log1p to safely handle zeros)
dt[, log_poly_volume := log1p(as.numeric(volumeNum))]
dt[, log_mcap := log(as.numeric(market_cap_usd_asof))]
dt[!is.finite(log_mcap), log_mcap := NA_real_]

dt[, log_analysts := log1p(as.numeric(analysts_covering_asof))]
dt[, log_turnover := log1p(as.numeric(turnover_6m_sum_volume))]
dt[, stock_volatility_6m := as.numeric(volatility_6m)]
dt[, abs_surprise := abs(as.numeric(val_surprise))]
dt[, eps_dispersion := as.numeric(val_eikon_eps_stddev_estimate)]

# Factors
dt[, horizon := as.factor(horizon)]
dt[, gics_sector := collapse_rare_levels(gics_sector, min_n = 25, other = "Other")]
dt[, event_month := as.factor(event_month)]

# Keep analysis columns
panel <- dt %>%
  transmute(
    id, ric, ticker,
    horizon, horizon_seconds,
    loss_polymarket,
    log_poly_volume, market_open_days,
    log_mcap, log_analysts, log_turnover,
    stock_volatility_6m, abs_surprise, eps_dispersion,
    gics_sector, event_month,
    umaEndDate
  ) %>%
  # Don’t silently drop everything; keep only rows usable for main regression
  filter(is.finite(loss_polymarket))

# Save analysis panel (CSV + JSONL)
panel_csv <- file.path(OUT_DIR, "analysis_panel.csv")
panel_jsonl <- file.path(OUT_DIR, "analysis_panel.jsonl")
data.table::fwrite(as.data.table(panel), panel_csv)
write_jsonl(panel, panel_jsonl)

# --------------------------- Main regression models --------------------------
# Interpretation: coefficients < 0 imply lower Brier loss => higher accuracy.

# Guard against degenerate factors (can happen after filtering)
if (nlevels(panel$horizon) < 2) stop("horizon has <2 levels after filtering; cannot include horizon fixed effects.")
if (nlevels(panel$gics_sector) < 2) panel$gics_sector <- factor("All")
if (nlevels(panel$event_month) < 2) panel$event_month <- factor("All")

# Model 1: main hypotheses (volume + market open length), horizon FE, cluster by market
m1 <- feols(
  loss_polymarket ~ log_poly_volume + market_open_days | horizon,
  cluster = ~ id,
  data = panel
)

# Model 2: add standard firm controls + sector FE
m2 <- feols(
  loss_polymarket ~ log_poly_volume + market_open_days +
    log_mcap + log_analysts + log_turnover +
    stock_volatility_6m + abs_surprise + eps_dispersion | horizon + gics_sector,
  cluster = ~ id,
  data = panel
)

# Model 3: add time FE (event month) as robustness
m3 <- feols(
  loss_polymarket ~ log_poly_volume + market_open_days +
    log_mcap + log_analysts + log_turnover +
    stock_volatility_6m + abs_surprise + eps_dispersion | horizon + gics_sector + event_month,
  cluster = ~ id,
  data = panel
)

message("\n==================== Main models (clustered by id) ====================\n")
print(etable(m1, m2, m3, fitstat = c("n", "r2", "rmse")))

# Save LaTeX tables (nice for paper)
tab_main_tex <- etable(m1, m2, m3, fitstat = c("n", "r2", "rmse"), tex = TRUE)
writeLines(tab_main_tex, con = file.path(OUT_DIR, "tables_main.tex"))

# ----------------------- By-horizon regressions (robustness) -----------------
# This answers: “Does volume matter more close to resolution than far away?”
models_by_h <- list()
h_levels <- levels(panel$horizon)

for (h in h_levels) {
  sub <- panel %>% filter(horizon == h)
  # Need variation
  if (nrow(sub) < 50) next
  # Drop unused levels inside the subset
  sub$gics_sector <- droplevels(sub$gics_sector)
  sub$event_month <- droplevels(sub$event_month)
  
  fe_vars <- c()
  if (nlevels(sub$gics_sector) >= 2) fe_vars <- c(fe_vars, "gics_sector")
  if (nlevels(sub$event_month) >= 2) fe_vars <- c(fe_vars, "event_month")
  
  base_rhs <- paste(
    "loss_polymarket ~ log_poly_volume + market_open_days +",
    "log_mcap + log_analysts + log_turnover +",
    "stock_volatility_6m + abs_surprise + eps_dispersion"
  )
  
  if (length(fe_vars) > 0) {
    fml <- as.formula(paste(base_rhs, "|", paste(fe_vars, collapse = " + ")))
  } else {
    fml <- as.formula(base_rhs)  # no fixed effects if none have 2+ levels
  }
  
  models_by_h[[h]] <- feols(
    fml,
    cluster = ~ id,
    data = sub
  )
  
}

if (length(models_by_h) > 0) {
  message("\n==================== By-horizon models (one per horizon) ====================\n")
  print(etable(models_by_h, fitstat = c("n", "r2", "rmse")))
  tab_by_h_tex <- etable(models_by_h, fitstat = c("n", "r2", "rmse"), tex = TRUE)
  writeLines(tab_by_h_tex, con = file.path(OUT_DIR, "tables_by_horizon.tex"))
}

# ----------------------------- Coefficient plot ------------------------------
# Focus on your two key hypotheses: volume & open length (from m3).
# Color palette per your thesis convention:
COL_GREY1 <- "#808080"
COL_GREY2 <- "#A9A9A9"
COL_RED   <- "#E3170A"

coefs <- broom::tidy(m3, conf.int = TRUE) %>%
  filter(term %in% c("log_poly_volume", "market_open_days")) %>%
  mutate(term = recode(term,
                       "log_poly_volume" = "log(1 + Polymarket volume)",
                       "market_open_days" = "Market open length (days)"))

p <- ggplot(coefs, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = COL_GREY2) +
  geom_point(size = 3, color = COL_RED) +
  geom_errorbar(aes(xmin = conf.low, xmax = conf.high),
                width = 0.2, orientation = "y", color = COL_GREY1) +
  labs(
    title = "Drivers of Brier loss (lower is better): key coefficients",
    x = "Coefficient (with 95% CI)",
    y = NULL
  ) +
  theme_minimal(base_size = 12)

ggsave(filename = file.path(OUT_DIR, "coefplot_volume_open_days.png"), plot = p, width = 8, height = 4, dpi = 200)

# --------------------- Optional: selection correction (Heckman) --------------
# We implement a simple, transparent control-function approach:
# 1) Probit: selected_on_polymarket ~ Z
# 2) IMR = phi(xb)/Phi(xb) for selected
# 3) Outcome regression includes IMR as additional regressor
#
# IMPORTANT for “scientifically correct” use:
# - You should ideally have an exclusion restriction: a variable that affects
#   selection (listing on Polymarket) but does not directly affect forecast
#   accuracy conditional on covariates.
# - Here we *use exchange variables* (primary_exchange, exchange_country) in
#   selection only by default.
#
# If heck_universe is not available, we skip.

if (has_heck) {
  
  # Load heckman universe (CSV expected; if you have parquet, convert to CSV first)
  if (grepl("\\.csv$", heck_path, ignore.case = TRUE)) {
    hu <- data.table::fread(heck_path, na.strings = c("", "NA", "NaN"))
  } else {
    message("Heckman universe is not CSV. Please export to CSV (heckman_universe_events.csv) and re-run.")
    hu <- NULL
  }
  
  if (!is.null(hu)) {
    
    # Required columns for a workable selection model
    hu_required <- c(
      "ric", "event_date",
      "market_cap_usd_asof_evt", "analysts_covering_asof_evt",
      "turnover_lookback_sum_volume_asof_evt", "volatility_lookback_asof_evt",
      "gics_sector", "primary_exchange", "exchange_country"
    )
    hu_missing <- setdiff(hu_required, names(hu))
    if (length(hu_missing) > 0) {
      message("Skipping selection correction. heck_universe is missing: ", paste(hu_missing, collapse = ", "))
    } else {
      
      # Create an event-level outcome at a single “final snapshot” per market:
      # choose the row with smallest horizon_seconds (closest to resolution).
      final_snapshot <- panel %>%
        group_by(id) %>%
        slice_min(order_by = horizon_seconds, n = 1, with_ties = FALSE) %>%
        ungroup() %>%
        mutate(event_date = as.Date(umaEndDate))
      
      final_csv <- file.path(OUT_DIR, "analysis_final_snapshot.csv")
      final_jsonl <- file.path(OUT_DIR, "analysis_final_snapshot.jsonl")
      data.table::fwrite(as.data.table(final_snapshot), final_csv)
      write_jsonl(final_snapshot, final_jsonl)
      
      # Build selection indicator on the universe
      hu <- as.data.table(hu)
      hu[, event_date := as.Date(event_date)]
      sample_keys <- final_snapshot %>% distinct(ric, event_date) %>%
        mutate(key = paste(ric, event_date, sep = "__")) %>%
        pull(key)
      
      hu[, key := paste(ric, event_date, sep = "__")]
      hu[, selected := as.integer(key %in% sample_keys)]
      
      # Covariates for selection
      hu[, log_mcap_evt := log(as.numeric(market_cap_usd_asof_evt))]
      hu[!is.finite(log_mcap_evt), log_mcap_evt := NA_real_]
      hu[, log_analysts_evt := log1p(as.numeric(analysts_covering_asof_evt))]
      hu[, log_turnover_evt := log1p(as.numeric(turnover_lookback_sum_volume_asof_evt))]
      hu[, stock_vol_evt := as.numeric(volatility_lookback_asof_evt)]
      
      hu[, gics_sector := collapse_rare_levels(gics_sector, min_n = 25, other = "Other")]
      hu[, primary_exchange := collapse_rare_levels(primary_exchange, min_n = 25, other = "Other")]
      hu[, exchange_country := collapse_rare_levels(exchange_country, min_n = 25, other = "Other")]
      
      # Keep complete cases for selection model
      sel <- hu %>%
        as.data.frame() %>%
        select(selected, log_mcap_evt, log_analysts_evt, log_turnover_evt, stock_vol_evt,
               gics_sector, primary_exchange, exchange_country, ric, event_date) %>%
        filter(is.finite(selected)) %>%
        filter(is.finite(log_mcap_evt), is.finite(log_analysts_evt), is.finite(log_turnover_evt), is.finite(stock_vol_evt))
      
      # Selection probit (include exchange vars here as exclusion restrictions)
      # Ensure factors are dropleveled
      sel$gics_sector <- droplevels(factor(sel$gics_sector))
      sel$primary_exchange <- droplevels(factor(sel$primary_exchange))
      sel$exchange_country <- droplevels(factor(sel$exchange_country))
      
      # Drop any factor with <2 levels to avoid contrasts errors
      cand_factors <- c("gics_sector", "primary_exchange", "exchange_country")
      good_factors <- cand_factors[sapply(cand_factors, function(v) nlevels(sel[[v]]) >= 2)]
      
      rhs_terms <- c(
        "log_mcap_evt", "log_analysts_evt", "log_turnover_evt", "stock_vol_evt",
        good_factors
      )
      
      sel_fml <- as.formula(paste("selected ~", paste(rhs_terms, collapse = " + ")))
      
      sel_mod <- glm(
        sel_fml,
        family = binomial(link = "probit"),
        data = sel
      )
      
      
      # Compute IMR for selected observations
      xb <- as.numeric(predict(sel_mod, type = "link"))
      Phi <- pnorm(xb)
      phi <- dnorm(xb)
      imr <- rep(NA_real_, length(xb))
      # For selected=1: IMR = phi/Phi
      imr[sel$selected == 1] <- phi[sel$selected == 1] / pmax(Phi[sel$selected == 1], 1e-12)
      
      sel$imr <- imr
      
      # Merge IMR into final snapshot outcome data
      final2 <- final_snapshot %>%
        mutate(event_date = as.Date(umaEndDate)) %>%
        left_join(sel %>% filter(selected == 1) %>% select(ric, event_date, imr),
                  by = c("ric", "event_date"))
      
      # Outcome regression with IMR (selection-corrected)
      # Note: identification relies on the exclusion vars being valid.
      m_uncorrected <- feols(
        loss_polymarket ~ log_poly_volume + market_open_days +
          log_mcap + log_analysts + log_turnover +
          stock_volatility_6m + abs_surprise + eps_dispersion | gics_sector + event_month,
        cluster = ~ ric,
        data = final2
      )
      
      m_corrected <- feols(
        loss_polymarket ~ log_poly_volume + market_open_days +
          log_mcap + log_analysts + log_turnover +
          stock_volatility_6m + abs_surprise + eps_dispersion + imr | gics_sector + event_month,
        cluster = ~ ric,
        data = final2
      )
      
      message("\n==================== Selection correction (final snapshot) ====================\n")
      print(etable(m_uncorrected, m_corrected, fitstat = ~ n + r2 + rmse))
      writeLines(
        etable(m_uncorrected, m_corrected, fitstat = ~ n + r2 + rmse, tex = TRUE),
        con = file.path(OUT_DIR, "tables_selection_corrected.tex")
      )
    }
  }
}

message("\nDone. Outputs written to: ", OUT_DIR)
