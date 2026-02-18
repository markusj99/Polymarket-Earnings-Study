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

