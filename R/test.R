#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/scripts/brier_plot_bss_tests.R
# Purpose: Plot Brier Scores (with CI) for 3 models across time horizons,
#          compute Brier Skill Score vs baselines, and test whether Polymarket
#          is significantly more accurate per horizon.
#
# Assumptions:
# - dataset_long is already loaded in the environment.
# - loss_polymarket, loss_dice, loss_hist are already computed as (p - y)^2.
# - status == "ok" indicates valid/non-stale data points (per your pipeline).
#
# Outputs (relative to project root):
# - outputs/brier_analysis/brier_scores_by_horizon.csv
# - outputs/brier_analysis/brier_scores_by_horizon.jsonl
# - outputs/brier_analysis/brier_skill_scores_by_horizon.csv
# - outputs/brier_analysis/brier_skill_scores_by_horizon.jsonl
# - outputs/brier_analysis/brier_tests_by_horizon.csv
# - outputs/brier_analysis/brier_tests_by_horizon.jsonl
# - figures/brier_scores_by_horizon.png
# - figures/brier_scores_by_horizon.pdf
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

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(jsonlite)
})

# ---------------------------- Helpers ----------------------------------------

find_project_root <- function(start_dir = getwd()) {
  # Walk upward until we find a directory that looks like the project root
  # (contains "data" and "R"). This avoids hardcoding absolute paths.
  cur <- normalizePath(start_dir, winslash = "/", mustWork = FALSE)
  
  for (i in 1:50) {
    has_data <- dir.exists(file.path(cur, "data"))
    has_R    <- dir.exists(file.path(cur, "R"))
    if (has_data && has_R) return(cur)
    
    parent <- normalizePath(dirname(cur), winslash = "/", mustWork = FALSE)
    if (identical(parent, cur)) break
    cur <- parent
  }
  
  stop("Could not find project root (expected folders 'data' and 'R' somewhere above: ", start_dir, ").")
}

write_jsonl <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  con <- file(path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  
  rows <- split(df, seq_len(nrow(df)))
  for (r in rows) {
    # Convert one-row data.frame to a named list with proper types
    obj <- as.list(r[1, , drop = TRUE])
    line <- jsonlite::toJSON(obj, auto_unbox = TRUE, na = "null")
    writeLines(line, con = con, sep = "\n")
  }
}

boot_mean_ci_by_id <- function(df, id_col, value_col, n_boot = 2000, conf = 0.95, seed = 1) {
  # Bootstrap CI for mean, resampling IDs (cluster bootstrap).
  # If multiple rows per id exist in df, we average within id first.
  set.seed(seed)
  
  id_sym <- rlang::ensym(id_col)
  v_sym  <- rlang::ensym(value_col)
  
  per_id <- df %>%
    dplyr::filter(!is.na(!!id_sym), !is.na(!!v_sym)) %>%
    dplyr::group_by(!!id_sym) %>%
    dplyr::summarise(v = mean(!!v_sym, na.rm = TRUE), .groups = "drop")
  
  v <- per_id$v
  n <- length(v)
  
  if (n == 0) {
    return(tibble(n = 0L, mean = NA_real_, sd = NA_real_, ci_low = NA_real_, ci_high = NA_real_))
  }
  
  mean_hat <- mean(v)
  sd_hat   <- stats::sd(v)
  
  boot_means <- replicate(n_boot, mean(sample(v, size = n, replace = TRUE)))
  
  alpha <- (1 - conf) / 2
  ci <- stats::quantile(boot_means, probs = c(alpha, 1 - alpha), names = FALSE, na.rm = TRUE)
  
  tibble(
    n = as.integer(n),
    mean = as.numeric(mean_hat),
    sd = as.numeric(sd_hat),
    ci_low = as.numeric(ci[1]),
    ci_high = as.numeric(ci[2])
  )
}

# ---------------------------- Main -------------------------------------------

if (!exists("dataset_long")) {
  stop("Object 'dataset_long' not found. Load it first, then run this script.")
}

ROOT <- find_project_root(getwd())
OUT_DIR <- file.path(ROOT, "outputs", "brier_analysis")
FIG_DIR <- file.path(ROOT, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Palette (per your scheme)
COL_POLY <- "#E3170A"  # red
COL_DICE <- "#808080"  # grey
COL_HIST <- "#00008B"  # dark blue

# 1) Clean + restrict to comparable sample
#    For fair “against each other” comparisons, use rows where ALL three losses exist.
df0 <- dataset_long %>%
  dplyr::mutate(
    loss_polymarket = as.numeric(loss_polymarket),
    loss_dice       = as.numeric(loss_dice),
    loss_hist       = as.numeric(loss_hist)
  ) %>%
  dplyr::filter(status == "ok") %>%
  dplyr::filter(!is.na(horizon), !is.na(horizon_seconds)) %>%
  dplyr::filter(!is.na(loss_polymarket), !is.na(loss_dice), !is.na(loss_hist))

# Horizon ordering (longest -> shortest)
horizon_levels <- df0 %>%
  dplyr::distinct(horizon, horizon_seconds) %>%
  dplyr::arrange(dplyr::desc(horizon_seconds)) %>%
  dplyr::pull(horizon)

df0 <- df0 %>%
  dplyr::mutate(horizon = factor(horizon, levels = horizon_levels))

# 2) Brier score summaries (bootstrap CI), by horizon and model
loss_long <- df0 %>%
  dplyr::select(id, horizon, horizon_seconds, loss_polymarket, loss_dice, loss_hist) %>%
  tidyr::pivot_longer(
    cols = c(loss_polymarket, loss_dice, loss_hist),
    names_to = "model",
    values_to = "loss"
  ) %>%
  dplyr::mutate(
    model = dplyr::recode(
      model,
      loss_polymarket = "Polymarket",
      loss_dice       = "Dice (0.5)",
      loss_hist       = "Historical rate"
    ),
    model = factor(model, levels = c("Polymarket", "Dice (0.5)", "Historical rate"))
  )

brier_summary <- loss_long %>%
  dplyr::group_by(horizon, horizon_seconds, model) %>%
  tidyr::nest() %>%
  dplyr::mutate(
    stats = purrr::map(
      data,
      ~ boot_mean_ci_by_id(.x, id_col = id, value_col = loss, n_boot = 2000, conf = 0.95, seed = 1)
    )
  ) %>%
  dplyr::select(-data) %>%
  tidyr::unnest(stats) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(dplyr::desc(horizon_seconds), model)

# Save Brier summary
brier_csv   <- file.path(OUT_DIR, "brier_scores_by_horizon.csv")
brier_jsonl <- file.path(OUT_DIR, "brier_scores_by_horizon.jsonl")
write.csv(brier_summary, brier_csv, row.names = FALSE)
write_jsonl(brier_summary, brier_jsonl)

# 3) Brier Skill Score (Polymarket vs baselines), by horizon
bss_tbl <- df0 %>%
  dplyr::group_by(horizon, horizon_seconds) %>%
  dplyr::summarise(
    n = dplyr::n_distinct(id),
    BS_polymarket = mean(loss_polymarket),
    BS_dice       = mean(loss_dice),
    BS_hist       = mean(loss_hist),
    BSS_vs_dice = 1 - (mean(loss_polymarket) / mean(loss_dice)),
    BSS_vs_hist = 1 - (mean(loss_polymarket) / mean(loss_hist)),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(horizon_seconds))

bss_csv   <- file.path(OUT_DIR, "brier_skill_scores_by_horizon.csv")
bss_jsonl <- file.path(OUT_DIR, "brier_skill_scores_by_horizon.jsonl")
write.csv(bss_tbl, bss_csv, row.names = FALSE)
write_jsonl(bss_tbl, bss_jsonl)

# 4) Statistical tests per horizon:
#    Paired tests using differences in loss for each market (same id, same horizon).
#    H1: Polymarket loss < baseline loss  (i.e., mean(diff) < 0)
test_one_horizon <- function(df_h) {
  # df_h is already restricted to complete cases for all losses
  diff_dice <- df_h$loss_polymarket - df_h$loss_dice
  diff_hist <- df_h$loss_polymarket - df_h$loss_hist
  
  # One-sample t-test on paired differences
  t_dice <- t.test(diff_dice, mu = 0, alternative = "less")
  t_hist <- t.test(diff_hist, mu = 0, alternative = "less")
  
  tibble(
    n = dplyr::n_distinct(df_h$id),
    
    mean_diff_poly_minus_dice = mean(diff_dice),
    t_stat_vs_dice = unname(t_dice$statistic),
    p_value_vs_dice = t_dice$p.value,
    
    mean_diff_poly_minus_hist = mean(diff_hist),
    t_stat_vs_hist = unname(t_hist$statistic),
    p_value_vs_hist = t_hist$p.value
  )
}

tests_tbl <- df0 %>%
  dplyr::group_by(horizon, horizon_seconds) %>%
  tidyr::nest() %>%
  dplyr::mutate(test = purrr::map(data, test_one_horizon)) %>%
  dplyr::select(-data) %>%
  tidyr::unnest(test) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(dplyr::desc(horizon_seconds)) %>%
  dplyr::mutate(
    p_adj_holm_vs_dice = p.adjust(p_value_vs_dice, method = "holm"),
    p_adj_holm_vs_hist = p.adjust(p_value_vs_hist, method = "holm")
  )

tests_csv   <- file.path(OUT_DIR, "brier_tests_by_horizon.csv")
tests_jsonl <- file.path(OUT_DIR, "brier_tests_by_horizon.jsonl")
write.csv(tests_tbl, tests_csv, row.names = FALSE)
write_jsonl(tests_tbl, tests_jsonl)

# 5) Plot: Brier score means + bootstrap 95% CI by horizon for 3 models
plot_df <- brier_summary %>%
  dplyr::mutate(horizon = factor(horizon, levels = horizon_levels))

p <- ggplot(plot_df, aes(x = horizon, y = mean, group = model, color = model)) +
  geom_line() +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15) +
  scale_color_manual(values = c(
    "Polymarket" = COL_POLY,
    "Dice (0.5)" = COL_DICE,
    "Historical rate" = COL_HIST
  )) +
  labs(
    title = "Brier Score by Time Horizon (with 95% bootstrap CI)",
    x = "Time snapshot / horizon",
    y = "Brier Score (mean of (p - y)^2)",
    color = "Model",
    caption = "CIs are cluster bootstrap over market ids; sample restricted to rows with all 3 models available (status == 'ok')."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)

png_path <- file.path(FIG_DIR, "brier_scores_by_horizon.png")
pdf_path <- file.path(FIG_DIR, "brier_scores_by_horizon.pdf")
ggsave(filename = png_path, plot = p, width = 10, height = 6, dpi = 300)
ggsave(filename = pdf_path, plot = p, width = 10, height = 6)

# 6) Console output (helpful when running interactively)
cat("\n=== Brier Scores by Horizon (mean + bootstrap 95% CI) ===\n")
print(brier_summary)

cat("\n=== Brier Skill Score (Polymarket vs baselines) ===\n")
print(bss_tbl)

cat("\n=== Paired tests per horizon (H1: Polymarket loss < baseline loss) ===\n")
print(tests_tbl)

cat("\nSaved outputs to:\n")
cat("  ", OUT_DIR, "\n")
cat("Saved figures to:\n")
cat("  ", FIG_DIR, "\n")
