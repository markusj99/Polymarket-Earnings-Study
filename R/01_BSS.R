#!/usr/bin/env Rscript
# =============================================================================
# File:    Polymarket-Earnings-Study/R/scripts/brier_plot_bss_tests.R
# Purpose: Plot Brier Scores (with CI) for 3 models across time horizons,
#          compute Brier Skill Score vs baselines, and test whether Polymarket
#          is significantly more accurate per horizon.
#
# Prints "publication-style" tables (gt) + saves them to HTML.
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

# ---------- Load data (shared loader) ----------
source(file.path(ROOT, "R", "utils", "load_data.R"))
D <- load_project_data(ROOT)
dataset_long <- D$dataset_long

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(jsonlite)
  library(stats)
  library(gt)
})

# ---------------------------- Helpers ----------------------------------------

write_jsonl <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  con <- file(path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  
  rows <- split(df, seq_len(nrow(df)))
  for (r in rows) {
    obj <- as.list(r[1, , drop = TRUE])
    line <- jsonlite::toJSON(obj, auto_unbox = TRUE, na = "null")
    writeLines(line, con = con, sep = "\n")
  }
}

fmt_mean_ci <- function(mean, lo, hi, digits = 4) {
  if (any(!is.finite(c(mean, lo, hi)))) return(NA_character_)
  sprintf(paste0("%.", digits, "f [%.", digits, "f, %.", digits, "f]"), mean, lo, hi)
}

fmt_num <- function(x, digits = 4) {
  x <- as.numeric(x)
  out <- rep(NA_character_, length(x))
  ok <- is.finite(x)
  out[ok] <- sprintf(paste0("%.", digits, "f"), x[ok])
  out
}


sig_stars <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    p < 0.1   ~ ".",
    TRUE ~ ""
  )
}

boot_mean_ci_by_id <- function(df, id_col, value_col, n_boot = 2000, conf = 0.95, seed = 1) {
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

make_gt_table <- function(df, title, subtitle = NULL, note = NULL, rowname_col = NULL) {
  g <- gt::gt(df, rowname_col = rowname_col) %>%
    gt::tab_header(title = title, subtitle = subtitle) %>%
    gt::opt_table_font(font = list(gt::google_font("Source Sans Pro"), gt::default_fonts())) %>%
    gt::tab_options(
      table.font.size = px(12),
      heading.title.font.size = px(16),
      heading.subtitle.font.size = px(12)
    )
  
  if (!is.null(note)) {
    g <- g %>% gt::tab_source_note(note)
  }
  g
}

save_gt_html <- function(gt_obj, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  gt::gtsave(gt_obj, path)
}

# ---------------------------- Main -------------------------------------------

if (!exists("dataset_long")) {
  stop("Object 'dataset_long' not found. Load it first, then run this script.")
}

OUT_DIR <- file.path(ROOT, "statistics", "brier_analysis")
FIG_DIR <- file.path(ROOT, "statistics", "brier_analysis", "figures")
TAB_DIR <- file.path(OUT_DIR, "tables")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

# Palette (per your scheme)
COL_POLY <- "#E3170A"  # red
COL_DICE <- "#808080"  # grey
COL_HIST <- "#00008B"  # dark blue

# 1) Clean + restrict to comparable sample
df0 <- dataset_long %>%
  dplyr::mutate(
    loss_polymarket = as.numeric(loss_polymarket),
    loss_dice       = as.numeric(loss_dice),
    loss_hist       = as.numeric(loss_hist)
  ) %>%
  dplyr::filter(status == "ok") %>%
  dplyr::filter(!is.na(horizon), !is.na(horizon_seconds)) %>%
  dplyr::filter(!is.na(loss_polymarket), !is.na(loss_dice), !is.na(loss_hist))

# Exclude long horizons
exclude_horizons <- c("4w", "3w", "2w")
df0 <- df0 %>% dplyr::filter(!(horizon %in% exclude_horizons))

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

# Save Brier summary (machine-readable)
brier_csv   <- file.path(OUT_DIR, "brier_scores_by_horizon.csv")
brier_jsonl <- file.path(OUT_DIR, "brier_scores_by_horizon.jsonl")
write.csv(brier_summary, brier_csv, row.names = FALSE)
write_jsonl(brier_summary, brier_jsonl)

# ---- Pretty Brier table (wide) with CI shown clearly ----
n_by_h <- brier_summary %>%
  dplyr::group_by(horizon, horizon_seconds) %>%
  dplyr::summarise(N = min(n, na.rm = TRUE), .groups = "drop")

brier_pretty <- brier_summary %>%
  dplyr::mutate(Brier_95CI = purrr::pmap_chr(list(mean, ci_low, ci_high), fmt_mean_ci)) %>%
  dplyr::select(horizon, horizon_seconds, model, Brier_95CI) %>%
  tidyr::pivot_wider(names_from = model, values_from = Brier_95CI) %>%
  dplyr::left_join(n_by_h, by = c("horizon", "horizon_seconds")) %>%
  dplyr::arrange(dplyr::desc(horizon_seconds)) %>%
  dplyr::mutate(horizon = as.character(horizon)) %>%
  dplyr::select(horizon, N, `Polymarket`, `Dice (0.5)`, `Historical rate`)

brier_gt <- make_gt_table(
  brier_pretty,
  title = "Brier Score by horizon (mean with 95% CI)",
  subtitle = "CI is cluster bootstrap over market ids; sample restricted to complete cases (status == 'ok')",
  note = "Cell format: mean [CI_low, CI_high].",
  rowname_col = "horizon"
)

save_gt_html(brier_gt, file.path(TAB_DIR, "table_brier_scores_by_horizon.html"))

# 3) Brier Skill Score (Polymarket vs baselines), by horizon
bss_tbl <- df0 %>%
  dplyr::group_by(horizon, horizon_seconds) %>%
  dplyr::summarise(
    n = dplyr::n_distinct(id),
    BS_polymarket = mean(loss_polymarket),
    BS_dice       = mean(loss_dice),
    BS_hist       = mean(loss_hist),
    BSS_vs_dice   = 1 - (mean(loss_polymarket) / mean(loss_dice)),
    BSS_vs_hist   = 1 - (mean(loss_polymarket) / mean(loss_hist)),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(horizon_seconds))

bss_csv   <- file.path(OUT_DIR, "brier_skill_scores_by_horizon.csv")
bss_jsonl <- file.path(OUT_DIR, "brier_skill_scores_by_horizon.jsonl")
write.csv(bss_tbl, bss_csv, row.names = FALSE)
write_jsonl(bss_tbl, bss_jsonl)

bss_pretty <- bss_tbl %>%
  dplyr::mutate(
    horizon = as.character(horizon),
    N = n,
    `BS (Polymarket)` = fmt_num(BS_polymarket, 4),
    `BS (Dice)`       = fmt_num(BS_dice, 4),
    `BS (Historical)` = fmt_num(BS_hist, 4),
    `BSS vs Dice`     = fmt_num(BSS_vs_dice, 4),
    `BSS vs Historical` = fmt_num(BSS_vs_hist, 4)
  ) %>%
  dplyr::select(horizon, N, `BS (Polymarket)`, `BS (Dice)`, `BS (Historical)`, `BSS vs Dice`, `BSS vs Historical`)

bss_gt <- make_gt_table(
  bss_pretty,
  title = "Brier Skill Score (BSS) by horizon",
  subtitle = "BSS = 1 - BS_model / BS_baseline (higher is better)",
  rowname_col = "horizon"
)

save_gt_html(bss_gt, file.path(TAB_DIR, "table_brier_skill_score_by_horizon.html"))

# 4) Statistical tests per horizon (paired one-sided t-tests)
test_one_horizon <- function(df_h) {
  diff_dice <- df_h$loss_polymarket - df_h$loss_dice
  diff_hist <- df_h$loss_polymarket - df_h$loss_hist
  
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

# ---- Pretty "regression-like" paired tests table ----
tests_pretty <- tests_tbl %>%
  dplyr::mutate(
    horizon = as.character(horizon),
    
    se_vs_dice = abs(mean_diff_poly_minus_dice / t_stat_vs_dice),
    se_vs_hist = abs(mean_diff_poly_minus_hist / t_stat_vs_hist),
    
    stars_dice = sig_stars(p_adj_holm_vs_dice),
    stars_hist = sig_stars(p_adj_holm_vs_hist),
    
    `ΔLoss: Poly − Dice` = sprintf("%.4f%s\n(%.4f)", mean_diff_poly_minus_dice, stars_dice, se_vs_dice),
    `p (Holm) vs Dice`   = sprintf("%.3g", p_adj_holm_vs_dice),
    
    `ΔLoss: Poly − Hist` = sprintf("%.4f%s\n(%.4f)", mean_diff_poly_minus_hist, stars_hist, se_vs_hist),
    `p (Holm) vs Hist`   = sprintf("%.3g", p_adj_holm_vs_hist),
    
    N = n
  ) %>%
  dplyr::select(
    horizon, N,
    `ΔLoss: Poly − Dice`, `p (Holm) vs Dice`,
    `ΔLoss: Poly − Hist`, `p (Holm) vs Hist`
  )

tests_gt <- make_gt_table(
  tests_pretty,
  title = "Paired accuracy tests by horizon",
  subtitle = "H1: Polymarket loss < baseline loss (paired one-sided t-tests); stars use Holm-adjusted p-values",
  note = "Entries show mean(Δloss) with SE in parentheses. Stars: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10 (Holm-adjusted).",
  rowname_col = "horizon"
)

save_gt_html(tests_gt, file.path(TAB_DIR, "table_paired_tests_by_horizon.html"))

# 5) Plot: Brier score means + 95% bootstrap CI by horizon for 3 models
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
    title = "Brier Score by Time Horizon (with 95% CI)",
    x = "Time snapshot / horizon",
    y = "Brier Score (mean of (p - y)^2)",
    color = "Model",
    caption = "CIs are cluster bootstrap over market ids; sample restricted to rows with all 3 models available (status == 'ok')."
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

png_path <- file.path(FIG_DIR, "brier_scores_by_horizon.png")
pdf_path <- file.path(FIG_DIR, "brier_scores_by_horizon.pdf")
ggsave(filename = png_path, plot = p, width = 10, height = 6, dpi = 300)
ggsave(filename = pdf_path, plot = p, width = 10, height = 6)

# 6) Console output + nice tables (gt)
cat("\n=== Brier Score by horizon (pretty) ===\n")
print(brier_pretty, n = nrow(brier_pretty))
cat("\nHTML saved: ", file.path(TAB_DIR, "table_brier_scores_by_horizon.html"), "\n", sep = "")

cat("\n=== Brier Skill Score by horizon (pretty) ===\n")
print(bss_pretty, n = nrow(bss_pretty))
cat("\nHTML saved: ", file.path(TAB_DIR, "table_brier_skill_score_by_horizon.html"), "\n", sep = "")

cat("\n=== Paired tests by horizon (pretty, regression-style) ===\n")
print(tests_pretty, n = nrow(tests_pretty))
cat("\nHTML saved: ", file.path(TAB_DIR, "table_paired_tests_by_horizon.html"), "\n", sep = "")

# Print gt tables (shows in Viewer / RStudio)
print(brier_gt)
print(bss_gt)
print(tests_gt)

cat("\nSaved outputs to:\n  ", OUT_DIR, "\n", sep = "")
cat("Saved figures to:\n  ", FIG_DIR, "\n", sep = "")
cat("Saved HTML tables to:\n  ", TAB_DIR, "\n", sep = "")
