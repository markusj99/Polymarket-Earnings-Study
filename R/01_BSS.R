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

fmt_ci <- function(lo, hi, digits = 4) {
  if (any(!is.finite(c(lo, hi)))) return(NA_character_)
  sprintf(paste0("[%.", digits, "f, %.", digits, "f]"), lo, hi)
}

bootstrap_mean_ci <- function(x, n_boot = 25000, conf = 0.95, seed = 1L) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  
  n <- length(x)
  if (n == 0) {
    return(tibble(
      boot_n = 0L,
      boot_mean = NA_real_,
      boot_ci_low = NA_real_,
      boot_ci_high = NA_real_
    ))
  }
  
  mean_hat <- mean(x)
  
  if (n == 1) {
    return(tibble(
      boot_n = 1L,
      boot_mean = mean_hat,
      boot_ci_low = mean_hat,
      boot_ci_high = mean_hat
    ))
  }
  
  set.seed(seed)
  boot_means <- replicate(
    n_boot,
    mean(sample(x, size = n, replace = TRUE))
  )
  
  alpha <- (1 - conf) / 2
  ci <- stats::quantile(
    boot_means,
    probs = c(alpha, 1 - alpha),
    names = FALSE,
    na.rm = TRUE,
    type = 7
  )
  
  tibble(
    boot_n = as.integer(n),
    boot_mean = as.numeric(mean_hat),
    boot_ci_low = as.numeric(ci[1]),
    boot_ci_high = as.numeric(ci[2])
  )
}

run_paired_robustness <- function(x, conf = 0.95, n_boot = 25000, seed = 1L) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  
  n <- length(x)
  x_nonzero <- x[x != 0]
  
  if (n == 0) {
    return(tibble(
      n = 0L,
      n_nonzero_sign = 0L,
      mean_diff = NA_real_,
      median_diff = NA_real_,
      t_ci_low = NA_real_,
      t_ci_high = NA_real_,
      boot_ci_low = NA_real_,
      boot_ci_high = NA_real_,
      p_value_t_less = NA_real_,
      p_value_wilcox_less = NA_real_,
      p_value_sign_less = NA_real_,
      share_negative = NA_real_,
      share_zero = NA_real_,
      share_positive = NA_real_
    ))
  }
  
  t_less <- if (n >= 2) {
    stats::t.test(x, mu = 0, alternative = "less")
  } else {
    NULL
  }
  
  t_two <- if (n >= 2) {
    stats::t.test(x, mu = 0, alternative = "two.sided", conf.level = conf)
  } else {
    NULL
  }
  
  wilcox_less <- if (n >= 1 && any(x != 0)) {
    tryCatch(
      suppressWarnings(
        stats::wilcox.test(
          x,
          mu = 0,
          alternative = "less",
          exact = FALSE,
          conf.int = FALSE
        )
      ),
      error = function(e) NULL
    )
  } else {
    NULL
  }
  
  sign_less <- if (length(x_nonzero) >= 1) {
    stats::binom.test(
      x = sum(x_nonzero < 0),
      n = length(x_nonzero),
      p = 0.5,
      alternative = "greater"
    )
  } else {
    NULL
  }
  
  boot <- bootstrap_mean_ci(
    x,
    n_boot = n_boot,
    conf = conf,
    seed = seed
  )
  
  tibble(
    n = as.integer(n),
    n_nonzero_sign = as.integer(length(x_nonzero)),
    mean_diff = mean(x),
    median_diff = stats::median(x),
    t_ci_low = if (!is.null(t_two)) as.numeric(t_two$conf.int[1]) else NA_real_,
    t_ci_high = if (!is.null(t_two)) as.numeric(t_two$conf.int[2]) else NA_real_,
    boot_ci_low = boot$boot_ci_low,
    boot_ci_high = boot$boot_ci_high,
    p_value_t_less = if (!is.null(t_less)) as.numeric(t_less$p.value) else NA_real_,
    p_value_wilcox_less = if (!is.null(wilcox_less)) as.numeric(wilcox_less$p.value) else NA_real_,
    p_value_sign_less = if (!is.null(sign_less)) as.numeric(sign_less$p.value) else NA_real_,
    share_negative = mean(x < 0),
    share_zero = mean(x == 0),
    share_positive = mean(x > 0)
  )
}

summarise_paired_diff <- function(x, conf = 0.95, n_boot = 25000, seed = 1L) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  
  n <- length(x)
  if (n == 0) {
    return(tibble(
      n = 0L,
      mean_diff = NA_real_,
      sd_diff = NA_real_,
      median_diff = NA_real_,
      q25_diff = NA_real_,
      q75_diff = NA_real_,
      min_diff = NA_real_,
      max_diff = NA_real_,
      share_negative = NA_real_,
      share_zero = NA_real_,
      share_positive = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_,
      boot_ci_low = NA_real_,
      boot_ci_high = NA_real_
    ))
  }
  
  q <- stats::quantile(
    x,
    probs = c(0.25, 0.75),
    names = FALSE,
    type = 7,
    na.rm = TRUE
  )
  
  tt_two_sided <- if (n >= 2) {
    stats::t.test(x, mu = 0, alternative = "two.sided", conf.level = conf)
  } else {
    NULL
  }
  
  boot <- bootstrap_mean_ci(
    x,
    n_boot = n_boot,
    conf = conf,
    seed = seed
  )
  
  tibble(
    n = as.integer(n),
    mean_diff = mean(x),
    sd_diff = if (n >= 2) stats::sd(x) else NA_real_,
    median_diff = stats::median(x),
    q25_diff = as.numeric(q[1]),
    q75_diff = as.numeric(q[2]),
    min_diff = min(x),
    max_diff = max(x),
    share_negative = mean(x < 0),
    share_zero = mean(x == 0),
    share_positive = mean(x > 0),
    ci_low = if (!is.null(tt_two_sided)) as.numeric(tt_two_sided$conf.int[1]) else NA_real_,
    ci_high = if (!is.null(tt_two_sided)) as.numeric(tt_two_sided$conf.int[2]) else NA_real_,
    boot_ci_low = boot$boot_ci_low,
    boot_ci_high = boot$boot_ci_high
  )
}

save_paired_diff_plots <- function(df, baseline_label, baseline_slug, horizon_levels,
                                   out_dir, fill_col, zero_col, mean_col) {
  plot_df <- df %>%
    dplyr::filter(baseline == baseline_label) %>%
    dplyr::mutate(horizon = factor(horizon, levels = horizon_levels))
  
  mean_lines <- plot_df %>%
    dplyr::group_by(horizon) %>%
    dplyr::summarise(mean_diff = mean(diff, na.rm = TRUE), .groups = "drop")
  
  p_hist <- ggplot(plot_df, aes(x = diff)) +
    geom_histogram(bins = 20, fill = fill_col, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = zero_col, linewidth = 0.7) +
    geom_vline(
      data = mean_lines,
      aes(xintercept = mean_diff),
      inherit.aes = FALSE,
      color = mean_col,
      linewidth = 0.8
    ) +
    facet_wrap(~ horizon, scales = "free_y") +
    labs(
      title = paste0("Histogram of paired loss differences: Polymarket - ", baseline_label),
      subtitle = "Dashed line = 0; solid line = sample mean difference",
      x = "d_i = loss_polymarket - loss_baseline",
      y = "Count"
    ) +
    theme_minimal(base_size = 12)
  
  p_qq <- ggplot(plot_df, aes(sample = diff)) +
    stat_qq(color = fill_col, alpha = 0.8) +
    stat_qq_line(color = mean_col, linewidth = 0.8) +
    facet_wrap(~ horizon, scales = "free") +
    labs(
      title = paste0("Normal Q-Q plot of paired loss differences: Polymarket - ", baseline_label),
      subtitle = "One panel per horizon",
      x = "Theoretical quantiles",
      y = "Sample quantiles"
    ) +
    theme_minimal(base_size = 12)
  
  ggsave(
    filename = file.path(out_dir, paste0("hist_paired_diff_", baseline_slug, ".png")),
    plot = p_hist, width = 12, height = 8, dpi = 300
  )
  ggsave(
    filename = file.path(out_dir, paste0("hist_paired_diff_", baseline_slug, ".pdf")),
    plot = p_hist, width = 12, height = 8
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("qq_paired_diff_", baseline_slug, ".png")),
    plot = p_qq, width = 12, height = 8, dpi = 300
  )
  ggsave(
    filename = file.path(out_dir, paste0("qq_paired_diff_", baseline_slug, ".pdf")),
    plot = p_qq, width = 12, height = 8
  )
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

boot_mean_ci_by_id <- function(df, id_col, value_col, n_boot = 25000, conf = 0.95, seed = 1) {
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
DIAG_DIR <- file.path(OUT_DIR, "paired_difference_diagnostics")
DIAG_FIG_DIR <- file.path(DIAG_DIR, "figures")

dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG_FIG_DIR, recursive = TRUE, showWarnings = FALSE)
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
      ~ boot_mean_ci_by_id(.x, id_col = id, value_col = loss, n_boot = 25000, conf = 0.95, seed = 1)
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
  subtitle = "CI is cluster bootstrap over market ids; sample restricted to complete cases.",
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

# 4) Paired-difference diagnostics by horizon and baseline
diff_long <- df0 %>%
  dplyr::mutate(
    diff_vs_dice = loss_polymarket - loss_dice,
    diff_vs_hist = loss_polymarket - loss_hist
  ) %>%
  dplyr::select(id, horizon, horizon_seconds, diff_vs_dice, diff_vs_hist) %>%
  tidyr::pivot_longer(
    cols = c(diff_vs_dice, diff_vs_hist),
    names_to = "baseline",
    values_to = "diff"
  ) %>%
  dplyr::mutate(
    baseline = dplyr::recode(
      baseline,
      diff_vs_dice = "Dice (0.5)",
      diff_vs_hist = "Historical rate"
    ),
    baseline = factor(baseline, levels = c("Dice (0.5)", "Historical rate")),
    horizon = factor(horizon, levels = horizon_levels)
  )

paired_diff_summary <- diff_long %>%
  dplyr::group_by(horizon, horizon_seconds, baseline) %>%
  tidyr::nest() %>%
  dplyr::mutate(
    stats = purrr::map(data, ~ summarise_paired_diff(.x$diff, conf = 0.95))
  ) %>%
  dplyr::select(-data) %>%
  tidyr::unnest(stats) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(dplyr::desc(horizon_seconds), baseline)

paired_diff_csv   <- file.path(DIAG_DIR, "paired_difference_summary_by_horizon.csv")
paired_diff_jsonl <- file.path(DIAG_DIR, "paired_difference_summary_by_horizon.jsonl")
write.csv(paired_diff_summary, paired_diff_csv, row.names = FALSE)
write_jsonl(paired_diff_summary, paired_diff_jsonl)


paired_diff_pretty <- paired_diff_summary %>%
  dplyr::mutate(
    horizon = as.character(horizon),
    baseline = as.character(baseline),
    N = n,
    `Mean Δloss` = fmt_num(mean_diff, 4),
    `95% CI for mean Δloss (t)` = purrr::map2_chr(ci_low, ci_high, fmt_ci),
    `95% CI for mean Δloss (bootstrap)` = purrr::map2_chr(boot_ci_low, boot_ci_high, fmt_ci),
    `SD` = fmt_num(sd_diff, 4),
    `Median` = fmt_num(median_diff, 4),
    `IQR` = purrr::map2_chr(q25_diff, q75_diff, fmt_ci),
    `Min` = fmt_num(min_diff, 4),
    `Max` = fmt_num(max_diff, 4),
    `% (Δloss < 0)` = sprintf("%.1f%%", 100 * share_negative)
  ) %>%
  dplyr::select(
    horizon, baseline, N,
    `Mean Δloss`,
    `95% CI for mean Δloss (t)`,
    `95% CI for mean Δloss (bootstrap)`,
    `SD`, `Median`, `IQR`, `Min`, `Max`, `% (Δloss < 0)`
  )

paired_diff_gt <- make_gt_table(
  paired_diff_pretty,
  title = "Paired loss-difference diagnostics by horizon",
  subtitle = "Δloss = loss_polymarket - loss_baseline; both t-based and bootstrap 95% CIs are reported for the mean paired difference",
  note = "Negative Δloss means Polymarket is more accurate than the baseline. Bootstrap CI is a percentile bootstrap for the mean paired difference.",
  rowname_col = NULL
)

save_gt_html(
  paired_diff_gt,
  file.path(TAB_DIR, "table_paired_difference_diagnostics_by_horizon.html")
)

# Histogram + QQ plots, one panel per horizon
save_paired_diff_plots(
  df = diff_long,
  baseline_label = "Dice (0.5)",
  baseline_slug = "dice",
  horizon_levels = horizon_levels,
  out_dir = DIAG_FIG_DIR,
  fill_col = COL_DICE,
  zero_col = COL_HIST,
  mean_col = COL_POLY
)

save_paired_diff_plots(
  df = diff_long,
  baseline_label = "Historical rate",
  baseline_slug = "historical_rate",
  horizon_levels = horizon_levels,
  out_dir = DIAG_FIG_DIR,
  fill_col = COL_HIST,
  zero_col = COL_DICE,
  mean_col = COL_POLY
)

# 5) Statistical tests per horizon (paired one-sided t-tests with two-sided 95% CI)
test_one_horizon <- function(df_h, conf = 0.95) {
  diff_dice <- df_h$loss_polymarket - df_h$loss_dice
  diff_hist <- df_h$loss_polymarket - df_h$loss_hist
  
  t_dice_less <- stats::t.test(diff_dice, mu = 0, alternative = "less")
  t_hist_less <- stats::t.test(diff_hist, mu = 0, alternative = "less")
  
  t_dice_two <- stats::t.test(diff_dice, mu = 0, alternative = "two.sided", conf.level = conf)
  t_hist_two <- stats::t.test(diff_hist, mu = 0, alternative = "two.sided", conf.level = conf)
  
  n_dice <- sum(is.finite(diff_dice))
  n_hist <- sum(is.finite(diff_hist))
  
  tibble(
    n = as.integer(min(n_dice, n_hist)),
    
    mean_diff_poly_minus_dice = mean(diff_dice, na.rm = TRUE),
    sd_diff_vs_dice = stats::sd(diff_dice, na.rm = TRUE),
    ci_low_vs_dice = as.numeric(t_dice_two$conf.int[1]),
    ci_high_vs_dice = as.numeric(t_dice_two$conf.int[2]),
    t_stat_vs_dice = unname(t_dice_less$statistic),
    p_value_vs_dice = t_dice_less$p.value,
    
    mean_diff_poly_minus_hist = mean(diff_hist, na.rm = TRUE),
    sd_diff_vs_hist = stats::sd(diff_hist, na.rm = TRUE),
    ci_low_vs_hist = as.numeric(t_hist_two$conf.int[1]),
    ci_high_vs_hist = as.numeric(t_hist_two$conf.int[2]),
    t_stat_vs_hist = unname(t_hist_less$statistic),
    p_value_vs_hist = t_hist_less$p.value
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
    
    stars_dice = sig_stars(p_adj_holm_vs_dice),
    stars_hist = sig_stars(p_adj_holm_vs_hist),
    
    `Mean Δloss: Poly − Dice` = sprintf("%.4f%s", mean_diff_poly_minus_dice, stars_dice),
    `95% CI vs Dice`          = purrr::map2_chr(ci_low_vs_dice, ci_high_vs_dice, fmt_ci),
    `p (Holm) vs Dice`        = sprintf("%.3g", p_adj_holm_vs_dice),
    
    `Mean Δloss: Poly − Hist` = sprintf("%.4f%s", mean_diff_poly_minus_hist, stars_hist),
    `95% CI vs Hist`          = purrr::map2_chr(ci_low_vs_hist, ci_high_vs_hist, fmt_ci),
    `p (Holm) vs Hist`        = sprintf("%.3g", p_adj_holm_vs_hist),
    
    N = n
  ) %>%
  dplyr::select(
    horizon, N,
    `Mean Δloss: Poly − Dice`, `95% CI vs Dice`, `p (Holm) vs Dice`,
    `Mean Δloss: Poly − Hist`, `95% CI vs Hist`, `p (Holm) vs Hist`
  )

tests_gt <- make_gt_table(
  tests_pretty,
  title = "Paired accuracy tests by horizon",
  subtitle = "P-values are from one-sided paired t-tests (H1: Polymarket loss < baseline loss); confidence intervals are two-sided 95% CIs for the mean paired difference",
  note = "Δloss = loss_polymarket - loss_baseline. Negative values favor Polymarket. Stars use Holm-adjusted p-values: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10.",
  rowname_col = "horizon"
)

save_gt_html(tests_gt, file.path(TAB_DIR, "table_paired_tests_by_horizon.html"))

# 6) Robustness tests by horizon and baseline:
#    paired t-test + Wilcoxon signed-rank + sign test + bootstrap CI
robustness_tbl <- diff_long %>%
  dplyr::group_by(horizon, horizon_seconds, baseline) %>%
  tidyr::nest() %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    seed = 1000L + dplyr::row_number(),
    test = purrr::map2(
      data,
      seed,
      ~ run_paired_robustness(.x$diff, conf = 0.95, n_boot = 25000, seed = .y)
    )
  ) %>%
  dplyr::select(-data, -seed) %>%
  tidyr::unnest(test) %>%
  dplyr::arrange(dplyr::desc(horizon_seconds), baseline) %>%
  dplyr::group_by(baseline) %>%
  dplyr::mutate(
    p_adj_holm_t = p.adjust(p_value_t_less, method = "holm"),
    p_adj_holm_wilcox = p.adjust(p_value_wilcox_less, method = "holm"),
    p_adj_holm_sign = p.adjust(p_value_sign_less, method = "holm")
  ) %>%
  dplyr::ungroup()

robustness_csv   <- file.path(OUT_DIR, "paired_robustness_tests_by_horizon.csv")
robustness_jsonl <- file.path(OUT_DIR, "paired_robustness_tests_by_horizon.jsonl")
write.csv(robustness_tbl, robustness_csv, row.names = FALSE)
write_jsonl(robustness_tbl, robustness_jsonl)

robustness_pretty <- robustness_tbl %>%
  dplyr::mutate(
    horizon = as.character(horizon),
    baseline = as.character(baseline),
    N = n,
    `N sign` = n_nonzero_sign,
    `Mean Δloss` = fmt_num(mean_diff, 4),
    `Median Δloss` = fmt_num(median_diff, 4),
    `95% CI (t)` = purrr::map2_chr(t_ci_low, t_ci_high, fmt_ci),
    `95% CI (bootstrap)` = purrr::map2_chr(boot_ci_low, boot_ci_high, fmt_ci),
    `p t-test (Holm)` = sprintf("%.3g", p_adj_holm_t),
    `p Wilcoxon (Holm)` = sprintf("%.3g", p_adj_holm_wilcox),
    `p Sign (Holm)` = sprintf("%.3g", p_adj_holm_sign),
    `% (Δloss < 0)` = sprintf("%.1f%%", 100 * share_negative)
  ) %>%
  dplyr::select(
    horizon, baseline, N, `N sign`,
    `Mean Δloss`, `Median Δloss`,
    `95% CI (t)`, `95% CI (bootstrap)`,
    `p t-test (Holm)`, `p Wilcoxon (Holm)`, `p Sign (Holm)`,
    `% (Δloss < 0)`
  )

robustness_gt <- make_gt_table(
  robustness_pretty,
  title = "Robustness tests for paired loss differences by horizon",
  subtitle = "Main test: paired one-sided t-test; sensitivity checks: Wilcoxon signed-rank test, sign test, and bootstrap CI for the mean paired difference",
  note = paste(
    "Δloss = loss_polymarket - loss_baseline.",
    "Negative values favor Polymarket.",
    "Wilcoxon uses the large-sample approximation (exact = FALSE) because ties are common.",
    "Sign test is an exact binomial test on non-zero differences only.",
    "Bootstrap CI is a two-sided percentile bootstrap for the mean paired difference."
  ),
  rowname_col = NULL
)

save_gt_html(
  robustness_gt,
  file.path(TAB_DIR, "table_paired_robustness_tests_by_horizon.html")
)

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
    caption = "CIs are cluster bootstrap over market ids; sample restricted to rows with all 3 models available."
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

cat("\n=== Paired difference diagnostics by horizon ===\n")
print(paired_diff_pretty, n = nrow(paired_diff_pretty))
cat("\nHTML saved: ", file.path(TAB_DIR, "table_paired_difference_diagnostics_by_horizon.html"), "\n", sep = "")
cat("Diagnostic figures saved to:\n  ", DIAG_FIG_DIR, "\n", sep = "")

cat("\n=== Paired tests by horizon (pretty, regression-style) ===\n")
print(tests_pretty, n = nrow(tests_pretty))
cat("\nHTML saved: ", file.path(TAB_DIR, "table_paired_tests_by_horizon.html"), "\n", sep = "")

cat("\n=== Robustness tests by horizon ===\n")
print(robustness_pretty, n = nrow(robustness_pretty))
cat("\nHTML saved: ", file.path(TAB_DIR, "table_paired_robustness_tests_by_horizon.html"), "\n", sep = "")

# Print gt tables (shows in Viewer / RStudio)
print(brier_gt)
print(bss_gt)
print(paired_diff_gt)
print(tests_gt)
print(robustness_gt)

cat("\nSaved outputs to:\n  ", OUT_DIR, "\n", sep = "")
cat("Saved paired-difference diagnostics to:\n  ", DIAG_DIR, "\n", sep = "")
cat("Saved figures to:\n  ", FIG_DIR, "\n", sep = "")
cat("Saved HTML tables to:\n  ", TAB_DIR, "\n", sep = "")
