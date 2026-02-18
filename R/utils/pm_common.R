# =============================================================================
# File:    Polymarket-Earnings-Study/R/utils/pm_common.R
# Purpose: Shared helpers used by (1) Brier-score script and (2) Heckman/EventStudy
# =============================================================================

options(stringsAsFactors = FALSE, scipen = 999)

# -----------------------------
# Packages (loaded on demand)
# -----------------------------
pm_required_pkgs <- c(
  "tidyverse", "lubridate", "janitor", "scales", "jsonlite", "glue", "fs",
  "broom", "sandwich", "lmtest"
)

pm_load_packages <- function() {
  missing <- pm_required_pkgs[!vapply(pm_required_pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop(
      "Missing packages: ", paste(missing, collapse = ", "), "\n",
      "If you use renv, run: renv::restore(). Otherwise, install.packages().",
      call. = FALSE
    )
  }
  invisible(lapply(pm_required_pkgs, library, character.only = TRUE))
}

# -----------------------------
# Color palette (project requirement)
# -----------------------------
COL_GREY_1   <- "#808080"
COL_GREY_2   <- "#A9A9A9"
COL_RED      <- "#E3170A"
COL_DARKBLUE <- "#00008B"
COL_BLUE     <- "#0000FF"

theme_corporate <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      axis.title = ggplot2::element_text(face = "bold"),
      legend.position = "bottom"
    )
}

# -----------------------------
# IO helpers
# -----------------------------
read_csv_required <- function(path) {
  if (!file.exists(path)) stop(glue::glue("Input file not found: {path}"), call. = FALSE)
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

read_csv_optional <- function(path) {
  if (!file.exists(path)) {
    message(glue::glue("NOTE: Optional input missing (skipping): {path}"))
    return(tibble::tibble())
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

write_table_triple <- function(df, stem, out_dir) {
  table_dir <- file.path(out_dir, "tables")
  fs::dir_create(table_dir)

  csv_path   <- file.path(table_dir, paste0(stem, ".csv"))
  jsonl_path <- file.path(table_dir, paste0(stem, ".jsonl"))
  json_path  <- file.path(table_dir, paste0(stem, ".json"))

  readr::write_csv(df, csv_path, na = "")

  con <- file(jsonl_path, open = "wt")
  on.exit(close(con), add = TRUE)
  jsonlite::stream_out(df, con = con, verbose = FALSE)

  jsonlite::write_json(
    x = df,
    path = json_path,
    dataframe = "rows",
    auto_unbox = TRUE,
    pretty = TRUE,
    na = "null"
  )

  list(csv = csv_path, jsonl = jsonl_path, json = json_path)
}

save_plot_png <- function(p, stem, out_dir, width = 10, height = 6, dpi = 300) {
  plot_dir <- file.path(out_dir, "plots")
  fs::dir_create(plot_dir)
  png_path <- file.path(plot_dir, paste0(stem, ".png"))
  ggplot2::ggsave(filename = png_path, plot = p, width = width, height = height, dpi = dpi)
  png_path
}

# -----------------------------
# Parsing + safety helpers
# -----------------------------
parse_ts_utc <- function(x) {
  if (inherits(x, "POSIXct")) return(lubridate::with_tz(x, tzone = "UTC"))
  if (is.numeric(x)) return(lubridate::as_datetime(x, tz = "UTC"))

  x_chr <- as.character(x)
  x_chr <- stringr::str_trim(x_chr)
  x_chr <- dplyr::na_if(x_chr, "")

  x_chr <- stringr::str_replace_all(x_chr, "T", " ")
  x_chr <- stringr::str_replace(x_chr, "Z$", "+00:00")

  suppressWarnings(
    lubridate::parse_date_time(
      x_chr,
      orders = c(
        "ymd HMSOSz", "ymd HMSz", "ymd HMSOS", "ymd HMS",
        "ymdHMSOSz", "ymdHMSz", "ymdHMSOS", "ymdHMS",
        "ymd"
      ),
      tz = "UTC",
      exact = FALSE
    )
  )
}

parse_date_utc <- function(x) {
  if (inherits(x, "Date")) return(x)
  if (inherits(x, "POSIXct")) return(as.Date(lubridate::with_tz(x, "UTC")))
  x_chr <- as.character(x)
  suppressWarnings(lubridate::ymd(x_chr))
}

normalize_ric <- function(x) {
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_to_upper(x)
  dplyr::na_if(x, "")
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

safe_log <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  log(pmax(x, 1))
}

# -----------------------------
# Semantics: make p_mkt := P(BEAT) coherent
# -----------------------------
yes_means_beat <- function(val_yes_semantics = NA_character_, question = NA_character_) {
  s <- stringr::str_to_upper(as.character(val_yes_semantics))
  q <- stringr::str_to_upper(as.character(question))

  if (!is.na(s) && nzchar(s)) {
    if (stringr::str_detect(s, "BEAT") && !stringr::str_detect(s, "MISS")) return(TRUE)
    if (stringr::str_detect(s, "MISS") && !stringr::str_detect(s, "BEAT")) return(FALSE)
  }

  if (!is.na(q) && nzchar(q)) {
    if (stringr::str_detect(q, "\\bBEAT\\b")) return(TRUE)
    if (stringr::str_detect(q, "\\bMISS\\b")) return(FALSE)
  }

  TRUE
}

prob_bin_20pct <- function(p) {
  cut(
    p,
    breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
    include.lowest = TRUE,
    right = TRUE,
    labels = c("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
  )
}

# -----------------------------
# Factor + robust inference helpers
# -----------------------------
factor_has_2plus <- function(x) {
  f <- droplevels(as.factor(x))
  nlevels(f) >= 2
}

vcov_cluster_or_hc <- function(model, cluster = NULL, type = "HC1") {
  if (!is.null(cluster)) {
    cl <- as.factor(cluster)
    n_cl <- length(unique(cl[!is.na(cl)]))
    if (n_cl >= 30) {
      return(sandwich::vcovCL(model, cluster = cl, type = type))
    }
  }
  sandwich::vcovHC(model, type = type)
}

tidy_coeftest <- function(ct, model_name) {
  mat <- as.matrix(ct)
  if (ncol(mat) < 4) stop("coeftest object has unexpected shape.", call. = FALSE)

  tibble::tibble(
    term      = rownames(mat),
    estimate  = as.numeric(mat[, 1]),
    std_error = as.numeric(mat[, 2]),
    statistic = as.numeric(mat[, 3]),
    p_value   = as.numeric(mat[, 4]),
    model     = model_name
  ) %>%
    dplyr::mutate(
      conf_low_95  = estimate - 1.96 * std_error,
      conf_high_95 = estimate + 1.96 * std_error
    )
}

one_sample_mean_test <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3) {
    return(tibble::tibble(
      N = n, mean = mean(x), sd = sd(x), se = NA_real_,
      t_stat = NA_real_, p_value = NA_real_,
      conf_low_95 = NA_real_, conf_high_95 = NA_real_
    ))
  }
  m <- mean(x)
  s <- stats::sd(x)
  se <- s / sqrt(n)
  t <- m / se
  p <- 2 * stats::pt(abs(t), df = n - 1, lower.tail = FALSE)
  ci_half <- stats::qt(0.975, df = n - 1) * se
  tibble::tibble(
    N = n,
    mean = m,
    sd = s,
    se = se,
    t_stat = t,
    p_value = p,
    conf_low_95 = m - ci_half,
    conf_high_95 = m + ci_half
  )
}

mean_ci_95 <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 2) {
    return(tibble::tibble(N = n, mean = mean(x), sd = sd(x), se = NA_real_, ci_low_95 = NA_real_, ci_high_95 = NA_real_))
  }
  m <- mean(x); s <- stats::sd(x); se <- s / sqrt(n)
  tcrit <- stats::qt(0.975, df = n - 1)
  tibble::tibble(
    N = n,
    mean = m,
    sd = s,
    se = se,
    ci_low_95 = m - tcrit * se,
    ci_high_95 = m + tcrit * se
  )
}

format_mean_ci <- function(mean, lo, hi, digits = 4) {
  # Vectorized: returns one formatted string per element.
  mean <- as.numeric(mean)
  lo   <- as.numeric(lo)
  hi   <- as.numeric(hi)
  
  n <- max(length(mean), length(lo), length(hi))
  if (length(mean) != n) mean <- rep(mean, length.out = n)
  if (length(lo)   != n) lo   <- rep(lo,   length.out = n)
  if (length(hi)   != n) hi   <- rep(hi,   length.out = n)
  
  ok <- is.finite(mean) & is.finite(lo) & is.finite(hi)
  
  out <- rep(NA_character_, n)
  out[ok] <- sprintf(
    paste0("%.", digits, "f [%.", digits, "f, %.", digits, "f]"),
    mean[ok], lo[ok], hi[ok]
  )
  out
}
