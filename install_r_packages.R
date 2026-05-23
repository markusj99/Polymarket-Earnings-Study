#!/usr/bin/env Rscript
# R package installer for the Polymarket Earnings Study pipeline.
# Run with:
#   Rscript install_r_packages.R

repos <- c(CRAN = "https://cloud.r-project.org")

required_packages <- c(
  # Project/runtime management
  "renv",

  # Tidyverse and data handling
  "tidyverse",
  "readr",
  "dplyr",
  "tidyr",
  "purrr",
  "stringr",
  "tibble",
  "forcats",
  "lubridate",
  "janitor",
  "jsonlite",
  "glue",
  "fs",
  "rlang",

  # Plotting/tables/reporting
  "ggplot2",
  "gt",
  "modelsummary",
  "scales",

  # Statistics/regression
  "broom",
  "lmtest",
  "sandwich",
  "sampleSelection",

  # Optional helper used when running interactively from RStudio
  "rstudioapi"
)

missing <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0L) {
  message("Installing missing R packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = repos, dependencies = TRUE)
} else {
  message("All required R packages are already installed.")
}

still_missing <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(still_missing) > 0L) {
  stop(
    "These R packages could not be loaded after installation: ",
    paste(still_missing, collapse = ", "),
    call. = FALSE
  )
}

message("R package setup complete.")
