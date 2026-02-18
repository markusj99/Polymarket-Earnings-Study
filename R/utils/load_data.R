# R/utils/load_data.R
# =============================================================================
# Purpose: Centralized data loader so multiple scripts share the same inputs.
# Returns: Named list of tibbles
# Usage:
#   source(file.path(ROOT, "R", "utils", "load_data.R"))
#   D <- load_project_data(ROOT)
#   dataset_long  <- D$dataset_long
#   stock_prices  <- D$stock_prices
#   heckman_events <- D$heckman_universe_events
# =============================================================================

load_project_data <- function(root) {
  stopifnot(is.character(root), length(root) == 1, nzchar(root))

  # ---------------------------------------------------------------------------
  # Define canonical input paths (relative to project root)
  # ---------------------------------------------------------------------------
  paths <- list(
    dataset_long            = file.path(root, "data", "complete_dataset_long.csv"),
    stock_prices            = file.path(root, "data", "stock_prices", "stock_prices_daily.csv"),
    heckman_universe_events = file.path(root, "data", "heckman_selection_model", "heckman_universe_events.csv")
  )

  # ---------------------------------------------------------------------------
  # Guardrails: prevent NULL path lookups
  # ---------------------------------------------------------------------------
  required_names <- c(
    "dataset_long",
    "stock_prices",
    "heckman_universe_events"
  )

  missing_names <- setdiff(required_names, names(paths))
  if (length(missing_names) > 0) {
    stop(
      "Bug in load_data.R: 'paths' is missing these names: ",
      paste(missing_names, collapse = ", "),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Fail fast with a clear message if any input file is missing
  # ---------------------------------------------------------------------------
  missing_files <- names(paths)[!vapply(paths, file.exists, logical(1))]
  if (length(missing_files) > 0) {
    msg <- paste0(
      "Input file(s) not found:\n",
      paste0(" - ", missing_files, ": ", unlist(paths[missing_files]), collapse = "\n"),
      "\n\nRoot used: ", root
    )
    stop(msg, call. = FALSE)
  }

  # ---------------------------------------------------------------------------
  # Read all files (UTF-8 locale for consistent parsing)
  # Use [[ ]] so a missing name cannot silently become NULL.
  # ---------------------------------------------------------------------------
  loc <- readr::locale(encoding = "UTF-8")

  data <- list(
    dataset_long            = readr::read_csv(paths[["dataset_long"]], show_col_types = FALSE, locale = loc),
    stock_prices            = readr::read_csv(paths[["stock_prices"]], show_col_types = FALSE, locale = loc),
    heckman_universe_events = readr::read_csv(paths[["heckman_universe_events"]], show_col_types = FALSE, locale = loc)
  )

  return(data)
}
