# R/constants.R
# Global constants for thesis project (Corporate Earnings / Polymarket)

# --- Time handling ---
TZ_UTC <- "UTC"

# Force R to treat times as UTC in this session
set_utc <- function() {
  Sys.setenv(TZ = TZ_UTC)
  options(tz = TZ_UTC)
}

# --- Snapshot labels (thesis decision) ---
SNAPSHOT_KEEP <- c("1w", "6d", "5d", "4d", "3d", "2d", "1d", "12h", "6h")
SNAPSHOT_EXCLUDE <- c("4w", "3w", "2w")

# Optional: enforce ordering everywhere
SNAPSHOT_ORDER <- SNAPSHOT_KEEP

# --- Convert snapshot label -> horizon hours ---
snapshot_to_hours <- function(lbl) {
  if (grepl("w$", lbl)) return(as.numeric(sub("w$", "", lbl)) * 7 * 24)
  if (grepl("d$", lbl)) return(as.numeric(sub("d$", "", lbl)) * 24)
  if (grepl("h$", lbl)) return(as.numeric(sub("h$", "", lbl)))
  stop("Unknown snapshot label: ", lbl)
}

# Returns TRUE if obs_ts is acceptable for the target_ts for that snapshot label
is_stale_snapshot <- function(obs_ts, target_ts, snapshot_label) {
  # obs_ts and target_ts should be POSIXct in UTC
  lag_hours <- abs(as.numeric(difftime(obs_ts, target_ts, units = "hours")))
  lag_hours > allowed_lag_hours(snapshot_label)
}
