############################################################
# 0) Setup: working directory + packages
############################################################

# Set your working directory (adjust if needed)
setwd("~/Desktop/Thesis/R")

# Install packages once (uncomment if needed)
# install.packages(c("dplyr", "readr", "ggplot2"))

library(dplyr)
library(readr)
library(ggplot2)

############################################################
# 1) Load data: Polymarket prices + validated outcomes
############################################################

prices <- read_csv("poly_prices_long.csv")   # Polymarket snapshots (p = price_yes)
val    <- read_csv("correct.csv")            # Correctly matched/validated outcomes

############################################################
# 2) Merge + create core variables for forecasting
#    y = realized outcome (1 if YES, else 0)
#    p = Polymarket implied probability (price_yes)
############################################################

df <- prices %>%
  inner_join(val %>% select(market_id, expected_resolution), by = "market_id") %>%
  mutate(
    y = ifelse(expected_resolution == "YES", 1, 0),
    p = price_yes
  )

############################################################
# 3) Brier Score: overall + by horizon (snapshot_label)
############################################################

brier_overall <- df %>%
  summarise(
    brier = mean((p - y)^2, na.rm = TRUE),
    N = sum(!is.na(p))
  )

brier_by_horizon <- df %>%
  filter(!is.na(p)) %>%
  group_by(snapshot_label) %>%
  summarise(
    N = n(),
    brier = mean((p - y)^2),
    .groups = "drop"
  ) %>%
  arrange(brier)

# Print
brier_overall
brier_by_horizon

############################################################
# 4) Percentage of Correct Predictions (Accuracy)
#    Using cutoff 0.5: predict YES if p>=0.5 else NO
############################################################

acc_by_horizon <- df %>%
  filter(!is.na(p)) %>%
  mutate(
    pred = ifelse(p >= 0.5, 1, 0),
    correct = (pred == y)
  ) %>%
  group_by(snapshot_label) %>%
  summarise(
    N = n(),
    accuracy = mean(correct),
    accuracy_pct = round(100 * accuracy, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(accuracy))

# Print
acc_by_horizon

############################################################
# 5) Calibration (Reliability) for 24h horizon
#    In your data: 24h = "1d"
#    Bins: 0–20, 20–40, 40–60, 60–80, 80–100
############################################################

cal_24h <- df %>%
  filter(snapshot_label == "1d", !is.na(p)) %>%
  mutate(
    p_bin = cut(
      p,
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
      include.lowest = TRUE,
      labels = c("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
    )
  ) %>%
  group_by(p_bin) %>%
  summarise(
    N = n(),
    avg_p = mean(p),
    realized = mean(y),
    .groups = "drop"
  )

# Print
cal_24h

############################################################
# 6) Plot: Calibration (24h) with 45-degree line
#    Point size = N in each bin; labels show bin + N
############################################################

p_cal24 <- ggplot(cal_24h, aes(x = avg_p, y = realized)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_point(aes(size = N)) +
  geom_text(aes(label = paste0(p_bin, "\nN=", N)), nudge_y = 0.03, show.legend = FALSE) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Calibration (24h before earnings)",
    x = "Average implied probability in bracket",
    y = "Realized frequency"
  ) +
  theme_minimal()

print(p_cal24)

############################################################
# 7) Save outputs: tables + figure + full session (optional)
############################################################

# Create folders
dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# Save tables
write.csv(brier_overall,     "outputs/tables/brier_overall.csv", row.names = FALSE)
write.csv(brier_by_horizon,  "outputs/tables/brier_by_horizon.csv", row.names = FALSE)
write.csv(acc_by_horizon,    "outputs/tables/accuracy_by_horizon.csv", row.names = FALSE)
write.csv(cal_24h,           "outputs/tables/calibration_24h_bins.csv", row.names = FALSE)

# Save figure (PNG for Word, PDF for Overleaf)
ggsave("outputs/figures/calibration_24h.png", plot = p_cal24, width = 7, height = 4.5, dpi = 300)
ggsave("outputs/figures/calibration_24h.pdf", plot = p_cal24, width = 7, height = 4.5)

# Optional: save entire session (all objects)
save.image("outputs/session_saved.RData")

############################################################
# Done
############################################################
############################################################
# 8) Brier Skill Score (BSS)
#    Benchmark 1: base-rate forecast (p = mean(y) in sample)
#    Optional Benchmark 2: coin-flip (p = 0.5)
############################################################

# Overall BSS vs base-rate benchmark
p_bar <- mean(df$y, na.rm = TRUE)  # base rate in full sample

brier_ref_base <- mean((p_bar - df$y)^2, na.rm = TRUE)
brier_mkt_all  <- mean((df$p - df$y)^2, na.rm = TRUE)

bss_overall_base <- 1 - (brier_mkt_all / brier_ref_base)

# Overall BSS vs 0.5 benchmark
brier_ref_05 <- mean((0.5 - df$y)^2, na.rm = TRUE)
bss_overall_05 <- 1 - (brier_mkt_all / brier_ref_05)

bss_overall <- data.frame(
  benchmark = c("base_rate", "0.5"),
  brier_market = c(brier_mkt_all, brier_mkt_all),
  brier_benchmark = c(brier_ref_base, brier_ref_05),
  bss = c(bss_overall_base, bss_overall_05)
)

bss_overall


# BSS by horizon vs base-rate (computed within each horizon sample)
bss_by_horizon <- df %>%
  filter(!is.na(p)) %>%
  group_by(snapshot_label) %>%
  summarise(
    N = n(),
    brier_market = mean((p - y)^2),
    p_bar_h = mean(y),  # base rate within this horizon sample
    brier_benchmark = mean((p_bar_h - y)^2),
    bss = 1 - brier_market / brier_benchmark,
    .groups = "drop"
  ) %>%
  arrange(snapshot_label)

bss_by_horizon

# (Optional) Save BSS tables
write.csv(bss_overall,    "outputs/tables/bss_overall.csv", row.names = FALSE)
write.csv(bss_by_horizon, "outputs/tables/bss_by_horizon.csv", row.names = FALSE)


############################################################
# 9) Calibration regressions for 24h (snapshot_label == "1d")
#    9a) OLS calibration regression: y = alpha + beta*p
#        Perfect calibration => alpha = 0 and beta = 1
#    9b) Logit/Probit calibration: y on log-odds / probit index
############################################################

df_24h <- df %>% filter(snapshot_label == "1d", !is.na(p))

# 9a) OLS calibration regression
ols_cal <- lm(y ~ p, data = df_24h)
summary(ols_cal)

# Joint test: alpha = 0 and beta = 1
# We'll use 'car' for linearHypothesis (install once if needed)
# install.packages("car")
library(car)

linearHypothesis(ols_cal, c("(Intercept) = 0", "p = 1"))

# Save OLS regression output (optional)
sink("outputs/tables/ols_calibration_24h.txt")
print(summary(ols_cal))
print(linearHypothesis(ols_cal, c("(Intercept) = 0", "p = 1")))
sink()


# 9b) Logit/Probit calibration models
# (i) Simple logit/probit using p directly (common and interpretable)
logit_p  <- glm(y ~ p, data = df_24h, family = binomial(link = "logit"))
probit_p <- glm(y ~ p, data = df_24h, family = binomial(link = "probit"))

summary(logit_p)
summary(probit_p)

# (ii) Optional: calibration on log-odds transform of p
# Avoid infinities when p=0 or p=1 using a small epsilon
eps <- 1e-6
df_24h <- df_24h %>%
  mutate(
    p_clip = pmin(pmax(p, eps), 1 - eps),
    log_odds = log(p_clip / (1 - p_clip))
  )

logit_logodds <- glm(y ~ log_odds, data = df_24h, family = binomial(link = "logit"))
summary(logit_logodds)

# Save logit/probit outputs (optional)
sink("outputs/tables/logit_probit_calibration_24h.txt")
cat("LOGIT (y ~ p)\n"); print(summary(logit_p))
cat("\nPROBIT (y ~ p)\n"); print(summary(probit_p))
cat("\nLOGIT (y ~ log_odds(p))\n"); print(summary(logit_logodds))
sink()
