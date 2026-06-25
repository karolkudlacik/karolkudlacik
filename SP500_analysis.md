S&P 500 — Market Risk Analysis
================
Karol Kudłacik
2026-06-24

- [Objective](#objective)
- [1 Data loading & preparation](#1-data-loading--preparation)
- [2 Return computation](#2-return-computation)
- [3 Historical Value-at-Risk (99%)](#3-historical-value-at-risk-99)
- [4 Expected Shortfall (97.5%)](#4-expected-shortfall-975)
- [5 VaR vs. ES comparison](#5-var-vs-es-comparison)
- [6 Simplified IMA-style charge](#6-simplified-ima-style-charge)
- [7 Basel backtesting (real S&P 500)](#7-basel-backtesting-real-sp-500)
- [8 Appendix A — classifier sanity-check (synthetic, **not** an S&P
  result)](#8-appendix-a--classifier-sanity-check-synthetic-not-an-sp-result)
- [9 Summary](#9-summary)
- [Reproducibility](#reproducibility)

*Historical VaR · Expected Shortfall (97.5%) · simplified IMA-style
charge · Basel traffic-light backtesting with formal coverage tests.*

## Objective

Quantify the downside market risk of the S&P 500 and, crucially, **check
whether a standard 99% Value-at-Risk model actually holds up
out-of-sample** — the validation question a risk function has to answer
before relying on a model for capital. The pipeline goes from raw prices
to regulatory-style metrics: Historical VaR, Expected Shortfall, a
(deliberately simplified) FRTB IMA-style charge, and a Basel backtest of
the model on the *realised* index returns.

**Data.** Daily S&P 500 OHLC prices, file `spx_d.csv` (source:
[Stooq](https://stooq.pl/q/d/l/?s=%5Espx&i=d); schema
`Date,Open,High,Low,Close,Volume`, only `Date`/`Open`/`Close` used),
sample from 1995-01-01 onwards.

**Reproducibility.** Everything on the real index is deterministic. The
only stochastic part is *Appendix A* — a synthetic sanity-check of the
traffic-light classifier (not a result for the S&P 500) — seeded with
`set.seed(42)`. The analysis uses **base R only**; `knitr` is used here
purely to render the tables.

## 1 Data loading & preparation

Prices are sorted, filtered from 1995, and reduced to the columns we
need.

``` r
sp500_data       <- read.csv("spx_d.csv")
sp500_data$Date  <- as.Date(sp500_data$Date)
sp500_data       <- sp500_data[order(sp500_data$Date), ]

cutoff_date      <- as.Date("1995-01-01")
sp500_filtered   <- sp500_data[sp500_data$Date >= cutoff_date, c("Date", "Open", "Close")]
sp500_filtered   <- sp500_filtered[complete.cases(sp500_filtered), ]

close_vec        <- sp500_filtered$Close
dates_vec        <- sp500_filtered$Date

cat("Observations after 1995-01-01:", nrow(sp500_filtered), "\n")
```

    #> Observations after 1995-01-01: 7857

``` r
cat("Date range:", format(min(dates_vec)), "to", format(max(dates_vec)), "\n")
```

    #> Date range: 1995-01-03 to 2026-03-23

## 2 Return computation

The simple (arithmetic) return over a holding period $h$ is
$r_t^{(h)} = (P_t - P_{t-h})/P_{t-h}$, with $h=1$ for daily and $h=10$
for the ten-trading-day Basel horizon.

> **Note on the 10-day series.** These are *overlapping* 10-day returns
> (each shares 9 days with its neighbour). They are convenient but
> serially correlated by construction — which matters for exception
> counting and is flagged again below.

``` r
returns_1d  <- diff(close_vec, lag = 1)  / head(close_vec, -1)
returns_10d <- diff(close_vec, lag = 10) / head(close_vec, -10)

dates_1d    <- dates_vec[-1]
dates_10d   <- dates_vec[-(1:10)]

cat("1-day  return observations:", length(returns_1d),  "\n")
```

    #> 1-day  return observations: 7856

``` r
cat("10-day return observations:", length(returns_10d), "\n")
```

    #> 10-day return observations: 7847

## 3 Historical Value-at-Risk (99%)

A rolling 252-day Historical VaR at 99% confidence, using the $(n+1)\,p$
plotting-position quantile (R’s `quantile` type 6). With $n=252$ and
$p=0.01$ the tail position is $h=(252+1)\times 0.01 = 2.53$, so

$$\text{VaR}_t = -\Big[\,r_{(2)} + 0.53\,\big(r_{(3)} - r_{(2)}\big)\Big],$$

where $r_{(k)}$ is the $k$-th smallest return in the window. VaR is
reported as a **positive** loss number.

> **Alignment.** $\text{VaR}_t$ is estimated from returns
> $[t-252,\ t-1]$ and never uses day $t$ itself, so the estimate is
> genuinely out-of-sample. The *same* alignment is reused everywhere
> below (including the backtest), so no separate “shift” step is needed.

``` r
hist_VaR <- function(returns_vec, window = 252, conf = 0.99) {
  n   <- length(returns_vec)
  p   <- 1 - conf                              # 0.01 for 99% VaR
  VaR <- rep(NA_real_, n)

  for (t in (window + 1):n) {
    window_returns <- returns_vec[(t - window):(t - 1)]   # 252 past returns
    sorted         <- sort(window_returns)                # ascending (worst first)

    h    <- (window + 1) * p                   # = 2.53
    lo   <- floor(h)                           # = 2
    hi   <- lo + 1                             # = 3
    frac <- h - lo                             # = 0.53

    VaR[t] <- -(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
  }
  VaR
}
```

``` r
VaR_1d  <- hist_VaR(returns_1d,  window = 252, conf = 0.99)
VaR_10d <- hist_VaR(returns_10d, window = 252, conf = 0.99)

cat("=== 1-Day Historical VaR (99%) ===\n");  print(summary(na.omit(VaR_1d)))
```

    #> === 1-Day Historical VaR (99%) ===

    #>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    #> 0.01378 0.02194 0.02952 0.03225 0.03620 0.08864

``` r
cat("\n=== 10-Day Historical VaR (99%) ===\n"); print(summary(na.omit(VaR_10d)))
```

    #> 
    #> === 10-Day Historical VaR (99%) ===

    #>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    #> 0.01877 0.04922 0.07310 0.08403 0.09970 0.23191

``` r
par(mfrow = c(2, 2), mar = c(3, 4, 3, 1))

plot(dates_1d, returns_1d, type = "l", col = "steelblue", lwd = 0.7,
     main = "S&P 500 - 1-Day Simple Returns", xlab = "", ylab = "Return", xaxt = "n")
axis.Date(1, at = seq(min(dates_1d), max(dates_1d), by = "5 years"), format = "%Y")
abline(h = 0, col = "black", lwd = 0.8)

plot(dates_10d, returns_10d, type = "l", col = "darkorange", lwd = 0.7,
     main = "S&P 500 - 10-Day Simple Returns", xlab = "", ylab = "Return", xaxt = "n")
axis.Date(1, at = seq(min(dates_10d), max(dates_10d), by = "5 years"), format = "%Y")
abline(h = 0, col = "black", lwd = 0.8)

plot(dates_1d, returns_1d, type = "l", col = "steelblue", lwd = 0.7,
     main = "1-Day Returns vs. Rolling 99% VaR", xlab = "", ylab = "Return / VaR", xaxt = "n")
axis.Date(1, at = seq(min(dates_1d), max(dates_1d), by = "5 years"), format = "%Y")
lines(dates_1d, -VaR_1d, col = "firebrick", lwd = 1.5)
abline(h = 0, col = "black", lwd = 0.8)
legend("bottomleft", legend = c("1-day return", "-VaR 99%"),
       col = c("steelblue", "firebrick"), lwd = c(0.7, 1.5), bty = "n", cex = 0.8)

plot(dates_10d, returns_10d, type = "l", col = "darkorange", lwd = 0.7,
     main = "10-Day Returns vs. Rolling 99% VaR", xlab = "", ylab = "Return / VaR", xaxt = "n")
axis.Date(1, at = seq(min(dates_10d), max(dates_10d), by = "5 years"), format = "%Y")
lines(dates_10d, -VaR_10d, col = "firebrick", lwd = 1.5)
abline(h = 0, col = "black", lwd = 0.8)
legend("bottomleft", legend = c("10-day return", "-VaR 99%"),
       col = c("darkorange", "firebrick"), lwd = c(0.7, 1.5), bty = "n", cex = 0.8)
```

<img src="SP500_analysis_files/figure-gfm/var-plots-1.png" style="display: block; margin: auto;" />

``` r
par(mfrow = c(1, 1))
```

### VaR exceedance counts (descriptive)

A 99% VaR is expected to be breached on ~1% of days. Formal coverage
tests are run on the 1-day series in the [backtesting
section](#7-basel-backtesting-real-sp-500).

``` r
breaches_1d  <- sum(-returns_1d  > VaR_1d,  na.rm = TRUE)
breaches_10d <- sum(-returns_10d > VaR_10d, na.rm = TRUE)
total_1d     <- sum(!is.na(VaR_1d))
total_10d    <- sum(!is.na(VaR_10d))

cat(sprintf("1-Day  VaR exceptions: %d / %d  (%.2f%%)  | expected 1.00%%\n",
            breaches_1d, total_1d, 100 * breaches_1d / total_1d))
```

    #> 1-Day  VaR exceptions: 97 / 7604  (1.28%)  | expected 1.00%

``` r
cat(sprintf("10-Day VaR exceptions: %d / %d  (%.2f%%)  | overlapping returns\n",
            breaches_10d, total_10d, 100 * breaches_10d / total_10d))
```

    #> 10-Day VaR exceptions: 127 / 7595  (1.67%)  | overlapping returns

The 10-day exceptions are serially dependent (overlap), so the count
**cannot** be compared to 1% the way the 1-day count can — treat it as
descriptive only.

## 4 Expected Shortfall (97.5%)

Expected Shortfall answers: *given that we are in the worst 2.5% of
days, what is the average loss?* The 97.5% level is the FRTB convention.
With $252\times0.025 = 6.3$ observations in the tail (6 fully + 0.3
weight on the 7th):

$$\text{ES}_t = -\frac{1}{6.3}\left[\sum_{k=1}^{6} r_{(k)} + 0.3\,r_{(7)}\right].$$

> **Convention note.** VaR above uses the $(n+1)\,p$ position; this ES
> uses the $n\,p$ tail mass ($252\times0.025=6.3$). They are
> deliberately different estimators — a quantile vs. a tail average.

``` r
hist_ES <- function(returns_vec, window = 252, conf = 0.975) {
  n   <- length(returns_vec)
  p   <- 1 - conf                          # 0.025
  ES  <- rep(NA_real_, n)

  tail_n    <- window * p                  # 6.3
  tail_full <- floor(tail_n)               # 6
  tail_frac <- tail_n - tail_full          # 0.3

  for (t in (window + 1):n) {
    window_returns <- returns_vec[(t - window):(t - 1)]
    sorted         <- sort(window_returns)
    tail_sum <- sum(sorted[1:tail_full]) + tail_frac * sorted[tail_full + 1]
    ES[t]    <- -tail_sum / tail_n
  }
  ES
}
```

``` r
ES_1d  <- hist_ES(returns_1d,  window = 252, conf = 0.975)
ES_10d <- hist_ES(returns_10d, window = 252, conf = 0.975)

cat("=== 1-Day Expected Shortfall (97.5%) ===\n");  print(summary(na.omit(ES_1d)))
```

    #> === 1-Day Expected Shortfall (97.5%) ===

    #>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    #> 0.01201 0.02009 0.02843 0.03022 0.03529 0.07785

``` r
cat("\n=== 10-Day Expected Shortfall (97.5%) ===\n"); print(summary(na.omit(ES_10d)))
```

    #> 
    #> === 10-Day Expected Shortfall (97.5%) ===

    #>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    #> 0.01711 0.04703 0.06976 0.07822 0.09462 0.20468

``` r
par(mfrow = c(2, 1), mar = c(3, 4, 3, 1))

plot(dates_1d, returns_1d, type = "l", col = "steelblue", lwd = 0.6,
     main = "S&P 500 - 1-Day Returns | Rolling 99% VaR & 97.5% ES",
     xlab = "", ylab = "Return / Risk", xaxt = "n",
     ylim = range(c(returns_1d, -VaR_1d, -ES_1d), na.rm = TRUE))
axis.Date(1, at = seq(min(dates_1d), max(dates_1d), by = "5 years"), format = "%Y")
lines(dates_1d, -VaR_1d, col = "firebrick",  lwd = 1.5)
lines(dates_1d, -ES_1d,  col = "darkorchid", lwd = 1.5)
abline(h = 0, col = "black", lwd = 0.5)
legend("bottomleft", legend = c("1-day return", "-VaR 99%", "-ES 97.5%"),
       col = c("steelblue", "firebrick", "darkorchid"), lwd = c(0.6, 1.5, 1.5),
       bty = "n", cex = 0.8)

plot(dates_10d, returns_10d, type = "l", col = "darkorange", lwd = 0.6,
     main = "S&P 500 - 10-Day Returns | Rolling 99% VaR & 97.5% ES",
     xlab = "", ylab = "Return / Risk", xaxt = "n",
     ylim = range(c(returns_10d, -VaR_10d, -ES_10d), na.rm = TRUE))
axis.Date(1, at = seq(min(dates_10d), max(dates_10d), by = "5 years"), format = "%Y")
lines(dates_10d, -VaR_10d, col = "firebrick",  lwd = 1.5)
lines(dates_10d, -ES_10d,  col = "darkorchid", lwd = 1.5)
abline(h = 0, col = "black", lwd = 0.5)
legend("bottomleft", legend = c("10-day return", "-VaR 99%", "-ES 97.5%"),
       col = c("darkorange", "firebrick", "darkorchid"), lwd = c(0.6, 1.5, 1.5),
       bty = "n", cex = 0.8)
```

<img src="SP500_analysis_files/figure-gfm/es-plots-1.png" style="display: block; margin: auto;" />

``` r
par(mfrow = c(1, 1))
```

### ES exceedance count — diagnostic only

Counting how often the loss exceeds the ES level is a monitoring
**diagnostic, not a validation of ES**. ES is not elicitable, so there
is no clean “expected frequency” to test against; a proper ES backtest
would use e.g. the Acerbi–Székely (2014) tests.

``` r
exc_ES_1d  <- sum(-returns_1d  > ES_1d,  na.rm = TRUE)
exc_ES_10d <- sum(-returns_10d > ES_10d, na.rm = TRUE)

cat(sprintf("1-Day  ES exceedances: %d / %d  (%.2f%%)\n",
            exc_ES_1d, sum(!is.na(ES_1d)), 100 * exc_ES_1d / sum(!is.na(ES_1d))))
```

    #> 1-Day  ES exceedances: 110 / 7604  (1.45%)

``` r
cat(sprintf("10-Day ES exceedances: %d / %d  (%.2f%%)\n",
            exc_ES_10d, sum(!is.na(ES_10d)), 100 * exc_ES_10d / sum(!is.na(ES_10d))))
```

    #> 10-Day ES exceedances: 150 / 7595  (1.97%)

## 5 VaR vs. ES comparison

``` r
comp <- data.frame(
  Metric   = c("VaR 99% - mean", "VaR 99% - max",
               "ES 97.5% - mean", "ES 97.5% - max", "ES / VaR ratio (avg)"),
  `1-Day`  = c(mean(VaR_1d, na.rm = TRUE), max(VaR_1d, na.rm = TRUE),
               mean(ES_1d, na.rm = TRUE),  max(ES_1d, na.rm = TRUE),
               mean(ES_1d / VaR_1d, na.rm = TRUE)),
  `10-Day` = c(mean(VaR_10d, na.rm = TRUE), max(VaR_10d, na.rm = TRUE),
               mean(ES_10d, na.rm = TRUE),  max(ES_10d, na.rm = TRUE),
               mean(ES_10d / VaR_10d, na.rm = TRUE)),
  check.names = FALSE
)
knitr::kable(comp, digits = 4, caption = "Rolling VaR and ES, full sample.")
```

| Metric               |  1-Day | 10-Day |
|:---------------------|-------:|-------:|
| VaR 99% - mean       | 0.0322 | 0.0840 |
| VaR 99% - max        | 0.0886 | 0.2319 |
| ES 97.5% - mean      | 0.0302 | 0.0782 |
| ES 97.5% - max       | 0.0779 | 0.2047 |
| ES / VaR ratio (avg) | 0.9461 | 0.9439 |

Rolling VaR and ES, full sample.

**`√10` cross-check.** The regulatory shortcut for a 10-day horizon
scales the 1-day VaR by $\sqrt{10}$ under an i.i.d. assumption.
Comparing it to the direct (overlapping) 10-day VaR is a sanity check,
not a claim that one is correct — a gap is expected because real returns
are neither i.i.d. nor free of serial correlation.

``` r
cat(sprintf("Direct overlapping 10-day VaR (mean): %.4f\n", mean(VaR_10d, na.rm = TRUE)))
```

    #> Direct overlapping 10-day VaR (mean): 0.0840

``` r
cat(sprintf("sqrt(10) * 1-day VaR (mean)         : %.4f\n", sqrt(10) * mean(VaR_1d, na.rm = TRUE)))
```

    #> sqrt(10) * 1-day VaR (mean)         : 0.1020

## 6 Simplified IMA-style charge

This captures the **spirit** of the FRTB Internal Models Approach —
capital driven by the larger of a stressed-period ES and a current ES at
97.5%:

$$\text{Charge} = \max\!\big(\text{ES}_{\text{stress}},\ \text{ES}_{\text{recent}}\big).$$

> **It is a deliberate simplification, not the regulatory IMA number.**
> It omits the capital multiplier ($m_c \ge 1.5$), liquidity-horizon
> scaling, the reduced-set calibration, the non-modellable-risk-factor
> (NMRF/SES) add-on, the default-risk charge (DRC), and cross-risk-class
> aggregation.

The stress window is found by scanning **every** 252-day window for the
highest ES; recent ES is the last 252 days.

``` r
window    <- 252
conf      <- 0.975
p         <- 1 - conf
tail_n    <- window * p          # 6.3 (globals used by es_single_window below)
tail_full <- floor(tail_n)       # 6
tail_frac <- tail_n - tail_full  # 0.3

es_single_window <- function(ret_window) {
  sorted   <- sort(ret_window)
  tail_sum <- sum(sorted[1:tail_full]) + tail_frac * sorted[tail_full + 1]
  -tail_sum / tail_n
}

find_stress_period <- function(returns_vec, dates_vec, window = 252) {
  n         <- length(returns_vec)
  n_windows <- n - window + 1
  es_all    <- numeric(n_windows)
  for (i in seq_len(n_windows)) {
    es_all[i] <- es_single_window(returns_vec[i:(i + window - 1)])
  }
  best_i <- which.max(es_all)
  list(stress_ES = es_all[best_i],
       stress_start = dates_vec[best_i],
       stress_end   = dates_vec[best_i + window - 1],
       stress_index = best_i,
       all_es       = es_all)
}

stress_1d  <- find_stress_period(returns_1d,  dates_1d)
stress_10d <- find_stress_period(returns_10d, dates_10d)

recent_ES_1d  <- es_single_window(tail(returns_1d,  window))
recent_ES_10d <- es_single_window(tail(returns_10d, window))

IMA_1d  <- max(stress_1d$stress_ES,  recent_ES_1d)
IMA_10d <- max(stress_10d$stress_ES, recent_ES_10d)
```

``` r
stress_tbl <- data.frame(
  Horizon            = c("1-day", "10-day"),
  `Stress window`    = c(paste(format(stress_1d$stress_start),  "to", format(stress_1d$stress_end)),
                         paste(format(stress_10d$stress_start), "to", format(stress_10d$stress_end))),
  `Stress ES (97.5%)`= c(stress_1d$stress_ES, stress_10d$stress_ES),
  check.names = FALSE
)
knitr::kable(stress_tbl, digits = 4, caption = "Auto-identified stress period (highest-ES 252-day window).")
```

| Horizon | Stress window            | Stress ES (97.5%) |
|:--------|:-------------------------|------------------:|
| 1-day   | 2007-12-03 to 2008-12-01 |            0.0779 |
| 10-day  | 2019-03-25 to 2020-03-23 |            0.2047 |

Auto-identified stress period (highest-ES 252-day window).

``` r
ima_tbl <- data.frame(
  Metric   = c("Stress ES (97.5%)", "Recent ES (97.5%)",
               "Simplified IMA-style charge", "Binding constraint"),
  `1-Day`  = c(sprintf("%.4f", stress_1d$stress_ES), sprintf("%.4f", recent_ES_1d),
               sprintf("%.4f", IMA_1d),
               ifelse(stress_1d$stress_ES  >= recent_ES_1d,  "STRESS", "RECENT")),
  `10-Day` = c(sprintf("%.4f", stress_10d$stress_ES), sprintf("%.4f", recent_ES_10d),
               sprintf("%.4f", IMA_10d),
               ifelse(stress_10d$stress_ES >= recent_ES_10d, "STRESS", "RECENT")),
  check.names = FALSE
)
knitr::kable(ima_tbl, caption = "Simplified IMA-style charge = max(stress ES, recent ES).")
```

| Metric                      | 1-Day  | 10-Day |
|:----------------------------|:-------|:-------|
| Stress ES (97.5%)           | 0.0779 | 0.2047 |
| Recent ES (97.5%)           | 0.0352 | 0.0904 |
| Simplified IMA-style charge | 0.0779 | 0.2047 |
| Binding constraint          | STRESS | STRESS |

Simplified IMA-style charge = max(stress ES, recent ES).

``` r
par(mfrow = c(2, 2), mar = c(3, 4, 3, 1))

dates_scan_1d  <- dates_1d [1:(length(returns_1d)  - window + 1)]
dates_scan_10d <- dates_10d[1:(length(returns_10d) - window + 1)]

plot(dates_scan_1d, stress_1d$all_es, type = "l", col = "steelblue", lwd = 1,
     main = "ES of Every 252-Day Window - 1-Day", xlab = "", ylab = "ES (97.5%)", xaxt = "n")
axis.Date(1, at = seq(min(dates_scan_1d), max(dates_scan_1d), by = "5 years"), format = "%Y")
abline(v = stress_1d$stress_start, col = "firebrick",  lwd = 2,   lty = 2)
abline(h = stress_1d$stress_ES,    col = "firebrick",  lwd = 1.5, lty = 3)
abline(h = recent_ES_1d,           col = "darkorchid", lwd = 1.5, lty = 3)
legend("topright", legend = c("Rolling ES", "Stress start", "Stress ES", "Recent ES"),
       col = c("steelblue","firebrick","firebrick","darkorchid"),
       lwd = c(1,2,1.5,1.5), lty = c(1,2,3,3), bty = "n", cex = 0.75)

plot(dates_scan_10d, stress_10d$all_es, type = "l", col = "darkorange", lwd = 1,
     main = "ES of Every 252-Day Window - 10-Day", xlab = "", ylab = "ES (97.5%)", xaxt = "n")
axis.Date(1, at = seq(min(dates_scan_10d), max(dates_scan_10d), by = "5 years"), format = "%Y")
abline(v = stress_10d$stress_start, col = "firebrick",  lwd = 2,   lty = 2)
abline(h = stress_10d$stress_ES,    col = "firebrick",  lwd = 1.5, lty = 3)
abline(h = recent_ES_10d,           col = "darkorchid", lwd = 1.5, lty = 3)
legend("topright", legend = c("Rolling ES", "Stress start", "Stress ES", "Recent ES"),
       col = c("darkorange","firebrick","firebrick","darkorchid"),
       lwd = c(1,2,1.5,1.5), lty = c(1,2,3,3), bty = "n", cex = 0.75)

plot(dates_1d, returns_1d, type = "l", col = "steelblue", lwd = 0.6,
     main = "1-Day Returns - Stress Period Highlighted", xlab = "", ylab = "Return", xaxt = "n")
axis.Date(1, at = seq(min(dates_1d), max(dates_1d), by = "5 years"), format = "%Y")
rect(stress_1d$stress_start, par("usr")[3], stress_1d$stress_end, par("usr")[4],
     col = adjustcolor("firebrick", alpha.f = 0.15), border = NA)
abline(h = 0, lwd = 0.5)

plot(dates_10d, returns_10d, type = "l", col = "darkorange", lwd = 0.6,
     main = "10-Day Returns - Stress Period Highlighted", xlab = "", ylab = "Return", xaxt = "n")
axis.Date(1, at = seq(min(dates_10d), max(dates_10d), by = "5 years"), format = "%Y")
rect(stress_10d$stress_start, par("usr")[3], stress_10d$stress_end, par("usr")[4],
     col = adjustcolor("firebrick", alpha.f = 0.15), border = NA)
abline(h = 0, lwd = 0.5)
```

<img src="SP500_analysis_files/figure-gfm/stress-plots-1.png" style="display: block; margin: auto;" />

``` r
par(mfrow = c(1, 1))
```

## 7 Basel backtesting (real S&P 500)

The **primary** backtest: realised 1-day returns vs. the out-of-sample
99% VaR computed above. Exceptions are counted, then the Basel **traffic
light** is evaluated over the most recent 250 trading days (the
regulatory one-year window).

``` r
# Exception indicator on the REAL series (NA where VaR is absent -> FALSE)
breach_1d_vec <- (-returns_1d > VaR_1d) & !is.na(VaR_1d)
valid_idx_1d  <- which(!is.na(VaR_1d))

n_valid_1d  <- length(valid_idx_1d)
n_breach_1d <- sum(breach_1d_vec[valid_idx_1d])

# Basel traffic light over the most recent 250 trading days
tl_window       <- 250
recent_idx      <- tail(valid_idx_1d, tl_window)
recent_dates    <- dates_1d[recent_idx]
recent_breaches <- sum(breach_1d_vec[recent_idx])
tl_status <- if (recent_breaches <= 4) "GREEN" else if (recent_breaches <= 9) "AMBER" else "RED"

# Basel (1996) plus-factor: addition to the 'k' multiplier of 3
multiplier_table <- data.frame(
  breaches = 0:10,
  zone     = c(rep("GREEN", 5), rep("AMBER", 5), "RED"),
  addition = c(0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.50, 0.65, 0.75, 0.85, 1.00)
)
k_addition <- multiplier_table$addition[multiplier_table$breaches == min(recent_breaches, 10)]

cat(sprintf("Full sample : %d / %d exceptions (%.2f%%, expected 1.00%%)\n",
            n_breach_1d, n_valid_1d, 100 * n_breach_1d / n_valid_1d))
```

    #> Full sample : 97 / 7604 exceptions (1.28%, expected 1.00%)

``` r
cat(sprintf("Last 250 days: %s to %s\n", format(min(recent_dates)), format(max(recent_dates))))
```

    #> Last 250 days: 2025-03-25 to 2026-03-23

``` r
cat(sprintf("              %d exceptions -> [ %s ]  (k add-on +%.2f)\n",
            recent_breaches, tl_status, k_addition))
```

    #>               2 exceptions -> [ GREEN ]  (k add-on +0.00)

``` r
valid_bt      <- !is.na(VaR_1d)
plot_dates    <- dates_1d[valid_bt]
plot_returns  <- returns_1d[valid_bt]
plot_VaR      <- VaR_1d[valid_bt]
plot_breaches <- breach_1d_vec[valid_bt]

plot(plot_dates, plot_returns, type = "l", col = "steelblue", lwd = 0.6,
     main = "S&P 500 - Realised 1-Day Returns vs. Rolling 99% VaR (out-of-sample)",
     xlab = "Date", ylab = "Daily return", xaxt = "n",
     ylim = range(c(plot_returns, -plot_VaR), na.rm = TRUE))
axis.Date(1, at = seq(min(plot_dates), max(plot_dates), by = "5 years"), format = "%Y")
lines(plot_dates, -plot_VaR, col = "firebrick", lwd = 1.4)
points(plot_dates[plot_breaches], plot_returns[plot_breaches], col = "red", pch = 19, cex = 0.5)
rect(min(recent_dates), par("usr")[3], max(recent_dates), par("usr")[4],
     col = adjustcolor("gold", alpha.f = 0.15), border = NA)
tl_col <- switch(tl_status, GREEN = "#2ecc71", AMBER = "#f39c12", RED = "#e74c3c")
legend("bottomleft",
       legend = c("1-day return", "-VaR 99%", "Exception",
                  sprintf("Last 250d (%s: %d exc.)", tl_status, recent_breaches)),
       col = c("steelblue", "firebrick", "red", adjustcolor("gold", 0.6)),
       pch = c(NA, NA, 19, 15), lwd = c(0.6, 1.4, NA, NA), pt.cex = c(NA, NA, 0.5, 1.5),
       bty = "n", cex = 0.75, text.col = c("black", "black", "black", tl_col))
abline(h = 0, col = "black", lwd = 0.4)
```

<img src="SP500_analysis_files/figure-gfm/backtest-plot-1.png" style="display: block; margin: auto;" />

### Formal coverage tests — Kupiec & Christoffersen

Run on the **full-sample** 1-day exceptions ($N \approx 7{,}600$), where
the chi-square asymptotics are reliable. They are **not** run on the
10-day series: overlapping returns make exceptions serially dependent by
construction, which violates the independence assumption. (On the short
250-day window the LR asymptotics are also unreliable, so the zone above
uses the Basel exact-binomial cut-offs instead.)

``` r
# Safe log-likelihood term: count*log(prob) with the convention 0*log(0) = 0
g <- function(count, prob) if (count == 0) 0 else count * log(prob)

kupiec_pof <- function(exceptions, p = 0.01) {
  exceptions <- exceptions[!is.na(exceptions)]
  N <- length(exceptions); x <- sum(exceptions); pi_hat <- x / N
  ll_null <- g(N - x, 1 - p)      + g(x, p)
  ll_alt  <- g(N - x, 1 - pi_hat) + g(x, pi_hat)
  LR <- -2 * (ll_null - ll_alt)
  list(N = N, x = x, rate = pi_hat, LR = LR, p_value = pchisq(LR, 1, lower.tail = FALSE))
}

christoffersen_ind <- function(exceptions) {
  ex <- as.integer(exceptions[!is.na(exceptions)]); n <- length(ex)
  prev <- ex[-n]; curr <- ex[-1]
  n00 <- sum(prev == 0 & curr == 0); n01 <- sum(prev == 0 & curr == 1)
  n10 <- sum(prev == 1 & curr == 0); n11 <- sum(prev == 1 & curr == 1)
  pi01 <- if ((n00 + n01) > 0) n01 / (n00 + n01) else 0
  pi11 <- if ((n10 + n11) > 0) n11 / (n10 + n11) else 0
  pi   <- (n01 + n11) / (n00 + n01 + n10 + n11)
  ll_null <- g(n00 + n10, 1 - pi)   + g(n01 + n11, pi)
  ll_alt  <- g(n00, 1 - pi01) + g(n01, pi01) + g(n10, 1 - pi11) + g(n11, pi11)
  LR <- -2 * (ll_null - ll_alt)
  list(LR = LR, p_value = pchisq(LR, 1, lower.tail = FALSE))
}

exc_1d <- as.integer(breach_1d_vec[valid_idx_1d])   # 0/1, contiguous in time
uc     <- kupiec_pof(exc_1d, p = 0.01)
ind    <- christoffersen_ind(exc_1d)
LR_cc  <- uc$LR + ind$LR
p_cc   <- pchisq(LR_cc, df = 2, lower.tail = FALSE)

cov_tbl <- data.frame(
  Test      = c("Kupiec POF (unconditional)", "Christoffersen (independence)", "Conditional coverage"),
  LR        = c(uc$LR, ind$LR, LR_cc),
  df        = c(1, 1, 2),
  `p-value` = c(uc$p_value, ind$p_value, p_cc),
  Decision  = c(ifelse(uc$p_value < 0.05, "reject", "do not reject"),
                ifelse(ind$p_value < 0.05, "reject", "do not reject"),
                ifelse(p_cc < 0.05, "reject", "do not reject")),
  check.names = FALSE
)
knitr::kable(cov_tbl, digits = 3,
             caption = sprintf("Backtest of 99%% VaR on %d observations, %d exceptions (%.2f%%).",
                               uc$N, uc$x, 100 * uc$rate))
```

| Test                          |     LR |  df | p-value | Decision |
|:------------------------------|-------:|----:|--------:|:---------|
| Kupiec POF (unconditional)    |  5.368 |   1 |   0.021 | reject   |
| Christoffersen (independence) | 21.491 |   1 |   0.000 | reject   |
| Conditional coverage          | 26.859 |   2 |   0.000 | reject   |

Backtest of 99% VaR on 7604 observations, 97 exceptions (1.28%).

On this sample the model is rejected on all three tests: the exception
rate exceeds 1% (Kupiec) **and** the exceptions cluster in time
(Christoffersen independence strongly rejected). Both are the textbook
failure mode of equal-weighted historical simulation — it reacts slowly
to volatility changes and does not capture volatility clustering, so
breaches bunch together in crisis periods. This is a known limitation of
the method, not a coding artefact; a conditional model (EWMA- or
GARCH-filtered historical simulation) would target both effects.

## 8 Appendix A — classifier sanity-check (synthetic, **not** an S&P result)

> This is **not** a backtest of the S&P 500 and **not** a result about
> the index. Its only purpose is to confirm that the breach-counting and
> zone logic behave correctly when fed a series with a known,
> deliberately stressed regime. The series is simulated from a
> regime-switching Normal model (no fat tails, no realistic clustering);
> the resulting zone depends on the RNG seed and is meaningless as a
> risk statement.

``` r
set.seed(42)

dates_syn <- seq(as.Date("2007-01-02"), Sys.Date(), by = "day")
dates_syn <- dates_syn[!weekdays(dates_syn) %in% c("Saturday", "Sunday")]
n_syn     <- length(dates_syn)

returns_syn <- numeric(n_syn)
for (i in seq_len(n_syn)) {
  d <- dates_syn[i]
  if (d >= as.Date("2008-09-01") & d <= as.Date("2009-06-30")) {
    returns_syn[i] <- rnorm(1, mean = -0.0015, sd = 0.028)   # GFC
  } else if (d >= as.Date("2020-02-20") & d <= as.Date("2020-05-31")) {
    returns_syn[i] <- rnorm(1, mean = -0.0008, sd = 0.025)   # COVID crash
  } else if ((d >= as.Date("2011-07-01") & d <= as.Date("2011-12-31")) |
             (d >= as.Date("2018-10-01") & d <= as.Date("2018-12-31")) |
             (d >= as.Date("2022-01-01") & d <= as.Date("2022-12-31"))) {
    returns_syn[i] <- rnorm(1, mean = 0.0000, sd = 0.016)    # elevated vol
  } else {
    returns_syn[i] <- rnorm(1, mean = 0.0003, sd = 0.010)    # normal
  }
}

VaR_syn        <- hist_VaR(returns_syn, window = 252, conf = 0.99)   # same OOS estimator
breach_syn_vec <- (-returns_syn > VaR_syn) & !is.na(VaR_syn)
valid_idx_syn  <- which(!is.na(VaR_syn))
recent_breaches_syn <- sum(breach_syn_vec[tail(valid_idx_syn, 250)])
tl_status_syn <- if (recent_breaches_syn <= 4) "GREEN" else if (recent_breaches_syn <= 9) "AMBER" else "RED"

cat(sprintf("Synthetic series: %d days | full-sample exceptions %.2f%% | last-250 zone: %s (%d)\n",
            n_syn, 100 * sum(breach_syn_vec[valid_idx_syn]) / length(valid_idx_syn),
            tl_status_syn, recent_breaches_syn))
```

    #> Synthetic series: 5082 days | full-sample exceptions 1.18% | last-250 zone: GREEN (3)

``` r
plot_dates_syn <- dates_syn[valid_idx_syn]
plot(plot_dates_syn, returns_syn[valid_idx_syn], type = "l", col = "grey50", lwd = 0.5,
     main = "Appendix A - Synthetic series (classifier sanity-check, NOT S&P)",
     xlab = "Date", ylab = "Synthetic return", xaxt = "n",
     ylim = range(c(returns_syn[valid_idx_syn], -VaR_syn[valid_idx_syn]), na.rm = TRUE))
axis.Date(1, at = seq(min(plot_dates_syn), max(plot_dates_syn), by = "2 years"), format = "%Y")
lines(plot_dates_syn, -VaR_syn[valid_idx_syn], col = "firebrick", lwd = 1.2)
points(dates_syn[breach_syn_vec], returns_syn[breach_syn_vec], col = "red", pch = 19, cex = 0.4)
abline(h = 0, col = "black", lwd = 0.4)
legend("bottomleft", legend = c("synthetic return", "-VaR 99%", "exception"),
       col = c("grey50", "firebrick", "red"), pch = c(NA, NA, 19),
       lwd = c(0.5, 1.2, NA), pt.cex = c(NA, NA, 0.4), bty = "n", cex = 0.75)
```

<img src="SP500_analysis_files/figure-gfm/appendix-synthetic-1.png" style="display: block; margin: auto;" />

## 9 Summary

On the realised index the 99% VaR averages 3.22% (1-day) and peaks at
8.86% during the 2007-12-03 – 2008-12-01 stress window; the 97.5% ES
averages 3.02%. The simplified IMA-style charge is bound by the stressed
period (7.79% for 1-day, 20.47% for 10-day). Backtesting the model on
the real returns gives 97/7604 exceptions (1.28%): unconditional
coverage is rejected (Kupiec *p* = 0.021) and so is independence
(Christoffersen *p* = 0.000) — i.e. breaches cluster, the expected
behaviour of equal-weighted historical simulation. Over the most recent
250 days the model sits in the **GREEN** zone (2 exceptions).

**Known limitations.**

- Equal-weighted historical-simulation VaR reacts slowly to volatility
  changes (ghosting; no volatility scaling) and does not model
  volatility clustering, so it under-reacts to regime shifts — visible
  here in both the above-1% exception rate and the rejected independence
  test (clustered breaches).
- The 10-day figures use **overlapping** returns and are therefore
  serially correlated; their exception counts are descriptive only, not
  comparable to 1% (so the formal coverage tests run on the 1-day
  series).
- The IMA-style charge is a **simplification** (no regulatory
  multiplier, liquidity horizons, NMRF/SES add-on, DRC, or cross-class
  aggregation).
- ES is reported via exceedance counts — a diagnostic, **not** a formal
  ES backtest (ES is not elicitable; see Acerbi–Székely 2014).

## Reproducibility

``` r
sessionInfo()
```

    #> R version 4.3.3 (2024-02-29)
    #> Platform: x86_64-pc-linux-gnu (64-bit)
    #> Running under: Ubuntu 24.04.4 LTS
    #> 
    #> Matrix products: default
    #> BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.0 
    #> LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.12.0
    #> 
    #> locale:
    #> [1] C
    #> 
    #> time zone: Etc/UTC
    #> tzcode source: system (glibc)
    #> 
    #> attached base packages:
    #> [1] stats     graphics  grDevices utils     datasets  methods   base     
    #> 
    #> loaded via a namespace (and not attached):
    #>  [1] compiler_4.3.3  fastmap_1.1.1   cli_3.6.2       tools_4.3.3    
    #>  [5] htmltools_0.5.7 yaml_2.3.8      rmarkdown_2.25  highr_0.10     
    #>  [9] knitr_1.45      xfun_0.41       digest_0.6.34   rlang_1.1.3    
    #> [13] evaluate_0.23
