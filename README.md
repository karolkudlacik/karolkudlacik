# Karol Kudłacik

Applied Mathematics Student · Aspiring Quantitative / Risk Analyst

*Applying quantitative methods and machine learning to market risk, model validation and financial forecasting.*

---

## 📊 Featured Projects

### 📉 S&P 500 Market Risk Analysis – VaR, ES & Basel Backtesting

- **Problem:** Quantify the downside market risk of the S&P 500 index using industry-standard risk measures, and validate the model's reliability through regulatory backtesting — the core challenge financial institutions face when sizing capital reserves.
- **Approach:** Using daily S&P 500 closing prices from 1995 onwards, 1-day and 10-day simple returns were computed. A rolling 252-day Historical VaR (99%) was implemented with linear interpolation between tail observations. Expected Shortfall (ES) at 97.5% was then added as a deeper tail measure, averaging the worst 2.5% of outcomes (6.3 observations, with fractional weighting). Following the Basel III / FRTB Internal Models Approach, a stress period was identified by scanning all possible 252-day windows to find the episode with the highest ES — anchoring the IMA capital charge as max(Stress ES, Recent ES). Finally, a regime-based simulated return series (GFC, COVID crash, elevated-vol periods) was used for Basel Traffic Light backtesting, counting VaR breaches in the most recent 250-day window and assigning a GREEN/AMBER/RED classification with the corresponding capital multiplier.
- **Result:** The analysis produced rolling VaR and ES curves, breach rates, and a full IMA charge comparison table across 1-day and 10-day horizons. The automatically identified stress period served as the binding capital constraint, and the backtesting framework output a traffic-light status and Basel k-multiplier add-on, mirroring real regulatory reporting logic.
- **Limitations:** Historical-simulation VaR is sensitive to the look-back window (ghosting effect) and applies no volatility scaling; 10-day overlapping returns introduce autocorrelation; and the traffic-light test here runs on simulated stress regimes rather than realized P&L.
- **Tech:** R, zoo, dplyr
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/SP500_analysis.html)
- **Data:** [S&P 500 daily prices — Stooq](https://stooq.pl/q/d/l/?s=^spx&f=19900501&t=20260327&i=d)

### 🎲 Options Wheel vs. Buy & Hold – Monte Carlo Strategy Comparison

- **Problem:** Compare a systematic options-selling strategy (the "Wheel") against passive Buy & Hold on an ETF, and quantify how each behaves on return, risk and stability across different market regimes — i.e. whether harvesting option premium delivers a better risk-adjusted payoff than simply holding. Presented at the 14th Cracow Conference of Financial Mathematics (2026).
- **Approach:** Built a Monte Carlo engine simulating 10,000 GBM price paths in each of three regimes (bull μ=15% / σ=12%, bear μ=−20% / σ=35%, sideways μ=3% / σ=18%), $100k capital, r=4%. The Wheel agent systematically sells cash-secured puts then covered calls at the 0.30-delta strike (21-day tenor), priced with Black–Scholes; implied volatility is set to realized + 2% to capture the volatility risk premium (VRP), and covered-call strikes are floored at the assignment price to protect the cost basis. Performance measured with CAGR, annualized Sharpe, average maximum drawdown, CVaR (95%) and the path-by-path probability P(Wheel > B&H).
- **Result:** Buy & Hold won nominally only in the strong bull market (CAGR 15.8% vs 11.0%) — the Wheel caps upside — but with a worse risk profile (Sharpe 0.74 vs 1.19, deeper drawdowns). The Wheel outperformed in the bear (P = 87%) and sideways (P = 71%) regimes and showed consistently lighter left tails (better CVaR) across all three. In short, it acts as a volatility-management overlay: a more defensive, predictable payoff in exchange for capped upside, with the VRP as the structural source of edge.
- **Limitations:** GBM has no fat tails, so crash / black-swan risk is understated; transaction costs are omitted, which favours the higher-turnover Wheel; perfect liquidity at Black–Scholes + VRP is assumed; and the edge is sensitive to the flat +2% VRP, which in reality varies and can turn negative in stress.
- **Tech:** Python: numpy, scipy.stats
- **Repository:** [View Presentation (PDF)](https://github.com/karolkudlacik/karolkudlacik/blob/main/Options_strategy_wheel_vs_Buy_paper_from_kkmf.pdf) (PL)
                  [View code (python notebook)]) (PL)
- **Data:** Synthetic (Monte Carlo / GBM simulation)

### 🏦 Corporate Credit Risk – Probability of Default (PD)

- **Problem:** Estimate the 12-month probability of default for a Middle-Market Wholesale corporate book and decide which modelling approach a risk function should actually deploy. The predictors here are not raw financials but six qualitative credit-expert opinion scores (market position, management quality, access to credit, profitability, short- and medium-term liquidity), recorded across several years per customer — a panel structure that makes naive row-level train/test splitting unsafe.
- **Approach:** From ~5,800 firm-year assessments (2000–2008), empty export-artefact rows were dropped and two customer-history features engineered (cumulative assessments and prior defaults). To prevent leakage in the panel, the train/test split was performed by customer (GroupShuffleSplit) rather than by row, so no client appears on both sides. Each predictor was supervised-binned to its Weight of Evidence on the training set only, screened by Information Value (keeping IV > 0.1) and pruned for multicollinearity above 0.75 (replicating caret::findCorrelation). Three model families were fitted on the WoE features — logistic regression, probit, and a linear probability model — and benchmarked against the credit experts' fixed scorecard, PD = 1 / (1 + exp(-0.1 × Score)), with the weights specified in the brief. Discrimination was assessed via AUC/Gini and the Kolmogorov–Smirnov statistic with a paired bootstrap confidence interval, parsimony via AIC/BIC, and a segmentation analysis by financial-holding type compared one overall model against two bespoke models.
- **Result:** Logistic and probit were statistically indistinguishable — the 95% bootstrap interval for their Gini difference contained zero — so the logit was retained for its log-odds interpretability and regulatory familiarity. The linear probability model was rejected because a large share of its fitted probabilities fell outside [0, 1], and the experts' scorecard proved both badly miscalibrated (mean PD ≈ 0.99 against an observed default rate ≈ 0.11) and inverted in its ranking. The notebook outputs a full model-comparison table with ROC and bootstrap-CI Gini plots, mirroring how a model-validation report is structured. The very high headline discrimination (Gini ≈ 0.94) reflects the expert-assessment nature of this learning dataset rather than production-grade performance.
- **Limitations:** Because the six predictors are expert risk opinions already strongly aligned with the eventual default, they behave as near-leaking overlays — the probit's quasi-separation warning is a direct symptom — so discrimination is inflated relative to a real corporate-PD model (typically Gini ≈ 0.4–0.7). Validation is also in-sample in time: there is no out-of-time split or Population Stability Index (PSI) check across years, and the panel is thin (most customers appear only once or twice). A production build would rebuild on raw financial ratios and add through-the-cycle stability testing.
- **Tech:** Python, scorecardpy, statsmodels, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/credit_risk_pd.ipynb)
- **Data:** HSBC Quants Academy – Project 2 (Corporate Credit Risk); corporate credit assessments, name: credit-risk_data.xls

### 📈 Credit Card Approval Prediction

- **Problem:** Financial institutions face challenges in automating credit card approval decisions while balancing two competing goals: minimizing risk by rejecting unreliable applicants, and maximizing profit by not turning away creditworthy ones. The dataset was highly imbalanced (~99.5% approvals), making standard classification approaches insufficient.
- **Approach:** Built and compared five classification models — Logistic Regression, SVM, Decision Tree, Random Forest, and a Neural Network. The preprocessing pipeline included one-hot encoding of categorical features, StandardScaler normalization (applied separately to train/test to prevent data leakage), and SMOTETomek resampling to address class imbalance. Hyperparameter tuning was performed via RandomizedSearchCV for the traditional models and manual grid search for the neural network. Models were evaluated using Precision, F1-score, and AUC-ROC.
- **Result:** Logistic Regression and SVM produced the highest AUC-ROC scores (close to 1.0), while Decision Tree and Random Forest performed well but were sensitive to hyperparameter choices. The Neural Network — despite architectural improvements (BatchNormalization, Dropout, L2 regularization) — was outperformed by the simpler models, confirming that classical ML can be more effective on structured tabular data. These near-ceiling scores largely reflect the relative simplicity of this learning dataset rather than production-grade performance.
- **Tech:** Python, TensorFlow/Keras, Scikit-learn, Imbalanced-learn (SMOTETomek), Pandas, NumPy, Matplotlib, Seaborn
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/credit_card_project.ipynb)
- **Data:** [Kaggle – Application Data](https://www.kaggle.com/datasets/caesarmario/application-data)

### 📈 Time Series – Monthly Sales Forecasting

- **Problem:** Given monthly product sales data from 2015 onwards, the goal was to analyze the temporal structure of the series and produce reliable forecasts for the first three quarters of 2026 (Jan–Sep).
- **Approach:** The raw series was tested for stationarity using the Augmented Dickey-Fuller test (p = 0.84 → non-stationary), then first-differenced to achieve stationarity (p = 0.01). ACF/PACF plots of the differenced series suggested an AR process, guiding model selection. A periodogram and a smoothed spectral density estimator (Daniell kernel) identified a dominant cycle of ~16 months. Four ARIMA candidates were fitted and compared — ARIMA(1,1,0), ARIMA(1,2,0), ARIMA(2,1,0), and ARIMA(2,2,0) — with selection based on AIC/BIC and residual diagnostics. ARIMA(2,1,0) was chosen, avoiding over-differencing while achieving the lowest information criteria.
- **Result:** The ARIMA(2,1,0) model produced a 9-month forecast for Jan–Sep 2026, revealing a continuing gradual decline in sales (from ~1582 in January to ~1432 in September), with widening prediction intervals reflecting growing uncertainty further into the future.
- **Note:** The same toolkit (stationarity testing, ARIMA, residual diagnostics) transfers directly to modelling financial returns and volatility.
- **Tech:** R, astsa, tseries, forecast, bestNormalize
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/Time_Series_Project_task_1.html)
- **Data:** [Monthly sales data (CSV)](https://github.com/karolkudlacik/karolkudlacik/blob/main/SWK-14.csv)

---

## 🛠️ Technical Stack

**Languages:** Python, R

**ML & Data Science:** Scikit-learn, Imbalanced-learn, Pandas, NumPy, Matplotlib, Seaborn, scorecardpy

**R packages:** astsa, tseries, forecast, zoo, dplyr, bestNormalize

**Finance & Quantitative:** Market risk (VaR, Expected Shortfall), Basel III / FRTB backtesting, PD, credit risk modelling, time-series analysis, statistical analysis

---

## 📬 Connect

📧 **Email:** [karol.kudlacik@outlook.com](mailto:karol.kudlacik@outlook.com)  
🔗 **LinkedIn:** [linkedin.com/in/karol-kudlacik](https://www.linkedin.com/in/karol-kudlacik)  
🐙 **GitHub:** [github.com/karolkudlacik](https://github.com/karolkudlacik)
