# Karol Kudlacik

**Mathematician**

Focused on applying machine learning and time series forecasting to quantitative finance and financial technology problems.

---

## 📊 Featured Projects

### 📈 Credit Card Approval Prediction
- **Problem:** Financial institutions face challenges in automating credit card approval decisions while balancing two competing goals: minimizing risk by rejecting unreliable applicants, and maximizing profit by not turning away creditworthy ones. The dataset was highly imbalanced (~99.5% approvals), making standard classification approaches insufficient.
- **Approach:** Built and compared five classification models — Logistic Regression, SVM, Decision Tree, Random Forest, and a Neural Network. The preprocessing pipeline included one-hot encoding of categorical features, StandardScaler normalization (applied separately to train/test to prevent data leakage), and SMOTETomek resampling to address class imbalance. Hyperparameter tuning was performed via RandomizedSearchCV for traditional models and manual grid search for the neural network. Models were evaluated using Precision, F1-score, and AUC-ROC.
- **Result:** Logistic Regression and SVM achieved near-perfect classification with AUC-ROC scores close to 1.0. Decision Tree and Random Forest performed well but were sensitive to hyperparameter choices. The Neural Network, despite architectural improvements (BatchNormalization, Dropout, L2 regularization), was outperformed by the simpler traditional models — confirming that for structured tabular data, classical ML approaches can be more effective.
- **Tech:** Python, TensorFlow/Keras, Scikit-learn, Imbalanced-learn (SMOTETomek), Pandas, NumPy, Matplotlib, Seaborn
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/credit_card_project.ipynb)
- **Data:**[kaggle](https://www.kaggle.com/datasets/caesarmario/application-data)
### 📈 Time Series – Monthly Sales Forecasting

- **Problem:** Given monthly product sales data spanning from 2015, the goal was to analyze the underlying temporal structure of the series and produce reliable sales forecasts for all 9 months of the first three quarters of 2026.
- **Approach:** The raw series was tested for stationarity using the Augmented Dickey-Fuller test (p = 0.84 → non-stationary), then first-differenced to achieve stationarity (p = 0.01). ACF/PACF plots of the differenced series suggested an AR process, guiding model selection. A periodogram and smoothed spectral density estimator (Daniell kernel) were computed to identify dominant cycles — a strong peak was found around a ~16-month period. Four ARIMA candidates were fitted and compared — ARIMA(1,1,0), ARIMA(1,2,0), ARIMA(2,1,0), and ARIMA(2,2,0) — with model selection based on AIC/BIC and residual diagnostics. ARIMA(2,1,0) was chosen as the best model, avoiding over-differencing while achieving the lowest information criteria.
- **Result:** The ARIMA(2,1,0) model produced a 9-month forecast for Jan–Sep 2026, revealing a continuing gradual decline in sales (from ~1582 in January to ~1432 in September), with widening prediction intervals reflecting growing uncertainty further into the future.
- **Tech:** R, astsa, tseries, forecast, bestNormalize
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/Time_Series_Project_task_1.html)
- **Data:** [task1_data](https://github.com/karolkudlacik/karolkudlacik/blob/main/SWK-14.csv)
  
### 📉 S&P 500 Market Risk Analysis – VaR, ES & Basel Backtesting

- **Problem:** Problem: Quantify the downside market risk of the S&P 500 index using industry-standard financial risk measures, and validate the model's reliability through regulatory backtesting — addressing the core challenge faced by financial institutions when sizing capital reserves.
- **Approach:** Using daily S&P 500 closing prices from 1995 onwards, 1-day and 10-day simple returns were computed. A rolling 252-day Historical VaR (99%) was implemented with linear interpolation between tail observations. Expected Shortfall (ES) at 97.5% was then added as a deeper tail measure, averaging the worst 2.5% of outcomes (6.3 observations, with fractional weighting). Following the Basel III / FRTB Internal Models Approach, a stress period was identified by scanning all possible 252-day windows to find the episode with the highest ES — anchoring the IMA capital charge as max(Stress ES, Recent ES). Finally, a regime-based simulated return series (GFC, COVID crash, elevated vol periods) was used for Basel Traffic Light backtesting, counting VaR breaches in the most recent 250-day window and assigning a GREEN/AMBER/RED classification with the corresponding capital multiplier.
- **Result:** The analysis produced rolling VaR and ES curves, breach rates, and a full IMA charge comparison table across 1-day and 10-day horizons. The stress period (automatically identified as the worst historical window) served as the binding capital constraint. The backtesting framework output a traffic light status and Basel k-multiplier addition, mirroring real regulatory reporting logic.
- **Tech:** R, zoo, dplyr
- **Repository:** [View Project](https://github.com/karolkudlacik/karolkudlacik/blob/main/SP500_analysis.html)
- **Data:**[SP500_Data](https://stooq.pl/q/d/l/?s=^spx&f=19900501&t=20260327&i=d)
---

## 🛠️ Technical Stack

**Languages:** Python, R

**ML & Data Science:** TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

**Finance & Quantitative:** Time Series Analysis, Risk Modeling, Statistical Analysis

---

## 📬 Connect

📧 Email: karol.kudlacik@outlook.com
🔗 LinkedIn: 
🐙 GitHub: https://github.com/karolkudlacik/karolkudlacik
