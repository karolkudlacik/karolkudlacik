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
