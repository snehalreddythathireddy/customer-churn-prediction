# 📉 Telco Customer Churn Prediction
A complete end-to-end machine learning project to predict customer churn for a telecom company using **Exploratory Data Analysis**, **XGBoost**, and **SHAP explainability**.
---
## 📁 Project Structurecustomer-churn-prediction/
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   └── 02_model.ipynb      # XGBoost Model + SHAP
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── .gitignore
└── README.md---

## 📊 Dataset
- **Source:** IBM Sample Dataset (Telco Customer Churn)
- **Rows:** 7,043 customers
- **Features:** 21 (demographics, services, billing)
- **Target:** Churn (Yes/No)

---

## 🔍 Key EDA Findings
- **26.6%** of customers churned — imbalanced dataset
- Customers with **month-to-month contracts** churn at ~42%
- **Fiber optic** internet users churn the most (~42%)
- **Senior citizens** churn at nearly double the average rate
- **Electronic check** payment method has highest churn rate (~45%)
- Customers with **low tenure** churn heavily in the first 10 months

---

## 🤖 Model Performance
| Metric | Score |
|---|---|
| ROC-AUC | **0.8344** |
| Churn Recall | **79%** |
| Churn Precision | 50% |
| Accuracy | 73% |

> Model is tuned for high recall — catching churners is more valuable than avoiding false alarms.

---

## 🔎 SHAP Insights (Top Churn Drivers)
1. **Contract type** — Two-year contracts strongly prevent churn
2. **Tenure** — Long-term customers rarely churn
3. **Internet Service** — Fiber optic increases churn risk
4. **Monthly Charges** — Higher bills = more churn
5. **Payment Method** — Electronic check is a high-risk signal

---

## 🚀 How to Run
```bash
git clone https://github.com/snehalreddythathireddy/customer-churn-prediction.git
cd customer-churn-prediction
pip install pandas numpy matplotlib seaborn xgboost shap scikit-learn
jupyter notebook
```
Then run `01_eda.ipynb` first, followed by `02_model.ipynb`.

---

## 🛠️ Tech Stack
- Python 3.13
- Pandas 2.3.3 | XGBoost 3.2.0
- Scikit-learn | SHAP | Seaborn | Matplotlib should i paste all this in readme
