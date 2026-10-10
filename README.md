# Predicitive-Analytics-Workflow-9.9

# Supervised Learning Models: Time Series, Regression & Classification

A collection of supervised machine learning projects covering **time series forecasting**, **regression**, and **classification**, built with Python and applied to real-world data.

## 📖 Overview

This repository walks through the full supervised learning workflow: data cleaning, exploratory analysis, feature engineering, model training, evaluation, and deployment. Each project shows how to frame a business question as a predictive problem and measure how well the model answers it.

## 🧠 Models Covered

| Type | Goal | Example algorithms | Key metrics |
|------|------|--------------------|-------------|
| **Regression** | Predict a continuous value | Linear Regression, Ridge/Lasso, Random Forest, Gradient Boosting | MAE, RMSE, R² |
| **Classification** | Predict a category | Logistic Regression, Decision Tree, Random Forest, XGBoost | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Time Series** | Forecast future values over time | ARIMA/SARIMA, Prophet, lag-feature ML models | MAE, RMSE, MAPE |

## 🛠️ Tech Stack

- **Python**: pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn
- **SQL**: data extraction and preparation
- **Power BI / Excel**: reporting and visualization
- **Streamlit**: deploying models as interactive apps

## 📁 Repository Structure

```
├── data/                  # Raw and processed datasets
├── notebooks/
│   ├── 01_regression.ipynb
│   ├── 02_classification.ipynb
│   └── 03_time_series.ipynb
├── src/                   # Reusable functions and scripts
├── models/                # Saved trained models
├── app/                   # Streamlit app
├── requirements.txt
└── README.md
```

## 🔄 Workflow

1. **Data collection & cleaning**: handle missing values, duplicates, and outliers
2. **Exploratory data analysis**: understand distributions, trends, and relationships
3. **Feature engineering**: encoding, scaling, and for time series, lag and rolling features
4. **Model training**: train/test split (chronological split for time series) and cross-validation
5. **Evaluation**: compare models using the metrics above
6. **Deployment**: serve the best model through a Streamlit app



## 👤 Author

**Michael Clayman Budu-Ainooson**
Data Analyst | Excel · SQL · Python · Power BI · Predictive Analytics

- LinkedIn: www.linkedin.com/in/michael-budu-ainooson-b39b0a2a6
- Email: claymanainooson@gmail.com

