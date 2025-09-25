# 🚦 Time Series Traffic Prediction

## 📌 Overview
This project focuses on predicting traffic flow using machine learning models.  
We experimented with different **window sizes** for feature engineering, applied **cross-validation** to evaluate performance, and compared multiple models to find the optimal solution.

## 🎯 Objectives
- Explore and preprocess traffic time series data.
- Handle outliers using **interpolation** for `users < 500`.
- Build predictive features using a sliding `window_size`.
- Compare multiple models:
  - **RandomForestRegressor**
  - **GradientBoostingRegressor**
  - **XGBoost Regressor**
- Use **cross-validation** to evaluate metrics (MAE, RMSE, R²).
- Summarize results in Markdown with plots for easier interpretation.

## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost)
- **Cross-validation** with `cross_val_score`
- **Git & GitHub** for version control
- **Markdown** for results documentation

## 📂 Project Structure
