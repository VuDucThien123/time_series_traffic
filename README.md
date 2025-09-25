#  Time Series Traffic Prediction

## 📌 Overview
This project focuses on predicting traffic flow using machine learning models.  
We experimented with different **window sizes** for feature engineering and compared multiple models to find the optimal solution.

## 🎯 Objectives
- Explore and preprocess web-traffic time series data.
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
- **Git & GitHub** for version control
- **Markdown** for results documentation

## 📊 Methodology
1. **Data Preprocessing**
   - Interpolated values where `users < 500`.
   - Normalized / scaled features.
   - Sliding window for lag features.

2. **Model Training**
   - Metrics used:
     - **MAE (Mean Absolute Error)**
     - **MSE (Mean Squared Error)**
     - **RMSE (Root Mean Squared Error)**
     - **R² (Coefficient of Determination)**
     - **MAPE (Mean Absolute Percentage Error)**

3. **Model Comparison**
   - RandomForestRegressor (baseline)
   - GradientBoostingRegressor
   - XGBoost Regressor

## 📈 Results
- Metrics were saved into [`results/metrics_results.md`](results/model_metrics_comparison.md).
- Visualizations show performance differences across window sizes and models.

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/time_series_traffic.git
   cd time_series_traffic
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run model training:**
   ```bash
   python src/main.py
4. **Check results:**
   - Metrics in results/metrics_results.md
   - Visualizations in results/plots/
  
## 📌 Future Improvements
  - Try LSTM / Deep Learning models for sequence data.
  - Hyperparameter tuning with Optuna or GridSearchCV.
  - Deploy as a small dashboard app with Streamlit / Gradio.





