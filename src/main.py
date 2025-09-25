import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
data = pd.read_csv("D:/project AI_ML/time_series_traffic/data/web-traffic.csv")

data["date"] = pd.to_datetime(data["date"], format="%d/%m/%y")
data = data.set_index("date")

# Create range date
full_range = pd.date_range(start=data.index.min(),
                           end=data.index.max(),
                           freq="D")

# Reindex data for fill missing value
data = data.reindex(full_range)
data.index.name = "date"
data = data.reset_index()

# function extract data
def extract_data(data, window_size = 5):
    i = 1
    while i < window_size:
        data["users_{}".format(i)] = data["users"].shift(-i)
        i+= 1
    data['target'] = data['users'].shift(-i)
    data = data.dropna(axis = 0)
    return data

def interpolate_data(data):
    mask = data["users"] < 500
    data.loc[mask, "users"] = None
    data["users"] = data["users"].interpolate(method="linear")
    return data
data = interpolate_data(data)
data = extract_data(data, 7)

x = data.drop(["date", 'target'], axis = 1)
y = data['target']

train_size = 0.7
num_samples = len(x)
x_train = x[:int(train_size * num_samples)]
y_train = y[:int(train_size * num_samples)]
x_test = x[int(train_size * num_samples):]
y_test = y[int(train_size * num_samples):]


# scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Các mô hình để so sánh
models = {
    "RF": RandomForestRegressor(),
    "GBR": GradientBoostingRegressor(),
    "XGB": XGBRegressor()
}

# Bảng kết quả
results = []

for name, model in models.items():
    # Train
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    # Tính metrics
    metrics = {
        "Model": name,
        "R2": r2_score(y_test, y_predict),
        "MAE": mean_absolute_error(y_test, y_predict),
        "MSE": mean_squared_error(y_test, y_predict),
        "RMSE": root_mean_squared_error(y_test, y_predict),
        "MAPE": mean_absolute_percentage_error(y_test, y_predict),
    }
    results.append(metrics)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["date"][:int(train_size * num_samples)],
            data["users"][:int(train_size * num_samples)], label="train")
    ax.plot(data["date"][int(train_size * num_samples):],
            data["users"][int(train_size * num_samples):], label="test")
    ax.plot(data["date"][int(train_size * num_samples):], y_predict, label="predict")
    ax.set_title(f"Web Traffic Prediction using {name}")
    ax.set_xlabel("date")
    ax.set_ylabel("users")
    ax.legend()
    plt.savefig(f"web-traffic-{name}.png")
    plt.close()

results_df = pd.DataFrame(results)
# Lưu kết quả thành file Markdown
with open("model_metrics_comparison.md", "w") as f:
    f.write(results_df.to_markdown(index=False))

