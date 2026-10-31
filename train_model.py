


import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/hourly_energy.csv")
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").copy()

# Time features
df["Hour"] = df["Time"].dt.hour
df["Day"] = df["Time"].dt.day
df["DayOfWeek"] = df["Time"].dt.day_of_week

# Previous-hour and previous-24-hour features
for col in ["Demand", "Wind", "Solar"]:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag24"] = df[col].shift(24)

# Remove rows without historical data
df = df.dropna().reset_index(drop=True)

features = [
    "Hour", "Day", "DayOfWeek",
    "Demand_lag1", "Wind_lag1", "Solar_lag1",
    "Demand_lag24", "Wind_lag24", "Solar_lag24"
]

X = df[features]
y = df["Demand"]

# Chronological split
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    )
}

results = []
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# Select best model
best_model_name = min(results, key=lambda x: x["MAE"])["Model"]
best_model = models[best_model_name]

# Save files
joblib.dump(best_model, "models/demand_model.pkl")
joblib.dump(features, "models/feature_columns.pkl")

pd.DataFrame(results).to_csv(
    "data/model_results.csv", index=False
)

print("\nBEST MODEL:", best_model_name)
print("MODEL SAVED SUCCESSFULLY!")
