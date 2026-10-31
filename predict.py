


import pandas as pd
import joblib

# Load model and features
model = joblib.load("models/demand_model.pkl")
features = joblib.load("models/feature_columns.pkl")

# Load hourly data
df = pd.read_csv("data/hourly_energy.csv")
df["Time"] = pd.to_datetime(df["Time"])

# Latest and 24-hour-old records
latest = df.iloc[-1]
previous_24 = df.iloc[-24]

# Next hour
next_time = latest["Time"] + pd.Timedelta(hours=1)

# Prepare features
input_data = pd.DataFrame([{
    "Hour": next_time.hour,
    "Day": next_time.day,
    "DayOfWeek": next_time.day_of_week,

    "Demand_lag1": latest["Demand"],
    "Wind_lag1": latest["Wind"],
    "Solar_lag1": latest["Solar"],

    "Demand_lag24": previous_24["Demand"],
    "Wind_lag24": previous_24["Wind"],
    "Solar_lag24": previous_24["Solar"]
}])

input_data = input_data[features]

# Predict
prediction = model.predict(input_data)[0]

print("\nENERGY DEMAND FORECAST")
print("----------------------")
print("Forecast time:", next_time)
print("Predicted demand:", round(prediction, 2))
print("Unit: Dataset's original demand unit")
