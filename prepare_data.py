

import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_energy.csv")

# Convert Time into datetime
df["Time"] = pd.to_datetime(df["Time"])

# Set Time as index
df = df.set_index("Time")

# Convert 10-second readings into hourly averages
hourly_df = df.resample("1h").mean()

# Save hourly dataset
hourly_df.to_csv("data/hourly_energy.csv")

print("HOURLY DATA CREATED!")
print("Shape:", hourly_df.shape)
print(hourly_df.head())
print("\nMissing values:")
print(hourly_df.isnull().sum())

