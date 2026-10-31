

import pandas as pd

# Step 1: Read original Excel file
df = pd.read_excel(
    "data/April 2023.xlsx",
    sheet_name="Sheet1",
    header=1
)

# Step 2: Rename time column
df = df.rename(columns={"Unnamed: 0": "Time"})

# Step 3: Select useful columns
df = df[[
    "Time",
    "NLDC_DEMAND|P",
    "ALL_INDIA_WIND|P",
    "Solar"
]].copy()

# Step 4: Rename columns
df = df.rename(columns={
    "NLDC_DEMAND|P": "Demand",
    "ALL_INDIA_WIND|P": "Wind",
    "Solar": "Solar"
})

# Step 5: Mark invalid zero-demand readings
df.loc[df["Demand"] == 0, ["Demand", "Wind", "Solar"]] = float("nan")

# Step 6: Replace negative solar with zero
df["Solar"] = df["Solar"].clip(lower=0)

# Step 7: Interpolate invalid readings
df[["Demand", "Wind", "Solar"]] = (
    df[["Demand", "Wind", "Solar"]].interpolate(method="linear")
)

# Step 8: Save cleaned dataset
df.to_csv("data/cleaned_energy.csv", index=False)

print("DATA CLEANING COMPLETED!")
print("Rows:", len(df))
print("Missing values:")
print(df.isnull().sum())
print("Saved: data/cleaned_energy.csv")
