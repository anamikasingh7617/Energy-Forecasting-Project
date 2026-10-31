

import pandas as pd

file_path = "data/April 2023.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Sheet1",
    header=1
)

# Rename date column
df.rename(columns={"Unnamed: 0": "Time"}, inplace=True)

# Check time interval
time_diff = df["Time"].diff().value_counts()

print("DATE RANGE:")
print(df["Time"].min(), "to", df["Time"].max())

print("\nTIME INTERVALS:")
print(time_diff.head())

print("\nDUPLICATE TIMESTAMPS:")
print(df["Time"].duplicated().sum())

print("\nNEGATIVE VALUES:")
print((df.select_dtypes(include="number") < 0).sum())

print("\nIMPORTANT COLUMN STATISTICS:")
print(
    df[[
        "ALL_INDIA_WIND|P",
        "ALL_IND_SOLAR|P",
        "NLDC_DEMAND|P"
    ]].describe()
)


print("\nZERO VALUES:")
print((df.select_dtypes(include="number") == 0).sum())

print("\nNEGATIVE SOLAR READINGS:")
print(
    df.loc[
        df["ALL_IND_SOLAR|P"] < 0,
        ["Time", "ALL_IND_SOLAR|P", "Solar", "NLDC_DEMAND|P"]
    ].head(10)
)

print("\nZERO DEMAND READINGS:")
print(
    df.loc[
        df["NLDC_DEMAND|P"] == 0,
        ["Time", "NLDC_DEMAND|P"]
    ].head(10)
)


print("\nSOLAR COLUMN COMPARISON:")
print("Negative raw solar:", (df["ALL_IND_SOLAR|P"] < 0).sum())
print("Zero derived solar:", (df["Solar"] == 0).sum())

print("\nCHECK SOLAR RELATIONSHIP:")
print(
    df.loc[
        df["ALL_IND_SOLAR|P"] < 0,
        ["Time", "ALL_IND_SOLAR|P", "Solar"]
    ].head(15).to_string(index=False)
)

print("\nDEMAND ZERO - NEARBY READINGS:")
zero_index = df.index[df["NLDC_DEMAND|P"] == 0]

if len(zero_index) > 0:
    i = zero_index[0]
    print(
        df.loc[
            max(0, i-3):i+3,
            ["Time", "NLDC_DEMAND|P", "ALL_INDIA_WIND|P",
             "ALL_IND_SOLAR|P", "Solar"]
        ].to_string(index=False)
    )

print("\nSOLAR COLUMN COMPARISON:")
print("Negative raw solar:", (df["ALL_IND_SOLAR|P"] < 0).sum())
print("Zero derived solar:", (df["Solar"] == 0).sum())


print("\nSOLAR COMPARISON CHECK:")

print("Raw solar negative:",
      (df["ALL_IND_SOLAR|P"] < 0).sum())

print("Raw solar zero:",
      (df["ALL_IND_SOLAR|P"] == 0).sum())

print("Solar zero:",
      (df["Solar"] == 0).sum())

print("Solar and raw solar differences:")
print(
    (df["Solar"] - df["ALL_IND_SOLAR|P"])
    .describe()
)

print("\nDEMAND ZERO TOTAL:")
print((df["NLDC_DEMAND|P"] == 0).sum())
raw = df["ALL_IND_SOLAR|P"]
solar = df["Solar"]

print("\nFINAL SOLAR CHECK:")
print("Positive raw solar values:", (raw > 0).sum())
print(
    "Positive values with difference:",
    ((raw > 0) & ((solar - raw).abs() > 0.001)).sum()
)

print("\nZERO DEMAND CHECK:")
zero_rows = df[df["NLDC_DEMAND|P"] == 0]

print(zero_rows[[
    "Time",
    "TOTAL_THM_ONLY|P",
    "ALL_INDIA_WIND|P",
    "Solar",
    "NLDC_DEMAND|P"
]].to_string(index=False))
