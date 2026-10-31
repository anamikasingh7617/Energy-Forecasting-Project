

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/hourly_energy.csv")

df["Time"] = pd.to_datetime(df["Time"])

# Demand graph
plt.figure(figsize=(12, 5))
plt.plot(df["Time"], df["Demand"])
plt.title("Hourly Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/demand_graph.png")
plt.show()

# Wind and Solar graph
plt.figure(figsize=(12, 5))
plt.plot(df["Time"], df["Wind"], label="Wind")
plt.plot(df["Time"], df["Solar"], label="Solar")
plt.title("Hourly Renewable Energy Production")
plt.xlabel("Date")
plt.ylabel("Production")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/renewable_graph.png")
plt.show()

print("EDA GRAPHS CREATED!")
