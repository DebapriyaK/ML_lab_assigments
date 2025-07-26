import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

df = pd.read_csv("data1.csv")

numeric_df = df.select_dtypes(include=[np.number])

# Drop rows with too many NaNs but keep some
numeric_df = numeric_df.dropna(thresh=2)  # Keep rows with at least 2 non-NaN values

# Check if we have at least 2 rows
if len(numeric_df) < 2:
    raise ValueError("Not enough numeric rows to compare. Try reducing NaN dropping.")

# Take any two vectors
vec1 = numeric_df.iloc[0].fillna(0).values  # Fill NaN with 0s if needed
vec2 = numeric_df.iloc[1].fillna(0).values

# Calculate Minkowski distances
r_values = range(1, 11)
distances = [distance.minkowski(vec1, vec2, p=r) for r in r_values]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(r_values, distances, marker='o', linestyle='-', color='darkgreen')
plt.title("Minkowski Distance (r=1 to 10) Between Two Feature Vectors")
plt.xlabel("r value")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
print("Available numeric rows:", len(numeric_df))
print("Available numeric columns:", numeric_df.columns.tolist())
