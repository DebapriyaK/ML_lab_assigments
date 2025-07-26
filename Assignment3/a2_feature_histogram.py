import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data1.csv")

# Step 1: Choose a feature
# You can change this to any numeric feature name from your dataset
feature_name = 'frame.len'  # Change this if needed

# Step 2: Check if the feature exists and is numeric
if feature_name not in df.columns:
    raise ValueError(f"'{feature_name}' not found in dataset columns.")

if not np.issubdtype(df[feature_name].dtype, np.number):
    raise ValueError(f"'{feature_name}' is not numeric. Please choose a numeric feature.")

# Step 3: Drop missing values from that column
feature_data = df[feature_name].dropna()

# Step 4: Calculate mean and variance
mean_val = np.mean(feature_data)
var_val = np.var(feature_data)

print(f"Feature selected: {feature_name}")
print(f"Mean: {mean_val}")
print(f"Variance: {var_val}")

# Step 5: Plot histogram using buckets
plt.figure(figsize=(10, 6))
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black')  # 20 buckets
plt.title(f"Histogram of '{feature_name}'")
plt.xlabel(feature_name)
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
