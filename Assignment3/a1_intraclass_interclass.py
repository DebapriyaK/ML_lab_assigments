import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("data1.csv")

# Step 1: Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_df = df[numeric_cols].copy()

# Step 2: Add back the label column
numeric_df['label'] = df['label']

# Step 3: Encode labels to integers
label_encoder = LabelEncoder()
numeric_df['label'] = label_encoder.fit_transform(numeric_df['label'])

# Get readable class names
class_names = label_encoder.classes_

# Step 4: Print class distribution
print("Class distribution:\n", numeric_df['label'].value_counts())
print("Corresponding class names:", list(class_names), "\n")

# Step 5: Group by label and compute centroids and spreads (handle NaNs)
grouped = numeric_df.groupby('label')

centroids = {}
std_devs = {}

for label, group in grouped:
    data = group.drop(columns='label')

    # Handle missing values: fill NaNs with 0
    data_clean = data.fillna(0).values

    # Compute centroid and spread
    centroids[label] = np.mean(data_clean, axis=0)
    std_devs[label] = np.std(data_clean, axis=0)

    print(f"Class '{class_names[label]}' centroid:\n{centroids[label]}")
    print(f"Class '{class_names[label]}' standard deviation (spread):\n{std_devs[label]}\n")

# Step 6: Compute Euclidean distance between two classes
if len(centroids) >= 2:
    labels = list(centroids.keys())
    centroid1 = np.nan_to_num(centroids[labels[0]])
    centroid2 = np.nan_to_num(centroids[labels[1]])

    distance = np.linalg.norm(centroid1 - centroid2)
    print(f"Euclidean distance between class '{class_names[labels[0]]}' and '{class_names[labels[1]]}': {distance}")
else:
    print("Only one class found. Interclass distance can't be computed.")
