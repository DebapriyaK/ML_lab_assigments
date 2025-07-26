import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data1.csv")

# Encode class labels if needed
if df['label'].dtype == 'object':
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Select only top 2 classes for binary classification
top_two = df['label'].value_counts().index[:2].tolist()
filtered_df = df[df['label'].isin(top_two)]

# Keep only numeric columns
numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])

# Separate features and label
X = numeric_df.drop('label', axis=1)
y = numeric_df['label']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# --- A8: kNN for k = 1 to 11 ---
k_values = range(1, 12)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # Show detailed classification report for k = 1 and k = 3
    if k == 1 or k == 3:
        print(f"\nClassification Report for k = {k}:\n")
        print(classification_report(y_test, y_pred))

# Plot Accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.xticks(k_values)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('kNN Classifier Accuracy vs. k')
plt.grid(True)
plt.tight_layout()
plt.show()
