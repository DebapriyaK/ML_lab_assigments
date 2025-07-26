import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Load dataset
df = pd.read_csv("data1.csv")

# Encode class labels 
if df['label'].dtype == 'object':
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Filter top 2 classes for binary classification
top_two = df['label'].value_counts().index[:2].tolist()
filtered_df = df[df['label'].isin(top_two)]

# Select numeric features only
numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])

# Separate features and target
X = numeric_df.drop('label', axis=1)
y = numeric_df['label']

# Handle NaNs with column mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Train model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on train and test data
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion Matrix & Reports

print("\n--- TRAINING SET EVALUATION ---")
print("Confusion Matrix (Train):\n", confusion_matrix(y_train, y_train_pred))
print("Classification Report (Train):\n", classification_report(y_train, y_train_pred))
print("Accuracy (Train):", accuracy_score(y_train, y_train_pred))

print("\n--- TEST SET EVALUATION ---")
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))
print("Accuracy (Test):", accuracy_score(y_test, y_test_pred))
