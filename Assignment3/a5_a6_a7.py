import pandas as pd
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

# Separate X and y
X = numeric_df.drop('label', axis=1)
y = numeric_df['label']

# Handle NaNs by filling them with column-wise mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)

print("\n Model Trained Successfully!")
print(" Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#---------a7-----------
test_vect = X_test[0].reshape(1, -1)  # Because X_test is a NumPy array
predicted_class = knn.predict(test_vect)
actual_class = y_test.iloc[0]  # Still a pandas Series

print("\n--- A7: Test Vector Prediction ---")
print("Test Vector:", test_vect)
print("Predicted Class:", predicted_class[0])
print("Actual Class:", actual_class)
