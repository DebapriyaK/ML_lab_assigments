import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data1.csv")

# Encode label column if it's not already numeric
if df['label'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Count classes to pick top 2
class_counts = df['label'].value_counts()
top_two_classes = class_counts.index[:2].tolist()

print(f"\nSelected class labels: {top_two_classes}")

# Filter dataset to only those two classes
filtered_df = df[df['label'].isin(top_two_classes)]

# Drop non-numeric columns if any (e.g., timestamps, IPs)
numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])

# Separate X (features) and y (labels)
X = numeric_df.drop('label', axis=1)
y = numeric_df['label']

# Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print shapes and confirmation
print("\nTrain/Test Split Done!")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
