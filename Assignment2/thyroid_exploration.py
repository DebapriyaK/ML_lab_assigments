import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the data from the Excel file, replace '?' with NaN for easier processing
def load_thyroid_data(filepath: str):
    df = pd.read_excel(filepath, sheet_name="thyroid0387_UCI")
    df.replace('?', np.nan, inplace=True)  # Convert '?' placeholders to proper NaN values
    return df

# Identify data type (nominal, ordinal, numeric) based on values and datatype
def detect_attribute_types(df):
    print("\n== Attribute Data Types ==")
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        dtype = df[col].dtype
        if dtype == object:
            if len(unique_values) < 10:
                print(f"{col}: Categorical (Nominal or Ordinal)")
            else:
                print(f"{col}: High-cardinality Nominal or Mixed")
        elif np.issubdtype(dtype, np.number):
            print(f"{col}: Numeric")
        else:
            print(f"{col}: Other - {dtype}")


# Convert specific known numeric columns from string (with '?') to float
def convert_numeric(df):
    numeric_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # '?' will become NaN
    return df

# Print min and max values for each numeric column
def numeric_ranges(df):
    print("\n== Data Ranges for Numeric Attributes ==")
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        print(f"{col}: Min = {numeric_df[col].min()}, Max = {numeric_df[col].max()}")

# Check for missing values in all columns
def missing_values(df):
    print("\n== Missing Values in Each Attribute ==")
    print(df.isnull().sum())

# Detect outliers using the Interquartile Range (IQR) method
def detect_outliers(df):
    print("\n== Outlier Detection (IQR Method) ==")
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
        print(f"{col}: {outliers} outliers")

# Calculate mean and standard deviation for each numeric column
def numeric_stats(df):
    print("\n== Mean and Standard Deviation for Numeric Attributes ==")
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        print(f"{col}: Mean = {numeric_df[col].mean():.2f}, Std Dev = {numeric_df[col].std():.2f}")

# Main function to orchestrate all tasks
def main():
    filepath = "Lab Session Data.xlsx"
    df = load_thyroid_data(filepath)

    # Step 1: Identify types (categorical, numeric)
    detect_attribute_types(df)

    # Step 2: Clean and convert numeric columns
    df = convert_numeric(df)

    # Step 4: Numeric ranges and missing value report
    numeric_ranges(df)
    missing_values(df)

    # Step 5: Outlier detection and basic statistics
    detect_outliers(df)
    numeric_stats(df)


# Run the script
if __name__ == "__main__":
    main()
