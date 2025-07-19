import pandas as pd
import numpy as np

# Load and clean the dataset
def load_thyroid_data(filepath: str):
    df = pd.read_excel(filepath, sheet_name="thyroid0387_UCI")
    df.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
    return df

# Convert 't'/'f' to 1/0 for binary attributes
def convert_binary_columns(df):
    binary_like_cols = [col for col in df.columns if df[col].dropna().isin(['t', 'f']).all()]
    for col in binary_like_cols:
        df[col] = df[col].map({'t': 1, 'f': 0})
    return df

# ompute similarity measures (JC and SMC)
def similarity_measure(df):
    print("\n== A5: Similarity Measure (JC and SMC) ==")

    #Select only binary (0/1) columns
    binary_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]
    
    if len(binary_cols) == 0:
        print("No binary columns found.")
        return

    binary_df = df[binary_cols]

    #Get the first two rows
    v1 = binary_df.iloc[0].astype(int)
    v2 = binary_df.iloc[1].astype(int)

    #Compute f11, f10, f01, f00
    f11 = ((v1 == 1) & (v2 == 1)).sum()
    f10 = ((v1 == 1) & (v2 == 0)).sum()
    f01 = ((v1 == 0) & (v2 == 1)).sum()
    f00 = ((v1 == 0) & (v2 == 0)).sum()

    #JC and SMC
    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) else 0
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)

    #Display
    print(f"Binary Attributes Used: {len(binary_cols)}")
    print(f"f11 (1,1): {f11}, f10 (1,0): {f10}, f01 (0,1): {f01}, f00 (0,0): {f00}")
    print(f"Jaccard Coefficient (JC): {jc:.4f}")
    print(f"Simple Matching Coefficient (SMC): {smc:.4f}")

# Main function
def main():
    filepath = "Lab Session Data.xlsx"
    df = load_thyroid_data(filepath)
    df = convert_binary_columns(df)
    similarity_measure(df)

# Run
if __name__ == "__main__":
    main()
