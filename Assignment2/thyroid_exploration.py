import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the data from the Excel file, replace '?' with NaN for easier processing
def load_thyroid_data(filepath: str):
    df = pd.read_excel(filepath, sheet_name="thyroid0387_UCI")
    df.replace('?', np.nan, inplace=True)  # Convert '?' placeholders to proper NaN values
    return df

def detect_attribute_types(df):
    