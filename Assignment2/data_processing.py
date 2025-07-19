import pandas as pd
import numpy as np

#purchase data
def load_data(filepath: str, sheet_name: str = "Purchase data"):
    # Load the sheet into a DataFrame
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Extract matrix A: columns for Candies, Mangoes, Milk Packets
    A = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values

    # Extract matrix/vector C: Payment
    C = df["Payment (Rs)"].values

    return A, C

def load_classification_data(filepath: str, sheet_name: str = "Purchase data"):
    """
    Loads features (X) and classification labels (y) for logistic regression.
    Labels: rich if payment>200, else pOOR.

    Returns:
        X (np.ndarray): Features - Candies, Mangoes, Milk Packets.
        y (np.ndarray): Labels - 1 for rich, 0 for pOOR.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Features: same as before
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values

    # Label: 1 if Payment > 200, else 0
    y = (df["Payment (Rs)"] > 200).astype(int).values

    return X, y

# stock data
def load_irctc_data(file_path):
    # Load the "IRCTC Stock Price" sheet
    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

    # Rename columns for easier access
    df.columns = ["Date", "Month_Abbr", "Weekday_Abbr", "Price", "Open", "High", "Low", "Volume", "Chg%"]

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert Month Abbr to full month names
    df['Month'] = df['Date'].dt.month_name()

    # Convert Chg% from string like "-2.52%" to float -2.52
    df['Chg%'] = pd.to_numeric(df['Chg%'].astype(str).str.replace('%', ''), errors='coerce')

    return df


