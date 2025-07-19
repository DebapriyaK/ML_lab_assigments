import numpy as np
from data_processing import load_data

def main():
    # Load matrices A and C from the Excel sheet
    A, C = load_data("Lab Session Data.xlsx")

    # 1. Dimensionality of the vector space = number of product types (columns in A)
    dimensionality = A.shape[1]
    print(f"Dimensionality of the vector space: {dimensionality}")

    # 2. Number of vectors = number of customer entries (rows in A)
    num_vectors = A.shape[0]
    print(f"Number of vectors in the vector space: {num_vectors}")

    # 3. Rank of matrix A
    rank = np.linalg.matrix_rank(A)
    print(f"Rank of matrix A: {rank}")

    # 4. Compute pseudo-inverse of A and solve for X
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C

    # Print the computed cost of each product
    product_names = ["Candy (Rs per unit)", "Mango (Rs per Kg)", "Milk Packet (Rs per unit)"]
    print("Computed cost of each product:")
    for name, price in zip(product_names, X):
        print(f"{name}: ₹{price:.2f}")

if __name__ == "__main__":
    main()
