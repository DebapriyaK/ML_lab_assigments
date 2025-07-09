def mult(A,B): #multiply mmatrices
    n = len(A)
    result = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_power(A, m):
    result = A
    for _ in range(m - 1):
        result = mult(result, A)
    return result


def main():
    n = int(input("Enter size of square matrix (n x n): "))
    print("Enter the matrix row by row:")
    A = []
    for i in range(n):
        row = list(map(int, input().split()))
        if len(row)!= n:
            print("Invalid row length.")
            return
        A.append(row)

    m = int(input("Enter power m (positive integer): "))
    if m <= 0:
        print("Power must be positive.")
        return

    result = matrix_power(A, m)
    print(f"\nMatrix A^{m} is:")
    for row in result:
        print(*row)


if __name__ == "__main__":
    main()
