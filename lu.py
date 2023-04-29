# Hava Haviv 211737440
import numpy as np


# define the function 'lu' that takes a matrix and a result vector as inputs and returns the solution vector 'X'
def lu(matrix, result):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    isU = False

    # compute elements of the first row of the U matrix
    for i in range(0, n, 1):
        U[0, i] = matrix[0, i]
        # set the diagonal elements of L to 1
        L[i, i] = 1

    # compute elements of the first column of the L matrix
    for i in range(1, n, 1):
        L[i, 0] = matrix[i, 0] / U[0, 0]

    # compute the remaining elements of L and U
    for i in range(1, n, 1):
        j = 1
        while j < n:
            if isU:
                U[i][j] = matrix[i][j] - sigma(L, U, i, j, i)
            elif j < i:
                L[i][j] = (1 / U[j][j]) * (matrix[i][j] - sigma(L, U, i, j, j))

            # switch to computing elements of the U matrix once we reach the diagonal of L
            if j == n-1 and not isU:
                isU = True
                j = i - 1
            j += 1
        isU = False

    # solve for Y and X vectors using the L and U matrices
    Y = np.zeros(n)
    X = np.zeros(n)
    for i in range(n):
        Y[i] = result[i] - sum(L[i, j]*Y[j] for j in range(i))
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - sum(U[i, j] * X[j] for j in range(i + 1, n))) / U[i, i]
    return X


# computes the sum of the products of elements in a row of L and a column of U
def sigma(l, u, i, j, des):
    sum = 0
    for k in range(0, des, 1):
        sum += l[i][k] * u[k][j]
    return sum


if __name__ == '__main__':
    A = np.array([[2, 0, -3, 0.5, -1], [0, -2, 3, -1, 0], [0.5, 0, 1, -5, 0], [0, 1, 4, -1, -1], [0.1, 1, 0.5, 0, 4]])
    B = np.array([-7.5, -6, 5, 10, 5.3])
    x = lu(A, B)
    print(x)
