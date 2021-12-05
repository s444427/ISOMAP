import copy
import math

import sklearn
from sklearn import datasets
import random
from plotly.express import scatter_3d
import numpy
import numpy as np
import matplotlib.pyplot as plt


def Floyd_Warshall(W):
    D = W.copy()
    n = D.shape[0]
    for k in range(n):
        for i in range(n):
            D[i, :] = np.minimum(D[i, :], D[i, k] + D[k, :])
    return D


def k_weight_matrix(D, k=2):
    # Calculate k closest neighbours
    # Creating D_copy has no other point then to show the difference and not "spoil" the original matrix D
    n = int(len(D))

    infinity = 10.
    W = np.full((n, n), infinity)

    D_copy = D.copy()
    for i in range(0, len(D_copy)):
        for j in range(k):
            W[i][i] = 0

            max_idx = np.argmin(D_copy[i])
            W[i][max_idx] = D[i][max_idx]
            D_copy[i][max_idx] = 0

    if (W == W.T).all():
        print("Weight matrix check: correct")
    else:
        print("Weight matrix check: failed")
    return W


def distance_matrix(X):
    n = int(len(X))
    D = np.zeros(shape=(n, n))

    # Get Cartesian distance
    for i in range(n):
        for j in range(n):
            D[i][j] = math.sqrt(abs(X[i][0] - X[j][0]) + abs(X[i][1] - X[j][1]) + abs(X[i][2] - X[j][2]))

    return D


def generate_data():
    random.seed(1234)
    # print("Hi")
    X, Y = sklearn.datasets.make_swiss_roll(10)
    # print("Before data")
    # print(X)
    # print(smoke_it)

    # show the data
    # fig = scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=Y)
    # fig.show()
    return X, Y


def gram_matrix(D, d):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = - H @ (D * D) @ H / 2
    Lambda, V = np.linalg.eigh(G)
    Lambda, V = Lambda[:-d - 1:-1], V[:, :-d - 1:-1]
    Z = np.diag(np.sqrt(Lambda)) @ V.T
    # print("After data")
    # print(Z)
    return Z


if __name__ == '__main__':
    X, Y = generate_data()
    D = distance_matrix(X)
    W = k_weight_matrix(D, 5)
    D = Floyd_Warshall(W)
    Z = gram_matrix(D, 2)
    # print(len(Z))
    # print(len(Z[0]))

    # x1 = Z[0]
    # y1 = Z[1]
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)

    # ax1.scatter(x1, y1, s=10, c='blue', marker="o", label='non-whitened')
    # plt.legend(loc='lower left')
    # plt.savefig('scatter_plot.jpg')
