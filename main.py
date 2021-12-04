import copy
import math

import sklearn
from sklearn import datasets
import random
from plotly.express import scatter_3d
import numpy
import numpy as np
import matplotlib.pyplot as plt


def neighbourhood_matrix(X, k=1):
    n = int(len(X))

    D = np.zeros(shape=(n, n))
    # W = np.zeros(shape=(n, n))
    infinity = 10.
    W = np.full((n, n), infinity)

    # Get Cartesian distance
    for i in range(n):
        for j in range(n):
            D[i][j] = math.sqrt(abs(X[i][0] - X[j][0]) + abs(X[i][1] - X[j][1]) + abs(X[i][2] - X[j][2]))

    print(D)
    # Calculate k closest neighbours
    D_copy = D.copy()
    for j in range(k):
        for i in range(0, len(D_copy)):
            W[i][i] = 0

            max_idx = np.argmax(D_copy[i])
            W[i][max_idx] = D[i][max_idx]
    print(W)
    return W


def generate_data():
    random.seed(1234)
    print("Hi")
    X, smoke_it = sklearn.datasets.make_swiss_roll(3)
    # print(X)
    # print(smoke_it)

    # show the data
    # fig = scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=X[:, 2])
    # fig.show()
    neighbourhood_matrix(X)


if __name__ == '__main__':
    generate_data()
