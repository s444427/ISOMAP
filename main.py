import math
import plotly.graph_objects as po
import sklearn
from sklearn import datasets
import random
from plotly.express import scatter_3d
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

    infinity = 1000.
    W = np.full((n, n), infinity)

    D_copy = D.copy()

    for o in range(n):
        D_copy[o][o] = 1000

    for i in range(0, len(D_copy)):
        for j in range(0, k):
            W[i][i] = 0

            max_idx = np.argmin(D_copy[i])
            W[i][max_idx] = D[i][max_idx]
            D_copy[i][max_idx] = 1000

    if (W == W.T).all():
        print("Weight matrix check: correct")
    else:
        print("Weight matrix check: failed")
    return W


def e_weight_matrix(D, e=1):
    # Calculate closest neighbours in range e
    # Creating D_copy has no other point then to show the difference and not "spoil" the original matrix D
    n = int(len(D))

    infinity = 1000.
    W = np.full((n, n), infinity)

    D_copy = D.copy()
    for i in range(0, len(D_copy)):
        W[i][i] = 0
        for j in range(len(D_copy[i])):
            if D_copy[i][j] <= e:
                W[i][j] = D[i][j]

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


def generate_data(n=800, seed=1234):
    random.seed(seed)
    X, Y = sklearn.datasets.make_swiss_roll(n)
    return X, Y


def show_swiss(X, Y):
    # show the data
    fig = scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=Y)
    fig.show()


def gram_matrix(D, d):
    # D = Distances matrix
    # d = number of output dimentions

    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = - H @ (D * D) @ H / 2
    Lambda, V = np.linalg.eigh(G)
    Lambda, V = Lambda[:-d - 1:-1], V[:, :-d - 1:-1]
    Z = np.diag(np.sqrt(Lambda)) @ V.T
    return Z


def plot_graph_3d(x, W, color=None):
    n = W.shape[0]
    edges_x, edges_y, edges_z = [], [], []

    for i in range(0, n):
        for j in range(0, i):
            if 0 < W[i, j] < 1000:
                edges_x.extend([x[0, i], x[0, j], None])
                edges_y.extend([x[1, i], x[1, j], None])
                edges_z.extend([x[2, i], x[2, j], None])

    edges_trace = po.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode='lines')
    nodes_trace = po.Scatter3d(x=x[0], y=x[1], z=x[2], mode='markers',
                               marker=dict(color=color, size=5.0))

    figure = po.Figure(data=[edges_trace, nodes_trace])
    figure.show()


if __name__ == '__main__':
    X, Y = generate_data(n=800)
    # show_swiss(X, Y)
    D = distance_matrix(X)

    k = 7
    e = 2.5

    W_k = k_weight_matrix(D, k)
    W_e = e_weight_matrix(D, e)
    plot_graph_3d(X.T, W_k, Y)

    D_optimised = Floyd_Warshall(W_k)

    Z = gram_matrix(D_optimised, 2)

    # print(len(Z))
    # print(len(Z[0]))

    x1 = Z[0]
    y1 = Z[1]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x=Z[0], y=Z[1], c=Y)
    plt.savefig('scatter_plot.jpg')
