import numpy as np

# pointer = np.array([
#     [1, 1],
#     [2, 2],
#     [3, 3],
#     [4, 4],
#     [5, 5]
# ])


def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''
    n, p = X.shape
    W = np.zeros((n, n))
    for i in range(n):
        dis = [np.sum(np.square(X[i, :] - X[j, :])) for j in range(n)]
        idx = np.argpartition(dis, k)[:k]
        for j in idx:
            W[i, j] = dis[j] if dis[j] > threshold else 0
    return W
    # YOUR CODE HERE
    # begin answer
    # end answer


# knn_graph(pointer, 4, 1)
