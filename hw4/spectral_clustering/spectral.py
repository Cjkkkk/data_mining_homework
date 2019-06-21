import numpy as np
import kmeans


def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    n, _ = W.shape
    # begin answer
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = np.sum(W[i, :])

    D_negative_sqrt = np.copy(D)
    for i in range(n):
        D_negative_sqrt[i, i] = 1 / np.sqrt(D[i, i])
    A = np.matmul(np.matmul(D_negative_sqrt, D - W), D_negative_sqrt)
    eigen_values, eigen_vectors = np.linalg.eig(A)
    sorted_idx = np.argsort(eigen_values)
    low = eigen_vectors[:, sorted_idx[1]]
    # idx = np.random.randint(0, k, n)
    # idx[z < 0] = 0
    # idx[z > 0] = 1
    # return idx
    return kmeans.kmeans(low.reshape(n, 1), k)
    # end answer
