import numpy as np


def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    D = np.vstack((np.ones((1, N)), X))
    in_matrix = np.zeros((P, N))
    for i in range(N):
        if y[0, i] > 0:
            pos = 1
        else:
            pos = 0
        in_matrix[pos, i] = 1

    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(D, D.T) + lmbda * np.eye(P+1, P+1)), D), in_matrix.T)
    w = w[:, 1] - w[:, 0]
    # end answer
    return w
