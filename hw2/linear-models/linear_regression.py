import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    D = np.vstack((np.ones((1, N)), X))

    in_matrix = np.zeros((P, N))
    for i in range(N):
        if y[0, i] > 0:
            pos = 1
        else:
            pos = 0
        in_matrix[pos, i] = 1

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(D, D.T)), D), in_matrix.T)
    w = w[:, 1] - w[:, 0]
    # print(w)

    return w
