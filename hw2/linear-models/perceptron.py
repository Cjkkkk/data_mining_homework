import numpy as np


def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    D = np.vstack((np.ones((1, N)), X))
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    MAX_ITERATION = 5000
    while iters < MAX_ITERATION:
        flag = True
        for i in range(N):
            iters += 1
            x = D[:, i].reshape((P + 1, 1))
            label = np.sign(np.matmul(w.T, x))[0, 0]
            if label != y[0, i]:
                flag = False
                w = w + x * y[0, i]
        if flag:
            break

    # error = 0
    # for i in range(N):
    #     if predict[0, i] != y[0, i]:
    #         error += (np.matmul(w.T, D[:, i].reshape((P + 1, 1)))[0, 0] * y[0, i])
    # print(error)
    return w, iters