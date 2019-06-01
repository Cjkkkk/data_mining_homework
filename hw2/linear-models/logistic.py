import numpy as np


def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    learning_rate = 0.1
    # 计算分子
    D = np.vstack((np.ones((1, N)), X))
    for _ in range(1000):
        # 计算分母
        exp = np.exp(np.matmul(w.T, D))
        delta = np.zeros((P+1, N))
        for i in range(N):
            delta[:, i] = D[:, i] * exp[0, i] / (1 + exp[0, i]) - D[:, i] * y[0, i]
        # 求和
        delta = np.sum(delta, axis=1)
        if learning_rate * np.linalg.norm(delta) < 0.001:
             break
        w -= learning_rate * delta.reshape((-1, 1))
    # end answer
    
    return w
