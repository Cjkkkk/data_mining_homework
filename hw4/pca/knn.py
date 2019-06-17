import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    # end answer

    N_test, P = x.shape
    N_train, _ = x_train.shape

    y = np.zeros((N_test,))
    for i in range(N_test):
        # 计算test中的每一个点的k个最近的点
        idx = np.argpartition(
            np.array([np.sum(np.square(x[i, :] - x_train[j, :])) for j in range(N_train)])
            , k)[:k]
        # 找到每个标签中出现的次数
        values, counts = np.unique(y_train[idx], return_counts=True)
        # 投票找出最大值
        y[i] = values[np.argmax(counts)]
    # end answer
    return y
