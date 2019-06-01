import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    MAX_ITERATIONS = 1000
    n, p = x.shape

    ran_idx = np.random.randint(n, size=k)
    idx = np.random.randint(0, k, (n, ))
    ctrs = x[ran_idx, :]

    iter_ctrs = np.zeros((MAX_ITERATIONS+1, k, p))
    iter_ctrs[0, :, :] = ctrs  # 随机生成的centers放在index=0的位置

    for i in range(1, MAX_ITERATIONS + 1):
        # 分配到k个cluster
        error = 0
        for j in range(n):
            new_label = np.argmin([np.sum(np.square(ctrs[m, :] - x[j, :])) for m in range(k)])
            if idx[j] != new_label:
                error += 1
                idx[j] = new_label
        if error == 0:
            print("iterations times: ", i)
            iter_ctrs.resize((i, k, p))
            return idx, ctrs, iter_ctrs
        # 计算新的center
        for m in range(k):
            ctrs[m] = np.zeros((p, ))
            index = np.array(idx == m)
            if index.sum() != 0:  # 没有该标签的样本
                ctrs[m] = np.mean(x[index, :], axis=0)
        iter_ctrs[i, :, :] = ctrs
    # end answer
    print("reach max iterations times: ", MAX_ITERATIONS)
    return idx, ctrs, iter_ctrs
