import numpy as np
import math


def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    # Your code HERE
    # begin answer

    det = [np.linalg.det(Sigma[:, :, k]) for k in range(K)]
    inv = [np.linalg.inv(Sigma[:, :, j]) for j in range(K)]
    
    exp = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            x_minus_mean = X[:, i] - Mu[:, j]
            exp[i][j] = \
                math.exp(
                    -0.5 * (
                        np.matmul(
                            np.matmul(x_minus_mean.T, inv[j]),
                            x_minus_mean)
                        ))
    # 计算likelihood
    l = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            l[i][j] = 1 / (2 * math.pi * math.sqrt(abs(det[j]))) * exp[i][j]
    # 计算posterior
    p = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            p[i][j] = l[i][j] * Phi[j]
        p[i, :] = p[i, :] / np.sum(p[i, :])

    # end answer
    # print(p)
    return p


def main():
    X = np.array([[1, 1], [1, 1]])
    Mu = np.array([[0, 1], [0, 1]])
    Phi = np.array([0.5, 0.5])
    Sigma = np.array([
        [
            [1, 1],
            [0, 0]
        ],
        [
            [0, 0],
            [1, 1]
        ],
    ])

    print(gaussian_pos_prob(X, Mu, Sigma, Phi))
