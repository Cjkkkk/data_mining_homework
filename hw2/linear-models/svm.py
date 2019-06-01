import numpy as np
import scipy.optimize as opt


def cont(w, X, y):
    return np.multiply(y[0, :], np.matmul(w.T, X)) - 1


def func(w):
    return 0.5 * (np.linalg.norm(w[1:, ]) ** 2)


def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.ones((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    D = np.vstack((np.ones((1, N)), X))
    cons = {'type': 'ineq', 'fun': cont, 'args': (D, y)}
    res = opt.minimize(func, w, constraints=cons, method='SLSQP')

    # 计算support vector的数目
    num = len(list(filter(lambda x: 0.95 < x < 1.05, np.multiply(y[0, :], np.matmul(w.T, D))[0, :])))
    # end answer
    return res.x, num

