import numpy as np
from likelihood import likelihood


def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    each_class_number = np.sum(x, axis=1)
    total = np.sum(x)

    prior = each_class_number / total
    p = np.zeros((C, N))
    for i in range(C):
        for j in range(N):
            p[i][j] = l[i][j] * prior[i]
    p = p / sum(p)
    # begin answer
    # end answer
    
    return p
