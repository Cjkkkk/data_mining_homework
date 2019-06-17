import numpy as np
import pca
import matplotlib.pyplot as plt


def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)
    # YOUR CODE HERE
    # begin answer
    h, w, d = img_r.shape[0], img_r.shape[1], img_r.shape[2]
    img_r_reshape = img_r.reshape((h * w, d))
    value, vector = pca.PCA(img_r_reshape)

    # 求出均值和方差
    mean = np.mean(img_r_reshape, axis=0)
    # std = np.std(img_r_reshape)

    # 取前三个特征向量
    A = vector[:, 0:3]
    image = (np.matmul(np.matmul((img_r_reshape - mean), A), A.T)) + mean

    # 去除不符合的元素
    image = image.astype(np.int)
    image[image < 0] = 0
    image[image > 255] = 255

    # 重新生成图片
    return image.reshape((h, w, d))
    # end answer
