import numpy as np
import pca
import matplotlib.pyplot as plt
import math
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # 转为灰度图像
    gray = rgb2gray(img_r)

    # 找到所有有颜色的点
    colored_point = []
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            if gray[i, j] > 1:
                colored_point.append([i, j])

    colored_point = np.array(colored_point)
    eigen_values, eigen_vectors = pca.PCA(colored_point)
    # 重新生成图片
    degree = math.atan(eigen_vectors[1, 0] / eigen_vectors[0, 0]) / math.pi * 180
    degree = 90 - degree if degree > 0 else - 90 - degree
    print("rotate degree: ", degree)
    img = Image.fromarray(img_r.astype('uint8'))
    return img, img.rotate(degree)
    # end answer
