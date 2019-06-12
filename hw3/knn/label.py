import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 读取data目录下的图片并显示， 停留2s后等待用户输入标签，并将结果保存在文件i.txt中，i为[0..100]
for i in range(100):
    path = 'data/' + str(i) +".jpg"
    ImageItself = Image.open(path)
    ImageNumpyFormat = np.asarray(ImageItself)
    plt.imshow(ImageNumpyFormat)
    plt.draw()
    plt.pause(2)  # pause how many seconds
    plt.close()
    label = input("input label: ")
    f = open("data/" + str(i) + ".txt", "w")
    f.write(label)
    f.close()