import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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