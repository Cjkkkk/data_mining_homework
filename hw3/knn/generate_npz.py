import extract_image
import numpy as np
import show_image
num_images = 100

data = {
    'x_train': np.zeros((100*5, 140)),
    'y_train': np.zeros((100*5, )),
}
for i in range(num_images):
    # 处理图片
    image_data = extract_image.extract_image('data/' + str(i) + '.jpg')
    data['x_train'][i*5: (i+1)*5, :] = image_data

    # 处理标签
    image_label = open('data/' + str(i) + '.txt').read()
    image_label = list(map(lambda x: int(x), image_label))
    image_label = np.array(image_label, dtype=int)
    data['y_train'][i*5: (i+1)*5] = image_label

    # 保存数据到 npz
# show_image.show_image(data['x_train'][5:10, :])
# print(data['y_train'][5:10])
np.savez_compressed("hack_data.npz", x_train=data['x_train'], y_train=data['y_train'])
