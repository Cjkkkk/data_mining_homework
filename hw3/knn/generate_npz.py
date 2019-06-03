import extract_image
import numpy as np
num_images = 100

data = {
    'x_train': np.zeros((100*5, 140)),
    'y_train': np.zeros((100*5, )),
}
for i in range(num_images):
    # 处理图片
    image_data = extract_image('data/' + str(i) + '.jpg')
    image_data = extract_image(image_data)
    data['x_train'][i*5: (i+1)*5, :] = image_data

    # 处理标签
    image_label = open('data/label' + str(i) + '.txt').split()
    image_label = list(map(lambda x: int(x), image_label))
    data['y_train'][i*5: (i+1)*5] = np.array(image_label)

    # 保存数据到 npz
    np.save(data, "hack_data.npz")
