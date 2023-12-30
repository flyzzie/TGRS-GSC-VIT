import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from dataset import load_mat_hsi


def DrawResult(labels):

    row = height
    col = width
    palette = np.array([[37, 58, 150],
                        # [47, 78, 161],
                        # [56, 87, 166],
                        # [56, 116, 186],
                        [51, 181, 232],
                        [112, 204, 216],
                        # [119, 201, 168],
                        [148, 204, 120],
                        [188, 215, 78],
                        [238, 234, 63],
                        # [246, 187, 31],
                        [244, 127, 33],
                        [239, 71, 34],
                        [238, 33, 35],
                        # [180, 31, 35],
                        [123, 18, 20]])
    # palette = np.array([[37, 58, 150],
    #                     # [47, 78, 161],
    #                     [56, 87, 166],
    #                     # [56, 116, 186],
    #                     [51, 181, 232],
    #                     [112, 204, 216],
    #                     [119, 201, 168],
    #                     [148, 204, 120],
    #                     [188, 215, 78],
    #                     [238, 234, 63],
    #                     [246, 187, 31],
    #                     [244, 127, 33],
    #                     [239, 71, 34],
    #                     [238, 33, 35],
    #                     [180, 31, 35],
    #                     [123, 18, 20]])
    palette = palette[:class_num]
    palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0], 3))
    for i in range(class_num):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]
    X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result


X, y, labels = load_mat_hsi('flt', '../datasets')

class_num = 9

height = y.shape[0]
width = y.shape[1]
y = y.flatten()
img = DrawResult(np.reshape(y, -1))

plt.imsave("FLT" + '.png', img)

