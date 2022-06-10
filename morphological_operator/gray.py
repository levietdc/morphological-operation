import numpy as np


def erode(img, kernel):
    img_shape = img.shape
    kernel_shape = kernel.shape
    kernel_center = (kernel_shape[0] // 2, kernel_shape[1] // 2)
    eroded_img = np.zeros((img_shape[0] + kernel_shape[0] - 1, img_shape[1] + kernel_shape[1] - 1))

    x_append = np.zeros((img.shape[0], kernel_shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel_shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            x, y = i + kernel_center[0], j + kernel_center[1]
            tmp = np.zeros((kernel_shape[0], kernel_shape[1]))
            if np.all(img[i:i_, j:j_] != 0):
                tmp = img[i:i_, j:j_] - kernel[0:kernel.shape[0], 0:kernel.shape[1]]
                eroded_img[x, y] = tmp.min()

    return eroded_img[:img_shape[0], :img_shape[1]] / 255.0


def dilate(img, kernel):
    kernel_shape = kernel.shape
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    dilated_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            tmp = np.zeros((kernel_shape[0], kernel_shape[1]))
            if (img[i + kernel_center[0], j + kernel_center[0]] != 0):
                tmp = img[i:i_, j:j_] + kernel[0:kernel.shape[0], 0:kernel.shape[1]]
                for m in range(i, i_):
                    for n in range(j, j_):
                        new_value = img[i + kernel_center[0], j + kernel_center[0]] + kernel[
                            kernel_center[0], kernel_center[1]]
                        if (dilated_img[m, n] < new_value):
                            dilated_img[m, n] = new_value
                dilated_img[i + kernel_center[0], j + kernel_center[1]] = tmp.max()
    return dilated_img[:img_shape[0], :img_shape[1]] / 255.0


def open(img, kernel):
    return dilate(erode(img, kernel) * 255.0, kernel)


def close(img, kernel):
    return erode(dilate(img, kernel) * 255.0, kernel)


def gradient(img, kernel):
    return dilate(img, kernel) - erode(img, kernel)


def top_hat(img, kernel):
    return (img - open(img, kernel) * 255.0) / 255.0


def black_hat(img, kernel):
    return (close(img, kernel) * 255.0 - img) / 255.0
