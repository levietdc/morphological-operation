import numpy as np
import cv2


def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 255

    return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''


def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    dilate_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] // 2))
    img = np.append(img, x_append, axis=1)
    img = np.append(x_append, img, axis=1)

    y_append = np.zeros((kernel.shape[0] // 2, img.shape[1]))
    img = np.append(img, y_append, axis=0)
    img = np.append(y_append, img, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if (img[i + kernel_center[0], j + kernel_center[1]] == 255):
                i_ = i + kernel.shape[0]
                j_ = j + kernel.shape[1]
                for x in range(i, i_):
                    for y in range(j, j_):
                        if dilate_img[x][y] != 0 or kernel[x - i][y - j] != 0:
                            dilate_img[x][y] = 255

    return dilate_img[:img_shape[0], :img_shape[1]]


def opening(img, kernel):
    img_erosion = erode(img, kernel)
    img_dilation = dilate(img_erosion, kernel)
    return img_dilation[:img_dilation.shape[0], :img_dilation.shape[1]]


def closing(img, kernel):
    img_dilation = dilate(img, kernel)
    img_erosion = erode(img_dilation, kernel)
    return img_erosion[:img_erosion.shape[0], :img_erosion.shape[1]]


def hit_or_miss(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    hit_miss_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            ck = True
            for x in range(i, i_):
                for y in range(j, j_):
                    if (kernel[x - i][y - j] == 1 and img[x][y] != 255) or (
                            kernel[x - i][y - j] == -1 and img[x][y] != 0):
                        ck = False
                        break
            if ck:
                hit_miss_img[i][j] = 255

    return hit_miss_img[:img_shape[0], :img_shape[1]]


def thinningIteration(img, iter):
    out_img = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            p2 = img[i - 1][j]
            p3 = img[i - 1][j + 1]
            p4 = img[i][j + 1]
            p5 = img[i + 1][j + 1]
            p6 = img[i + 1][j]
            p7 = img[i + 1][j - 1]
            p8 = img[i][j - 1]
            p9 = img[i - 1][j - 1]

            A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)

            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            m1 = (p2 * p4 * p6) if iter == 0 else (p2 * p4 * p8)
            m2 = (p4 * p6 * p8) if iter == 0 else (p2 * p6 * p8)
            if A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0:
                out_img[i, j] = 1
    return out_img


def thinning(img):
    np.true_divide(img, 255)
    pre = np.zeros(img.shape)
    diff = None
    while True:
        img = thinningIteration(img, 0)
        img = thinningIteration(img, 1)
        cv2.absdiff(img, pre, diff)
        pre = np.copy(img)
        if cv2.countNonZero(diff) == 0:
            break
    img *= 255
    return img


def extract_boundary(img, kernel):
    return (img - erode(img, kernel))
