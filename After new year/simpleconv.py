import numpy as np
import cv2 as cv

# Kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 2, -1],
                   [-1, -1, -1]])

# small blur
kernel = np.array([[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]])

# cool!
kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                   [1/25, 1/25, 1/25, 1/25, 1/25],
                   [1/25, 1/25, 1/25, 1/25, 1/25],
                   [1/25, 1/25, 1/25, 1/25, 1/25],
                   [1/25, 1/25, 1/25, 1/25, 1/25]])

def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    pad = kh // 2
    padded_image = np.pad(image, pad_width= pad, mode="constant", constant_values = 0)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

image = cv.imread("SVP_EN_43.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray_image)
conv_img = convolve(gray_image, kernel=kernel)
cv.imshow("conv", conv_img)

cv.waitKey(0)
cv.destroyAllWindows()