# ps2
import os
import numpy as np
from cv2 import cv2
from disparity_ssd import disparity_ssd


def normalize_disparity_image(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    return normalized_image.astype(np.uint8)


def question1():
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

    D_L = disparity_ssd(L, R)
    normalized_d_l = normalize_disparity_image(D_L)

    D_R = disparity_ssd(R, L)
    normalized_d_r = normalize_disparity_image(D_R)

    cv2.imwrite(os.path.join('output', 'ps2-1-a-1.png'), normalized_d_l)
    cv2.imwrite(os.path.join('output', 'ps2-1-a-2.png'), normalized_d_r)


def ps2_question2():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)

    R = cv2.imread(os.path.join('input', 'pair1-R.png'))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    D_L = disparity_ssd(L, R)
    normalized_d_l = normalize_disparity_image(D_L)

    cv2.imwrite(os.path.join('output', 'ps2-2-a-1.png'), normalized_d_l)
    print('Done processing disparity L')

    D_R = disparity_ssd(R, L)
    normalized_d_r = normalize_disparity_image(D_R)

    cv2.imwrite(os.path.join('output', 'ps2-2-a-2.png'), normalized_d_r)


def ps2_question3():
    rng = np.random.default_rng()

    L = cv2.imread(os.path.join('input', 'pair1-L.png'))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    L += rng.normal(0, 100, L.shape)

    R = cv2.imread(os.path.join('input', 'pair1-R.png'))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    R += rng.normal(0, 100, L.shape)

    D_L = disparity_ssd(L, R)
    normalized_d_l = normalize_disparity_image(D_L)

    cv2.imwrite(os.path.join('output', 'ps2-3-a-1.png'), normalized_d_l)
    print('Done processing disparity L')

    D_R = disparity_ssd(R, L)
    normalized_d_r = normalize_disparity_image(D_R)

    cv2.imwrite(os.path.join('output', 'ps2-3-a-2.png'), normalized_d_r)


ps2_question2()
