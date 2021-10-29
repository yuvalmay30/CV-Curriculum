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

    D_L = disparity_ssd(L, R, 8, 30)
    normalized_d_l = normalize_disparity_image(D_L)

    D_R = disparity_ssd(R, L, 8, 30)
    normalized_d_r = normalize_disparity_image(D_R)

    cv2.imwrite(os.path.join('output', 'ps2-1-a-1.png'), normalized_d_l)
    cv2.imwrite(os.path.join('output', 'ps2-1-a-2.png'), normalized_d_r)


def question2():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)

    R = cv2.imread(os.path.join('input', 'pair1-R.png'))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    D_L = disparity_ssd(L, R, 4, 80)
    normalized_d_l = np.abs(D_L)
    normalized_d_l = normalize_disparity_image(normalized_d_l)

    cv2.imwrite(os.path.join('output', 'ps2-2-a-1.png'), normalized_d_l)

    print('Done processing disparity L')

    D_R = disparity_ssd(R, L, 4, 80)
    normalized_d_r = np.abs(D_R)
    normalized_d_r = normalize_disparity_image(normalized_d_r)

    cv2.imwrite(os.path.join('output', 'ps2-2-a-2.png'), normalized_d_r)


def question3():
    rng = np.random.default_rng()

    L = cv2.imread(os.path.join('input', 'pair1-L.png'))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    L = L.astype(float)
    L += rng.normal(0, 20, L.shape)
    L = L.astype(np.uint8)

    R = cv2.imread(os.path.join('input', 'pair1-R.png'))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    R = R.astype(float)
    R += rng.normal(0, 20, R.shape)
    R = R.astype(np.uint8)

    D_L = disparity_ssd(L, R, 4, 80)
    normalized_d_l = np.abs(D_L)
    normalized_d_l = normalize_disparity_image(normalized_d_l)

    cv2.imwrite(os.path.join('output', 'ps2-3-a-1.png'), normalized_d_l)
    print('Done processing disparity L')

    D_R = disparity_ssd(R, L, 4, 80)
    normalized_d_r = np.abs(D_R)
    normalized_d_r = normalize_disparity_image(normalized_d_r)

    cv2.imwrite(os.path.join('output', 'ps2-3-a-2.png'), normalized_d_r)


def question4():
    # rng = np.random.default_rng()

    L = cv2.imread(os.path.join('input', 'pair1-L.png'))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    # L = L.astype(float)
    # L += rng.normal(0, 20, L.shape)
    L = L.astype(np.uint8)

    R = cv2.imread(os.path.join('input', 'pair1-R.png'))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    # R = R.astype(float)
    # R += rng.normal(0, 20, R.shape)
    R = R.astype(np.uint8)

    D_L = disparity_ssd(L, R, 8, 80, 'ccoeff_normed')
    normalized_d_l = np.abs(D_L)
    normalized_d_l = normalize_disparity_image(normalized_d_l)

    cv2.imwrite(os.path.join('output', 'ps2-4-a-1.png'), normalized_d_l)
    print('Done processing disparity L')

    # D_R = disparity_ssd(R, L, 4, 80, 'ccoeff_normed')
    # normalized_d_r = np.abs(D_R)
    # normalized_d_r = normalize_disparity_image(normalized_d_r)
    #
    # cv2.imwrite(os.path.join('output', 'ps2-4-a-2.png'), normalized_d_r)


question4()
