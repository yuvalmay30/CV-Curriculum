from cv2 import cv2
import numpy as np

rng = np.random.default_rng()

linux_image = cv2.imread('./input/linux.png')

noisy_blue = linux_image[:, :, 0] + rng.normal(0, 100, linux_image.shape[:2])
normalized_blue = cv2.normalize(noisy_blue, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

linux_image[:, :, 0] = normalized_blue

cv2.imshow('image', linux_image)
cv2.imwrite('./output/ps0-5-b-1.png', linux_image)

cv2.waitKey(0)
