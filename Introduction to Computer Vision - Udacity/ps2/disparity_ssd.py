from cv2 import cv2
import numpy as np
from tqdm import tqdm


def disparity_ssd(source_image, template_image, window_edge_size=2, max_offset=10, matching_function=None):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    half_window_size = int(window_edge_size / 2)
    disparity_image = np.zeros(source_image.shape)

    padded_source_image = pad_image_with_zeros(source_image, window_edge_size)
    padded_template_image = pad_image_with_zeros(template_image, window_edge_size)

    for row_index in tqdm(range(source_image.shape[0])):
        for column_index in range(source_image.shape[1]):
            disparity_image[row_index, column_index] = \
                calculate_disparity_for_pixel(row_index + half_window_size,
                                              column_index + half_window_size,
                                              window_edge_size,
                                              padded_source_image,
                                              padded_template_image,
                                              max_offset,
                                              matching_function)

    return disparity_image


def calculate_disparity_for_pixel(row_index, column_index, window_edge_size,
                                  source_image, template_image, max_offset, matching_function):
    half_window_size = int(window_edge_size / 2)
    minimal_ssd, minimal_ssd_index = None, None

    source_window = get_window_around_pixel(row_index, column_index, window_edge_size, source_image)

    left_bound_window_index = max(column_index - max_offset, half_window_size)
    right_bound_window_index = min(column_index + max_offset, template_image.shape[1] - half_window_size)

    for window_column_index in range(left_bound_window_index, right_bound_window_index + 1):
        template_window = get_window_around_pixel(row_index, window_column_index, window_edge_size, template_image)
        ssd = None

        if matching_function in [None, 'ssd']:
            ssd = calculate_ssd(source_window, template_window)
        else:
            ssd = cv2.matchTemplate(source_window, template_window, cv2.TM_CCOEFF_NORMED)

        if minimal_ssd is None or ssd < minimal_ssd:
            minimal_ssd = ssd
            minimal_ssd_index = window_column_index

    return minimal_ssd_index - column_index


def get_window_around_pixel(row, column, window_size, image):
    half_window_size = int(window_size / 2)

    return image[
           row - half_window_size: row + half_window_size,
           column - half_window_size: column + half_window_size
           ]


def calculate_ssd(source_window, template_window):
    windows_differences = source_window - template_window
    return np.sum(np.power(windows_differences, 2))


def pad_image_with_zeros(image, pad_size):
    padded_image_size = tuple(map(lambda x: x + pad_size, image.shape))
    padded_image = np.zeros(padded_image_size)

    num_of_rows, num_of_columns = padded_image.shape
    half_pad_size = int(pad_size / 2)

    padded_image[half_pad_size: num_of_rows - half_pad_size, half_pad_size: num_of_columns - half_pad_size] = image

    padded_image = padded_image.astype(np.uint8)
    return padded_image
