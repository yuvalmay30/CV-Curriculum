import operator
import numpy as np
from cv2 import cv2
import math


def calculate_derivatives(image, ksize):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)

    return grad_x, grad_y


def generate_gradient_pair_images(image_path, output_file_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image_grad_x, image_grad_y = calculate_derivatives(image, 3)

    image_grad_x = cv2.normalize(image_grad_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_grad_y = cv2.normalize(image_grad_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    image_gradient_pair = np.concatenate((image_grad_x, image_grad_y), axis=1)

    cv2.imwrite(output_file_path, image_gradient_pair)


def ps4_1_a():
    generate_gradient_pair_images('transA.jpg', 'output/ps4-1-a-1.jpg')
    generate_gradient_pair_images('simA.jpg', 'output/ps4-1-a-2.jpg')


def get_gaussian_kernel_matrix(window_size=5):
    pre_gaussian_kerner = np.zeros((window_size, window_size))
    pre_gaussian_kerner[int(window_size / 2), int(window_size / 2)] = 1.0

    gaussian_kernel = cv2.GaussianBlur(pre_gaussian_kerner, (window_size, window_size), 1)
    return gaussian_kernel


def get_window_in_image(image, x_center_index, y_center_index, half_window_size):
    return image[x_center_index - half_window_size: x_center_index + half_window_size + 1,
           y_center_index - half_window_size: y_center_index + half_window_size + 1]


def sum_gaussian_kernel_in_window(image, x_center_index, y_center_index, half_window_size, gaussian_kernel):
    window = get_window_in_image(image, x_center_index, y_center_index, half_window_size)

    return np.sum(gaussian_kernel * window)


def create_R(image_path, output_file_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image_grad_x, image_grad_y = calculate_derivatives(image, 3)

    squared_grad_x = image_grad_x * image_grad_x
    grad_x_multiply_grad_y = image_grad_x * image_grad_y
    squared_grad_y = image_grad_y * image_grad_y

    window_size = 7
    half_window_size = int(window_size / 2)

    gaussian_kernel = get_gaussian_kernel_matrix(window_size)

    R = np.zeros(image.shape, dtype=np.float32)
    alpha = 0.04

    for x_index in range(half_window_size, image.shape[0] - half_window_size):
        for y_index in range(half_window_size, image.shape[1] - half_window_size):
            summed_gaussed_squared_x = sum_gaussian_kernel_in_window(squared_grad_x, x_index, y_index, half_window_size,
                                                                     gaussian_kernel)
            summed_gaussed_squared_y = sum_gaussian_kernel_in_window(squared_grad_y, x_index, y_index, half_window_size,
                                                                     gaussian_kernel)
            summed_gaussed_grad_x_multiply_grad_y = sum_gaussian_kernel_in_window(
                grad_x_multiply_grad_y, x_index, y_index, half_window_size, gaussian_kernel)

            M = np.array([
                [summed_gaussed_squared_x, summed_gaussed_grad_x_multiply_grad_y],
                [summed_gaussed_grad_x_multiply_grad_y, summed_gaussed_squared_y]
            ])

            det_M = np.linalg.det(M)
            trace_M = np.trace(M)

            R[x_index, y_index] = det_M - alpha * np.power(trace_M, 2)

    norm_R = cv2.normalize(R, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite(output_file_path, norm_R)


def ps4_1_b():
    create_R('transA.jpg', 'output/ps4-1-b-1.jpg')
    create_R('transB.jpg', 'output/ps4-1-b-2.jpg')
    create_R('simA.jpg', 'output/ps4-1-b-3.jpg')
    create_R('simB.jpg', 'output/ps4-1-b-4.jpg')


def threshold_and_non_maximal_suppression(image_path, output_path, threshold, radius):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold = image.max() * threshold

    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)

    non_maximal_suppressioned_image = np.zeros_like(thresholded_image)
    half_radius = int(radius / 2)

    for x_index in range(half_radius, non_maximal_suppressioned_image.shape[0] - half_radius):
        for y_index in range(half_radius, non_maximal_suppressioned_image.shape[1] - half_radius):
            window = get_window_in_image(image, x_index, y_index, half_radius)

            max_value_in_window = window.max()
            if thresholded_image[x_index, y_index] == max_value_in_window:
                non_maximal_suppressioned_image[x_index, y_index] = max_value_in_window

    cv2.imshow('image', non_maximal_suppressioned_image)
    cv2.imwrite(output_path, non_maximal_suppressioned_image)


def ps_1_c():
    threshold_and_non_maximal_suppression('output/ps4-1-b-1.jpg', 'output/ps4-1-c-1.jpg', 0.25, 11)
    threshold_and_non_maximal_suppression('output/ps4-1-b-2.jpg', 'output/ps4-1-c-2.jpg', 0.25, 11)
    threshold_and_non_maximal_suppression('output/ps4-1-b-3.jpg', 'output/ps4-1-c-3.jpg', 0.25, 11)
    threshold_and_non_maximal_suppression('output/ps4-1-b-4.jpg', 'output/ps4-1-c-4.jpg', 0.25, 11)


def calculate_angle_image(grad_x_image, grad_y_image):
    angle_image = np.zeros_like(grad_x_image)

    for x_index in range(grad_x_image.shape[0]):
        for y_index in range(grad_x_image.shape[1]):
            y_grad_value = grad_y_image[x_index, y_index]
            x_grad_value = grad_x_image[x_index, y_index]

            angle_image[x_index, y_index] = math.atan2(y_grad_value, x_grad_value)

    return angle_image


def generate_key_points(source_image, harris_corners_image):
    grad_x, grad_y = calculate_derivatives(source_image, 11)
    angle_image = calculate_angle_image(grad_x, grad_y)

    threshold_value = 0.25 * harris_corners_image.max()
    _, harris_corners_image = cv2.threshold(harris_corners_image, threshold_value, 255, cv2.THRESH_BINARY)

    coordinates_indices_seperated = np.where(harris_corners_image == 255)
    coordinates = zip(coordinates_indices_seperated[0], coordinates_indices_seperated[1])

    key_points = []

    for (x_index, y_index) in coordinates:
        angle = angle_image[x_index, y_index]
        key_point = cv2.KeyPoint(float(y_index), float(x_index), size=20, angle=angle, octave=0)
        key_points.append(key_point)

    return key_points


def get_image_of_key_points_with_angle(source_image, harris_corners_image):
    key_points = generate_key_points(source_image, harris_corners_image)

    source_image_with_key_points = cv2.drawKeypoints(source_image, key_points, None, (255, 0, 0),
                                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return source_image_with_key_points


def ps4_2_a():
    transA = cv2.imread('transA.jpg', cv2.IMREAD_GRAYSCALE)
    transB = cv2.imread('transB.jpg', cv2.IMREAD_GRAYSCALE)

    transA_harris_corners_image = cv2.imread('output/ps4-1-c-1.jpg', cv2.IMREAD_GRAYSCALE)
    transB_harris_corners_image = cv2.imread('output/ps4-1-c-2.jpg', cv2.IMREAD_GRAYSCALE)

    transA_with_key_points = get_image_of_key_points_with_angle(transA, transA_harris_corners_image)
    transB_with_key_points = get_image_of_key_points_with_angle(transB, transB_harris_corners_image)

    transAB_concatenated = np.concatenate((transA_with_key_points, transB_with_key_points), axis=1)
    cv2.imshow('image', transAB_concatenated)
    cv2.imwrite('output/ps4-2-a-1.png', transAB_concatenated)

    simA = cv2.imread('simA.jpg', cv2.IMREAD_GRAYSCALE)
    simB = cv2.imread('simB.jpg', cv2.IMREAD_GRAYSCALE)

    simA_harris_corners_image = cv2.imread('output/ps4-1-c-3.jpg', cv2.IMREAD_GRAYSCALE)
    simB_harris_corners_image = cv2.imread('output/ps4-1-c-4.jpg', cv2.IMREAD_GRAYSCALE)

    simA_with_key_points = get_image_of_key_points_with_angle(simA, simA_harris_corners_image)
    simB_with_key_points = get_image_of_key_points_with_angle(simB, simB_harris_corners_image)

    simAB_concatenated = np.concatenate((simA_with_key_points, simB_with_key_points), axis=1)
    cv2.imshow('image', simAB_concatenated)
    cv2.imwrite('output/ps4-2-a-2.png', simAB_concatenated)


def create_matching_image(image1, image2, key_points1, key_points2, points_matches: cv2.DMatch):
    pair_image = np.concatenate((image1, image2), axis=1)

    for match in points_matches:
        image1_key_point = key_points1[match.queryIdx]
        image2_key_point = key_points2[match.trainIdx]

        image1_coordinate = tuple(int(coordinate) for coordinate in image1_key_point.pt)
        image2_coordinate = tuple(int(coordinate) for coordinate in image2_key_point.pt)

        image2_coordinate = tuple(map(operator.add, image2_coordinate, (image1.shape[0], 0)))

        pair_image = cv2.line(pair_image, image1_coordinate, image2_coordinate, (0, 255, 0), thickness=1)

    return pair_image


def ps4_2_b():
    transA = cv2.imread('transA.jpg', cv2.IMREAD_GRAYSCALE)
    transB = cv2.imread('transB.jpg', cv2.IMREAD_GRAYSCALE)

    transA_harris_corners_image = cv2.imread('output/ps4-1-c-1.jpg', cv2.IMREAD_GRAYSCALE)
    transB_harris_corners_image = cv2.imread('output/ps4-1-c-2.jpg', cv2.IMREAD_GRAYSCALE)

    # Get key points
    transA_key_points = generate_key_points(transA, transA_harris_corners_image)
    transB_key_points = generate_key_points(transB, transB_harris_corners_image)

    # Initialize SIFT
    sift = cv2.SIFT_create()

    #   Extracting SIFT descriptors for both images
    points_transA, descriptors_transA = sift.compute(transA, transA_key_points)
    points_transB, descriptors_transB = sift.compute(transB, transB_key_points)

    #   Initialize matcher
    bfm = cv2.BFMatcher()
    matches = bfm.match(descriptors_transA, descriptors_transB)

    matching_image_trans = create_matching_image(transA, transB, points_transA, points_transB, matches)
    cv2.imwrite('output/ps4-2-b-1.jpg', matching_image_trans)


    simA = cv2.imread('simA.jpg', cv2.IMREAD_GRAYSCALE)
    simB = cv2.imread('simB.jpg', cv2.IMREAD_GRAYSCALE)

    simA_harris_corners_image = cv2.imread('output/ps4-1-c-3.jpg', cv2.IMREAD_GRAYSCALE)
    simB_harris_corners_image = cv2.imread('output/ps4-1-c-4.jpg', cv2.IMREAD_GRAYSCALE)

    # Get key points
    simA_key_points = generate_key_points(simA, simA_harris_corners_image)
    simB_key_points = generate_key_points(simB, simB_harris_corners_image)

    #   Extracting SIFT descriptors for both images
    points_simA, descriptors_simA = sift.compute(simA, simA_key_points)
    points_simB, descriptors_simB = sift.compute(simB, simB_key_points)

    #   Initialize matcher
    matches = bfm.match(descriptors_simA, descriptors_simB)

    matching_image_sim = create_matching_image(simA, simB, points_simA, points_simB, matches)
    cv2.imwrite('output/ps4-2-b-2.jpg', matching_image_sim)


def calculate_consensus_set_by_match(matches, dx, dy, key_points1, key_points2, tolerance=10):
    consensus_set = []

    for match in matches:
        (x1, y1) = key_points1[match.queryIdx].pt
        (x2, y2) = key_points2[match.trainIdx].pt

        match_dx = x1 - x2
        match_dy = y1 - y2

        if abs(dx - match_dx) < tolerance and abs(dy - match_dy) < tolerance:
            consensus_set.append(match)

    return consensus_set


def find_consensus_set(matches, key_points1, key_points2):
    max_consensus_set_size = 0
    best_consensus_set = None

    for match in matches:
        (x1, y1) = key_points1[match.queryIdx].pt
        (x2, y2) = key_points2[match.trainIdx].pt

        dx = x1 - x2
        dy = y1 - y2

        consensus_set = calculate_consensus_set_by_match(matches, dx, dy, key_points1, key_points2)
        if len(consensus_set) > max_consensus_set_size:
            max_consensus_set_size = len(consensus_set)
            best_consensus_set = consensus_set

    return best_consensus_set


def ps4_3_a():
    transA = cv2.imread('transA.jpg', cv2.IMREAD_GRAYSCALE)
    transB = cv2.imread('transB.jpg', cv2.IMREAD_GRAYSCALE)

    transA_harris_corners_image = cv2.imread('output/ps4-1-c-1.jpg', cv2.IMREAD_GRAYSCALE)
    transB_harris_corners_image = cv2.imread('output/ps4-1-c-2.jpg', cv2.IMREAD_GRAYSCALE)

    # Get key points
    transA_key_points = generate_key_points(transA, transA_harris_corners_image)
    transB_key_points = generate_key_points(transB, transB_harris_corners_image)

    # Initialize SIFT
    sift = cv2.SIFT_create()

    #   Extracting SIFT descriptors for both images
    points_transA, descriptors_transA = sift.compute(transA, transA_key_points)
    points_transB, descriptors_transB = sift.compute(transB, transB_key_points)

    #   Initialize matcher
    bfm = cv2.BFMatcher()
    matches = bfm.match(descriptors_transA, descriptors_transB)

    consensus_set = find_consensus_set(matches, points_transA, points_transB)

    matched_image = np.array([])
    # draw matches with biggest consensus
    matched_image = cv2.drawMatches(transA, points_transA, transB, points_transB, consensus_set,
                                    flags=2, outImg=matched_image)

    cv2.imwrite('output/ps4-3-a-1.png', matched_image)
    # cv2.imshow('image', matched_image)
    # cv2.waitKey(0)


ps4_3_a()
