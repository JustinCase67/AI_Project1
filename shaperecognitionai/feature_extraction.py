import numpy as np
from math import pi


class FeatureExtractor:

    # aire de la forme / aire du cercle pseudo circonscrit

    @staticmethod
    def get_an_extreme_point(img, stop_value, axis, pos_start):
        extreme_tab = np.argmax(img[:] if pos_start == 0 else img[::-1, ::-1] == stop_value, axis=axis)
        mask = extreme_tab > 0
        extreme_index = np.argmax(mask)
        extreme_point = (extreme_tab[extreme_index], extreme_index)
        if pos_start:
            extreme_point = (img.shape[0] - 1 - extreme_point[0], img.shape[1] - 1 - extreme_point[1])
        return extreme_point

    @staticmethod
    def get_extreme_points_2D(img):
        extreme_left_point = FeatureExtractor.get_an_extreme_point(img, 1, 0, 0)
        extreme_right_point = FeatureExtractor.get_an_extreme_point(img, 1, 0, 1)
        extreme_top_point = FeatureExtractor.get_an_extreme_point(img, 1, 1, 0)
        extreme_bottom_point = FeatureExtractor.get_an_extreme_point(img, 1, 1, 1)
        return extreme_top_point, extreme_bottom_point, extreme_left_point, extreme_right_point

    # take two tuples of position points(x,y)
    @staticmethod
    def distance_between_two_points(centroid, point):
        return ((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2) ** 0.5

    # all tuple of position points
    @staticmethod
    def distances_from_a_point(ensemble_points, centroid_position):
        distances = np.zeros(len(ensemble_points))
        for i in range(len(ensemble_points)):
            distances[i] = FeatureExtractor.distance_between_two_points(centroid_position, ensemble_points[i])
        return distances

    @staticmethod
    def get_max_distance(distances):
        return np.max(distances)

    @staticmethod
    def get_min_distance(distances):
        return np.min(distances)

    @staticmethod
    def area_of_circle(radius):
        return pi * radius ** 2

    @staticmethod
    def ratio_area(img, max_distance_from_centroid):
        pseudo_concentric_circle_area = FeatureExtractor.area_of_circle(max_distance_from_centroid)
        return FeatureExtractor.area(img) / pseudo_concentric_circle_area

    @staticmethod
    def ratio_min_max_distances(max_distance, min_distance):
        return min_distance / max_distance

    @staticmethod
    def area(image):
        return np.sum(image)

    @staticmethod
    def centroid(image):
        c, r = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        return (np.sum(r * image), np.sum(c * image)) / FeatureExtractor.area(image)

    @staticmethod
    def create_image(size):
        return np.zeros((size[0], size[1]), dtype=np.uint8)

    @staticmethod
    def draw_rectangle(image, top_left, bottom_right):
        top, left = top_left
        bottom, right = bottom_right
        image[top:bottom + 1, left:right + 1] = 1

        # returns an image containing only the perimeter

    @staticmethod
    def transform_into_perimeter(image):
        neighbours = np.zeros((image.shape[1], image.shape[0]))
        neighbours[1:-1, 1:-1] = (image[:-2, :-2] + image[:-2, 1:-1] + image[:-2, 2:] +
                                  image[1:-1, :-2] + image[1:-1, 2:] +
                                  image[2:, :-2] + image[2:, 1:-1] + image[2:, 2:])
        neighbours = neighbours.astype(int)

        rule_test = np.array((0, 1, 1, 1, 1, 1, 1, 1, 0))

        return image * rule_test[neighbours]

    # returns the value of the perimeter
    @staticmethod
    def perimeter(image):
        return FeatureExtractor.area(FeatureExtractor.transform_into_perimeter(image))

    # return complexity of form
    @staticmethod
    def complexity(image):
        return 1 - (4*pi*FeatureExtractor.area(image)) / (FeatureExtractor.perimeter(image)**2)

    @staticmethod
    def get_diagonals_from_point(array, start_row, start_col):
        # Get dimensions of the array
        rows, cols = array.shape

        # Flatten the array
        flattened_array = array.flatten()

        # Calculate the index of the starting point in the flattened array
        start_index = start_row * cols + start_col

        # Create an array of indices for main diagonal and anti-diagonal
        diag1_indices = np.arange(min(rows - start_row, cols - start_col))
        diag2_indices = np.arange(min(start_row + 1, cols - start_col))
        diag3_indices = np.arange(min(rows - start_row, start_col + 1))
        diag4_indices = np.arange(min(start_row + 1, start_col + 1))

        # Calculate indices for main diagonal and anti-diagonal in the flattened array
        diag1_flat_indices = start_index + diag1_indices * (cols + 1)
        diag2_flat_indices = start_index - diag2_indices * (cols - 1)
        diag3_flat_indices = start_index + diag3_indices * (cols - 1)
        diag4_flat_indices = start_index - diag4_indices * (cols + 1)

        # Create views for the diagonals from the flattened array
        diag1 = flattened_array[diag1_flat_indices]
        diag2 = flattened_array[diag2_flat_indices]
        diag3 = flattened_array[diag3_flat_indices]
        diag4 = flattened_array[diag4_flat_indices]

        return diag1, diag2, diag3, diag4


    @staticmethod
    def get_views_from_point(array, start_row, start_col):
        # Get dimensions of the array
        rows, cols = array.shape

        # Flatten the array
        flattened_array = array.flatten()

        # Calculate the index of the starting point in the flattened array
        start_index = start_row * cols + start_col

        # Calculate indices for left, right, up, and down views in the flattened array
        left_indices = np.arange(start_index % cols, -1, -1)
        right_indices = np.arange(0, cols - start_index % cols)
        up_indices = np.arange(start_index // cols, -1, -1)
        down_indices = np.arange(0, rows - start_index // cols)

        # Calculate indices for left, right, up, and down views in the flattened array
        left_flat_indices = start_index - left_indices
        right_flat_indices = start_index + right_indices
        up_flat_indices = start_index - up_indices * cols
        down_flat_indices = start_index + down_indices * cols

        # Create views for the left, right, up, and down directions from the flattened array
        left_view = flattened_array[left_flat_indices[::-1]]
        right_view = flattened_array[right_flat_indices]
        up_view = flattened_array[up_flat_indices[::-1]]
        down_view = flattened_array[down_flat_indices]

        return left_view, right_view, up_view, down_view

    @staticmethod
    def get_closest_distance_from_point_2d(array, centroid_x, centroid_y):
        diagonals = FeatureExtractor.get_diagonals_from_point(array, centroid_x, centroid_y)
        lines = FeatureExtractor.get_views_from_point(array, centroid_x, centroid_y)
        nearest_one_index = []
        for i in range(len(diagonals)):
            nearest_one_index.append(np.argmax(diagonals[i][:] == (1 if array[centroid_x, centroid_y] == 0 else 0)))

        for y in range(len(lines)):
            nearest_one_index.append(np.argmax(lines[y][:] == (1 if array[centroid_x, centroid_y] == 0 else 0)))
        np_indexes = np.array(nearest_one_index)
        min_above_zero = np.argmin(np_indexes[np_indexes > 0])
        return nearest_one_index[min_above_zero]

    @staticmethod
    def get_metrics(img):
        points = FeatureExtractor.get_extreme_points_2D(img)
        centroid = FeatureExtractor.centroid(img)
        centroid_x , centroid_y = centroid
        r_centroid_x = round(centroid_x)
        r_centroid_y = round(centroid_y)
        distances = FeatureExtractor.distances_from_a_point(points, centroid)
        max_distance = FeatureExtractor.get_max_distance(distances)
        ratio_perimeter_area = FeatureExtractor.complexity(img)
        ratio_area_form_circle = FeatureExtractor.ratio_area(img, max_distance)
        min_pixel_from_point = FeatureExtractor.get_closest_distance_from_point_2d(img, r_centroid_x, r_centroid_y)
        ratio_tiny_circle_big_circle = FeatureExtractor.area_of_circle(min_pixel_from_point) / FeatureExtractor.area_of_circle(max_distance)
        return ratio_perimeter_area, ratio_area_form_circle, ratio_tiny_circle_big_circle


