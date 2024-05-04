import numpy as np

class util:
    @staticmethod
    def centroid_from_coordinates(coordinates):
        num_points = len(coordinates)
        if num_points == 0:
            return None

        coordinates_array = np.array(coordinates)
        centroid = np.mean(coordinates_array, axis=0)
        return centroid
    @staticmethod
    def euclidean_distance_squared(point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        return (np.sum((point1 - point2) ** 2))