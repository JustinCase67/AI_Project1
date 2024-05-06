from typing import Tuple
import numpy.typing as npt
import numpy as np

ImageData = Tuple[
    int, str, int, str, int, int, memoryview, memoryview, bool, bool, bool, str]
ImageType = Tuple[str, npt.NDArray]


class Util:
    @staticmethod
    def centroid_from_coordinates(coordinates) -> npt.NDArray | None:
        num_points = len(coordinates)
        if num_points == 0:
            return None

        coordinates_array = np.array(coordinates)
        centroid = np.mean(coordinates_array, axis=0)
        return centroid

    @staticmethod
    def euclidean_distance_squared(point1: npt.NDArray, point2: npt.NDArray) -> float:
        point1 = np.array(point1)
        point2 = np.array(point2)
        return (np.sum((point1 - point2) ** 2))
