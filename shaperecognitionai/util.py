import numpy as np


class Util:
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

    @staticmethod
    def distance_from_centroid(coordinates_tuple, index_tuples_references_tab,
                               coordinates_categorie):
        liste_centroid = []
        liste_distances = []
        for form_coordinates in coordinates_tuple:
            liste_centroid.append(Util.centroid_from_coordinates(form_coordinates))
        for centroid in liste_centroid:
            liste_distances.append(
                Util.euclidean_distance_squared(centroid, coordinates_categorie))
        return np.array(liste_distances)
