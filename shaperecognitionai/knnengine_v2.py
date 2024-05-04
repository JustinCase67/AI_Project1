import numpy as np
import numpy.typing as npt

import util
from feature_extraction import FeatureExtractor


class KNNEngine:
    def __init__(self):
        self.__k = Parameter("K", 1, 10, 3)
        self.__max_distance = Parameter("Max distance", 0, 1, 1)
        self.__training_data = None  # vide au debut, change apres extract (grosseur *4, on veut le tag qui est le type complexe)
        self.__known_categories = []

    @property
    def k(self):
        return self.__k

    @property
    def max_distance(self):
        return self.__max_distance

    @property
    def training_data(self):
        return self.__training_data

    @training_data.setter
    def training_data(self, data):
        self.__training_data = data

    @property
    def known_categories(self):
        return self.__known_categories

    def __lookup_categorie(self, tag: str) -> int:
        if not self.__known_categories:
            self.__known_categories.append("Undefined")
        if tag not in self.__known_categories:
            self.__known_categories.append(tag)
        return self.__known_categories.index(tag)

    def prepare_data(self, raw_data, raw: bool):
        extracted_data = np.zeros(4)
        metrics = FeatureExtractor.get_metrics(raw_data[1])
        extracted_data[:len(metrics)] = metrics
        if raw:
            tag = self.__lookup_categorie(raw_data[0])
            extracted_data[-1] = tag
        return extracted_data

    def assess_data_distance(self, test_image):
        training_data_metrics = self.__training_data[:, :-1]
        img_metrics = test_image[:-1]
        metrics_distance = np.zeros(len(self.__training_data))
        for i, metric in enumerate(training_data_metrics):
            metrics_distance[i] = util.euclidean_distance_squared(metric,
                                                                  img_metrics)
        return metrics_distance

    def get_neighbor(self, distances):
        acceptable_distances = np.where(
            distances <= self.__max_distance.current)
        return np.argsort(acceptable_distances)[:self.__k.current]

    def get_tags_index(self, neighbor):
        tags_index = np.zeros(len(neighbor), dtype=np.int64)
        for i, neighb in enumerate(neighbor):
            tags_index[i] = self.__training_data[neighb][-1]
        return tags_index

    def classify(self, test_image):
        distances = self.assess_data_distance(test_image)
        neighbor = self.get_neighbor(distances)
        try:
            tags_index = self.get_tags_index(neighbor)
            unique_values, counts = np.unique(tags_index, return_counts=True)
            unique_values_same_occurrences = unique_values[
                counts == counts.max()]
            if len(unique_values_same_occurrences) > 1:
                result = self.tie_breaker(test_image, neighbor,
                                          unique_values_same_occurrences)
            else:
                result = unique_values_same_occurrences[0]
            return self.__known_categories[result]
        except IndexError:
            return "Undefined"

    def tie_breaker(self, test_image, neighbor, ties):
        metrics = []
        tags = []
        for n in neighbor:
            if self.__training_data[n][-1] in ties:
                metrics.append(self.__training_data[n][:-1])
                tags.append(int(self.__training_data[n][-1]))
        distances = util.distance_from_centroid(metrics, tags,
                                                test_image[:-1])
        return np.argmin(distances)


class Parameter:
    def __init__(self, name: str, min: int, max: int, current: int):
        self.__name = name
        self.__min = min
        self.__max = max
        self.__current = current

    @property
    def current(self):
        return self.__current

    @current.setter
    def current(self, value: int):
        self.__current = value

    @property
    def name(self):
        return self.__name

    @property
    def min(self):
        return self.__min

    @property
    def max(self):
        return self.__max

    @max.setter
    def max(self, max):
        self.__max = max
