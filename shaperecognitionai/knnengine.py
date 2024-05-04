import numpy as np
import numpy.typing as npt

import tie_breaker
from feature_extraction import FeatureExtractor
from tie_breaker import neighbor_tie_breaker


class KNNEngine:
    def __init__(self):
        self.__k = 3
        self.__distance = 1
        self.__raw_data = None  # vide au debut, property pour modif quand selection
        self.__processed_data = None  # vide au debut, change apres extract (grosseur *4, on veut le tag qui est le type complexe)
        self.__img_data = None  # vide au debut, property pour modif quand selection
        self.__processed_img_data = np.zeros(4)
        self.__known_categories = []
        self.__metrics_distance = None  # index de processed_data et la valeur de la distance

    @property
    def raw_data(self):
        return self.__raw_data

    @raw_data.setter
    def raw_data(self, data):  # ajouter le type hinting
        self.__raw_data = data

    @property
    def img_data(self):
        return self.__img_data

    @img_data.setter
    def img_data(self, data):  # ajouter le type hinting
        self.__img_data = data

    @property
    def processed_data(self):
        return self.__processed_data

    @processed_data.setter
    def processed_data(self, data):  # ajouter le type hinting
        self.__processed_data = data

    @property
    def processed_img_data(self):
        return self.processed_img_data

    @processed_img_data.setter
    def processed_img_data(self, data):  # ajouter le type hinting
        self.processed_img_data = data

    def __lookup_categorie(self, tag: str) -> int:
        if tag not in self.__known_categories:
            self.__known_categories.append(tag)
        return self.__known_categories.index(tag)

    def extract_set_data(self):
        length = len(self.__raw_data)
        self.__processed_data = np.zeros(
            [length, 4])  # 3 dimensions + tag, à sortir du harcodage
        for i, data in enumerate(self.__raw_data):
            metrics = FeatureExtractor.get_metrics(data[1])
            self.__processed_data[i, :len(metrics)] = metrics
            self.__processed_data[i, -1] = self.__lookup_categorie(data[0])
        print("METRIQUES DATASET", self.__processed_data)
        return self.__processed_data
    # print(self.__known_categories)

    def get_known_forms(self):
        return self.__known_categories

    def extract_image_data(self):
        metrics = FeatureExtractor.get_metrics(self.__img_data[1])
        self.__processed_img_data[:len(metrics)] = metrics
        return self.__processed_img_data

    # print("METRIQUES IMAGE", self.__processed_img_data)

    def calculate_distance(self):
        data_metrics = self.__processed_data[:, :-1]
        img_metrics = self.__processed_img_data[:-1]
        distance = []
        for i, data in enumerate(data_metrics):
            squares_sum = tie_breaker.euclidean_distance_squared(data, img_metrics)
            distance.append(squares_sum)
        self.__metrics_distance = np.array(distance)
        print(self.__metrics_distance)
        return self.classify()

    def get_neighbor(self):
        return np.argsort(self.__metrics_distance)[:self.__k]

    def classify(self):
        neighbor = self.get_neighbor()
        print("NEIGH", neighbor)
        tags_index = np.zeros(len(neighbor), dtype=np.int64)
        for i, neighb in enumerate(neighbor):
            tags_index[i] = self.__processed_data[neighb][-1]
            # tags_index[n] = self.__processed_data[n][-1]
        print(tags_index)
        # Count occurrences of values at index 0
        unique_values, counts = np.unique(tags_index, return_counts=True)

        # Find unique values that occur the same number of times
        unique_values_same_occurrences = unique_values[counts == counts.max()]
        print("Unique values at index 0 with the same occurrences:",
              unique_values_same_occurrences)
        if len(unique_values_same_occurrences) > 1:
            metrics = []
            tags = []
            for n in neighbor:
                if self.__processed_data[n][-1] in unique_values_same_occurrences:
                    metrics.append(self.__processed_data[n][:-1])
                    tags.append(int(self.__processed_data[n][-1]))
            result = neighbor_tie_breaker(metrics, tags, self.__processed_img_data[:-1])
        else:
            result = unique_values_same_occurrences[0]
        print(self.__known_categories[result])
        return self.__known_categories[result]

