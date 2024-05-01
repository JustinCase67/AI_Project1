import numpy as np
import numpy.typing as npt

from feature_extraction import FeatureExtractor


class KNNEngine:
    def __init__(self):
        self.__k = 1
        self.__distance = 1
        self.__raw_data = None  # vide au debut, property pour modif quand selection
        self.__processed_data = None  # vide au debut, change apres extract (grosseur *4, on veut le tag qui est le type complexe)
        self.__img_data = None  # vide au debut, property pour modif quand selection
        self.__processed_img_data = np.zeros(4)
        self.__known_categories = {}
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

    def extract_set_data(self):
        length = len(self.__raw_data)
        self.__processed_data = np.zeros([length, 4]) # 3 dimensions + tag, à sortir du harcodage

        for i, data in enumerate(self.__raw_data): # insérer par row et par colonne en numpy?
            metrics = FeatureExtractor.get_metrics(data[1])
            for j, metric in enumerate(metrics):
                self.__processed_data[i][j] = metrics[j]
        print("METRIQUES DATASET", self.__processed_data)


    def extract_image_data(self):
        metrics = FeatureExtractor.get_metrics(self.__img_data[1])
        for j, metric in enumerate(metrics):
            self.__processed_img_data[j] = metrics[j] # emeric pas content
        print("METRIQUES IMAGE", self.__processed_img_data)