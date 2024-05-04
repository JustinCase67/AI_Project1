import sys
import numpy
import numpy as np

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QScreen, QPixmap, QImage, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox
from PySide6.QtWidgets import QWidget, QLabel, QScrollBar, QGroupBox, QComboBox
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout

from db_credential import PostgreSQLCredential
from klustr_dao import PostgreSQLKlustRDAO
from scatter_3d_viewer import QScatter3dViewer, QColorSequence
from klustr_utils import *
from klustr_dao import *
from knnengine_v2 import KNNEngine, Parameter

from __feature__ import snake_case, true_property

class QParameterPicker(QWidget):
    def __init__(self, *parameters: Parameter):
        super().__init__()
        self.__widget_title = QLabel()
        self.__widget_title.text = "Parameters"
        self.__test = []

        self.__central_layout = QVBoxLayout()
        for _ in parameters:
            p = QParameter(_)
            self.__test.append(p)
            self.__central_layout.add_widget(p)
        self.set_layout(self.__central_layout)



class QParameter(QWidget):
    def __init__(self, parameter: Parameter):
        super().__init__()
        self.parameter_scroll_bar = QScrollBar()
        self.parameter_value = QLabel()
        self.parameter_title = parameter.name

        parameter_layout = QHBoxLayout()
        parameter_layout.add_layout(
            self.__create_parameter(parameter,
                                    self.parameter_scroll_bar,
                                    self.parameter_value))

        self.set_layout(parameter_layout)


    def __create_parameter(self, parameter, scroll_bar: QScrollBar,
                           value_label: QLabel) -> QHBoxLayout:
        parameter_label = QLabel()
        parameter_label.text = parameter.name + " = "
        scroll_bar.orientation = Qt.Horizontal
        scroll_bar.set_range(parameter.min,
                             parameter.max)  # à changer pour des valeurs qui varient
        scroll_bar.value = parameter.current  # à changer pour une valeur par défaut
        scroll_bar.minimum_width = 50  # à changer pour un calcul
        value_label.set_num(scroll_bar.value)
        scroll_bar.valueChanged.connect(value_label.set_num)
        scroll_bar.valueChanged.connect(
            lambda value: self.set_current(parameter, value))

        layout = QHBoxLayout()
        layout.add_widget(parameter_label)
        layout.add_widget(value_label)
        layout.add_widget(scroll_bar)

        return layout

    @Slot()
    def set_current(self, parameter: Parameter, value: int):
        parameter.current = value


class QClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # valeurs de test
        self.__current_data_set = None
        # fin des valeurs de test
        self.__window_title = "Klustr KNN Classification"
        self.__window_width = 1024
        self.__window_height = 768
        self.resize(self.__window_width, self.__window_height)
        self.__knn_engine = KNNEngine()
        self.__color_sequence = QColorSequence()

        # Gestion DB
        credential = PostgreSQLCredential(password='AAAaaa123')
        self.__klustr_dao = PostgreSQLKlustRDAO(credential)

        # Génération du bouton About et du menu
        self.__data = self.__klustr_dao.available_datasets
        """mylist = [klustr_dao.labels_from_dataset(i[1]) for i in __data]
        print(mylist)
        
        mylist2 = [(klustr_dao.image_from_dataset(i[1], False), klustr_dao.image_from_dataset(i[1], True)) for i in __data]
        print(mylist2)
        
        mylist2 = [klustr_dao.image_from_label(mylist[i[0]]) for i in __data]
        print(mylist2)"""
        __items = [f"{i[1]} [{i[5]}] [{i[8]}]" for i in self.__data]
        

        self.__dataset = QGroupBox("Dataset")
        self.__dataset_layout = QVBoxLayout(self.__dataset)
        self.__dataset_dropmenu = QComboBox()
        self.__dataset_dropmenu.insert_items(0, __items)
        self.__dataset_dropmenu.activated.connect(
            lambda: self.update_data_set(__items[self.__dataset_dropmenu.current_index].split(maxsplit=1)[0]))
        self.__dataset_group_layout = QHBoxLayout(self.__dataset)
        # a encapsuler mieux
        self.__dataset_group1 = QGroupBox("Included in Data Set")
        self.__dataset_group2 = QGroupBox("Transformation")
        self.__dataset_group_layout.add_widget(self.__dataset_group1)
        self.__dataset_group_layout.add_widget(self.__dataset_group2)
        self.__dataset_layout.add_widget(self.__dataset_dropmenu)
        self.__dataset_layout.add_layout(self.__dataset_group_layout)

        self.__single_test = QGroupBox("Single Test")
        self.__single_test_layout = QVBoxLayout(self.__single_test)
        self.__single_test_dropmenu = QComboBox()
        self.__single_test_dropmenu.activated.connect(lambda: self.set_thumbnail(self.__single_test_dropmenu.current_index))
        self.__single_test_view_label = QLabel()
        self.__single_test_view_label.style_sheet = 'QLabel { background-color : #313D4A; padding : 10px 10px 10px 10px; }'  # 354A64
        self.__single_test_view_label.alignment = Qt.AlignCenter
        self.__single_test_button = QPushButton("Classify", self)
        self.__single_test_button.clicked.connect(
            lambda: self.classify_image(self.__single_test_dropmenu.current_index))
        self.__single_test_result_label = QLabel()
        self.__single_test_layout.add_widget(self.__single_test_dropmenu)
        self.__single_test_layout.add_widget(self.__single_test_view_label)
        self.__single_test_layout.add_widget(self.__single_test_button)
        self.__single_test_result = QLabel()
        self.__single_test_result.text = "not classified"
        self.__single_test_result.alignment = Qt.AlignCenter
        self.__single_test_layout.add_widget(self.__single_test_result)

        self.__parameters = QGroupBox("KNN parameters")
        self.__parameters_layout = QVBoxLayout(self.__parameters)
        self.__parameters_picker = QParameterPicker(self.__knn_engine.k, self.__knn_engine.max_distance)
        self.__parameters_layout.add_widget(self.__parameters_picker)


        self.about_button = QPushButton("About", self)
        self.about_button.clicked.connect(
            lambda: self.open_dialog("About KlustR KNN Classifier", "report.txt"))

        menu_layout = QVBoxLayout()
        menu_layout.add_widget(self.__dataset)
        menu_layout.add_widget(self.__single_test)
        menu_layout.add_widget(self.__parameters)
        menu_layout.add_widget(self.about_button)
        # menu_layout.add_widget(
        #    QParameterPicker(self.__k, self.__max_distance))
        # menu_layout.add_widget(self.about_button)
        # menu_widget = QWidget()
        # menu_widget.set_layout(menu_layout)

        # Combinaison du menu et du widget viewer
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        self.viewer_widget = QScatter3dViewer(parent=central_widget)
        central_layout.add_layout(menu_layout)
        central_layout.add_widget(self.viewer_widget)
        central_widget.set_layout(central_layout)
        self.set_central_widget(central_widget)

        # Initialisations
        self.update_data_set('ABC')
        #self.set_thumbnail(self.__single_test_dropmenu.current_index)

    @Slot()
    def open_dialog(self, title: str, source: str):
        try:
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()
                QMessageBox.about(self, title, content)
        except FileNotFoundError:
            print("Le fichier n'existe pas.")

    @Slot()
    def set_thumbnail(self, index):
        thumbnail = self.__current_data_set[index][6]
        img = qimage_argb32_from_png_decoding(thumbnail)
        self.__single_test_view_label.pixmap = QPixmap.from_image(img)
        self.__single_test_result.text = "not classified" # devrait etre ailleurs

    @Slot()
    def update_data_set(self, dataset_name):
        self.__current_data_set = self.__klustr_dao.image_from_dataset(dataset_name, False)
        self.set_training_data(dataset_name)
        self.set_single_test_dropmenu(self.__current_data_set)
        #self.test(dataset_name)
        self.set_thumbnail(self.__single_test_dropmenu.current_index)


    def set_single_test_dropmenu(self, data_set):
        items = [i[3] for i in data_set]
        self.__single_test_dropmenu.clear()
        self.__single_test_dropmenu.insert_items(0, items)

    def convert_query_to_img_tuple(self, query_result):
        tag = query_result[1]
        result_img = qimage_argb32_from_png_decoding(query_result[6])
        img_nparray = ndarray_from_qimage_argb32(result_img) ^ 1
        return tag, img_nparray

    @Slot()
    def set_training_data(self, dataset_name: str):
        query_result = self.__klustr_dao.image_from_dataset(dataset_name, True)
        self.__knn_engine.training_data = np.zeros([len(query_result), 4])
        for i, result in enumerate(query_result):
            raw_data = self.convert_query_to_img_tuple(result)
            self.__knn_engine.training_data[i] = self.__knn_engine.prepare_data(raw_data, True)
        self.add_points(self.__knn_engine.training_data, True)
        self.set_k_max(len(self.__knn_engine.training_data))

    def set_k_max(self, new_max):
        self.__knn_engine.k.max = new_max
        params = self.__parameters_picker.find_children(QParameter)
        for p in params:
            if p.parameter_title == self.__knn_engine.k.name:
                print('MAX', self.__knn_engine.k.max)
                p.parameter_scroll_bar.set_range(self.__knn_engine.k.min,
                                                 self.__knn_engine.k.max)

    @Slot()
    def classify_image(self, current_index):
        raw_img_data = self.convert_query_to_img_tuple(self.__current_data_set[current_index])
        extracted_img_data = self.__knn_engine.prepare_data(raw_img_data, False)
        self.add_points(extracted_img_data, False)
        result = self.__knn_engine.classify(extracted_img_data)
        self.__single_test_result.text = result
        return result


    def add_points(self, set_data, is_cloud):
        if is_cloud:
            sets = []
            for i in range(len(self.__knn_engine.known_categories)):
                indices = np.where(set_data[:, 3] == i)[0]
                valid_rows_view = set_data[indices]
                sets.append(valid_rows_view)
                print("valid_rows index", i, valid_rows_view[:, :-1])
            self.viewer_widget.clear()
            for i in range(len(self.__knn_engine.known_categories)):
                self.viewer_widget.add_serie(sets[i][:, :-1], self.__color_sequence.next(),
                                             title=self.__knn_engine.known_categories[i])

        else:
            self.viewer_widget.remove_serie('unkown shape')
            self.viewer_widget.add_serie(set_data[:-1].reshape(1, -1), QColor("black"), title="unkown shape")

    def test(self, dataset_name):
        file = open("test_results.txt", "a")
        file.write(dataset_name + "\n")
        compteur = 0
        for i, data in enumerate(self.__current_data_set):
            response = self.classify_image(i)
            if response == data[1]:
                compteur += 1
            else:
                file.write(str(i))
                file.write(': ' + data[1])
                file.write(' wrongfully identified as ' + response + '\n')
        file.write(str(compteur) + ' GOOD OUT OF' + str(len(self.__current_data_set)) + '\n')


def main():
    app = QApplication(sys.argv)
    window = QClassificationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
