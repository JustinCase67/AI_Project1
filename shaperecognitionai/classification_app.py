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
from knnengine import KNNEngine

from __feature__ import snake_case, true_property


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


class QParameterPicker(QWidget):
    def __init__(self, *parameters: Parameter):
        super().__init__()
        self.__widget_title = QLabel()
        self.__widget_title.text = "Parameters"

        self.__central_layout = QVBoxLayout()
        for _ in parameters:
            self.__central_layout.add_widget(QParameter(_))
        self.set_layout(self.__central_layout)


class QParameter(QWidget):
    def __init__(self, parameter: Parameter):
        super().__init__()
        self.__parameter_scroll_bar = QScrollBar()
        self.__parameter_value = QLabel()

        parameter_layout = QHBoxLayout()
        parameter_layout.add_layout(
            self.__create_parameter(parameter,
                                    self.__parameter_scroll_bar,
                                    self.__parameter_value))

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
        print(parameter.current)


class QClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # valeurs de test
        self.__k = Parameter("K", 5, 20, 0)
        self.__max_distance = Parameter("Max distance", 0, 100, 0)
        self.__current_data_set = None
        # fin des valeurs de test
        self.__window_title = "Klustr KNN Classification"
        self.__window_width = 1024
        self.__window_height = 768
        self.resize(self.__window_width, self.__window_height)
        self.__knn_engine = KNNEngine()
        self.__color_sequence = QColorSequence()

        # Gestion DB
        credential = PostgreSQLCredential(password='firen123')
        self.__klustr_dao = PostgreSQLKlustRDAO(credential)

        # Génération du bouton About et du menu
        __data = self.__klustr_dao.available_datasets
        """mylist = [klustr_dao.labels_from_dataset(i[1]) for i in __data]
        print(mylist)
        
        mylist2 = [(klustr_dao.image_from_dataset(i[1], False), klustr_dao.image_from_dataset(i[1], True)) for i in __data]
        print(mylist2)
        
        mylist2 = [klustr_dao.image_from_label(mylist[i[0]]) for i in __data]
        print(mylist2)"""

        __items = [f"{i[1]} [{i[5]}] [{i[8]}]" for i in __data]

        self.__dataset = QGroupBox("Dataset")
        self.__dataset_layout = QVBoxLayout(self.__dataset)
        self.__dataset_dropmenu = QComboBox()
        self.__dataset_dropmenu.insert_items(0, __items)
        self.__dataset_dropmenu.activated.connect(
            lambda: self.update_data_set(__items[self.__dataset_dropmenu.current_index].split(maxsplit=1)[0]))
        self.__dataset_group_layout = QHBoxLayout(self.__dataset)
        self.__dataset_group1 = QGroupBox("Included in Data Set")
        self.__dataset_group2 = QGroupBox("Transformation")
        self.__dataset_group_layout.add_widget(self.__dataset_group1)
        self.__dataset_group_layout.add_widget(self.__dataset_group2)
        self.__dataset_layout.add_widget(self.__dataset_dropmenu)
        self.__dataset_layout.add_layout(self.__dataset_group_layout)

        self.__single_test = QGroupBox("Single Test")
        self.__single_test_layout = QVBoxLayout(self.__single_test)
        self.__single_test_dropmenu = QComboBox()
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

        self.__parameters = QGroupBox("KNN parameters")
        self.__parameters_layout = QVBoxLayout(self.__parameters)
        self.__parameters_layout.add_widget(QParameterPicker(self.__k, self.__max_distance))

        self.about_button = QPushButton("About", self)
        self.about_button.clicked.connect(
            lambda: self.open_dialog("About KlustR KNN Classifier", "report.txt"))  # LE ficher n'existe pas

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

        self.viewer_widget.axis_x.title = "Axe X : A / P^2"
        self.viewer_widget.axis_y.title = "Axe Y : A shape / A pseudo circumscribed circle"
        self.viewer_widget.axis_z.title = "Axe Z : A pseudo inscribed circle / A pseudo circumscribed circle"
        self.viewer_widget.title = "KlustR KNN Classification"
        self.viewer_widget.axis_y.range = (0.0, 1.0)
        self.viewer_widget.axis_x.range = (0.0, 1.0)
        self.viewer_widget.axis_z.range = (0.0, 1.0)

    @Slot()
    def open_dialog(self, title: str, source: str):
        try:
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()
                QMessageBox.about(self, title, content)
        except FileNotFoundError:
            print("Le fichier n'existe pas.")

    @Slot()
    def update_data_set(self, dataset_name):
        self.__current_data_set = self.__klustr_dao.image_from_dataset(dataset_name, False)
        self.set_raw_data(dataset_name)
        self.set_single_test_dropmenu(self.__current_data_set)
        self.test(dataset_name)

    def set_single_test_dropmenu(self, data_set):
        items = [i[3] for i in data_set]
        self.__single_test_dropmenu.clear()
        self.__single_test_dropmenu.insert_items(0, items)

    @Slot()
    def set_raw_data(self, dataset_name: str):
        # Ajoute les points
        query_result = self.__klustr_dao.image_from_dataset(dataset_name, True)
        raw_data = []
        for result in query_result:
            tag = result[1]
            result_img = qimage_argb32_from_png_decoding(result[6])
            result_nparray = ndarray_from_qimage_argb32(result_img)
            result_nparray = result_nparray ^ 1
            raw_data.append((tag, result_nparray))
        self.__knn_engine.raw_data = raw_data
        set_data = self.__knn_engine.extract_set_data()
        self.add_points(set_data, True)


    @Slot()
    def classify_image(self, current_index):
        img_data = self.__current_data_set[current_index]
        tag = img_data[1]
        result_img = qimage_argb32_from_png_decoding(img_data[6])
        result_nparray = ndarray_from_qimage_argb32(result_img)
        result_nparray = result_nparray ^ 1
        self.__knn_engine.img_data = (tag, result_nparray)
        extracted_image_data = self.__knn_engine.extract_image_data()
        #self.add_points(extracted_image_data, False)
        return self.__knn_engine.calculate_distance()

    def add_points(self, set_data, is_cloud):
        if is_cloud:
            sets = []
            for i in range(len(self.__knn_engine.get_known_forms())):
                indices = np.where(set_data[:, 3] == i)[0]
                valid_rows_view = set_data[indices]
                sets.append(valid_rows_view)
                print("valid_rows index", i, valid_rows_view[:, :-1])
            self.viewer_widget.clear()
            for i in range(len(self.__knn_engine.get_known_forms())):
                self.viewer_widget.add_serie(sets[i][:, :-1], self.__color_sequence.next(),
                                             title=self.__knn_engine.get_known_forms()[i], size_percent=0.05)

        else:
            self.viewer_widget.remove_serie('unkown shape')
            self.viewer_widget.add_serie(set_data[:-1].reshape(1, -1), QColor("black"), title="unkown shape", size_percent=0.05)

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
        file.write(str(compteur) + ' GOOD OUT OF' + str(
            len(self.__current_data_set)) + '\n')


def main():
    app = QApplication(sys.argv)
    window = QClassificationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
