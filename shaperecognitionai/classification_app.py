import sys
import numpy

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QScreen, QPixmap, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox
from PySide6.QtWidgets import QWidget, QLabel, QScrollBar, QGroupBox, QComboBox
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout

from db_credential import PostgreSQLCredential
from klustr_dao import PostgreSQLKlustRDAO
from scatter_3d_viewer import QScatter3dViewer
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
        parameter_layout.add_layout(self.__create_parameter(parameter,self.__parameter_scroll_bar,self.__parameter_value))

        self.set_layout(parameter_layout)

    def __create_parameter(self, parameter, scroll_bar: QScrollBar,
                           value_label: QLabel) -> QHBoxLayout:
        parameter_label = QLabel()
        parameter_label.text = parameter.name + " = "
        scroll_bar.orientation = Qt.Horizontal
        scroll_bar.set_range(parameter.min,
                             parameter.max)  # à changer pour des valeurs qui varient
        scroll_bar.value = parameter.current  # à changer pour une valeur par défaut
        scroll_bar.minimum_width = 250  # à changer pour un calcul
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
        self.__k = Parameter("K", 1, 20, 0)
        self.__max_distance = Parameter("Max distance", 0, 100, 0)
        self.__current_data_set = None
        self.__knn_engine = KNNEngine()
        self.__data_info = []
        self.__gui_maker()

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
        # DEBUT DES TESTS
        self.test(dataset_name)

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
        self.__knn_engine.processed_data = self.__knn_engine.extract_set_data()

    @Slot()
    def classify_image(self, current_index):
        img_data = self.__current_data_set[current_index]
        tag = img_data[1]
        result_img = qimage_argb32_from_png_decoding(img_data[6])
        result_nparray = ndarray_from_qimage_argb32(result_img)
        result_nparray = result_nparray ^ 1
        self.__knn_engine.img_data = (tag, result_nparray)
        self.__knn_engine.processed_image_data = self.__knn_engine.extract_image_data()
        return self.__knn_engine.calculate_distance()

    @Slot()
    def get_data_info(self, index):
        selected_text = self.__dataset_dropmenu.item_text(index)
        for data in self.__klustr_dao.available_datasets:
            if data[1] in selected_text:
                self.__data_info = data
                test = data

    @Slot()
    def update_labels(self):
        self.category_count_value.text = str(self.__data_info[5])
        self.training_count_value.text = str(self.__data_info[6])
        self.test_count_value.text = str(self.__data_info[7])
        self.total_count_value.text = str(self.__data_info[8])
        self.translated_bool_value.text = str(self.__data_info[2])
        self.rotated_bool_value.text = str(self.__data_info[3])
        self.scaled_bool_value.text = str(self.__data_info[4])

    def __gui_maker(self):
        self.set_window_title("Klustr KNN Classification")
        self.resize(1024, 768)

        self.__dataset = self.__create_dataset_group()

        self.__single_test = self.__create_single_test()

        self.__parameters = self.__create_parameters_group()

        self.about_button = self.__about_button()

        menu_layout = self.__create_menu(self.__dataset, self.__single_test, self.__parameters, self.about_button)
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        viewer_widget = QScatter3dViewer(parent=central_widget)
        central_layout.add_layout(menu_layout)
        central_layout.add_widget(viewer_widget)
        central_widget.set_layout(central_layout)
        self.set_central_widget(central_widget)


    def __connect_signals(self):
        #self.__dataset_dropmenu.activated.connect(self.__update_data_set_info)
        #self.__single_test_button.clicked.connect(self.__classify_single_test)
        pass

    def __initialize_menu(self, layout):
        pass
    def ___dataset_group(self):
        # Premier groupe de boîtes
        group = QGroupBox("Dataset")
        layout1 = QVBoxLayout(group1)


        return group

    def __get_dropmenu(self):
        credential = PostgreSQLCredential(password='Ravens522!')
        self.__klustr_dao = PostgreSQLKlustRDAO(credential)
        __data = self.__klustr_dao.available_datasets
        __items = [f"{i[1]} [{i[5]}] [{i[8]}]" for i in __data]
        dropmenu = QComboBox()
        dropmenu.insert_items(0, __items)
        dropmenu.activated.connect(lambda index: self.get_data_info(index))
        dropmenu.activated.connect(lambda: self.update_labels())
        return dropmenu

    def __get_info(self):
        group = QGroupBox("Included in Data Set")
        group_layout = QVBoxLayout()  # Layout vertical pour le groupe
        self.__info_layout = QVBoxLayout(group)  # Layout en grille pour aligner les étiquettes et les valeurs
        self.__info_category = self.category_widget(self.__info_layout)
        self.__info_training = self.training_widget(self.__info_layout)
        self.__info_test = self.test_widget(self.__info_layout)
        self.__info_total = self.total_widget(self.__info_layout)
        group_layout.add_layout(self.__info_layout)
        group.set_layout(group_layout)
        return group

    def category_widget(self, layout):
        category = QWidget()
        layout.add_widget(category)
        category_layout = QHBoxLayout(category)
        category_count_label = QLabel("Category count:")
        self.category_count_value = QLabel(self)
        category_layout.add_widget(category_count_label)
        category_layout.add_widget(self.category_count_value)
        return category

    def training_widget(self, layout):
        training = QWidget()
        layout.add_widget(training)
        training_layout = QHBoxLayout(training)
        training_count_label = QLabel("Training image count:")
        self.training_count_value = QLabel()
        training_layout.add_widget(training_count_label)
        training_layout.add_widget(self.training_count_value)
        return training

    def test_widget(self, layout):
        test = QWidget()
        layout.add_widget(test)
        test_layout = QHBoxLayout(test)
        test_count_label = QLabel("Test image count:")
        self.test_count_value = QLabel()
        test_layout.add_widget(test_count_label)
        test_layout.add_widget(self.test_count_value)
        return test

    def total_widget(self, layout):
        total = QWidget()
        layout.add_widget(total)
        total_layout = QHBoxLayout(total)
        total_count_label = QLabel("Total image count:")
        self.total_count_value = QLabel()
        total_layout.add_widget(total_count_label)
        total_layout.add_widget(self.total_count_value)
        return total

    def __get_transformation(self):
        group = QGroupBox("Transformation")
        group_layout = QVBoxLayout()
        self.__transformation_layout = QVBoxLayout(group)
        self.__translation = self.translation_widget(self.__transformation_layout)
        self.__rotation = self.rotation_widget(self.__transformation_layout)
        self.__scale = self.scale_widget(self.__transformation_layout)
        group_layout.add_layout(self.__info_layout)
        group.set_layout(group_layout)
        return group

    def translation_widget(self, layout):
        translation = QWidget()
        layout.add_widget(translation)
        translation_layout = QHBoxLayout(translation)
        translated_label = QLabel("Translated:")
        self.translated_bool_value = QLabel()
        translation_layout.add_widget(translated_label)
        translation_layout.add_widget(self.translated_bool_value)
        return translation

    def rotation_widget(self, layout):
        rotation = QWidget()
        layout.add_widget(rotation)
        rotation_layout = QHBoxLayout(rotation)
        rotated_label = QLabel("Rotated :")
        self.rotated_bool_value = QLabel()
        rotation_layout.add_widget(rotated_label)
        rotation_layout.add_widget(self.rotated_bool_value)
        return rotation

    def scale_widget(self, layout):
        scale = QWidget()
        layout.add_widget(scale)
        scale_layout = QHBoxLayout(scale)
        scaled_label = QLabel("Scaled :")
        self.scaled_bool_value = QLabel()
        scale_layout.add_widget(scaled_label)
        scale_layout.add_widget(self.scaled_bool_value)

    def __create_dataset_group(self):
        dataset = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset)
        self.__dataset_dropmenu = self.__get_dropmenu()
        self.__dataset_group_layout = QHBoxLayout(dataset)
        dataset_group1 = self.__get_info()
        dataset_group2 = self.__get_transformation()
        self.__dataset_group_layout.add_widget(dataset_group1)
        self.__dataset_group_layout.add_widget(dataset_group2)
        dataset_layout.add_widget(self.__dataset_dropmenu)
        dataset_layout.add_layout(self.__dataset_group_layout)
        return dataset

    def __create_single_test(self):
        single_test = QGroupBox("Single Test")
        single_test_layout = QVBoxLayout(single_test)
        # ------------------------------------------------------------------------------------
        self.__single_test_dropmenu = QComboBox()
        self.__single_test_view_label = QLabel()
        self.__single_test_view_label.style_sheet = 'QLabel { background-color : #313D4A; padding : 10px 10px 10px 10px; }'  # 354A64
        self.__single_test_view_label.alignment = Qt.AlignCenter
        self.__single_test_button = QPushButton("Classify", self)
        self.__single_test_button.clicked.connect(lambda: self.classify_image(self.__single_test_dropmenu.current_index))
        self.__single_test_result_label = QLabel()
        single_test_layout.add_widget(self.__single_test_dropmenu)
        single_test_layout.add_widget(self.__single_test_view_label)
        single_test_layout.add_widget(self.__single_test_button)
        return single_test

    def __create_parameters_group(self):
        group = QGroupBox("KNN parameters")
        group.set_fixed_height(125)
        layout = QVBoxLayout(group)
        layout.add_widget(QParameterPicker(self.__k, self.__max_distance))
        group.set_layout(layout)
        return group

    def __about_button(self):
        button = QPushButton("About", self)
        button.clicked.connect(lambda: self.open_dialog("About KlustR KNN Classifier", "report.txt"))  # LE ficher n'existe pas
        return button

    def __create_menu(self, widget1, widget2, widget3, widget4):
        menu = QVBoxLayout()
        menu.add_widget(widget1)
        menu.add_widget(widget2)
        menu.add_widget(widget3)
        menu.add_widget(widget4)
        return menu




def main():
    app = QApplication(sys.argv)
    window = QClassificationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
