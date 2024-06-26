import sys
import numpy.typing as npt

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QMessageBox
from PySide6.QtWidgets import QWidget, QLabel, QScrollBar, QGroupBox, QComboBox
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout

from util import ImageData
from db_credential import PostgreSQLCredential
from scatter_3d_viewer import QScatter3dViewer, QColorSequence
from klustr_utils import *
from klustr_dao import *
from knnengine import KNNEngine, Parameter

from __feature__ import snake_case, true_property


class QParameterPicker(QWidget):
    def __init__(self, *parameters: Parameter):
        super().__init__()
        self.__widget_title = QLabel()
        self.__widget_title.text = "Parameters"

        self.__central_layout = QVBoxLayout()
        for p in parameters:
            self.__central_layout.add_widget(QParameter(p))
        self.set_layout(self.__central_layout)


class QParameter(QWidget):
    def __init__(self, parameter: Parameter):
        super().__init__()
        self.parameter_scroll_bar = QScrollBar()
        self.parameter_value = QLabel()
        self.parameter_title = parameter.name
        self.parameter_scale = parameter.scale

        parameter_layout = QHBoxLayout()
        parameter_layout.add_layout(
            self.__create_parameter(parameter,
                                    self.parameter_scroll_bar,
                                    self.parameter_value,
                                    self.parameter_scale))

        self.set_layout(parameter_layout)

    def __create_parameter(self, parameter, scroll_bar: QScrollBar,
                           value_label: QLabel, scale: float) -> QHBoxLayout:
        parameter_label = QLabel()
        parameter_label.text = parameter.name + " = "
        scroll_bar.orientation = Qt.Horizontal
        scroll_bar.set_range(parameter.min * scale,
                             parameter.max * scale)
        scroll_bar.value = parameter.current * scale
        scroll_bar.minimum_width = 250
        value_label.set_num(scroll_bar.value / scale)
        scroll_bar.valueChanged.connect(
            lambda value: value_label.set_num(value / scale))
        scroll_bar.valueChanged.connect(
            lambda value: self.set_current(parameter, value, scale))

        layout = QHBoxLayout()
        layout.add_widget(parameter_label)
        layout.add_widget(value_label)
        layout.add_widget(scroll_bar)

        return layout

    @Slot()
    def set_current(self, parameter: Parameter, value: int, scale: float):
        parameter.current = value / scale


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
        self.__klustr_dao = PostgreSQLKlustRDAO(
            PostgreSQLCredential(password='AAAaaa123'))
        self.__data_info = self.__klustr_dao.available_datasets[0]

        self.__gui_maker()

        # Initialisations
        # self.global_test()
        # self.single_test('ABC')

    @Slot()
    def open_dialog(self, title: str, source: str) -> None:
        try:
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()
                QMessageBox.about(self, title, content)
        except FileNotFoundError:
            print("Le fichier n'existe pas.")

    @Slot()
    def set_thumbnail(self, index: int) -> None:
        thumbnail = self.__current_data_set[index][6]
        img = qimage_argb32_from_png_decoding(thumbnail)
        self.__single_test_view_label.pixmap = QPixmap.from_image(img)
        self.__single_test_result_label.text = "not classified"

    @Slot()
    def update_data_set(self, dataset_name: str) -> None:
        self.__current_data_set = self.__klustr_dao.image_from_dataset(
            dataset_name, False)
        self.set_training_data(dataset_name)
        self.set_single_test_dropmenu(self.__current_data_set)
        self.set_thumbnail(self.__single_test_dropmenu.current_index)
        self.__single_test_result_label.text = "not classified"

    def set_single_test_dropmenu(self, data_set: list[ImageData]) -> None:
        items = [i[3] for i in data_set]
        self.__single_test_dropmenu.clear()
        self.__single_test_dropmenu.insert_items(0, items)

    def convert_query_to_img_tuple(self, query_result: ImageData) -> tuple[
        str, npt.NDArray]:
        tag = query_result[1]
        result_img = qimage_argb32_from_png_decoding(query_result[6])
        img_nparray = ndarray_from_qimage_argb32(result_img) ^ 1
        return tag, img_nparray

    @Slot()
    def set_training_data(self, dataset_name: str) -> None:
        query_result = self.__klustr_dao.image_from_dataset(dataset_name, True)
        self.__knn_engine.training_data = np.zeros([len(query_result), 4])
        for i, result in enumerate(query_result):
            raw_data = self.convert_query_to_img_tuple(result)
            self.__knn_engine.training_data[
                i] = self.__knn_engine.prepare_data(raw_data, True)
        self.add_points(self.__knn_engine.training_data, True)
        self.set_k_max(len(self.__knn_engine.training_data))

    def set_k_max(self, new_max: int) -> None:
        self.__knn_engine.k.max = new_max
        params = self.__parameters_picker.find_children(QParameter)
        for p in params:
            if p.parameter_title == self.__knn_engine.k.name:
                p.parameter_scroll_bar.set_range(self.__knn_engine.k.min,
                                                 self.__knn_engine.k.max)

    @Slot()
    def classify_image(self, image: ImageData) -> str:
        raw_img_data = self.convert_query_to_img_tuple(image)
        extracted_img_data = self.__knn_engine.prepare_data(raw_img_data,
                                                            False)
        self.add_points(extracted_img_data, False)
        result = self.__knn_engine.classify(extracted_img_data)
        self.__single_test_result_label.text = result
        return result

    def add_points(self, set_data: npt.NDArray, is_cloud: bool) -> None:
        if is_cloud:
            sets = []
            for i in range(len(self.__knn_engine.known_categories)):
                indices = np.where(set_data[:, 3] == i)[0]
                valid_rows_view = set_data[indices]
                sets.append(valid_rows_view)
            self.viewer_widget.clear()
            for i in range(len(self.__knn_engine.known_categories)):
                self.viewer_widget.add_serie(sets[i][:, :-1],
                                             self.__color_sequence.next(),
                                             title=
                                             self.__knn_engine.known_categories[
                                                 i], size_percent=0.05)

        else:
            self.viewer_widget.remove_serie('unkown shape')
            self.viewer_widget.add_serie(set_data[:-1].reshape(1, -1),
                                         QColor("black"), title="unkown shape",
                                         size_percent=0.05)


    def single_test(self, dataset_name) -> None:
        """Evaluates and write in a file the classification accuracy of a
            specific dataset. In case of failure, the shape's index in the dataset,
            the shape name and the misidentified shape name are provided"""
        file = open("test_results.txt", "a")
        file.write(dataset_name + "\n")
        compteur = 0
        for i, data in enumerate(self.__current_data_set):
            response = self.classify_image(data)
            if response == data[1]:
                compteur += 1
            else:
                file.write(str(i))
                file.write(': ' + data[1])
                file.write(' wrongfully identified as ' + response + '\n')
        file.write(str(compteur) + ' GOOD OUT OF' + str(
            len(self.__current_data_set)) + '\n')
        file.close()

    def global_test(self) -> None:
        """Evaluates and write in a file the classification accuracy of all
            datasets returned from the database. Gives a success rate for each
            dataset and all datasets combined."""
        file = open("test_results.txt", "a")
        total_score = []
        for dataset in self.__data:
            file.write(str(dataset[1]) + ' : ')
            compteur = 0
            self.set_training_data(dataset[1])
            test_images = self.__klustr_dao.image_from_dataset(dataset[1],
                                                               False)
            for i, image in enumerate(test_images):
                response = self.classify_image(image)
                if response == image[1]:
                    compteur += 1
            file.write(str(compteur) + ' GOOD OUT OF ' + str(
                len(test_images)) + ' (')
            percentage = compteur / len(test_images) * 100
            total_score.append(percentage)
            file.write(str(percentage) + ')' '\n')
            print(str(dataset[1]) + ' DONE')
        file.write('AVERAGE SCORE : ' + str(np.mean(total_score)) + '\n')
        file.close()

    @Slot()
    def get_data_info(self, index):
        selected_text = self.__dataset_dropmenu.item_text(index)
        for data in self.__klustr_dao.available_datasets:
            if data[1] in selected_text:
                self.__data_info = data

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

        menu_layout = self.__create_menu(self.__dataset, self.__single_test,
                                         self.__parameters, self.about_button)
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        self.viewer_widget = QScatter3dViewer(parent=central_widget)
        central_layout.add_layout(menu_layout)
        central_layout.add_widget(self.viewer_widget)
        central_widget.set_layout(central_layout)
        self.set_central_widget(central_widget)

        self.viewer_widget.axis_x.title = "X Axis : A / P^2"
        self.viewer_widget.axis_y.title = "Y Axis : A shape / A pseudo circumscribed circle"
        self.viewer_widget.axis_z.title = "Z Axis : A pseudo inscribed circle / A Shape"
        self.viewer_widget.title = "KlustR KNN Classification"
        self.viewer_widget.axis_y.range = (0.0, 1.0)
        self.viewer_widget.axis_x.range = (0.0, 1.0)
        self.viewer_widget.axis_z.range = (0.0, 1.0)
        self.update_data_set('ABC')
        self.update_labels()

    def __get_dropmenu(self):
        __data = self.__klustr_dao.available_datasets
        __items = [f"{i[1]} [{i[5]}] [{i[8]}]" for i in __data]
        dropmenu = QComboBox()
        dropmenu.insert_items(0, __items)
        dropmenu.activated.connect(lambda: self.update_data_set(
            __items[self.__dataset_dropmenu.current_index].split(maxsplit=1)[
                0]))
        dropmenu.activated.connect(lambda index: self.get_data_info(index))
        dropmenu.activated.connect(lambda: self.update_labels())
        return dropmenu

    def __get_info(self):
        group = QGroupBox("Included in Data Set")
        group_layout = QVBoxLayout()  # Layout vertical pour le groupe
        self.__info_layout = QVBoxLayout(
            group)  # Layout en grille pour aligner les étiquettes et les valeurs
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
        self.__translation = self.translation_widget(
            self.__transformation_layout)
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
        self.__single_test_dropmenu = QComboBox()
        self.__single_test_dropmenu.activated.connect(
            lambda: self.set_thumbnail(
                self.__single_test_dropmenu.current_index))
        self.__single_test_view_label = QLabel()
        self.__single_test_view_label.style_sheet = 'QLabel { background-color : #313D4A; padding : 10px 10px 10px 10px; }'  # 354A64
        self.__single_test_view_label.alignment = Qt.AlignCenter
        self.__single_test_button = QPushButton("Classify", self)
        self.__single_test_button.clicked.connect(lambda: self.classify_image(
            self.__current_data_set[
                self.__single_test_dropmenu.current_index]))
        self.__single_test_result_label = QLabel()
        self.__single_test_result_label.text = "not classified"
        self.__single_test_result_label.alignment = Qt.AlignCenter
        single_test_layout.add_widget(self.__single_test_dropmenu)
        single_test_layout.add_widget(self.__single_test_view_label)
        single_test_layout.add_widget(self.__single_test_button)
        single_test_layout.add_widget(self.__single_test_result_label)
        return single_test

    def __create_parameters_group(self):
        group = QGroupBox("KNN parameters")
        group.set_fixed_height(125)
        layout = QVBoxLayout(group)
        self.__parameters_picker = QParameterPicker(self.__knn_engine.k,
                                                    self.__knn_engine.max_distance)
        layout.add_widget(self.__parameters_picker)
        group.set_layout(layout)
        return group

    def __about_button(self):
        button = QPushButton("About", self)
        button.clicked.connect(
            lambda: self.open_dialog("About KlustR KNN Classifier",
                                     "project_report.txt"))
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
    window.show_maximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
