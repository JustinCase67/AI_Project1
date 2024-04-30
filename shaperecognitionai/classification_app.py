import sys

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox
from PySide6.QtWidgets import QWidget, QLabel, QScrollBar
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout

from __feature__ import snake_case, true_property

from db_credential import PostgreSQLCredential
from klustr_dao import PostgreSQLKlustRDAO
from shaperecognitionai.scatter_3d_viewer import QScatter3dViewer


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


class QDesktopWidget:
    pass


class QClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # valeurs de test
        self.__k = Parameter("K", 5, 20, 0)
        self.__max_distance = Parameter("Max distance", 0, 100, 0)
        # fin des valeurs de test
        self.__window_title = "Klustr KNN Classification"
        self.__window_width = 1024 # demander au prof pour resizable selon l'écran, qdesktop deprecated
        self.__window_height = 768
        self.resize(self.__window_width, self.__window_height)

        # Gestion DB
        credential = PostgreSQLCredential(password='AAAaaa123')
        klustr_dao = PostgreSQLKlustRDAO(credential)

        # Génération du bouton About et du menu
        self.about_button = QPushButton("About", self)
        self.about_button.clicked.connect(lambda: self.open_dialog("About KlustR KNN Classifier", "report.txt"))

        menu_layout = QVBoxLayout()
        menu_layout.add_widget(
            QParameterPicker(self.__k, self.__max_distance))
        menu_layout.add_widget(self.about_button)
        menu_widget = QWidget()
        menu_widget.set_layout(menu_layout)


        # Combinaison du menu et du widget viewer
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        viewer_widget = QScatter3dViewer(parent=central_widget)
        central_layout.add_widget(menu_widget)
        central_layout.add_widget(viewer_widget)
        central_widget.set_layout(central_layout)
        self.set_central_widget(central_widget)

    @Slot()
    def open_dialog(self, title: str, source: str):
        try:
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()
                QMessageBox.about(self, title, content)
        except FileNotFoundError:
            print("Le fichier n'existe pas.")

def main():
    app = QApplication(sys.argv)
    window = QClassificationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
