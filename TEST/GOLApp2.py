import sys

from PySide6.QtCore import Qt
from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWidgets import QWidget, QLabel, QScrollBar, QGroupBox, QPushButton
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QIcon, QPixmap, QColor

from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QTimer

from gol_engine_03_np import GOLEngine

import np_test

from __feature__ import snake_case, true_property

class QGOLInfoPanel(QGroupBox):
    
    def __init__(self):
        super().__init__()
        
        self.title = 'Info Panel'
        
        self.__gen_nom = QLabel()
        self.__gen_nom.text = 'Generation'
        self.__gen_num = QLabel()
        self.__gen_num.set_num(0)
        
        self.__alive_nom = QLabel()
        self.__alive_nom.text = 'Alive'
        self.__alive_num = QLabel()
        self.__alive_num.set_num(0)
        
        self.__dead_nom = QLabel()
        self.__dead_nom.text = 'Dead'
        self.__dead_num = QLabel()
        self.__dead_num.set_num(0)
        
        
        
        gen_layout = QHBoxLayout()
        gen_layout.add_widget(self.__gen_nom)
        gen_layout.add_widget(self.__gen_num)
        
        alive_layout = QHBoxLayout()
        alive_layout.add_widget(self.__alive_nom)
        alive_layout.add_widget(self.__alive_num)
        
        dead_layout = QHBoxLayout()
        dead_layout.add_widget(self.__dead_nom)
        dead_layout.add_widget(self.__dead_num)
        
        final_layout = QVBoxLayout()
        final_layout.add_layout(gen_layout)
        final_layout.add_layout(alive_layout)
        final_layout.add_layout(dead_layout)
        
        self.set_layout(final_layout)
       
    Slot() 
    def __update_info_panel(self, gen, alive, dead):
        self.__gen_num = gen
        self.__alive_num = alive
        self.__dead_num = dead
    
    
class QGOLControlPanel(QGroupBox):
        
    speedChanged = Signal(int)
    
    def __init__(self):
        super().__init__()
        
        self.__stop = QPushButton()
        self._stop.set_text('STOP')
        self.__step = QPushButton()
        self._step.set_text('STEP')
        #rendre unavailable
        self.__speed = QScrollBar()
        self.__speed.orientation = Qt.Horizontal
        self.__speed.set_range(0, 1000)
        self.__speed.value = 1000
        self.__value = QLabel()
        self.__value.set_num(0)
        
        final_layout = QVBoxLayout()
        final_layout.add_widget(self.__stop)
        final_layout.add_widget(self.__step)
        final_layout.add_widget(self.__speed)
        final_layout.add_widget(self.__value)
        
        self.set_layout(final_layout)
            
            
        

class QGOLWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.set_window_title('Game of Life')
        
        self.__label = QLabel()
        
        #self.__engine = gol_engine_01.GOLEngine(200, 150)
        witdh = 150
        height = 150
        self.__engine = GOLEngine(witdh, height)
        #self.__engine.randomize()
        image = np_test.create_image((height, witdh))
        #np_test.draw_rectangle(image, (100, 100), (300, 300))
        np_test.draw_circle(image, (75, 75), 50)
        self.__engine.set_world(image)
        
        self.__label.pixmap = self.draw()
        
        self.__label.alignment = Qt.AlignCenter
        self.set_central_widget(self.__label)
        
        self.__engine.process_np(image)
        #self.__timer = QTimer()
        #self.__timer.timeout.connect(self.__update_GOL)
        #self.__timer.start(100)
        
    def draw(self):
        image = QImage(self.__engine.width, self.__engine.height, QImage.Format_ARGB32)
        image.fill(Qt.black)
        
        painter = QPainter(image)
        painter.set_pen(Qt.white)
        for y in range(self.__engine.height):
            for x in range(self.__engine.width):
                if self.__engine.cell_value(x, y):
                    painter.draw_point(x, y)
                    
        painter.end()
        
        pixmap = QPixmap()
        pixmap.convert_from_image(image.scaled(self.__label.size, Qt.KeepAspectRatio))
        return pixmap
        
    @Slot()
    def __update_GOL(self):
        #self.__engine.process()
        self.__label.pixmap = self.draw()
        

def main():
    app = QApplication(sys.argv)
    
    window = QGOLWindow()
    window.show()

    sys.exit(app.exec())    
    
    
if __name__ == '__main__':
    main()