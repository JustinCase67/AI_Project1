import random
from copy import deepcopy
import numpy as np 


class GOLEngine:
    def __init__(self, width, height):
        self.__width = None
        self.__height = None
        self.__world = None
        self.__temp = None
        self.__current_generation = 0
        self.__cells_alive = 0
        
        # concept of LUT
        # indicating if cell is dead or alive from the
        # number of alive neighbours 
        #                    0  1  2  3  4  5  6  7  8
        #                    |  |  |  |  |  |  |  |  | 
        #                    v  v  v  v  v  v  v  v  v  
        self.__alive_rule = (0, 1, 1, 1, 1, 1, 1, 0, 0)
        self.__dead_rule  = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        # état cellule  0                 1
        self.__rules = (self.__dead_rule, self.__alive_rule)
        
        self.resize(width, height)
    
    @property
    def width(self):
        return self.__width
    
    @width.setter
    def width(self, value):
        self.resize(value, self.__height)
    
    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        self.resize(self.__width, value)
        
    @property
    def current_generation(self):
        return self.__current_generation
        
    @property
    def cells_count(self):
        return self.__width * self.__height
        
    @property
    def cells_alive(self):
        return self.__cells_alive
        
    @property
    def cells_dead(self):
        return self.cells_count - self.__cells_alive
        
    def cell_value(self, x, y):
        # no input validation for performance consideration
        return self.__world[x][y]
        
    def set_cell_value(self, x, y, value):
        # no input validation for performance consideration
        self.__world[x][y] = value
        
    def set_world(self, image):
        # no input validation for performance consideration
        self.__world = image
    
    def __validate_size(self, size):
        if not isinstance(size, int):
            raise TypeError('size must be an int')
        if not(3 <= size <= 2000):
            raise ValueError('size must be between 3 and 2000')
    
    def resize(self, width, height):
        # l'importance de cette fonction justifie ces étapes de validation
        # => mettre l'importance sur l'interface de programmation
        self.__validate_size(width)
        self.__validate_size(height)
        
        self.__width = width
        self.__height = height
        self.__world = []
        self.__temp = []
        
        self.__world = [[0 for _ in range(self.__height)] for _ in range(self.__width)]
        self.__temp = deepcopy(self.__world)
        
        self.__current_generation = 0
        self.__cells_alive = 0

    def randomize(self, percent = 0.5):
        self.__current_generation = 0
        self.__cells_alive = 0
        for y in range(1, self.__height - 1):
            for x in range(1, self.__width - 1):
                cell_value = int(random.random() > percent)
                self.__world[x][y] = cell_value
                self.__cells_alive += cell_value

    def process(self):
        self.__cells_alive = 0
        for x in range(1, self.__width-1):
            for y in range(1, self.__height-1):
                neighbours = sum(self.__world[x-1][y-1:y+2]) \
                           + sum(self.__world[x][y-1:y+2:2]) \
                           + sum(self.__world[x+1][y-1:y+2])

                cell_value = self.__world[x][y]
                self.__temp[x][y] = self.__rules[cell_value][neighbours]
                self.__cells_alive += cell_value

        self.__world, self.__temp = self.__temp, self.__world
        self.__current_generation += 1
    
    def process_np(image):
        #on veux image, c, r, neighboor et temp
        #pour chaque image[:] si = 0 erit 0 dans temp
        #si image = 1 compte son nombre de neighboor et ecrit le dans neighboor
        #on transforme neighboor avec la regle alive pour quil finisse avec des 0 et des 1
        #on ecrit toute les 1 de neighboor dans temp
        #temp est le perimetre de la forme (reste juste a calculer area ppour la valeur du perimetre)
        width = image.shape[1]
        height = image.shape[0]
        temp = np.zeros(width, height)
        __alive_rule = (0, 1, 1, 1, 1, 1, 1, 0, 0)
        __dead_rule  = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        # état cellule  0                 1
        __rules = (__dead_rule, __alive_rule)
        c, r = np.meshgrid(np.arrange(width), np.arrange(height))
        #temp[:] = __rules[image[:]][neighbours]
        pass
        

