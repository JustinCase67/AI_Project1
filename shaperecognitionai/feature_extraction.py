import numpy as np
from math import pi



class FeatureExtractor:
    
    # aire de la forme / aire du cercle pseudo concentrique
    @staticmethod
    def get_extreme_points(img):
        size_x, size_y = img.shape
               
        extreme_left_tab = np.argmax(img == 1, axis = 0)
        mask = extreme_left_tab > 0 
        extreme_left_index = np.argmax(mask)
        extreme_left_points = (extreme_left_tab[extreme_left_index],extreme_left_index)
        
        
        extreme_right_tab = np.argmax(img[::-1, ::-1] == 1, axis = 0)
        mask = extreme_right_tab > 0 
        extreme_right_y_index = np.argmax(mask)
        extreme_right_points = (extreme_right_tab[extreme_right_y_index],extreme_right_y_index)
        extreme_right_points = (size_x - 1 - extreme_right_points[0],size_y - 1 - extreme_right_points[1])

        
        extreme_top_tab = np.argmax(img == 1, axis = 1)
        mask = extreme_top_tab > 0 
        extreme_top_y_index = np.argmax(mask)
        extreme_top_points = (extreme_top_tab[extreme_top_y_index],extreme_top_y_index)
        
        extreme_bottom_tab = np.argmax(img[::-1, ::-1] == 1, axis = 1)
        mask = extreme_bottom_tab > 0 
        extreme_bottom_y_index = np.argmax(mask)
        extreme_bottom_points = (extreme_bottom_tab[extreme_bottom_y_index],extreme_bottom_y_index)
        extreme_bottom_points = (size_x - 1 - extreme_bottom_points[0],size_y - 1 - extreme_bottom_points[1])
        #TUPLE POSITIONS (TOP, BOTTOM, LEFT, RIGHT) (for each positions row first then column)
        return(extreme_top_points,extreme_bottom_points, extreme_left_points, extreme_right_points)
    
    
    
    #take two tuples of position points(x,y)
    @staticmethod
    def distance_between_two_points(centroid,point):
        return ((point[0]-centroid[0])**2 +(point[1]-centroid[1])**2 )**0.5
    
    #all tuple of position points
    @staticmethod
    def distances_from_a_point(ensemble_points, centroid_position):
        distances = np.zeros(len(ensemble_points))
        for i in range(len(ensemble_points)):
            distances[i] = FeatureExtractor.distance_between_two_points(centroid_position,ensemble_points[i])
        return distances
            
    
    @staticmethod
    def get_max_distance(distances):
        return np.max(distances)
    
    @staticmethod
    def area_of_circle(radius):
        return pi*radius**2
    
    @staticmethod
    def ratio_area(img):
        points = FeatureExtractor.get_extreme_points(img)
        distances = FeatureExtractor.distances_from_a_point(points,FeatureExtractor.centroid(img))
        maxdistance = FeatureExtractor.get_max_distance(distances)
        pseudo_concentric_circle_area = FeatureExtractor.area_of_circle(maxdistance)
        return FeatureExtractor.area(img) / pseudo_concentric_circle_area
    
    
    @staticmethod
    def area(image):
        return np.sum(image)
    
    @staticmethod
    def centroid(image):
        c, r = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        return (np.sum(r * image), np.sum(c * image)) / FeatureExtractor.area(image)

    @staticmethod
    def create_image(size):
        return np.zeros((size[0], size[1]), dtype=np.uint8)

    @staticmethod
    def draw_rectangle(image, top_left,bottom_right):
        top, left = top_left
        bottom, right = bottom_right
        image[top:bottom+1,left:right+1] = 1   
        
        
      
# img = FeatureExtractor.create_image((10,10))
# print(img)
# FeatureExtractor.draw_rectangle(img,(2,2),(4,8))
# FeatureExtractor.draw_rectangle(img,(4,5),(8,7))
# print(img)
# img[1,3] = 1
# print(img)
# points = FeatureExtractor.get_extreme_points(img)
# print(points)
# distances = FeatureExtractor.distances_from_a_point(points,FeatureExtractor.centroid(img))
# print(distances)
# maxdistance = FeatureExtractor.get_max_distance(distances)
# print(maxdistance)
# print(FeatureExtractor.area_of_circle(maxdistance))
# print(FeatureExtractor.area(img))
# print(FeatureExtractor.ratio_area(img))

