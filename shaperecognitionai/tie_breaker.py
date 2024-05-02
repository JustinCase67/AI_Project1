import numpy as np


#coordinates is a list of tuples of coordinates
def centroid_from_coordinates(coordinates):
    num_points = len(coordinates)
    if num_points == 0:
        return None
    
    coordinates_array = np.array(coordinates)
    centroid = np.mean(coordinates_array, axis=0)
    return centroid

def euclidean_distance_squared(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return (np.sum((point1 - point2)**2))



def neighbor_tie_breaker(coordinates_tuple, index_tuples_references_tab, coordinates_form):
    liste_centroid = []
    liste_distances = []
    for form_coordinates in coordinates_tuple:
        liste_centroid.append(centroid_from_coordinates(form_coordinates))
        
    for centroid in liste_centroid:
        liste_distances.append(euclidean_distance_squared(centroid, coordinates_form))

    np_distances = np.array(liste_distances)
    
    index_max_value = np.argmin(np_distances)
    
    return index_tuples_references_tab[index_max_value]








def main():
    coordinatesTriangle = ((5,10), (15, 10), (15, 20))
    coordinatesCircle = ((20, 30), (20, 40), (20, 50))
    
    coordinatesForms = (coordinatesTriangle,coordinatesCircle)
    coordinates_references_index_tag = (2,3)
    
    metrics_form = (15,15)
    
    tag = neighbor_tie_breaker(coordinatesForms, coordinates_references_index_tag, metrics_form)
    
    print(tag)
    # centroid = centroid_from_coordinates(coordinates)
    # print("Centroid:", centroid)
    # distance = euclidean_distance_squared(centroid, (18.0, 12.0, 65.0, 60.0))
    # print(distance)


if __name__ == '__main__':
    main()
    