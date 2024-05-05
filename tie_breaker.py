import numpy as np


# coordinates is a list of tuples of coordinates
def centroid_from_coordinates(coordinates):
    num_points = len(coordinates)
    if num_points == 0:
        return None

    coordinates_array = np.array(coordinates)
    centroid = np.mean(coordinates_array, axis=0)
    return tuple(centroid)


def euclidean_distance_squared(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return (np.sum((point1 - point2) ** 2))


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
    coordinatesTriangle = ((5, 10), (15, 10), (15, 20))
    coordinatesCircle = ((20, 30), (20, 40), (20, 50))

    coordinatesForms = (coordinatesTriangle, coordinatesCircle, ((15, 16), (15, 16), (15, 16)))
    coordinates_references_index_tag = (2, 3, 6)

    metrics_form = (15, 15)

    tag = neighbor_tie_breaker(coordinatesForms, coordinates_references_index_tag, metrics_form)

    print(tag)

    data = np.array([[2, 2, 3, 4],
                     [1, 4, 5, 5],
                     [0, 6, 7, 6],
                     [0, 8, 9, 0],
                     [4, 10, 11, 2],
                     [2, 10, 11, 5],
                     [3, 10, 11, 4]])

    # Count occurrences of values at index 0
    unique_values, counts = np.unique(data[:, 0], return_counts=True)

    # Find unique values that occur the same number of times
    unique_values_same_occurrences = unique_values[counts == counts.max()]

    print("Unique values at index 0 with the same occurrences:", unique_values_same_occurrences)

    coordinates_breaker_tab = np.empty(len(unique_values_same_occurrences), dtype=object)
    for i in range(len(unique_values_same_occurrences)):
        filter_value = unique_values_same_occurrences[i]
        filtered_rows = data[data[:, 0] == filter_value]
        values_to_compare = filtered_rows[:, 1:]
        unique_tuples = np.unique(values_to_compare, axis=0)
        coordinates_breaker_tab[i] = unique_tuples
    print(coordinates_breaker_tab[1])

    # # Define the filter value
    # filter_value = 1

    # # Get the rows where the first column equals the filter value
    # filtered_rows = data[data[:, 0] == filter_value]

    # # Extract values at indices 1 and 2 for comparison
    # values_to_compare = filtered_rows[:, 1:]

    # # Get the unique tuples of values at indices 1 and 2
    # unique_tuples = np.unique(values_to_compare, axis=0)

    # # Print the resulting unique tuples
    # print(unique_tuples)
    # for unique_tuple in unique_tuples:
    #     print("Unique Tuple:", tuple(unique_tuple))
    # centroid = centroid_from_coordinates(coordinates)
    # print("Centroid:", centroid)
    # distance = euclidean_distance_squared(centroid, (18.0, 12.0, 65.0, 60.0))
    # print(distance)


if __name__ == '__main__':
    main()
    coordinates = [(1, 2), (3, 4), (5, 6), (7, 8)]
    tags = [1, 2, 1, 3]

    # Create a dictionary to store coordinates for each tag
    tag_coordinates = {}

    # Iterate through the lists
    for i, tag in enumerate(tags):
        # Get the coordinate corresponding to the current index
        coordinate = coordinates[i]

        # Check if the tag already exists in the dictionary
        if tag in tag_coordinates:
            # If it exists, append the coordinate to the list of coordinates for that tag
            tag_coordinates[tag].append(coordinate)
        else:
            # If it doesn't exist, create a new list with the coordinate as the first element
            tag_coordinates[tag] = [coordinate]

    print(tag_coordinates)
    # Convert the dictionary values to tuples
    tag_coordinates = {tag: tuple(coords) for tag, coords in tag_coordinates.items()}

    # Print the result
    print(tag_coordinates)

