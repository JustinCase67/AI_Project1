def classify(self, test_image):
    distances = self.assess_data_distance(test_image)
    neighbor = self.get_neighbor(distances)
    try:
        tags_index = self.get_tags_index(neighbor)
        unique_values, counts = np.unique(tags_index, return_counts=True)
        unique_values_same_occurrences = unique_values[counts == counts.max()]
        if len(unique_values_same_occurrences) > 1:
            metrics = []
            tags = []
            for n in neighbor:
                if self.training_data[n][-1] in unique_values_same_occurrences:
                    metrics.append(self.training_data[n][:-1])
                    tags.append(int(self.training_data[n][-1]))
            result = self.tie_breaker(metrics, tags, test_image[:-1])
        else:
            result = unique_values_same_occurrences[0]
        return self.__known_categories[result]
    except:
        return "Undefined"

    def tie_breaker(self, metrics, tags, test_image):
        print("test_image : ", test_image)
        print("metrics : ", metrics)
        print("tags : ", tags)

        unique_tags = list(set(tags))
        unique_tags.sort()  # Sorting for consistent order

        # Create a dictionary to store unique tags and their associated coordinates
        unique_tags_coordinates = {tag: [] for tag in unique_tags}

        # Iterate through the coordinates and corresponding tags
        for coordinate, tag in zip(metrics, tags):
            # Append the coordinate to the list of coordinates for the corresponding tag
            unique_tags_coordinates[tag].append(coordinate)

        # Convert the dictionary to a list of tuples
        result = [(tag, tuple(coords)) for tag, coords in unique_tags_coordinates.items()]

        # Print the result
        print(result)

        liste_centroid = []
        liste_distances = []

        for i in range(len(result)):
            liste_centroid.append(Util.centroid_from_coordinates(result[i][1]))
        for centroid in liste_centroid:
            liste_distances.append(Util.euclidean_distance_squared(centroid, test_image))
        np_distances = np.array(liste_distances)
        # index_max_value = np.argmin(np_distances)
        index_min_value = np.argmin(np_distances)
        return result[index_min_value][0]