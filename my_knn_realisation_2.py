import math
import numpy as np
import operator
from collections import Counter

class my_knn:
    def __init__(self, k):
        self.k = k

    def fit(self, train_set):
        self.train_set = train_set

    def predict(self, test_set):
        test_results = []

        for i in test_set:
            distances = []
            array_of_neighbors = []
            for j in self.train_set:
                pairs_of_data = zip(i, j)
                sum_of_dist = np.sum([pow(a - b, 2) for (a, b) in pairs_of_data])
                distance_i_j = math.sqrt(sum_of_dist)
                distances.append(distance_i_j)
                array_of_neighbors.append((j, distance_i_j))
            i_sorted_neighbors = sorted(array_of_neighbors, key=operator.itemgetter(1))
            i_closest_neighbors = (i_sorted_neighbors[:self.k])
            classes = [neighbour[0][-1] for neighbour in i_closest_neighbors]
            count_classes = Counter(classes)
            i_result = count_classes.most_common()[0][0]
            test_results.append(i_result)

        return test_results







