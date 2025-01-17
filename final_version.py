import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Hierarchical_clustering():
    def __init__(self, data, method=’centroid’):
        self.data = data
        self.method = method
        self.transformed_data = None
        self.labels = None
        self.list_of_distances = None
        self.predicted_labels = []
    def find_labels(self, num_of_the_column):
        '''Saving the labels column'''
        self.labels = self.data[self.data.columns[num_of_the_column]]
        self.data = self.data.drop(self.data.columns[num_of_the_column], axis=1)
    def preprocess(self, scaler = StandardScaler):
        '''Preprocessing the data'''
        transformer = scaler().fit(self.data)
        transformed_data = transformer.transform(self.data)
        self.transformed_data = pd.DataFrame(transformed_data)
        self.transformed_data.columns = self.data.columns
    def Markov_moment(self, q):
        '''Get the appropriate clustering'''

        def delta_2(li):
            '''Approximation error'''
            new_li = np.array(li) - li[0]
            res = (1/245) * (19 * new_li[1] ** 2 - 11 * new_li[2] ** 2 + 41 * new_li[3] ** 2 + 12*new_li[1]*new_li[2]-64*new_li[1]*new_li[3]-46*new_li[2]*new_li[3])
            return res
        def delta(li, q):
            '''Convert to trend set'''
            res = []
            for i in range(len(li)):
                res.append(li[i] + q*i)
            return res
        flag = True
        list_of_distances = [0]
        df_start = self.transformed_data.copy()
        elements = df_start.index
        for _ in range(self.data.shape[0] - 1):
            distance_df = pd.DataFrame(np.zeros((len(elements), len(elements))))
            distance_df.index = elements
            distance_df.columns = elements
            for i in range(distance_df.shape[0]):
                obj_1 = [int(f) for f in str(distance_df.index[i]).split()]
                for j in range(distance_df.shape[0]):
                    obj_2 = [int(f) for f in str(distance_df.index[j]).split()]
                    # count the distance between clusters
                    if obj_1 == obj_2:
                        distance_df[distance_df.index[i]][distance_df.index[j]] = 0
                    else:
                        centroid_1 = sum(list(map(lambda x: df_start.iloc[x, :] / len(obj_1), obj_1)))
                        centroid_2 = sum(list(map(lambda x: df_start.iloc[x, :] / len(obj_2), obj_2)))
                        if self.method == 'ward':
                            distance_df[distance_df.index[i]][distance_df.index[j]] = (len(obj_1 * len(obj_2))) / (len(obj_1) + len(obj_2)) * np.linalg.norm(centroid_1 - centroid_2)
                        else:
                            distance_df[distance_df.index[i]][distance_df.index[j]] = np.linalg.norm(centroid_1 - centroid_2)
            # finding the minimum distance
            x_coord = 0
            y_coord = 0
            min_distance = 10 ** 5 + 1
            for i in range(distance_df.shape[0]):
                for j in range(distance_df.shape[1]):
                    if distance_df.iloc[i, j] < min_distance and distance_df.iloc[i, j] != 0:
                        min_distance = distance_df.iloc[i, j]
                        x_coord = i
                        y_coord = j
            if self.method == 'centroid' and min_distance < list_of_distances[-1]:
                min_distance = list_of_distances[-1]
            list_of_distances.append(min_distance)
            trend = delta(list_of_distances, q)
            if len(trend) >= 5:
                if delta_2(trend[len(trend) - 5:len(trend) - 1]) <= 0 and delta_2(trend[len(trend) - 4:len(trend)]) > 0 and flag:
                    #print('the nature of the increase has changed')
                    res_elements = list(map(lambda x: str(x).split(), elements))
                    for i in range(len(self.labels)):
                        for j in range(len(res_elements)):
                            if i in [int(elem) for elem in res_elements[j]]:
                                self.predicted_labels.append(j)
                    flag = False
            new_index = f'{distance_df.index[x_coord]} {distance_df.index[y_coord]}'
            # remove rows and columns in the similarity matrix
            distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=0)
            distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=1)
            # add a new cluster to the elements list
            elements = list(distance_df.index)
            elements.append(new_index)
        self.list_of_distances = list_of_distances
    
    def draw_sequence_of_distances(self):
        '''Display the sequence of the minimum distances'''
        x = np.arange(0, len(self.list_of_distances), 1)
        y = np.array(self.list_of_distances)
        fig = plt.figure(figsize=(10, 7))
        plt.plot(x,y, linewidth = 0.8, color = 'black', marker='o', ms= 7, markerfacecolor='skyblue')
        plt.grid(True, alpha = 0.5)
        plt.show()  

    def print_clustering(self):
        for i in np.unique(self.predicted_labels):
            print(f'{i} cluster: {list(self.labels[self.predicted_labels == i])}')

'''Upload data'''
data_2020 = pd.read_excel('dataset_path', sheet_name='2020', usecols=list(range(0,7)))
data_2020.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']

'''Applying the method'''
clustering = Hierarchical_clustering(data_2020, 'ward')
clustering.find_labels(0)
clustering.preprocess()
clustering.Markov_moment(q = 1.2)
clustering.print_clustering()
clustering.draw_sequence_of_distances()
