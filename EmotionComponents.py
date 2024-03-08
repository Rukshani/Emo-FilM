"""
@author Rukshani Somarathna

This class is responsible for generating hierarchical clustering plot of the emotion categories derived from CPM ratings
through hierarchical clustering based on Euclidean distance

"""

import pandas as pd
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import EmoFilMDataFields


class EmotionComponents:

    def __init__(self):
        # Accessing variables from EmoFilMDataFields class
        self.sorted_discrete_items = EmoFilMDataFields.sorted_discrete_items
        self.sorted_all_items = EmoFilMDataFields.sorted_all_items
        self.sorted_grid_items = EmoFilMDataFields.sorted_grid_items

    # Clustering of the emotion categories derived from CPM ratings through hierarchical clustering based on
    # Euclidean distance
    def generate_cpm_hc_plot(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_sort = df[self.sorted_all_items]

        df_discrete_grid = df_sort[self.sorted_discrete_items + self.sorted_grid_items]

        # Scale the data
        scaler = MinMaxScaler(feature_range=(1, 100))
        df_scaled = pd.DataFrame(scaler.fit_transform(df_discrete_grid))
        df_scaled.columns = df_discrete_grid.columns

        df_discrete_grid_scaled = df_scaled.copy()

        emo_dic = {}

        # Loop for 13 emotions
        for emo in range(len(self.sorted_discrete_items)):
            top_array = []

            # Loop for 37 CoreGRID items
            for grid in range(len(self.sorted_grid_items)):
                top = 0

                for row in range(df_discrete_grid_scaled.shape[0]):
                    e_value = df_discrete_grid_scaled[self.sorted_discrete_items[emo]].iloc[row]
                    grid_value = df_discrete_grid_scaled[self.sorted_grid_items[grid]].iloc[row]
                    mul_value = e_value * grid_value
                    top = top + mul_value

                top_array.append(top)
            emo_dic[self.sorted_discrete_items[emo]] = top_array

        emo_value = df_discrete_grid_scaled[self.sorted_discrete_items].sum(axis=0)

        grid_vector = {}

        for emo in self.sorted_discrete_items:
            value = emo_dic[emo] / emo_value[emo]
            grid_vector[emo] = value

        df_gew_grid_hc_scaled = pd.DataFrame.from_dict(grid_vector).T

        plt.figure(figsize=(8, 5))
        distance_matrix = shc.linkage(df_gew_grid_hc_scaled, method='ward', metric='euclidean')
        shc.set_link_color_palette(['limegreen', 'red'])
        shc.dendrogram(distance_matrix, labels=self.sorted_discrete_items, orientation='right')

        plt.ylabel("Discrete emotions", size=15)
        plt.xlabel("Distance", size=15)
        plt.yticks(size=14)
        plt.show()


emotion_components = EmotionComponents()
emotion_components.generate_cpm_hc_plot()
