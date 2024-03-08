"""
@author Rukshani Somarathna

This class is responsible for generating the Pearson correlation plot and hierarchical clustering of the 13 emotions
based on Pearson correlation matrix and Euclidean distance.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import seaborn as sns
from matplotlib import pyplot as plt


import EmoFilMDataFields


class OneCorrelation:

    def __init__(self):
        # Accessing variables from EmoFilMDataFields class
        self.dur_movies = EmoFilMDataFields.dur_movies
        self.movies = EmoFilMDataFields.movies
        self.sorted_discrete_items = EmoFilMDataFields.sorted_discrete_items
        self.sorted_all_items = EmoFilMDataFields.sorted_all_items
        self.sorted_grid_items = EmoFilMDataFields.sorted_grid_items

    # Calculate average emotion correlations at film level using Pearson correlation for 13 discrete emotion ratings
    def generate_correlation_plot(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_sort = df[self.sorted_all_items]

        # Split dataframe by 14 valid movies
        after_the_rain_df = df_sort.iloc[0: self.dur_movies[0] + 1, :]
        between_viewings_df = df_sort.iloc[sum(self.dur_movies[0:1]) + 1: sum(self.dur_movies[0:2]) + 1, :]
        big_buck_bunny_df = df_sort.iloc[sum(self.dur_movies[0:2]) + 1: sum(self.dur_movies[0:3]) + 1, :]
        chatter_df = df_sort.iloc[sum(self.dur_movies[0:3]) + 1: sum(self.dur_movies[0:4]) + 1, :]
        first_bite_df = df_sort.iloc[sum(self.dur_movies[0:4]) + 1: sum(self.dur_movies[0:5]) + 1, :]
        lesson_learned_df = df_sort.iloc[sum(self.dur_movies[0:5]) + 1: sum(self.dur_movies[0:6]) + 1, :]
        payload_df = df_sort.iloc[sum(self.dur_movies[0:6]) + 1: sum(self.dur_movies[0:7]) + 1, :]
        sintel_df = df_sort.iloc[sum(self.dur_movies[0:7]) + 1: sum(self.dur_movies[0:8]) + 1, :]
        spaceman_df = df_sort.iloc[sum(self.dur_movies[0:8]) + 1: sum(self.dur_movies[0:9]) + 1, :]
        superhero_df = df_sort.iloc[sum(self.dur_movies[0:9]) + 1: sum(self.dur_movies[0:10]) + 1, :]
        tears_of_steel_df = df_sort.iloc[sum(self.dur_movies[0:10]) + 1: sum(self.dur_movies[0:11]) + 1, :]
        the_secret_number_df = df_sort.iloc[sum(self.dur_movies[0:11]) + 1: sum(self.dur_movies[0:12]) + 1, :]
        to_claire_from_sonny_df = df_sort.iloc[sum(self.dur_movies[0:12]) + 1: sum(self.dur_movies[0:13]) + 1, :]
        you_again_df = df_sort.iloc[sum(self.dur_movies[0:13]) + 1: sum(self.dur_movies[0:]) + 1, :]

        array_of_movie_df = [after_the_rain_df, between_viewings_df, big_buck_bunny_df, chatter_df, first_bite_df,
                             lesson_learned_df, payload_df, sintel_df, spaceman_df, superhero_df, tears_of_steel_df,
                             the_secret_number_df, to_claire_from_sonny_df, you_again_df]

        # Calculate Pearson correlation at movie-level
        pe_corr_mat_list = []

        for movie_df in array_of_movie_df:
            pe_corr_mat = movie_df[self.sorted_discrete_items].corr(method="pearson")
            pe_corr_mat_list.append(pe_corr_mat.values.tolist())

        pe_corr_mat_array = np.array(pe_corr_mat_list)
        pe_corr_mat_mean_array = np.mean(pe_corr_mat_array, axis=0)

        plt.figure(figsize=(12, 8))  # dpi=200

        v_min, v_max = -1, 1
        c_map = sns.diverging_palette(10, 150, as_cmap=True, center='light', n=20, s=150)
        pe_mean_out_arr_map = sns.heatmap(pe_corr_mat_mean_array, cmap=c_map, center=0, vmin=v_min, vmax=v_max,
                                          annot=True, fmt=".2f", annot_kws={"fontsize": 14})
        pe_mean_out_arr_map.set_xticklabels(self.sorted_discrete_items, fontsize=15, rotation=90)
        pe_mean_out_arr_map.set_yticklabels(self.sorted_discrete_items, fontsize=15, rotation=0)

        plt.show()

        return pe_corr_mat_mean_array

    # Generate hierarchical clustering of the 13 emotions based on Pearson correlation and Euclidean distance
    def generate_hc(self, pe_corr_mat_mean_array_output):
        plt.figure(figsize=(8, 5))

        pairwise_distances = shc.distance.pdist(pe_corr_mat_mean_array_output, metric='euclidean')
        linkage = shc.linkage(pairwise_distances, method='average')
        shc.set_link_color_palette(['red', 'limegreen'])
        shc.dendrogram(linkage, labels=self.sorted_discrete_items, orientation='right')

        plt.xlabel("Distance", size=15)
        plt.ylabel("Discrete Emotions", size=15)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.show()


one_correlation = OneCorrelation()
pe_corr_mat_mean_array_out = one_correlation.generate_correlation_plot()
one_correlation.generate_hc(pe_corr_mat_mean_array_out)
