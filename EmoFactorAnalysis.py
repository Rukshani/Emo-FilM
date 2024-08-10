"""
@author Rukshani Somarathna

This class is responsible for generating Promax and Varimax factor analysis scores. The class uses the FactorAnalyzer
library to generate the factor analysis scores. It generates scatter plots of loadings.
"""

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from matplotlib import pyplot as plt
import seaborn as sns

import EmoFilMDataFields


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class EmoFactorAnalysis:

    def __init__(self):
        # Accessing variables from EmoFilMDataFields class
        self.sorted_discrete_items = EmoFilMDataFields.sorted_discrete_items
        self.sorted_all_items = EmoFilMDataFields.sorted_all_items
        self.sorted_all_items_shortened = EmoFilMDataFields.sorted_all_items_shortened

    # Generate Varimax or Promax factor analysis scores
    def generate_fa(self, rotation):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_discrete_grid_fa = df[self.sorted_all_items]

        n_components = 3

        df_discrete_grid_fa.columns = self.sorted_all_items_shortened
        print(df_discrete_grid_fa.columns)

        print(f'Run Factor Analysis with {n_components} Factors')
        fact_analyser = FactorAnalyzer(n_factors=n_components, rotation=rotation)
        fact_analyser.fit(df_discrete_grid_fa)

        ev_fact_an_5 = fact_analyser.get_eigenvalues()

        eigen_df = pd.DataFrame(ev_fact_an_5[0][0:15], columns=['Eigenvalues'])
        loadings_df = pd.DataFrame(fact_analyser.loadings_, index=df_discrete_grid_fa.columns)
        loadings_df.columns = ['Valence', 'Arousal', 'Power']
        variance_df = pd.DataFrame(fact_analyser.get_factor_variance(),
                                   index=['Variance', 'Proportional Var', 'Cumulative Var'])
        commun_df = pd.DataFrame(fact_analyser.get_communalities(), index=df_discrete_grid_fa.columns,
                                 columns=['Communalities'])

        print(f'Eigenvalues of the first  {eigen_df.shape[0]} factors')
        print(eigen_df)
        print()
        print('Factor loadings of each item')
        print(loadings_df)
        print()
        print('Explained variance by factor')
        print(variance_df)
        print()
        print('Communalities: Explained variance per item')
        print(commun_df)

        # Correlation plot of Valence, Arousal and Power
        pe_corr_mat = loadings_df[loadings_df.columns].corr(method="pearson")

        plt.figure(figsize=(6, 4))

        v_min, v_max = -1, 1
        c_map = sns.diverging_palette(10, 150, as_cmap=True, center='light', n=20, s=150)
        pe_mean_out_arr_map = sns.heatmap(pe_corr_mat, cmap=c_map, center=0, vmin=v_min, vmax=v_max,
                                          annot=True, fmt=".2f", annot_kws={"fontsize": 14})
        pe_mean_out_arr_map.set_xticklabels(loadings_df.columns, fontsize=15, rotation=90)
        pe_mean_out_arr_map.set_yticklabels(loadings_df.columns, fontsize=15, rotation=0)
        plt.savefig('ValenceArousalPower.png', bbox_inches='tight')
        plt.show()

        # Scatter plot of Valence and Arousal loadings
        c_col = fact_analyser.loadings_[:, 0:3]
        rgb = list(map(tuple, normalize_data(c_col)))

        plt.scatter(x=fact_analyser.loadings_[:, 0],
                    y=fact_analyser.loadings_[:, 1],
                    s=fact_analyser.get_communalities() * 50,
                    c=rgb)
        plt.xlabel("Valence")
        plt.ylabel("Arousal")
        for i, label in enumerate(df_discrete_grid_fa.columns):
            plt.text(fact_analyser.loadings_[i, 0], fact_analyser.loadings_[i, 1], label, size=8)
        plt.savefig('ValenceArousal.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Scatter plot of Valence and Power loadings
        plt.figure(figsize=(6, 4), dpi=100)
        plt.scatter(x=fact_analyser.loadings_[:, 0],
                    y=fact_analyser.loadings_[:, 2],
                    s=fact_analyser.get_communalities() * 50,
                    c=rgb)
        plt.xlabel("Valence")
        plt.ylabel("Power")

        for i, label in enumerate(df_discrete_grid_fa.columns):
            print(label, )
            plt.text(fact_analyser.loadings_[i, 0], fact_analyser.loadings_[i, 2], label, size=8)
        plt.savefig('ValencePower.png', dpi=300, bbox_inches='tight')
        plt.show()


emo_factor_analysis = EmoFactorAnalysis()
emo_factor_analysis.generate_fa("promax")
emo_factor_analysis.generate_fa("varimax")
