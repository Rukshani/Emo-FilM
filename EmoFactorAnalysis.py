"""
@author Rukshani Somarathna

This class is responsible for generating Promax and Varimax factor analysis scores. The class uses the FactorAnalyzer
library to generate the factor analysis scores.
"""

import pandas as pd
from factor_analyzer import FactorAnalyzer

import EmoFilMDataFields


class EmoFactorAnalysis:

    def __init__(self):
        # Accessing variables from EmoFilMDataFields class
        self.sorted_discrete_items = EmoFilMDataFields.sorted_discrete_items
        self.sorted_all_items = EmoFilMDataFields.sorted_all_items
        self.sorted_grid_items = EmoFilMDataFields.sorted_grid_items

    # Generate Promax factor analysis scores
    def generate_promax_fa(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_sort = df[self.sorted_all_items]

        n_components = 3

        df_discrete_grid_fa = df_sort[self.sorted_discrete_items + self.sorted_grid_items]

        print(f'Run Factor Analysis with {n_components} Factors')
        fact_analyser = FactorAnalyzer(n_factors=n_components, rotation='promax')
        fact_analyser.fit(df_discrete_grid_fa)

        ev_fact_an_5 = fact_analyser.get_eigenvalues()

        eigen_df = pd.DataFrame(ev_fact_an_5[0][0:15], columns=['Eigenvalues'])
        loadings_df = pd.DataFrame(fact_analyser.loadings_, index=df_discrete_grid_fa.columns)
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

    # Generate Varimax factor analysis scores
    def generate_varimax_fa(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_sort = df[self.sorted_all_items]

        n_components = 3

        df_discrete_grid_fa = df_sort[self.sorted_discrete_items + self.sorted_grid_items]

        print(f'Run Factor Analysis with {n_components} Factors')
        fact_analyser = FactorAnalyzer(n_factors=n_components, rotation='varimax')
        fact_analyser.fit(df_discrete_grid_fa)

        ev_fact_an_5 = fact_analyser.get_eigenvalues()

        eigen_df = pd.DataFrame(ev_fact_an_5[0][0:15], columns=['Eigenvalues'])
        loadings_df = pd.DataFrame(fact_analyser.loadings_, index=df_discrete_grid_fa.columns)
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


emo_factor_analysis = EmoFactorAnalysis()
emo_factor_analysis.generate_promax_fa()
emo_factor_analysis.generate_varimax_fa()
