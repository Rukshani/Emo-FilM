"""
@author Rukshani Somarathna

This class is responsible for Leave-One-Film-Out (LOFO) binary modelling at the movie level and saving and visualising
feature importance bubble plot.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import EmoFilMDataFields


class BinaryModelling:

    def __init__(self):
        # Accessing variables from EmoFilMDataFields class
        self.dur_movies = EmoFilMDataFields.dur_movies
        self.sorted_discrete_items = EmoFilMDataFields.sorted_discrete_items
        self.sorted_all_items = EmoFilMDataFields.sorted_all_items
        self.sorted_grid_items = EmoFilMDataFields.sorted_grid_items

    # Generate baseline and list of movies
    def get_baseline_and_movies_list(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')

        df_copy = df.copy()

        # Binary split of data using the median
        for em in self.sorted_discrete_items:
            df_copy[em] = df_copy[em].apply(lambda x: 1 if x >= df[em].median() else 0)
        df_copy.head(3)

        # Generating the chance level dictionary of all emotions. This can be used to compare the model accuracy with
        # statistical modelling.
        chance_dict = {}

        for emotion in self.sorted_discrete_items:
            chance = df_copy[emotion].value_counts().max() / (df_copy[emotion].value_counts().sum())
            chance_dict[emotion] = chance * 100
        print(chance_dict)

        # Split dataframe by 14 valid movies
        after_the_rain_df = df_copy.iloc[0: self.dur_movies[0] + 1, :]
        between_viewings_df = df_copy.iloc[sum(self.dur_movies[0:1]) + 1: sum(self.dur_movies[0:2]) + 1, :]
        big_buck_bunny_df = df_copy.iloc[sum(self.dur_movies[0:2]) + 1: sum(self.dur_movies[0:3]) + 1, :]
        chatter_df = df_copy.iloc[sum(self.dur_movies[0:3]) + 1: sum(self.dur_movies[0:4]) + 1, :]
        first_bite_df = df_copy.iloc[sum(self.dur_movies[0:4]) + 1: sum(self.dur_movies[0:5]) + 1, :]
        lesson_learned_df = df_copy.iloc[sum(self.dur_movies[0:5]) + 1: sum(self.dur_movies[0:6]) + 1, :]
        payload_df = df_copy.iloc[sum(self.dur_movies[0:6]) + 1: sum(self.dur_movies[0:7]) + 1, :]
        sintel_df = df_copy.iloc[sum(self.dur_movies[0:7]) + 1: sum(self.dur_movies[0:8]) + 1, :]
        spaceman_df = df_copy.iloc[sum(self.dur_movies[0:8]) + 1: sum(self.dur_movies[0:9]) + 1, :]
        superhero_df = df_copy.iloc[sum(self.dur_movies[0:9]) + 1: sum(self.dur_movies[0:10]) + 1, :]
        tears_of_steel_df = df_copy.iloc[sum(self.dur_movies[0:10]) + 1: sum(self.dur_movies[0:11]) + 1, :]
        the_secret_number_df = df_copy.iloc[sum(self.dur_movies[0:11]) + 1: sum(self.dur_movies[0:12]) + 1, :]
        to_claire_from_sonny_df = df_copy.iloc[sum(self.dur_movies[0:12]) + 1: sum(self.dur_movies[0:13]) + 1, :]
        you_again_df = df_copy.iloc[sum(self.dur_movies[0:13]) + 1: sum(self.dur_movies[0:]) + 1, :]

        array_of_movie_df = [after_the_rain_df, between_viewings_df, big_buck_bunny_df, chatter_df, first_bite_df,
                             lesson_learned_df, payload_df, sintel_df, spaceman_df, superhero_df, tears_of_steel_df,
                             the_secret_number_df, to_claire_from_sonny_df, you_again_df]

        return chance_dict, array_of_movie_df

    # Binary modelling at the movie level using Random Forest Classifier
    def model_rf_movie_level(self, array_of_movie_df):

        # Define your strategy of modelling with GPU or CPU

        # Update the 'emotion' variable to the emotion you want to model
        emotion = 'Love'
        print(emotion)

        movie_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        acc_list = list()

        df_fi_score = pd.DataFrame()

        # Update the 'features_use' variable to the set of features you want to model. For example, if you want to model
        # all CPM features, use the following line:
        print('All CPM')
        features_use = self.sorted_grid_items

        # If you want to model the Appraisal set of features, use the following lines:

        # print('Appraisal')
        # features_use = self.sorted_grid_items[1:10]
        # print('Expression')
        # features_use = self.sorted_grid_items[10:15]
        # print('Motivation')
        # features_use = self.sorted_grid_items[15:25]
        # print('Feeling')
        # features_use = self.sorted_grid_items[25:32]
        # print('Physiology')
        # features_use = self.sorted_grid_items[32:37]

        for mov_id in movie_id_list:
            movie_id_list_copy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            movie_id_list_copy.remove(mov_id)

            append_mov_data = []

            for train_mov_id in movie_id_list_copy:
                append_mov_data.append(array_of_movie_df[train_mov_id])

            train_append_mov_data = pd.concat(append_mov_data, ignore_index=True)
            test_data = array_of_movie_df[mov_id]

            x_train = train_append_mov_data[features_use].values
            y_train = train_append_mov_data[emotion].values
            x_test = test_data[features_use].values
            y_test = test_data[emotion].values

            # The hyperparameters were found using the GridSearchCV method. The best hyperparameters were used to model.
            x_classifier = RandomForestClassifier(random_state=42)

            x_classifier.fit(x_train, y_train)
            y_predict = x_classifier.predict(x_test)

            fi_scores = x_classifier.feature_importances_
            df_fi_score[mov_id] = fi_scores

            acc = accuracy_score(y_test, y_predict) * 100

            acc_list.append(acc)

        # Save feature importance dataframes for each emotion modelling
        mean_fi_column = np.mean(df_fi_score, axis=1)
        df_fi_score['MeanFI'] = mean_fi_column
        df_fi_score['GRID'] = features_use
        df_fi_score.to_csv('path_to_your_FI_Data_' + emotion + '.csv', index=False)

        print(np.mean(acc_list))
        print(np.std(acc_list, ddof=1))
        print('------------------------------------------')

        for acc_item in acc_list:
            print(acc_item)

    # This is a sample code to visualise the feature importance bubble plot. This is based on the FI_Data CSVs generated
    # from the model_rf_movie_level method.
    def generate_fi_plot(self):

        saved_fi_files = pd.DataFrame(columns=self.sorted_discrete_items)

        for sorted_discrete_item in self.sorted_discrete_items:
            saved_fi_emo = pd.read_csv('path_to_your_FI_Data_' + sorted_discrete_item + '.csv')
            saved_fi_files[sorted_discrete_item] = saved_fi_emo['MeanFI']

        saved_fi_trans = saved_fi_files.T
        saved_fi_trans.columns = self.sorted_grid_items

        fig = go.Figure()

        # Update the 'multiplier' variable to the value you want to use to scale the marker size in the plot.
        multiplier = 300

        # Visualisation for Appraisal component
        for g1 in range(0, 10):
            fig.add_trace(go.Scatter(x=self.sorted_discrete_items, y=[self.sorted_grid_items[g1] for i in range(0, 13)],
                                     mode='markers',
                                     marker_size=saved_fi_trans[saved_fi_trans.columns[g1]].values * multiplier,
                                     marker_color='mediumvioletred', showlegend=False))

        # Visualisation for Expression component
        for g2 in range(10, 15):
            fig.add_trace(go.Scatter(x=self.sorted_discrete_items, y=[self.sorted_grid_items[g2] for i in range(0, 13)],
                                     mode='markers',
                                     marker_size=saved_fi_trans[saved_fi_trans.columns[g2]].values * multiplier,
                                     marker_color='lightsalmon', showlegend=False))

        # Visualisation for Motivation component
        for g3 in range(15, 25):
            fig.add_trace(go.Scatter(x=self.sorted_discrete_items, y=[self.sorted_grid_items[g3] for i in range(0, 13)],
                                     mode='markers',
                                     marker_size=saved_fi_trans[saved_fi_trans.columns[g3]].values * multiplier,
                                     marker_color='mediumblue', showlegend=False))

        # Visualisation for Feeling component
        for g4 in range(25, 32):
            fig.add_trace(go.Scatter(x=self.sorted_discrete_items, y=[self.sorted_grid_items[g4] for i in range(0, 13)],
                                     mode='markers',
                                     marker_size=saved_fi_trans[saved_fi_trans.columns[g4]].values * multiplier,
                                     marker_color='green', showlegend=False))

        # Visualisation for Physiology component
        for g5 in range(32, 37):
            fig.add_trace(go.Scatter(x=self.sorted_discrete_items, y=[self.sorted_grid_items[g5] for i in range(0, 13)],
                                     mode='markers',
                                     marker_size=saved_fi_trans[saved_fi_trans.columns[g5]].values * multiplier,
                                     marker_color='violet', showlegend=False))

        fig.update_layout(height=1300, width=800)
        fig.update_xaxes(tickangle=90)

        fig.show()


binary_modelling = BinaryModelling()
chance_dictionary, array_of_movies = binary_modelling.get_baseline_and_movies_list()
binary_modelling.model_rf_movie_level(array_of_movies)

# Feature importance plot.
# Note: Uncomment for visualisation.
# binary_modelling.generate_fi_plot()
