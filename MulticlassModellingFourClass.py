"""
@author Rukshani Somarathna

This script models the 4-multiclass emotion prediction using the Leave-One-Film-Out (LOFO) classifier.
"""

import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import EmoFilMDataFields


class MulticlassModellingFourClass:

    def __init__(self):
        # Accessing variables from EmoFilMDataFields class
        self.dur_movies = EmoFilMDataFields.dur_movies
        self.sorted_discrete_items = EmoFilMDataFields.sorted_discrete_items
        self.sorted_all_items = EmoFilMDataFields.sorted_all_items
        self.sorted_grid_items = EmoFilMDataFields.sorted_grid_items

    # Scale the data
    def scale_data(self, df, scale_columns):
        scaler = MinMaxScaler(feature_range=(1, 100))
        df[scale_columns] = pd.DataFrame(scaler.fit_transform(df[scale_columns]))
        return df

    # Plot confusion matrix
    def plot_confusion_matrix_custom(self, y_test_list, predicted_labels_list, classes, normalize=True,
                                     title=''):
        cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure()

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(6, 4))
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size=15)
        plt.yticks(tick_marks, classes, size=15)

        fmt = '.2f' if normalize else 'd'
        thresh = cnf_matrix.max() / 2.

        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black", fontsize=15)

        plt.tight_layout()
        plt.ylabel('True label', size=15)
        plt.xlabel('Predicted label', size=15)
        plt.savefig('MulticlassModellingFourClassCM.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Leave-One-Film-Out (LOFO) XGBClassifier modelling for 4-multiclass emotion prediction
    def model_xgb_lofo_four_class(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_scaled_csv = self.scale_data(df, self.sorted_grid_items + self.sorted_discrete_items)

        df_scaled = df_scaled_csv.copy()
        df_scaled['MultiEmotion_Max'] = np.nan

        for index, row in df_scaled.iterrows():
            love_value = row['Love']
            regard_value = row['Regard']
            warm_heartedness_value = row['WarmHeartedness']
            pos_other_mean = (love_value + regard_value + warm_heartedness_value) / 3

            satisfaction_value = row['Satisfaction']
            happiness_value = row['Happiness']
            pride_value = row['Pride']
            pos_self_mean = (satisfaction_value + happiness_value + pride_value) / 3

            anxiety_value = row['Anxiety']
            fear_value = row['Fear']
            surprise_value = row['Surprise']
            sad_value = row['Sad']
            neg_self_mean = (anxiety_value + fear_value + surprise_value + sad_value) / 4

            disgust_value = row['Disgust']
            anger_value = row['Anger']
            guilt_value = row['Guilt']
            neg_other_mean = (disgust_value + anger_value + guilt_value) / 3

            mean_array = [pos_other_mean, pos_self_mean, neg_self_mean, neg_other_mean]
            max_index = np.argmax(mean_array)

            # 'POS_Other'
            if max_index == 0:
                df_scaled['MultiEmotion_Max'][index] = 0
            # 'POS_Self'
            elif max_index == 1:
                df_scaled['MultiEmotion_Max'][index] = 1
            # 'NEG_Self'
            elif max_index == 2:
                df_scaled['MultiEmotion_Max'][index] = 2
            # 'NEG_Other'
            elif max_index == 3:
                df_scaled['MultiEmotion_Max'][index] = 3

        # Print chance level which is used as the baseline
        print((df_scaled['MultiEmotion_Max'].value_counts().max() / df_scaled[
            'MultiEmotion_Max'].value_counts().sum()) * 100)

        valid_movies1 = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LeassonLearned',
                         'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsofSteel', 'Thesecretnumber',
                         'ToClaireFromSonny', 'YouAgain']

        movie_dfs = []

        for i, movie_name in enumerate(valid_movies1):
            if i == 0:
                start_index = 0
                end_index = self.dur_movies[0] + 1
            else:
                start_index = sum(self.dur_movies[0:i]) + 1
                end_index = sum(self.dur_movies[0:i + 1]) + 1
            movie_df = df_scaled.iloc[start_index:end_index, :]
            print(movie_name, movie_df.shape)
            movie_df['movie_name'] = movie_name
            movie_dfs.append(movie_df)

        df_with_movie_names_lomo_max = pd.concat(movie_dfs, axis=0)

        valid_movies_list = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite',
                             'LeassonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsofSteel',
                             'Thesecretnumber', 'ToClaireFromSonny', 'YouAgain']

        acc_list, predicted_targets, actual_targets, f1_list = [], [], [], []

        for mov_id in valid_movies_list:
            print(mov_id)
            valid_movies_list2 = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite',
                                  'LeassonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsofSteel',
                                  'Thesecretnumber', 'ToClaireFromSonny', 'YouAgain']
            valid_movies_list2.remove(mov_id)

            append_mov_data = []

            for train_mov_id in valid_movies_list2:
                append_mov_data.append(
                    df_with_movie_names_lomo_max[df_with_movie_names_lomo_max['movie_name'] == train_mov_id])

            train_append_mov_data = pd.concat(append_mov_data, ignore_index=True)
            test_data = df_with_movie_names_lomo_max[df_with_movie_names_lomo_max['movie_name'] == mov_id]

            x_train = train_append_mov_data[self.sorted_grid_items]
            y_train = train_append_mov_data['MultiEmotion_Max']
            x_test = test_data[self.sorted_grid_items]
            y_test = test_data['MultiEmotion_Max']

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            # The hyperparameters were found using the GridSearchCV method. The best hyperparameters were used to model.
            clf = RandomForestClassifier(random_state=42)
            clf.fit(x_train, y_train)

            y_predict = clf.predict(x_test)

            # Print and append accuracies
            accuracy = accuracy_score(y_test, y_predict) * 100
            print('Accuracy:', accuracy)
            acc_list.append(accuracy)

            # Print and append F1-scores
            f1 = f1_score(y_test, y_predict, average='weighted')
            print('F1-score:', f1)
            f1_list.append(f1)

            # Appending data for confusion matrix
            predicted_targets = np.append(predicted_targets, y_predict)
            actual_targets = np.append(actual_targets, y_test)
            print('------------------------------------------------------------------------')

        # Print final average results
        print(np.mean(acc_list))
        print(np.std(acc_list, ddof=1))
        print(np.mean(acc_list) - np.std(acc_list, ddof=1))
        print()
        print(np.mean(f1_list))
        print(np.std(f1_list, ddof=1))

        # Plot confusion matrix
        print(clf.classes_)
        # self.plot_confusion_matrix_custom(actual_targets, predicted_targets, clf.classes_)
        self.plot_confusion_matrix_custom(actual_targets, predicted_targets, ['POS_Other', 'POS_Self',
                                                                              'NEG_Self', 'NEG_Other'])


multi_class_modelling_all_emo = MulticlassModellingFourClass()
multi_class_modelling_all_emo.model_xgb_lofo_four_class()
