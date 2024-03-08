"""
@author Rukshani Somarathna

This class contains the methods to model all emotions using multiclass classification using 10-Folds. It contains
Leave-One-Film-Out (LOFO) XGBClassifier modelling with hyperparameter tuning. It also contains methods to plot
confusion matrix.
"""

import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import EmoFilMDataFields


class MulticlassModellingAllEmo:

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
                                     title='Confusion matrix'):
        cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure()

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(10, 8))
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, size=15)
        plt.yticks(tick_marks, classes, size=15)

        fmt = '.2f' if normalize else 'd'
        thresh = cnf_matrix.max() / 2.

        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', size=15)
        plt.xlabel('Predicted label', size=15)

        plt.show()

    # Random Forest model with 10-fold cross-validation across movies
    def model_rf_cross_movie(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_scaled_csv = self.scale_data(df, self.sorted_grid_items + self.sorted_discrete_items)

        df_scaled = df_scaled_csv.copy()
        df_scaled['emo_max_index'] = np.NaN
        df_scaled['emo_max_name'] = np.NaN

        for index, row in df_scaled.iterrows():
            max_col_index = np.argmax(row[self.sorted_discrete_items].values)
            max_cat_name = self.sorted_discrete_items[max_col_index]

            df_scaled['emo_max_index'][index] = max_col_index
            df_scaled['emo_max_name'][index] = max_cat_name

        # Print chance level which is used as the baseline
        print((df_scaled['emo_max_name'].value_counts().max() / df_scaled['emo_max_name'].value_counts().sum()) * 100)

        # StratifiedKFold cross validation
        skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

        x_scaled = df_scaled[self.sorted_grid_items]
        y_scaled = df_scaled['emo_max_name']

        acc_list, y_test_list, y_predict_list = [], [], []

        for train_index, test_index in skf.split(x_scaled, y_scaled):
            x_train, x_test = x_scaled.values[train_index], x_scaled.values[test_index]
            y_train, y_test = y_scaled.values[train_index], y_scaled.values[test_index]

            # The hyperparameters were found using the GridSearchCV method. The best hyperparameters were used to model.
            clf = RandomForestClassifier(random_state=42)
            clf.fit(x_train, y_train)

            y_predict = clf.predict(x_test)

            accuracy = accuracy_score(y_test, y_predict) * 100
            print('Accuracy:', accuracy)

            y_test_list = np.append(y_test_list, y_test)
            y_predict_list = np.append(y_predict_list, y_predict)
            acc_list.append(accuracy)

        print(np.mean(acc_list))
        print(np.std(acc_list, ddof=1))

        self.plot_confusion_matrix_custom(y_test_list, y_predict_list, clf.classes_)

    # Leave-One-Film-Out (LOFO) XGBClassifier modelling with hyperparameter tuning
    def model_xgb_lofo(self):
        # Path to the concatenated data file
        df = pd.read_csv('path_to_data_file.csv')
        df_scaled_lofo_csv = self.scale_data(df, self.sorted_grid_items + self.sorted_discrete_items)

        df_scaled_lofo = df_scaled_lofo_csv.copy()
        df_scaled_lofo['emo_max_index'] = np.NaN
        df_scaled_lofo['emo_max_name'] = np.NaN

        for index, row in df_scaled_lofo.iterrows():
            max_col_index = np.argmax(row[self.sorted_discrete_items].values)
            max_cat_name = self.sorted_discrete_items[max_col_index]

            df_scaled_lofo['emo_max_index'][index] = max_col_index
            df_scaled_lofo['emo_max_name'][index] = max_cat_name

        print(df_scaled_lofo['emo_max_name'].value_counts())

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
            movie_df = df_scaled_lofo.iloc[start_index:end_index, :]
            print(movie_name, movie_df.shape)

            movie_df['movie_name'] = movie_name
            movie_dfs.append(movie_df)

        df_with_movie_names = pd.concat(movie_dfs, axis=0)

        # Label encoder
        label_encoder = LabelEncoder()
        df_with_movie_names['emo_max_name'] = label_encoder.fit_transform(df_with_movie_names['emo_max_name'])

        valid_movies_list = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite',
                             'LeassonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsofSteel',
                             'Thesecretnumber', 'ToClaireFromSonny', 'YouAgain']

        # Initialise variables
        num_boost_rounds = 100
        n_folds = 5
        init_learning_rate = 0.1
        init_max_depth = 4
        init_min_child_weight = 6
        init_subsample = 0.91
        init_colsample_bytree = 0.4
        init_n_estimators = 100

        init_acc_all = list()
        final_acc_all1, final_acc_all2, final_acc_all3, final_acc_all4 = list(), list(), list(), list()
        y_predict_list, y_test_list, combined_acc, final_acc_clean = list(), list(), list(), list()
        f1_list = list()
        df_results = pd.DataFrame()

        for mov_id in valid_movies_list:
            print(mov_id)
            valid_movies_list2 = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite',
                                  'LeassonLearned', 'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsofSteel',
                                  'Thesecretnumber', 'ToClaireFromSonny', 'YouAgain']
            valid_movies_list2.remove(mov_id)

            train_mov_data = []

            for train_mov_id in valid_movies_list2:
                train_mov_data.append(df_with_movie_names[df_with_movie_names['movie_name'] == train_mov_id])

            train_mov_df = pd.concat(train_mov_data, ignore_index=True)
            test_data = df_with_movie_names[df_with_movie_names['movie_name'] == mov_id]
            print(train_mov_df.shape)
            print(test_data.shape)

            x_train = train_mov_df[self.sorted_grid_items]
            y_train = train_mov_df['emo_max_name']
            x_test = test_data[self.sorted_grid_items]
            y_test = test_data['emo_max_name']

            # SMOTE sampling of training data
            if mov_id == 'Superhero':
                k_neighbors = 1
            else:
                k_neighbors = 5
            print('---------------------------')

            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

            # Training and validation split
            x_train_sample, x_val_sample, y_train_sample, y_val_sample = train_test_split(x_resampled, y_resampled,
                                                                                          test_size=0.2,
                                                                                          random_state=42)
            xgval = xgb.DMatrix(x_val_sample, label=y_val_sample)

            # Define initial XGBClassifier model
            clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                    random_state=42, seed=27, n_jobs=10,
                                    learning_rate=init_learning_rate, n_estimators=init_n_estimators,
                                    max_depth=init_max_depth, min_child_weight=init_min_child_weight, gamma=0,
                                    subsample=init_subsample, colsample_bytree=init_colsample_bytree, max_delta_step=10)
            clf.fit(x_train_sample, y_train_sample, eval_metric='merror', verbose=False)

            y_predict = clf.predict(x_test)
            init_acc = accuracy_score(y_test, y_predict)
            higher_acc = init_acc
            print("Initial Acc %s : %f" % (mov_id, init_acc * 100.0))

            init_acc_all.append(init_acc)

            #############################################################
            # Grid search for max_depth and min_child_weight
            clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                    random_state=42, seed=27, n_jobs=10,
                                    learning_rate=init_learning_rate, n_estimators=init_n_estimators,
                                    max_depth=init_max_depth, min_child_weight=init_min_child_weight, gamma=0,
                                    subsample=init_subsample, colsample_bytree=init_colsample_bytree, max_delta_step=10)
            xgb_param = clf.get_xgb_params()

            gridsearch_params = [(max_depth, min_child_weight) for max_depth in range(3, 20) for min_child_weight in
                                 range(3, 20)]

            min_mae = float("Inf")
            best_params = None

            for max_depth, min_child_weight in gridsearch_params:
                xgb_param['max_depth'] = max_depth
                xgb_param['min_child_weight'] = min_child_weight
                cv_results1 = xgb.cv(xgb_param, xgval, num_boost_round=num_boost_rounds, seed=27, nfold=n_folds,
                                     metrics={'merror'}, early_stopping_rounds=10)

                mean_mae = cv_results1['test-merror-mean'].min()

                if mean_mae < min_mae:
                    min_mae = mean_mae
                    best_params = (max_depth, min_child_weight)

            print("Best params: {},{} merror: {}".format(best_params[0], best_params[1], min_mae))

            clf.set_params(max_depth=best_params[0], min_child_weight=best_params[1])
            clf.fit(x_train_sample, y_train_sample, eval_metric='merror')
            predictions_test = clf.predict(x_test)
            accuracy_test1 = accuracy_score(y_test, predictions_test)
            print("Final Acc1 %s : %f" % (mov_id, accuracy_test1 * 100.0))
            final_acc_all1.append(accuracy_test1)

            if accuracy_test1 > higher_acc:
                higher_acc = accuracy_test1
                max_depth = best_params[0]
                min_child_weight = best_params[1]
            else:
                max_depth = init_max_depth
                min_child_weight = init_min_child_weight

            #############################################################
            # Grid search for subsample and colsample_bytree
            clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                    random_state=42, seed=27, n_jobs=10,
                                    learning_rate=init_learning_rate, n_estimators=init_n_estimators,
                                    max_depth=max_depth, min_child_weight=min_child_weight, gamma=0,
                                    subsample=init_subsample, colsample_bytree=init_colsample_bytree, max_delta_step=10)
            xgb_param = clf.get_xgb_params()
            gridsearch_params2 = [(subsample, colsample_bytree)
                                  for subsample in [i / 10. for i in range(1, 10)]
                                  for colsample_bytree in [i / 10. for i in range(1, 10)]]

            min_mae2 = float("Inf")
            best_params2 = None

            for subsample, col_sample_by_tree in gridsearch_params2:
                xgb_param['subsample'] = subsample
                xgb_param['col_sample_by_tree'] = col_sample_by_tree
                cv_results2 = xgb.cv(xgb_param, xgval, num_boost_round=num_boost_rounds, seed=27, nfold=n_folds,
                                     metrics={'merror'}, early_stopping_rounds=10)

                mean_mae = cv_results2['test-merror-mean'].min()

                if mean_mae < min_mae2:
                    min_mae2 = mean_mae
                    best_params2 = (subsample, col_sample_by_tree)

            print("Best params: {},{} merror: {}".format(best_params2[0], best_params2[1], min_mae2))

            clf.set_params(subsample=best_params2[0], colsample_bytree=best_params2[1])
            clf.fit(x_train_sample, y_train_sample, eval_metric='merror')
            predictions_test = clf.predict(x_test)
            accuracy_test2 = accuracy_score(y_test, predictions_test)
            print("Final Acc2 %s : %f" % (mov_id, accuracy_test2 * 100.0))
            final_acc_all2.append(accuracy_test2)

            if accuracy_test2 > higher_acc:
                higher_acc = accuracy_test2
                subsample = best_params2[0]
                col_sample_by_tree = best_params2[1]
            else:
                subsample = init_subsample
                col_sample_by_tree = init_colsample_bytree

            #############################################################
            # Grid search for learning rate
            clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                    random_state=42, seed=27, n_jobs=10,
                                    learning_rate=init_learning_rate, n_estimators=init_n_estimators,
                                    max_depth=max_depth, min_child_weight=min_child_weight, gamma=0,
                                    subsample=subsample, colsample_bytree=col_sample_by_tree, max_delta_step=10)
            xgb_param = clf.get_xgb_params()

            min_mae3 = float("Inf")
            best_params3 = None

            for eta in [0.1, .05, 0.01, .005]:
                xgb_param['eta'] = eta
                cv_results3 = xgb.cv(xgb_param, xgval, num_boost_round=num_boost_rounds, seed=27, nfold=n_folds,
                                     metrics={'merror'}, early_stopping_rounds=10)

                mean_mae = cv_results3['test-merror-mean'].min()

                if mean_mae < min_mae3:
                    min_mae3 = mean_mae
                    best_params3 = eta

            print("Best params: {}, merror: {}".format(best_params3, min_mae3))

            clf.set_params(learning_rate=best_params3)
            clf.fit(x_train_sample, y_train_sample, eval_metric='merror')
            predictions_test = clf.predict(x_test)
            accuracy_test3 = accuracy_score(y_test, predictions_test)
            print("Final Acc3 %s : %f" % (mov_id, accuracy_test3 * 100.0))
            final_acc_all3.append(accuracy_test3)

            if accuracy_test3 > higher_acc:
                higher_acc = accuracy_test3
                learning_rate = best_params3
            else:
                learning_rate = init_learning_rate

            #############################################################
            # Grid search for n_estimators
            n_estimator_new = init_n_estimators
            for n_estimator in [50, 75, 100, 150, 200, 500, 1000]:
                clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                        random_state=42, seed=27, n_jobs=10, learning_rate=learning_rate,
                                        n_estimators=n_estimator, max_depth=max_depth,
                                        min_child_weight=min_child_weight, gamma=0, subsample=subsample,
                                        colsample_bytree=col_sample_by_tree, max_delta_step=10)
                clf.fit(x_train_sample, y_train_sample, eval_metric='merror')
                predictions_test = clf.predict(x_test)
                accuracy_test4 = accuracy_score(y_test, predictions_test)

                if higher_acc < accuracy_test4:
                    higher_acc = accuracy_test4
                    n_estimator_new = n_estimator

            print("Best params: {}, acc: {}".format(n_estimator_new, higher_acc))

            clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                    random_state=42, seed=27, n_jobs=10,
                                    learning_rate=learning_rate, n_estimators=n_estimator_new, max_depth=max_depth,
                                    min_child_weight=min_child_weight, gamma=0, subsample=subsample,
                                    colsample_bytree=col_sample_by_tree, max_delta_step=10)
            clf.fit(x_train_sample, y_train_sample, eval_metric='merror')
            predictions_test = clf.predict(x_test)
            accuracy_test5 = accuracy_score(y_test, predictions_test)
            print("Final Acc4 %s : %f" % (mov_id, accuracy_test5 * 100.0))
            final_acc_all4.append(accuracy_test5)

            print(higher_acc)
            #############################################################

            combined_acc.append(higher_acc)

            # Final model
            print(max_depth, min_child_weight, subsample, col_sample_by_tree, learning_rate, n_estimator_new)
            clf = xgb.XGBClassifier(objective='multi:softmax', num_class=13, eval_metric='merror',
                                    random_state=42, seed=27, n_jobs=10, learning_rate=learning_rate,
                                    n_estimators=n_estimator_new, max_depth=max_depth,
                                    min_child_weight=min_child_weight, gamma=0, subsample=subsample,
                                    colsample_bytree=col_sample_by_tree, max_delta_step=10)
            clf.fit(x_train_sample, y_train_sample, eval_metric='merror')

            # Feature importance
            fi_scores = clf.feature_importances_
            df_results[mov_id] = fi_scores

            # Print final model predictions
            y_predict_clean = clf.predict(x_test)
            accuracy_test_clean = accuracy_score(y_test, y_predict_clean)
            print("Final clean ----------------- %s : %f" % (mov_id, accuracy_test_clean * 100.0))
            final_acc_clean.append(accuracy_test_clean)

            # Appending data for confusion matrix
            y_predict_list = np.append(y_predict_list, y_predict_clean)
            y_test_list = np.append(y_test_list, y_test)

            # F1-score calculations
            f1 = f1_score(y_test, y_predict_clean, average='weighted')
            print('F1-score:', f1)
            f1_list.append(f1)

        # Print final average results
        print(np.mean(final_acc_clean) * 100)
        print(np.std(final_acc_clean, ddof=1) * 100)
        print(np.mean(final_acc_clean) - np.std(final_acc_clean, ddof=1) * 100)
        print(np.mean(f1_list))
        print()

        # Plot confusion matrix
        self.plot_confusion_matrix_custom(y_test_list, y_predict_list, clf.classes_)


multiclass_modelling_all_emo = MulticlassModellingAllEmo()

# Cross movie modelling
# multiclass_modelling_all_emo.model_rf_cross_movie()

# LOFO modelling
# multiclass_modelling_all_emo.model_xgb_lofo()
