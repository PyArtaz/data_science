# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os

import keras.regularizers
import scipy.spatial.distance
import sklearn.decomposition
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import tpot.builtins

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import joblib
import numpy as np
from sklearn.utils import class_weight
import sklearn.model_selection
import plots
import pandas as pd
from tpot import TPOTClassifier
import collections
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tpot.export_utils import set_param_recursive
import pickle

# activate for GPU acceleration
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


################################################################################################################################################################
# Training parameters
################################################################################################################################################################
# dataset_path = prep.dataset_path  # Enter the directory of the training images
dataset_path = 'dataset/hand_landmarks/Own/Own_landmarks_bb_squarePix_Digits_with_enter-space-del.csv'
class_labels = ['A', 'B', 'C', 'D', 'DEL', 'E', 'ENTER', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


################################################################################################################################################################
# Training FUNCTIONS
################################################################################################################################################################

# load image data and convert it to the right dimensions to train the model. Image data augmentation is uses to generate training data
def load_dataframe(dataset_path, resampling='over'):
    df = pd.read_csv(dataset_path, index_col=0)

    print("\nLoaded dataset into dataframe:")
    print(df.head())

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = y.astype(str)

    # plot_class_occurrences(y)

    if resampling == 'over':
        X_resampled, y_resampled = oversample_class_occurrences(X, y)
    elif resampling == 'under':
        X_resampled, y_resampled = undersample_class_occurrences(X, y)
    else:
        pass

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, shuffle=True, stratify=y_resampled)

    return X_train, X_test, y_train, y_test


def plot_class_occurrences(class_list):
    # visualize the target variable
    import seaborn as sns
    import matplotlib.pyplot as plt

    # class_labels = [str(class_label) for class_label in list(set(class_list))]
    string_ints = [str(int) for int in class_list.values]
    class_labels = pd.Series(string_ints).drop_duplicates().tolist()
    g = sns.countplot(class_list)
    g.set_xticklabels(labels=class_labels)
    plt.show()


def oversample_class_occurrences(X, y):


    # define oversampling strategy
    over = SMOTE()  # RandomOverSampler(random_state=42, sampling_strategy=mean_class_occurrence/max_class_occurrence)
    # fit and apply the transform
    X_resampled, y_resampled = over.fit_resample(X, y)

    class_occurrences = collections.Counter(y)
    class_occurrences_resampled = collections.Counter(y_resampled)
    max_class_occurrence = class_occurrences[max(class_occurrences, key=class_occurrences.get)]
    min_class_occurrence = class_occurrences[max(class_occurrences, key=class_occurrences.get)]
    mean_class_occurrence = int(len(y)/len(class_occurrences)+0.5)       # number of data points divided by number of classes (added 0.5 for correct int rounding)

    print('Original dataset shape:', dict(sorted(class_occurrences.items(), key=lambda i: i[0])))
    print('Resample dataset shape:', dict(sorted(class_occurrences_resampled.items(), key=lambda i: i[0])))

    return X_resampled, y_resampled


def undersample_class_occurrences(X, y):
    # define undersampling strategy
    under = RandomUnderSampler(random_state=42)
    # fit and apply the transform
    X_resampled, y_resampled = under.fit_resample(X, y)

    class_occurrences = collections.Counter(y)
    class_occurrences_resampled = collections.Counter(y_resampled)

    print('Original dataset shape:', dict(sorted(class_occurrences.items(), key=lambda i: i[0])))
    print('Resample dataset shape:', dict(sorted(class_occurrences_resampled.items(), key=lambda i: i[0])))

    return X_resampled, y_resampled


def print_statistics(y_test, predictions):
    # print('ROCAUC score:', "%.2f" % roc_auc_score(y_test, predictions, multi_class='ovr'))
    print('Accuracy score:', "%.2f" % (accuracy_score(y_test, predictions) * 100))
    print('Precision score:', "%.2f" % (precision_score(y_test, predictions, average='weighted') * 100))
    print("Recall:", "%.2f" % (recall_score(y_test, predictions, average='weighted') * 100))
    print('F1 score:', "%.2f" % (f1_score(y_test, predictions, average='weighted') * 100))

    # print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # ToDo: plot Confusion Matrix and ROC
    # class_labels = [str(class_label) for class_label in list(set(predictions))]
    # plots.plot_confusion_matrix(y_test, predictions, class_labels)



def perform_svc_grid_search(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10], 'degree': [3, 5, 7], 'gamma': [1, 5, 10]}  # , 'kernel': ['rbf', 'poly', 'sigmoid']}

    # we can add class_weight='balanced' to add panalize mistake
    model = SVC(class_weight='balanced', probability=True, kernel='poly')

    grid = GridSearchCV(model, param_grid, refit=True, verbose=4, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_)

    return grid


def perform_tpot_search(X_train, X_test, y_train, y_test):
    # define search
    tpot = TPOTClassifier(generations=2, population_size=2, cv=10,
                                    random_state=42, verbosity=2, n_jobs=-1)

    # tpot = TPOTClassifier(generations=50, population_size=100, cv=10,
    #                       random_state=42, verbosity=2, n_jobs=-1)

    # perform the search
    tpot.fit(X_train, y_train)

    # export the best model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    tpot.export('tpot_model_' + timestr + '.py')

    # print(tpot.score(X_test, y_test))

    return tpot


def save_trained_model(model):
    # create model filename
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'dataset/saved_model/' + timestr + '_model'

    # now you can save it to a file
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(model, f)

    # alternative to pickle
    joblib.dump(model, filename + '.sav')

    print('Saved model to disk: ' + filename)

if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    start_time = time.time()

    # load hand_landmark data
    X_train, X_test, y_train, y_test = load_dataframe(dataset_path, resampling='over')

    # either: search for suitable model by manual grid search or automated evolutionary search
    # model = perform_svc_grid_search(X_train.values, y_train)                              # manual grid search
    # model = perform_tpot_search(X_train.values, X_test.values, y_train, y_test)           # automated evolutionary algorithm search

    # or: use previously found optimal model
    model = SVC(class_weight='balanced', probability=True, kernel='poly', degree=3, C=0.1, gamma=5)      # was best model so far

    model.fit(X_train.values, y_train)

    save_trained_model(model)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    predictions = model.predict(X_test)  # check performance

    # ToDo: 10- fold cross-validation

    print_statistics(y_test, predictions)

    # plot confusion matrix in unnormalized and normalized fashion
    plots.plot_cm(model, X_test, y_test)

