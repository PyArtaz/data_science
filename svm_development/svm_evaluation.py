import os
import glob
import numpy as np
import pandas as pd
import joblib
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import RepeatedStratifiedKFold
from svm_training import undersample_class_occurrences
from svm_plots import plot_cm, plot_multiclass_roc


# load the latest saved model in the model folder
def load_latest_model():
    # Directory path containing saved models
    directory = '../dataset/saved_model/'
    # necessary to load the latest saved model in the model folder
    list_of_files = glob.glob(directory + '*.sav')  # '*' means all if need specific format then e.g.: '*.sav'
    if len(list_of_files) == 0:
        print("Could not find any model in directory:", directory)
    else:
        latest_file = max(list_of_files, key=os.path.getmtime)
        print("Loaded model: " + str(latest_file))

        # load trained model
        model = joblib.load(latest_file)

        return model


# load another unseen dataset containing hand landmark coordinated and undersample the majority classes to test the model with balanced data
def load_test_dataframe(test_directory):
    df = pd.read_csv(test_directory, index_col=0)

    X, y = df.iloc[:, :-1], df.iloc[:, -1].values
    X, y = undersample_class_occurrences(X, y)
    y = y.astype(str)

    return X, y


# calculate metrics (accuracy, precision, recall and F1-score) to evaluate model performance
# the evaluation is performed by stratified 10-fold cross validation and gets repeated 10 times and averaged
def calculate_scores(model, X_test, y_test):
    # define metrics to evaluate model performance
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    # Create RepeatedStratifiedKFold object.
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = sklearn.model_selection.cross_validate(model, X_test, y_test, cv=rskf, scoring=scoring)

    # Print the output.
    for metric in scoring:
        dict_key = 'test_' + metric
        print('\nMaximum ', metric, ':', max(scores[dict_key]) * 100, '%')
        print('Minimum ', metric, ':', min(scores[dict_key]) * 100, '%')
        print('Overall ', metric, ':', np.mean(scores[dict_key]) * 100, '%')
        print('Standard Deviation of ', metric, ':', np.std(scores[dict_key]))


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # define directory of unseen test data
    dataset_path_test = '../dataset/hand_landmarks/asl_alphabet_train/asl_alphabet+digits_replaced_landmarks_bb.csv'

    # define class labels in the dataset
    class_labels = ['A', 'B', 'C', 'D', 'Del', 'E', 'Enter', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # load test images
    X_test, y_test = load_test_dataframe(dataset_path_test)

    # load and create latest created model
    model = load_latest_model()

    # Generate predictions for samples
    predictions = model.predict(X_test)

    # calculate metrics for model evaluation (accuracy, precision, recall, f1_score)
    calculate_scores(model, X_test, y_test)

    # plot confusion matrix in unnormalized and normalized fashion
    plot_cm(model, X_test, y_test)

    # plot multiclass ROC curve
    plot_multiclass_roc(model, X_test, y_test, class_labels, n_classes=len(np.unique(y_test)), figsize=(16, 10))
