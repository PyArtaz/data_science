import os
import time
import joblib
import collections
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sklearn.decomposition
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
import svm_plots


################################################################################################################################################################
# Training parameters
################################################################################################################################################################
dataset_path_train = '../dataset/hand_landmarks/Own/Own_landmarks_bb_squarePix_Letters+Digits.csv'       # Enter the directory of the training images

################################################################################################################################################################
# Training FUNCTIONS
################################################################################################################################################################

# load hand landmark data and convert it to the right dimensions to train the model
def load_dataframe(dataset_path, resampling='over'):
    df = pd.read_csv(dataset_path, index_col=0)

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


# display the class occurrences in a bar plot
def plot_class_occurrences(class_list):
    # visualize the target variable
    import seaborn as sns
    import matplotlib.pyplot as plt

    string_ints = [str(int) for int in class_list.values]                   # convert digits occurring in class labels to strings
    class_labels = pd.Series(string_ints).drop_duplicates().tolist()
    g = sns.countplot(class_list)
    g.set_xticklabels(labels=class_labels)
    plt.show()


# oversample the minority classes by synthesizing new hand landmarks from the existing ones using SMOTE algorithm
def oversample_class_occurrences(X, y):
    # define oversampling strategy
    over = SMOTE(random_state=42)
    # fit and apply the transform
    X_resampled, y_resampled = over.fit_resample(X, y)

    class_occurrences = collections.Counter(y)
    class_occurrences_resampled = collections.Counter(y_resampled)

    print('Original dataset shape:', dict(sorted(class_occurrences.items(), key=lambda i: i[0])))
    print('Resample dataset shape:', dict(sorted(class_occurrences_resampled.items(), key=lambda i: i[0])))

    return X_resampled, y_resampled


# undersample the majority classes by deleting examples from the majority classes
def undersample_class_occurrences(X, y):
    # define undersampling strategy
    under = RandomUnderSampler(random_state=42)
    # fit and apply the transform
    X_resampled, y_resampled = under.fit_resample(X, y)

    class_occurrences = collections.Counter(y)
    class_occurrences_resampled = collections.Counter(y_resampled)

    print('Original dataset shape:\n', dict(sorted(class_occurrences.items(), key=lambda i: i[0])))
    print('Resampled dataset shape:\n', dict(sorted(class_occurrences_resampled.items(), key=lambda i: i[0])))

    return X_resampled, y_resampled


# calculate metrics (accuracy, precision, recall and F1-score) to evaluate model performance
def print_statistics(y_test, predictions):
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print('Accuracy score:', "%.2f" % (accuracy_score(y_test, predictions) * 100))
    print('Precision score:', "%.2f" % (precision_score(y_test, predictions, average='weighted') * 100))
    print("Recall:", "%.2f" % (recall_score(y_test, predictions, average='weighted') * 100))
    print('F1 score:', "%.2f" % (f1_score(y_test, predictions, average='weighted') * 100))


# save model to disk
def save_trained_model(model):
    folder_path = '../dataset/saved_model/'
    # check if folder already exists and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # create model filename
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = folder_path + timestr + '_model'

    # save it to a file
    joblib.dump(model, filename + '.sav')

    print('Saved model to disk: ' + filename)


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    start_time = time.time()

    # load hand_landmark data
    X_train, X_test, y_train, y_test = load_dataframe(dataset_path_train, resampling='over')

    # was best model found by TPOT search
    model = SVC(class_weight='balanced', probability=True, kernel='poly', degree=3, C=0.1, gamma=5)

    # Train the model
    model.fit(X_train.values, y_train)

    pred_time = time.time() - start_time
    print("\nRuntime", pred_time, "s")

    save_trained_model(model)

    # plot confusion matrix after training in unnormalized and normalized fashion
    svm_plots.plot_cm(model, X_test, y_test)

    # check performance on unseen test data
    predictions = model.predict(X_test)

    # calculate metrics to evaluate model performance on unseen test data
    print_statistics(y_test, predictions)
