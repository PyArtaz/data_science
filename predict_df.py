# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import collections
import os

import imblearn.under_sampling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import itertools

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import plots
import joblib
import glob
import seaborn as sns


class_labels = ['A', 'B', 'C', 'D', 'Del', 'E', 'Enter', 'F', 'G', 'H', 'I',
                'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_latest_model():
    # Directory path containing saved models
    directory = 'dataset/models/'
    # necessary to load the latest saved model in the model folder
    list_of_files = glob.glob(directory + 'model_Own_landmarks_bb_squarePix_Letters+Digits.sav')  # '*' means all if need specific format then e.g.: '*.sav'
    if len(list_of_files) == 0:
        print("Could not find any model in directory:", directory)
    else:
        latest_file = max(list_of_files, key=os.path.getmtime)
        print("Loaded model: " + str(latest_file))

        # load trained model
        model = joblib.load(latest_file)

        return model

def load_df(test_directory):
    df = pd.read_csv(test_directory, index_col=0)

    print("\nLoaded dataset into dataframe:")
    print(df.head())

    X, y = df.iloc[:, :-1], df.iloc[:, -1].values
    X, y = undersample_class_occurrences(X, y)
    y = y.astype(str)

    image_labels_dict = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R',
                         19:'S', 20:'T', 21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z', 27:'AE', 28:'OE', 29:'UE', 30:'SCH',
                         31:'1', 32:'2', 33:'3', 34:'4', 35:'5'}
    #y = [image_labels_dict.get(item, item) for item in y]

    return X, y

def undersample_class_occurrences(X, y):
    # define undersampling strategy
    under = imblearn.under_sampling.RandomUnderSampler(random_state=42)
    # fit and apply the transform
    X_resampled, y_resampled = under.fit_resample(X, y)

    class_occurrences = collections.Counter(y)
    class_occurrences_resampled = collections.Counter(y_resampled)

    print('Original dataset shape:', dict(sorted(class_occurrences.items(), key=lambda i: i[0])))
    print('Resampled dataset shape:', dict(sorted(class_occurrences_resampled.items(), key=lambda i: i[0])))

    return X_resampled, y_resampled

def plot_cm(model, X_test, y_test):
    # Generate confusion matrix
    matrix = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix')
    plt.tight_layout() # .tight_layout(top=0.977, bottom=0.044, left=0.008, right=0.992, hspace=0.2, wspace=0.2)
    plt.show()
    # ToDO: plt.tight_layout + savefig

def plot_cm_2(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=60)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_multiclass_roc(clf, X_test, y_test, class_labels, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], class_labels[i]))
    plt.legend(loc = 'lower right', prop = {'size': 7})
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


def calculate_scores(model, X_test, y_test):
    # define metrics to evaluate model performance
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    # Create StratifiedKFold object.
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = sklearn.model_selection.cross_validate(model, X_test, y_test, cv=rskf, scoring=scoring)

    # Print the output.
    for metric in scoring:
        # print('List of possible metrics that can be obtained from this model:', scores)
        dict_key = 'test_' + metric
        print('\nMaximum ', metric, ':', max(scores[dict_key]) * 100, '%')
        print('Minimum ', metric, ':', min(scores[dict_key]) * 100, '%')
        print('Overall ', metric, ':', np.mean(scores[dict_key]) * 100, '%')
        print('Standard Deviation of ', metric, ':', np.std(scores[dict_key]))


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # define directory of unseen test data
    # test_directory = 'dataset/hand_landmarks/asl_alphabet_train/asl_alphabet+digits_replaced_landmarks_bb.csv'  #  'dataset/hand_landmarks/asl_alphabet_train/asl_alphabet+digits_landmarks_bb.csv'  #  'dataset/hand_landmarks/Image/Image_landmarks_bb_squarePix_without_umlauts_or_digits.csv'
    #test_directory = 'dataset/hand_landmarks/asl_alphabet+digits_landmarks_bb.csv'
    test_directory = 'dataset/hand_landmarks/Own/Own_landmarks_bb_squarePix_Letters+Digits.csv'

    # load test images
    X_test, y_test = load_df(test_directory)

    # load and create latest created model
    model = load_latest_model()

    # Generate predictions for samples
    predictions = model.predict(X_test)

    # calculate metrics for model evaluation (accuracy, precision, recall, f1_score)
    calculate_scores(model, X_test, y_test)

    # plot confusion matrix in unnormalized and normalized fashion
    plots.plot_cm(model, X_test, y_test)        # ToDO: Werte ausblenden

    # plot multiclass ROC curve
    plot_multiclass_roc(model, X_test, y_test, class_labels, n_classes=len(np.unique(y_test)), figsize=(16, 10))


