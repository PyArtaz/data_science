# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import itertools

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import plots
import joblib
import glob
import seaborn as sns


class_labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I',
                'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_latest_model():
    # Directory path containing saved models
    directory = 'dataset/saved_model/'
    # necessary to load the latest saved model in the model folder
    list_of_files = glob.glob(directory + '*.sav')  # '*' means all if need specific format then e.g.: '*.sav'
    if len(list_of_files) == 0:
        print("Could not find any model in directory:", directory)
    else:
        latest_file = max(list_of_files, key=os.path.getmtime)

        # load trained model
        model = joblib.load(latest_file)

        return model

def load_df(test_directory):
    df = pd.read_csv(test_directory, index_col=0)

    print("\nLoaded dataset into dataframe:")
    print(df.head())

    X, y = df.iloc[:, :-1], df.iloc[:, -1].values
    y = y.astype(str)

    image_labels_dict = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R',
                         19:'S', 20:'T', 21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z', 27:'AE', 28:'OE', 29:'UE', 30:'SCH',
                         31:'1', 32:'2', 33:'3', 34:'4', 35:'5'}
    #y = [image_labels_dict.get(item, item) for item in y]

    return X, y

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

if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # define directory of unseen test data
    test_directory = 'dataset/hand_landmarks/Own/Own_landmarks_bb_squarePix.csv'  #  'dataset/hand_landmarks/asl_alphabet_train/asl_alphabet+digits_landmarks_bb.csv'  #  'dataset/hand_landmarks/Image/Image_landmarks_bb_squarePix_without_umlauts_or_digits.csv'
    #test_directory = 'dataset/hand_landmarks/asl_alphabet+digits_landmarks_bb.csv'

    # load test images
    X_test, y_test = load_df(test_directory)

    # load and create latest created model
    model = load_latest_model()

    # Generate predictions for samples
    predictions = model.predict(X_test)

    #scores = sklearn.model_selection.cross_val_score(model, X_test, y_test, cv=10)

    # Create StratifiedKFold object.
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = sklearn.model_selection.cross_val_score(model, X_test, y_test, cv=rskf)

    # Print the output.
    print('List of possible accuracy:', scores)
    print('\nMaximum Accuracy That can be obtained from this model is:',
          max(scores) * 100, '%')
    print('\nMinimum Accuracy:',
          min(scores) * 100, '%')
    print('\nOverall Accuracy:',
          np.mean(scores) * 100, '%')
    print('\nStandard Deviation is:', np.std(scores))



    # plot confusion matrix in unnormalized and normalized fashion
    plots.plot_cm(model, X_test, y_test)

    # plot multiclass ROC curve with
    #plots.plot_roc(y_test, predictions, class_labels)


