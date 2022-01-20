# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os

import matplotlib.pyplot as plt
import sklearn.metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    #y = y.astype(str)

    image_labels_dict = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R',
                         19:'S', 20:'T', 21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z', 27:'AE', 28:'OE', 29:'UE', 30:'SCH',
                         31:'1', 32:'2', 33:'3', 34:'4', 35:'5'}

    y = [image_labels_dict.get(item, item) for item in y]

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


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # define directory of unseen test data
    test_directory = 'dataset/hand_landmarks/Letters_Artur/Letters_Artur_landmarks_bb_squarePix_flip.csv'  #  'dataset/hand_landmarks/asl_alphabet_train/asl_alphabet+digits_landmarks_bb.csv'  #  'dataset/hand_landmarks/Image/Image_landmarks_bb_squarePix_without_umlauts_or_digits.csv'
    #test_directory = 'dataset/hand_landmarks/asl_alphabet+digits_landmarks_bb.csv'

    # load test images
    X_test, y_test = load_df(test_directory)

    # load and create latest created model
    model = load_latest_model()

    # Generate predictions for samples
    predictions = model.predict(X_test)

    # plot Confusion Matrix and Classification Report
    plot_cm(model, X_test, y_test)

    # plot multiclass ROC curve with
    #plots.plot_roc(y_test, predictions, class_labels)


