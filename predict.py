# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import preprocessing as prep
import plots
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import numpy as np


test_directory = 'dataset/digits_split/test'                                      # define directory of unseen test data


# class_labels = ['A', 'B', 'C', 'D', 'DEL', 'E', 'ENTER', 'F', 'G', 'H',
#                 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
#                 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def calculate_scores(y_true, predictions):
    y_pred = tf.argmax(predictions, axis=1)

    # define metrics to evaluate model performance
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print('F1 score: %f' % f1)


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    # load test images
    test_generator = prep.load_test_images(test_directory)

    # load and create latest created model
    #model = prep.load_latest_model()
    model = prep.load_model_from_name("dataset/saved_model/20220128-111210-pretrained_model_vgg-dataset_digits_split")  # 20220121-125432-pretrained_model_vgg-dataset_Own_complete_split")  #

    # tell the model what cost and optimization method to use
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
    # num_of_test_samples = test_generator.samples
    steps_per_epoch = test_generator.n // test_generator.batch_size

    # Generate predictions for samples
    predictions = model.predict(test_generator, steps=steps_per_epoch, verbose=1)

    # plot Confusion Matrix and Classification Report
    # plots.plot_confusion_matrix_cnn(test_generator.classes, predictions, class_labels)  # true_classes, predicted_classes, labels_of_classes

    # plot multiclass ROC curve with
    # plots.plot_roc(test_generator.classes, predictions, class_labels)

    calculate_scores(test_generator.classes, predictions)
