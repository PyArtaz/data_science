# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os

import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from itertools import cycle

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_model_structure(model_, detailed_model_name):
    if not os.path.exists('dataset/diagrams/'):
        os.makedirs('dataset/diagrams/')
    # creates the file model_plot.png with a diagram of the created modelmodel structure
    plot_model(model_, to_file='dataset/diagrams/' + detailed_model_name + '.png', show_shapes=True, show_layer_names=True)  # , rankdir='LR')  # for horizontal direction


# Plot the model Accuracy graph
def plot_training_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Training and Validation loss')

    # Plot the model Accuracy graph (Ideally, it should be Logarithmic shape)
    ax1.plot(history.history['accuracy'], 'r', linewidth=3.0, label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], 'b', linewidth=3.0, label='Validation Accuracy')
    ax1.legend(fontsize=12)
    ax1.set(xlabel='Epochs ', ylabel='Accuracy')
    ax1.set_title('Accuracy Curves')

    ax1.set_axisbelow(True)                                                     # Don't allow the axis to be on top of your data
    ax1.minorticks_on()                                                         # Turn on the minor TICKS, which are required for the minor GRID
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black')        # Customize the major grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')      # Customize the minor grid

    # Plot the model Loss graph (Ideally it should be Exponentially decreasing shape)
    ax2.plot(history.history['loss'], 'g', linewidth=3.0, label='Training Loss')
    ax2.plot(history.history['val_loss'], 'y', linewidth=3.0, label='Validation Loss')
    ax2.legend(fontsize=12)
    ax2.set(xlabel='Epochs ', ylabel='Loss')
    ax2.set_title('Loss Curves')

    ax2.set_axisbelow(True)                                                     # Don't allow the axis to be on top of your data
    ax2.minorticks_on()                                                         # Turn on the minor TICKS, which are required for the minor GRID
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')        # Customize the major grid
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')      # Customize the minor grid

    fig.tight_layout()
    plt.savefig(filename, dpi=fig.dpi)
    plt.show()


def plot_confusion_matrix(classes, predictions, labels):
    #labels = [str(class_label) for class_label in list(set(predictions))]

    # Confusion Matrix and Classification Report
    predicted_categories = tf.argmax(predictions, axis=1)
    cm = confusion_matrix(classes, predicted_categories)

    print('Confusion Matrix')
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print('Classification Report')
    print(classification_report(classes, predicted_categories, target_names=labels))


def plot_cm(model, X_test, y_test):
    # Generate confusion matrix
    matrix = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                   cmap=plt.cm.Blues,
                                   normalize='true')
    plt.title('Confusion matrix')
    plt.show()


def plot_roc(y_true, y_pred, class_labels):
    """
    Source: https://sites.google.com/site/nttrungmtwiki/home/it/data-science---python/multiclass-and-multilabel-roc-curve-plotting
    """
    # transforms strings in class_labels in integers for further processing
    integer_class_labels = list(range(len(class_labels)))

    # Binarize the output
    y_true = label_binarize(y_true, classes=integer_class_labels)       # label_binarize(y_true, classes=list(map(int, [1,2,3,6,4,5,6,3,2,3,2,3])))
    n_classes = y_true.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute fpr and tpr with roc_curve from the ytest true labels to the scores
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot each class  curve on single graph for multi-class one vs all classification
    colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    #fullpath = save_plot_path.joinpath(save_plot_path.stem +'_roc_curve.png')
    #plt.savefig(fullpath)
    plt.show()